"""Boundary preparation and differentiable physics operators for attention models."""

from __future__ import annotations

import torch

from src.numerics.differential import autograd_differential_terms

from src.losses.physics import (
    elliptic_grid_pde_residual,
    grid_line_smoothness_loss,
    jacobian_positive_loss,
    elliptic_grid_normalized_pde_loss,
    stencil_queries,
)
from src.data.backend.parametric_torch import evaluate_curves_torch, evaluate_surfaces_torch


FACE_NAMES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


def build_boundary_inputs(geometries, grid_shape, device, dtype, sample_counts=None):
    """Sample directed curves/surfaces into the attention encoder input format."""
    dimension = len(grid_shape)
    batch = len(geometries)
    if dimension == 2:
        nx, ny = map(int, grid_shape)
        counts = tuple(sample_counts or (nx, ny, nx, ny))
        if len(counts) != 4 or min(counts) < 2:
            raise ValueError("2-D boundary_sample_counts needs four values >= 2")
        edges = []
        for name, count in zip(("bottom", "right", "top", "left"), counts):
            t = torch.linspace(0, 1, int(count), device=device, dtype=dtype)
            query = t[None].expand(batch, -1)
            edges.append(evaluate_curves_torch([getattr(g, name) for g in geometries], query))
        return tuple(edges)
    if dimension != 3:
        raise ValueError("Only 2-D and 3-D boundaries are supported")
    nx, ny, nz = map(int, grid_shape)
    default = ((ny, nz), (ny, nz), (nx, nz), (nx, nz), (nx, ny), (nx, ny))
    counts = tuple(tuple(map(int, pair)) for pair in (sample_counts or default))
    if len(counts) != 6 or any(len(pair) != 2 or min(pair) < 2 for pair in counts):
        raise ValueError("3-D boundary_sample_counts needs six [nu, nv] pairs >= 2")
    faces = []
    for name, (nu, nv) in zip(FACE_NAMES, counts):
        u = torch.linspace(0, 1, nu, device=device, dtype=dtype)
        v = torch.linspace(0, 1, nv, device=device, dtype=dtype)
        uu, vv = torch.meshgrid(u, v, indexing="ij")
        uq = uu.reshape(1, -1).expand(batch, -1)
        vq = vv.reshape(1, -1).expand(batch, -1)
        value = evaluate_surfaces_torch([getattr(g, name) for g in geometries], uq, vq)
        faces.append(value.reshape(batch, nu, nv, 3))
    return tuple(faces)


def predict_attention(model, boundary_inputs, logical, geometries):
    """Evaluate an attention HMONet and concatenate physical coordinates."""
    outputs = model(
        *boundary_inputs,
        *logical.unbind(-1),
        boundary_geometry=geometries,
    )
    return torch.cat(outputs[: logical.shape[-1]], -1)


def sample_boundary_logical_points(batch_size, sample_count, dimension, device,
                                   dtype, offset=0.0, tangential_margin=0.05):
    """Sample every logical side/face for autograd boundary physics losses.

    ``sample_count`` is the total number of points per geometry.  Points are
    distributed as evenly as possible over all 2*dimension boundary entities.
    A zero offset evaluates the exact computational boundary; a positive value
    evaluates an inward boundary layer while preserving the same side/face.
    """
    batch_size, sample_count, dimension = map(
        int, (batch_size, sample_count, dimension)
    )
    if batch_size < 1 or sample_count < 2 * dimension or dimension not in (2, 3):
        raise ValueError(
            "boundary physics sampling needs a positive batch, dimension 2/3, "
            "and at least 2*dimension points"
        )
    offsets = torch.as_tensor(offset, device=device, dtype=dtype).flatten()
    if offsets.numel() == 1:
        offsets = offsets.expand(dimension)
    if offsets.numel() != dimension or bool(
        ((offsets < 0) | (offsets >= 0.5)).any()
    ):
        raise ValueError(
            "boundary_physics_offset must be scalar or one value per axis in [0, 0.5)"
        )
    margins = torch.as_tensor(
        tangential_margin, device=device, dtype=dtype
    ).flatten()
    if margins.numel() == 1:
        margins = margins.expand(dimension)
    if margins.numel() != dimension or bool(
        ((margins < 0) | (margins >= 0.5)).any()
    ):
        raise ValueError(
            "boundary_physics_tangential_margin must be scalar or one value "
            "per axis in [0, 0.5)"
        )
    entity_count = 2 * dimension
    base, remainder = divmod(sample_count, entity_count)
    parts = []
    for entity in range(entity_count):
        count = base + int(entity < remainder)
        values = torch.rand(1, count, dimension, device=device, dtype=dtype).expand(
            batch_size, -1, -1
        ).clone()
        values.mul_(1.0 - 2.0 * margins).add_(margins)
        axis, upper = divmod(entity, 2)
        values[..., axis] = 1.0 - offsets[axis] if upper else offsets[axis]
        parts.append(values)
    return torch.cat(parts, dim=1).requires_grad_(True)


def autograd_elliptic_residual(prediction, logical):
    """Compute the inverse-form elliptic residual through exact autograd."""
    return autograd_differential_terms(prediction, logical)["residual"]


def autograd_physics_losses(prediction, logical, jacobian_minimum=1e-4,
                            epsilon=1e-8):
    """Compute normalized PDE, positive-Jacobian and smoothness losses once."""
    terms = autograd_differential_terms(prediction, logical)
    residual, jacobian, metric = (
        terms["residual"], terms["jacobian"], terms["metric"]
    )
    trace = metric.diagonal(dim1=-2, dim2=-1).sum(-1)
    pde = (residual.square().sum(-1) / trace.square().clamp_min(float(epsilon))).mean()
    determinant = torch.linalg.det(jacobian)
    jacobian_loss = torch.relu(float(jacobian_minimum) - determinant).square().mean()
    smoothness = prediction.new_zeros(())
    dimension = prediction.shape[-1]
    for i in range(dimension):
        smoothness = smoothness + terms["second"][i][i].square().sum(-1).mean()
        for j in range(i + 1, dimension):
            smoothness = smoothness + 2 * terms["second"][i][j].square().sum(-1).mean()
    return {
        "pde_normalized": pde,
        "jacobian": jacobian_loss,
        "smoothness": smoothness,
        "minimum_jacobian": determinant.detach().min(),
    }


def autograd_normalized_pde_loss(prediction, logical, epsilon=1e-8):
    """Compute only the normalized elliptic PDE loss through autograd."""
    terms = autograd_differential_terms(prediction, logical)
    trace = terms["metric"].diagonal(dim1=-2, dim2=-1).sum(-1)
    return (
        terms["residual"].square().sum(-1)
        / trace.square().clamp_min(float(epsilon))
    ).mean()


def autograd_pde_loss(model, boundary_inputs, geometries, batch_size,
                      sample_count, dimension, device, dtype, margin=0.0):
    """Draw common interior points and return the autograd PDE loss."""
    margin = float(margin)
    if not 0 <= margin < 0.5:
        raise ValueError("autograd interior_margin must lie in [0, 0.5)")
    if margin == 0:
        margin = torch.finfo(dtype).eps
    logical = torch.rand(batch_size, sample_count, dimension, device=device, dtype=dtype)
    logical = (margin + (1 - 2*margin) * logical).requires_grad_(True)
    prediction = predict_attention(model, boundary_inputs, logical, geometries)
    residual = autograd_elliptic_residual(prediction, logical)
    return residual.square().sum(-1).mean()


def finite_difference_pde_loss(model, boundary_inputs, geometries, batch_size,
                               sample_count, steps, device, dtype):
    """Return the same elliptic PDE loss using centered finite differences."""
    h = torch.as_tensor(steps, device=device, dtype=dtype)
    dimension = h.numel()
    centers = h + torch.rand(sample_count, dimension, device=device, dtype=dtype) * (1 - 2*h)
    stencil = stencil_queries(centers, h)
    query = stencil.reshape(1, -1, dimension).expand(batch_size, -1, -1)
    values = predict_attention(model, boundary_inputs, query, geometries)
    values = values.reshape(batch_size, sample_count, stencil.shape[-2], dimension)
    return elliptic_grid_pde_residual(values, h).square().sum(-1).mean()


def finite_difference_physics_losses(model, boundary_inputs, geometries,
                                     batch_size, sample_count, steps, device,
                                     dtype, jacobian_minimum=1e-4):
    """Compute all three derivative losses from one centered-difference stencil."""
    h = torch.as_tensor(steps, device=device, dtype=dtype)
    dimension = h.numel()
    centers = h + torch.rand(sample_count, dimension, device=device, dtype=dtype) * (1 - 2*h)
    return finite_difference_physics_losses_at_points(
        model,
        boundary_inputs,
        geometries,
        batch_size,
        centers,
        h,
        jacobian_minimum,
    )


def finite_difference_normalized_pde_loss(
    model, boundary_inputs, geometries, batch_size, sample_count, steps,
    device, dtype,
):
    """Compute only the normalized PDE loss from centered differences."""
    h = torch.as_tensor(steps, device=device, dtype=dtype)
    dimension = h.numel()
    centers = h + torch.rand(
        sample_count, dimension, device=device, dtype=dtype
    ) * (1 - 2 * h)
    stencil = stencil_queries(centers, h)
    query = stencil.reshape(1, -1, dimension).expand(batch_size, -1, -1)
    values = predict_attention(model, boundary_inputs, query, geometries)
    values = values.reshape(batch_size, sample_count, stencil.shape[-2], dimension)
    return elliptic_grid_normalized_pde_loss(values, h)


def finite_difference_physics_losses_at_points(
    model,
    boundary_inputs,
    geometries,
    batch_size,
    centers,
    steps,
    jacobian_minimum=1e-4,
):
    """Evaluate centered finite-difference physics at supplied logical points."""
    h = torch.as_tensor(steps, device=centers.device, dtype=centers.dtype)
    dimension = h.numel()
    if centers.ndim != 2 or centers.shape[-1] != dimension:
        raise ValueError("finite-difference centers must have shape [points, dimension]")
    if torch.any(centers < h) or torch.any(centers > 1 - h):
        raise ValueError(
            "centered finite-difference centers must stay at least one step "
            "inside every logical-domain boundary"
        )
    sample_count = centers.shape[0]
    stencil = stencil_queries(centers, h)
    query = stencil.reshape(1, -1, dimension).expand(batch_size, -1, -1)
    values = predict_attention(model, boundary_inputs, query, geometries)
    values = values.reshape(batch_size, sample_count, stencil.shape[-2], dimension)
    plus = values[..., 1 : 1 + 2 * dimension : 2, :]
    minus = values[..., 2 : 1 + 2 * dimension : 2, :]
    derivatives = (plus - minus) / (2 * h[:, None])
    return {
        "pde_normalized": elliptic_grid_normalized_pde_loss(values, h),
        "jacobian": jacobian_positive_loss(values, h, jacobian_minimum),
        "smoothness": grid_line_smoothness_loss(values, h),
        "minimum_jacobian": torch.linalg.det(derivatives).detach().min(),
    }
