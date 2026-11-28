"""ResMetaMesh auto-decoder with an algebraic base and per-shape LoRA."""
from itertools import combinations
from math import isfinite, prod
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.data.backend.parametric_torch import (
    evaluate_boundary_curves_torch, evaluate_boundary_surfaces_torch,
    evaluate_surface_torch,
)
from src.losses.physics import elliptic_grid_pde_residual_from_derivatives
from src.numerics.differential import (
    autograd_gradient, finite_difference_derivatives, stencil_offsets, stencil_queries,
)


class ResMetaMesh(nn.Module):
    """Shared Tanh decoder and a trainable latent vector for each training task."""

    def __init__(self, dimension, task_count, *, latent_dim=8, width=50, hidden_layers=3):
        super().__init__()
        if dimension not in (2, 3) or min(task_count, latent_dim, width, hidden_layers) < 1:
            raise ValueError("Invalid ResMetaMesh dimensions")
        self.dimension = dimension
        self.latent_dim = latent_dim
        self.latents = nn.Parameter(torch.empty(task_count, latent_dim))
        nn.init.normal_(self.latents, std=0.01)
        sizes = [dimension + latent_dim, *([width] * hidden_layers), dimension]
        self.decoder = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))

    def forward(self, query, latent, lora=None):
        x = torch.cat((latent.expand(*query.shape[:-1], -1), query), dim=-1)
        for layer in self.decoder[:-1]:
            x = torch.tanh(layer(x))
        output = self.decoder[-1](x)
        if lora is not None:
            left, right = lora
            output = output + F.linear(F.linear(x, right), left)
        # Applied before differentiation so product-rule terms enter the PDE.
        bubble = (query * (1 - query)).prod(dim=-1, keepdim=True)
        return bubble * output


def logical_query_grid(shape, device=None):
    axes = [
        torch.arange(count, dtype=torch.float32, device=device) / (count - 1)
        for count in shape
    ]
    return torch.stack(torch.meshgrid(*axes[::-1], indexing="ij")[::-1], dim=-1).reshape(
        -1, len(shape)
    )


def _tfi_base(geometry, query):
    """Evaluate Coons or Gordon--Hall interpolation in the query precision."""
    dimension = query.shape[-1]
    coordinates = query.reshape(1, -1, dimension).unbind(-1)
    weights = [(1 - coordinate, coordinate) for coordinate in coordinates]
    if dimension == 2:
        xi, eta = coordinates
        faces = evaluate_boundary_curves_torch(geometry, xi, eta).unbind(-2)
        endpoints = xi.new_tensor([[0, 1]])
        corners = evaluate_boundary_curves_torch(
            geometry, endpoints, endpoints
        )[0, :, 2:4, :]
        lower, upper = corners.unbind(1)
        result = sum(weight[..., None] * face
                     for weight, face in zip((*weights[0], *weights[1]), faces))
        for i in (0, 1):
            result -= (weights[0][i] * weights[1][0])[..., None] * lower[i]
            result -= (weights[0][i] * weights[1][1])[..., None] * upper[i]
    else:
        faces = evaluate_boundary_surfaces_torch(geometry, *coordinates).unbind(-2)
        surfaces = (
            (geometry.x_min, geometry.x_max),
            (geometry.y_min, geometry.y_max),
            (geometry.z_min, geometry.z_max),
        )
        result = sum(weight[..., None] * face
                     for weight, face in zip((w for pair in weights for w in pair), faces))
        # Subtract the 12 exact curved edges using the same incident faces as
        # the reference Gordon--Hall formula, then restore the eight corners.
        for normal, fixed in combinations(range(3), 2):
            for i in (0, 1):
                for j in (0, 1):
                    u, v = (coordinates[a] for a in range(3) if a != normal)
                    if fixed == 2:
                        v = torch.full_like(v, j)
                    else:
                        u = torch.full_like(u, j)
                    edge = evaluate_surface_torch(surfaces[normal][i], u, v)
                    result -= (weights[normal][i] * weights[fixed][j])[..., None] * edge
        for k in (0, 1):
            corners = evaluate_surface_torch(
                surfaces[2][k], query.new_tensor([0, 0, 1, 1]),
                query.new_tensor([0, 1, 0, 1]),
            )
            for i in (0, 1):
                for j in (0, 1):
                    weight = weights[0][i] * weights[1][j] * weights[2][k]
                    result += weight[..., None] * corners[2 * i + j]
    return result.reshape_as(query)


def base_coordinates(geometry, shape, *, device=None):
    return _tfi_base(geometry, logical_query_grid(shape, device))


def collocation_grid(shape, count, seed, fd_step=None):
    """Select fixed interior centers and construct single-precision PDE stencils."""
    query = logical_query_grid(shape)
    if fd_step is None:
        inside = ((query > 0) & (query < 1)).all(dim=-1)
    else:
        step = query.new_tensor(fd_step)
        inside = ((query >= step) & (query <= 1 - step)).all(dim=-1)
    centers = inside.nonzero(as_tuple=True)[0]
    if not len(centers):
        raise ValueError("ResMetaMesh collocation needs at least three nodes per axis")
    if count is not None and count > len(centers):
        raise ValueError(f"ResMetaMesh requested {count} centers, but only {len(centers)} stencils fit")
    if count is not None and count < len(centers):
        selected = np.sort(np.random.default_rng(seed).choice(len(centers), count, replace=False))
        centers = centers[selected]
    points = query[centers]
    if fd_step is not None:
        return points, stencil_queries(points, fd_step)
    strides = torch.tensor((1, *shape[:-1]), dtype=torch.int64).cumprod(0)
    offsets = stencil_offsets(len(shape), dtype=torch.int64) @ strides
    return points, centers[:, None] + offsets


def base_derivatives(geometry, shape, stencil, fd_step=None):
    if fd_step is None:
        values = base_coordinates(geometry, shape, device=stencil.device)[stencil]
        steps = [1 / (n - 1) for n in shape]
    else:
        values = _tfi_base(geometry, stencil)
        steps = fd_step
    first, second, mixed = finite_difference_derivatives(values, steps)
    return first, second, *mixed


def pde_loss(model, query, base, latent, lora=None, *, fd_step=None):
    """Unnormalized zero-source inverse elliptic residual of base plus correction."""
    if fd_step is None:
        first, second, mixed = _autograd_correction_derivatives(model, query, latent, lora)
    else:
        correction = model(query, latent, lora)
        first, second, mixed = finite_difference_derivatives(correction, fd_step)
    residual = elliptic_grid_pde_residual_from_derivatives(
        base[0] + first, base[1] + second,
        [a + b for a, b in zip(base[2:], mixed)],
    )
    return residual.square().sum(dim=-1).mean()


def _autograd_correction_derivatives(model, query, latent, lora):
    """Reproduce the coordinate derivatives for archived hybrid checkpoints."""
    query = query.detach().requires_grad_(True)
    dimension = model.dimension
    values = model(query, latent, lora)
    first = torch.stack([autograd_gradient(v, query) for v in values.unbind(-1)], dim=-1)
    hessian = torch.stack(
        [
            torch.stack([autograd_gradient(v, query) for v in row.unbind(-1)], dim=-1)
            for row in first.unbind(-2)
        ],
        dim=-3,
    )
    second = torch.stack([hessian[..., i, i, :] for i in range(dimension)], dim=-2)
    mixed = [hessian[..., i, j, :] for i, j in combinations(range(dimension), 2)]
    return first, second, mixed


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def adapt_shape(model, geometry, options, seed, device):
    """Fit a fresh latent and last-layer LoRA, leaving the saved decoder untouched."""
    _synchronize(device)
    started = perf_counter()
    dtype = model.latents.dtype
    shape = tuple(options["collocation_shape"])
    fd_step = options.get("fd_step")
    centers, stencil = collocation_grid(shape, options["adaptation_collocation_count"], seed, fd_step)
    # Checkpoints from the all-FD implementation stored the step without a backend.
    correction_step = fd_step if options.get("correction_derivative_backend") is None else None
    stencil = stencil.to(device=device)
    query = stencil if correction_step is not None else centers.to(device=device)
    base = base_derivatives(geometry, shape, stencil, fd_step)
    _synchronize(device)
    base_seconds = perf_counter() - started
    started = perf_counter()
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = nn.Parameter(torch.randn(model.latent_dim, generator=generator, device=device, dtype=dtype) * 0.01)
    output = model.decoder[-1]
    rank = options["lora_rank"]
    left = nn.Parameter(torch.zeros(output.out_features, rank, device=device, dtype=dtype))
    right = nn.Parameter(torch.empty(rank, output.in_features, device=device, dtype=dtype))
    nn.init.kaiming_uniform_(right, a=5**0.5, generator=generator)
    parameters = [latent, left, right]
    optimizer = torch.optim.LBFGS(parameters, lr=options["adaptation_lr"], max_iter=1,
                                 history_size=options["lbfgs_history_size"])

    def closure():
        optimizer.zero_grad(set_to_none=True)
        total = query.new_zeros(())
        chunk = options["derivative_batch_size"]
        for start in range(0, len(query), chunk):
            part = slice(start, start + chunk)
            loss = pde_loss(
                model, query[part], [v[part] for v in base], latent, (left, right),
                fd_step=correction_step,
            )
            loss = loss * (len(query[part]) / len(query))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite ResMetaMesh adaptation residual")
            loss.backward()
            total += loss.detach()
        regularizer = options["latent_l2_weight"] * latent.square().sum()
        regularizer.backward()
        return total + regularizer.detach()

    with torch.enable_grad():
        for _ in range(options["adaptation_steps"]):
            optimizer.step(closure)
    if not all(torch.isfinite(v).all() for v in parameters):
        raise FloatingPointError("Non-finite ResMetaMesh adaptation parameters")
    _synchronize(device)
    return (latent.detach(), left.detach(), right.detach()), {
        "adaptation_base_seconds": base_seconds,
        "adaptation_seconds": perf_counter() - started,
        "adaptation_steps": options["adaptation_steps"],
        "adaptation_collocation_count": len(query),
    }


@torch.no_grad()
def decode_mesh(model, geometry, shape, latent, lora=None, chunk_size=65536):
    query = logical_query_grid(shape, model.latents.device)
    chunks = [
        _tfi_base(geometry, part) + model(part, latent, lora)
        for part in query.split(chunk_size)
    ]
    return torch.cat(chunks).cpu().numpy()


def validate_training_options(options, dimension):
    shape = options["collocation_shape"]
    if len(shape) != dimension or any(type(n) is not int or n < 3 for n in shape):
        raise ValueError("ResMetaMesh collocation_shape requires one node count >= 3 per axis")
    fd_step = options.get("fd_step")
    if fd_step is not None and (
        len(fd_step) != dimension or any(not 0 < h < 0.5 for h in fd_step)
    ):
        raise ValueError("ResMetaMesh fd_step needs one value in (0, 0.5) per axis")
    for key in ("derivative_batch_size", "adaptation_steps", "lora_rank", "lbfgs_history_size"):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f"ResMetaMesh {key} must be a positive integer")
    for key in ("collocation_count", "adaptation_collocation_count"):
        count = options[key]
        if count is not None and (type(count) is not int or not 1 <= count <= prod(n-2 for n in shape)):
            raise ValueError(f"ResMetaMesh {key} must fit the interior lattice or be null")
    for key in ("adaptation_lr", "latent_l2_weight"):
        value = float(options[key])
        if not isfinite(value) or value < 0 or (key == "adaptation_lr" and value == 0):
            raise ValueError(f"Invalid ResMetaMesh {key}")
