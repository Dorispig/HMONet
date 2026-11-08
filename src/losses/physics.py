"""Losses for the inverse elliptic structured-grid generation PDE.

Implements paper equations 15, 17, 21 and 23--27. The 3-D coefficients
are metric cofactors (correcting the inconsistent indices in equation 18).
Spatial derivatives use values only; parameter backpropagation stays intact.

The explicit ``elliptic_grid_*`` names identify the governing PDE family and
leave room for adding losses for other PDEs without ambiguous generic names.
"""

from itertools import combinations
import torch

from src.numerics.differential import (
    boundary_aware_finite_difference_derivatives,
    finite_difference_derivatives,
    stencil_offsets as _stencil_offsets,
    stencil_queries as _stencil_queries,
)


def elliptic_grid_pde_residual_from_derivatives(first, second, mixed):
    """Compute the inverse elliptic residual from precomputed derivatives."""
    dimension = first.shape[-2]
    if dimension not in (2, 3) or second.shape[-2] != dimension:
        raise ValueError("first and second derivatives must describe a 2-D or 3-D map")
    metric = first @ first.transpose(-1, -2)
    if dimension == 2:
        cofactors = torch.stack(
            [
                torch.stack([metric[..., 1, 1], -metric[..., 0, 1]], -1),
                torch.stack([-metric[..., 0, 1], metric[..., 0, 0]], -1),
            ],
            -2,
        )
    else:
        g11, g22, g33 = metric[..., 0, 0], metric[..., 1, 1], metric[..., 2, 2]
        g12, g13, g23 = metric[..., 0, 1], metric[..., 0, 2], metric[..., 1, 2]
        cofactors = torch.stack(
            [
                torch.stack([g22*g33-g23.square(), g13*g23-g12*g33, g12*g23-g13*g22], -1),
                torch.stack([g13*g23-g12*g33, g11*g33-g13.square(), g12*g13-g11*g23], -1),
                torch.stack([g12*g23-g13*g22, g12*g13-g11*g23, g11*g22-g12.square()], -1),
            ], -2,
        )
    residual = (cofactors.diagonal(dim1=-2, dim2=-1).unsqueeze(-1) * second).sum(-2)
    for value, (axis_i, axis_j) in zip(mixed, combinations(range(dimension), 2)):
        residual = residual + 2 * cofactors[..., axis_i, axis_j, None] * value
    return residual


def stencil_offsets(dimension, *, device=None, dtype=None):
    """Compatibility wrapper for :func:`src.numerics.stencil_offsets`."""
    return _stencil_offsets(dimension, device=device, dtype=dtype)


def stencil_queries(centers, steps):
    """Compatibility wrapper for :func:`src.numerics.stencil_queries`."""
    return _stencil_queries(centers, steps)


def elliptic_grid_pde_residual(values, steps):
    """Compute the inverse elliptic grid-generation PDE residual.

    Coordinate components are summed in the loss, rather than averaged: this
    matches the paper's sum of squared x/y(/z) residuals at each point.
    """
    dimension = values.shape[-1]
    if dimension not in (2, 3) or values.shape[-2] != (9 if dimension == 2 else 19):
        raise ValueError("Expected 9-point 2-D or 19-point 3-D stencil")
    h = torch.as_tensor(steps, device=values.device, dtype=values.dtype)
    center = values[..., 0, :]
    plus = values[..., 1 : 1 + 2 * dimension : 2, :]
    minus = values[..., 2 : 1 + 2 * dimension : 2, :]
    first = (plus - minus) / (2 * h[:, None])
    second = (plus - 2 * center.unsqueeze(-2) + minus) / h[:, None].square()
    mixed_values = []
    start = 1 + 2 * dimension
    for offset, (i, j) in enumerate(combinations(range(dimension), 2)):
        pp, pm, mp, mm = values[
            ..., start + 4 * offset : start + 4 * offset + 4, :
        ].unbind(-2)
        mixed = (pp - pm - mp + mm) / (4 * h[i] * h[j])
        mixed_values.append(mixed)
    return elliptic_grid_pde_residual_from_derivatives(first, second, mixed_values)


def elliptic_grid_boundary_aware_terms(values, centers, steps):
    """Return residual and derivatives for the inward boundary-aware stencil."""
    first, second, mixed = boundary_aware_finite_difference_derivatives(
        values, centers, steps
    )
    return elliptic_grid_pde_residual_from_derivatives(first, second, mixed), first


def elliptic_grid_boundary_aware_scale_invariant_pde_loss(
    values, centers, steps, epsilon=1e-18
):
    """Scale-invariant elliptic loss using inward one-sided boundary rules."""
    residual, first = elliptic_grid_boundary_aware_terms(values, centers, steps)
    metric_trace = first.square().sum(dim=(-2, -1))
    exponent = 2 * values.shape[-1] - 1
    denominator = metric_trace.pow(exponent).clamp_min(float(epsilon))
    return (residual.square().sum(-1) / denominator).mean()


def jacobian_positive_loss(values, steps, minimum=1e-4):
    """Penalize non-positive mappings using the elliptic finite-difference stencil.

    The PDE stencil already contains the positive and negative neighbour on
    every computational axis. Reusing those predictions avoids another model
    forward pass. Gradients still propagate through the predicted coordinates.
    """
    dimension = values.shape[-1]
    if dimension not in (2, 3) or values.shape[-2] != (9 if dimension == 2 else 19):
        raise ValueError("Expected 9-point 2-D or 19-point 3-D stencil")
    h = torch.as_tensor(steps, device=values.device, dtype=values.dtype)
    plus = values[..., 1 : 1 + 2 * dimension : 2, :]
    minus = values[..., 2 : 1 + 2 * dimension : 2, :]
    derivatives = (plus - minus) / (2 * h[:, None])
    determinant = torch.linalg.det(derivatives)
    return torch.relu(float(minimum) - determinant).square().mean()


def winslow_energy_and_jacobian_barrier(values, steps, minimum=1e-4, epsilon=1e-6):
    """Regularized 2-D Winslow energy and a dimensionless positive-area barrier.

    Both terms reuse the first derivatives in the elliptic-residual stencil.
    The smooth positive-part denominator stays defined on initially inverted
    predictions; the separate barrier prevents isotropic collapse below epsilon.
    """
    if values.shape[-1] != 2 or values.shape[-2] not in (5, 9):
        raise ValueError("Winslow energy requires a 5- or 9-point 2-D stencil")
    if minimum <= 0 or epsilon <= 0:
        raise ValueError("Winslow minimum and epsilon must be positive")
    h = torch.as_tensor(steps, device=values.device, dtype=values.dtype)
    plus = values[..., 1:5:2, :]
    minus = values[..., 2:5:2, :]
    first = (plus - minus) / (2 * h[:, None])
    d_xi, d_eta = first.unbind(-2)
    determinant = d_xi[..., 0] * d_eta[..., 1] - d_xi[..., 1] * d_eta[..., 0]
    stretch = d_xi.square().sum(-1) + d_eta.square().sum(-1)
    # Stable evaluation of (J + sqrt(J^2 + epsilon^2))/2 for J < 0.
    root = torch.sqrt(determinant.square() + float(epsilon) ** 2)
    positive = (determinant + root) / 2
    negative = float(epsilon) ** 2 / (2 * (root - determinant))
    denominator = torch.where(determinant >= 0, positive, negative)
    winslow = (stretch / denominator).mean()
    barrier = torch.relu(1.0 - determinant / float(minimum)).square().mean()
    return winslow, barrier


def _first_and_second_derivatives(values, steps):
    """Compatibility wrapper around the shared finite-difference operator."""
    return finite_difference_derivatives(values, steps)


def elliptic_grid_normalized_pde_loss(values, steps, epsilon=1e-8):
    """Return metric-trace-normalized inverse elliptic grid PDE loss."""
    residual = elliptic_grid_pde_residual(values, steps)
    first, _, _ = _first_and_second_derivatives(values, steps)
    metric_trace = first.square().sum(dim=(-2, -1))
    denominator = metric_trace.square().clamp_min(float(epsilon))
    return (residual.square().sum(-1) / denominator).mean()


def elliptic_grid_scale_invariant_pde_loss(values, steps, epsilon=1e-18):
    """Return a scale-invariant inverse elliptic grid PDE loss.

    In ``dimension`` dimensions the metric-cofactor residual scales as
    ``length ** (2 * dimension - 1)``.  Its squared norm is therefore divided
    by the metric trace raised to the same ``2 * dimension - 1`` power.  The
    denominator intentionally remains in the autograd graph: detaching it
    would restore a gradient incentive to collapse the physical mapping.
    """
    residual = elliptic_grid_pde_residual(values, steps)
    first, _, _ = _first_and_second_derivatives(values, steps)
    metric_trace = first.square().sum(dim=(-2, -1))
    exponent = 2 * values.shape[-1] - 1
    denominator = metric_trace.pow(exponent).clamp_min(float(epsilon))
    return (residual.square().sum(-1) / denominator).mean()


def normalized_mapping_barriers_from_jacobian(
    first,
    characteristic_length,
    *,
    jacobian_minimum=0.02,
    singular_value_minimum=0.05,
    reduction="active",
):
    """Penalize orientation loss and local collapse from a Jacobian tensor.

    ``first`` may be produced by finite differences or automatic
    differentiation. ``characteristic_length`` is one detached physical scale
    per geometry. ``reduction='active'`` averages only nonzero violations;
    ``reduction='all'`` averages the pointwise penalty over every sampled
    point. Both losses are exactly zero once every sampled point satisfies its
    lower bound.
    """
    if jacobian_minimum <= 0 or singular_value_minimum <= 0:
        raise ValueError("Normalized mapping minima must be positive")
    if reduction not in {"active", "all"}:
        raise ValueError("barrier reduction must be active or all")
    dimension = first.shape[-1]
    scale = torch.as_tensor(
        characteristic_length, device=first.device, dtype=first.dtype
    ).detach().clamp_min(torch.finfo(first.dtype).eps)
    while scale.ndim < first.ndim - 2:
        scale = scale.unsqueeze(-1)
    determinant = torch.linalg.det(first)
    normalized_jacobian = determinant / scale.pow(dimension)
    normalized_singular = torch.linalg.svdvals(first)[..., -1] / scale

    def violation_mean(value, minimum):
        violation = torch.relu(1.0 - value / float(minimum)).square()
        if reduction == "all":
            return violation.mean()
        active = (violation > 0).to(violation.dtype)
        return violation.sum() / active.sum().clamp_min(1.0)

    return (
        violation_mean(normalized_jacobian, jacobian_minimum),
        violation_mean(normalized_singular, singular_value_minimum),
    )


def normalized_mapping_barriers(
    values,
    steps,
    characteristic_length,
    *,
    jacobian_minimum=0.02,
    singular_value_minimum=0.05,
    reduction="active",
):
    """Finite-difference wrapper for normalized mapping barriers."""
    first, _, _ = _first_and_second_derivatives(values, steps)
    return normalized_mapping_barriers_from_jacobian(
        first,
        characteristic_length,
        jacobian_minimum=jacobian_minimum,
        singular_value_minimum=singular_value_minimum,
        reduction=reduction,
    )


def grid_line_smoothness_loss(values, steps):
    """Penalize pure and mixed second derivatives of physical coordinates."""
    _, second, mixed = _first_and_second_derivatives(values, steps)
    total = second.square().sum(dim=(-2, -1))
    if mixed:
        total = total + 2.0 * torch.stack(
            [value.square().sum(-1) for value in mixed], -1
        ).sum(-1)
    return total.mean()


def orthogonality_loss(values, steps, epsilon=1e-8):
    """Penalize squared cosines between every pair of coordinate directions."""
    first, _, _ = _first_and_second_derivatives(values, steps)
    terms = []
    for i, j in combinations(range(first.shape[-2]), 2):
        numerator = (first[..., i, :] * first[..., j, :]).sum(-1).square()
        denominator = (
            first[..., i, :].square().sum(-1)
            * first[..., j, :].square().sum(-1)
        ).clamp_min(float(epsilon))
        terms.append(numerator / denominator)
    return torch.stack(terms, -1).mean()


# Backward-compatible names. New code should use the PDE-qualified names above.
elliptic_residual = elliptic_grid_pde_residual
normalized_pde_loss = elliptic_grid_normalized_pde_loss
scale_invariant_pde_loss = elliptic_grid_scale_invariant_pde_loss

