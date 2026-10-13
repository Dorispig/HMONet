"""Central finite differences for the PI-MeshONet elliptic residual.

Implements paper equations 15, 17, 21 and 23--27. The 3-D coefficients
are metric cofactors (correcting the inconsistent indices in equation 18).
Spatial derivatives use values only; parameter backpropagation stays intact.
"""

from itertools import combinations
import torch


def stencil_offsets(dimension, *, device=None, dtype=None):
    """Return 9 (2-D) or 19 (3-D) offsets, including mixed-derivative corners."""
    eye = torch.eye(dimension, device=device, dtype=dtype)
    offsets = [torch.zeros(dimension, device=device, dtype=dtype)]
    for i in range(dimension):
        offsets.extend([eye[i], -eye[i]])
    for i, j in combinations(range(dimension), 2):
        offsets.extend(
            [eye[i] + eye[j], eye[i] - eye[j], -eye[i] + eye[j], -eye[i] - eye[j]]
        )
    return torch.stack(offsets)


def stencil_queries(centers, steps):
    """Expand [..., point, dimension] centers without leaving the unit cube."""
    steps = torch.as_tensor(steps, device=centers.device, dtype=centers.dtype)
    return (
        centers.unsqueeze(-2)
        + stencil_offsets(centers.shape[-1], device=centers.device, dtype=centers.dtype)
        * steps
    )


def elliptic_residual(values, steps):
    """Compute [..., point, dimension] residuals from ordered stencil values.

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
        # Cross products are rows of the spatial Jacobian's cofactor matrix.
        # Their Gram matrix equals adj(J J^T), without an unstable inverse.
        cross = torch.stack(
            [
                torch.linalg.cross(first[..., 1, :], first[..., 2, :]),
                torch.linalg.cross(first[..., 2, :], first[..., 0, :]),
                torch.linalg.cross(first[..., 0, :], first[..., 1, :]),
            ],
            -2,
        )
        cofactors = cross @ cross.transpose(-1, -2)
    residual = (cofactors.diagonal(dim1=-2, dim2=-1).unsqueeze(-1) * second).sum(-2)
    start = 1 + 2 * dimension
    for offset, (i, j) in enumerate(combinations(range(dimension), 2)):
        pp, pm, mp, mm = values[
            ..., start + 4 * offset : start + 4 * offset + 4, :
        ].unbind(-2)
        mixed = (pp - pm - mp + mm) / (4 * h[i] * h[j])
        residual = residual + 2 * cofactors[..., i, j, None] * mixed
    return residual


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


def _first_and_second_derivatives(values, steps):
    """Return first, pure-second and mixed derivatives from a PDE stencil."""
    dimension = values.shape[-1]
    if dimension not in (2, 3) or values.shape[-2] != (9 if dimension == 2 else 19):
        raise ValueError("Expected 9-point 2-D or 19-point 3-D stencil")
    h = torch.as_tensor(steps, device=values.device, dtype=values.dtype)
    center = values[..., 0, :]
    plus = values[..., 1 : 1 + 2 * dimension : 2, :]
    minus = values[..., 2 : 1 + 2 * dimension : 2, :]
    first = (plus - minus) / (2 * h[:, None])
    second = (plus - 2 * center.unsqueeze(-2) + minus) / h[:, None].square()
    mixed = []
    start = 1 + 2 * dimension
    for offset, (i, j) in enumerate(combinations(range(dimension), 2)):
        pp, pm, mp, mm = values[
            ..., start + 4 * offset : start + 4 * offset + 4, :
        ].unbind(-2)
        mixed.append((pp - pm - mp + mm) / (4 * h[i] * h[j]))
    return first, second, mixed


def normalized_pde_loss(values, steps, epsilon=1e-8):
    """Scale the elliptic residual by the squared metric trace."""
    residual = elliptic_residual(values, steps)
    first, _, _ = _first_and_second_derivatives(values, steps)
    metric_trace = first.square().sum(dim=(-2, -1))
    denominator = metric_trace.square().clamp_min(float(epsilon))
    return (residual.square().sum(-1) / denominator).mean()


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

