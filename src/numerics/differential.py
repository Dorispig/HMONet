"""Dimension-independent finite-difference and automatic-differentiation operators."""

from itertools import combinations

import torch


def stencil_offsets(dimension, *, device=None, dtype=None):
    """Return the centered 9-point 2-D or 19-point 3-D PDE stencil offsets."""
    if dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3")
    eye = torch.eye(dimension, device=device, dtype=dtype)
    offsets = [torch.zeros(dimension, device=device, dtype=dtype)]
    for axis in range(dimension):
        offsets.extend((eye[axis], -eye[axis]))
    for first, second in combinations(range(dimension), 2):
        offsets.extend(
            (
                eye[first] + eye[second],
                eye[first] - eye[second],
                -eye[first] + eye[second],
                -eye[first] - eye[second],
            )
        )
    return torch.stack(offsets)


def stencil_queries(centers, steps):
    """Expand logical centers into an ordered centered-difference stencil."""
    steps = torch.as_tensor(steps, device=centers.device, dtype=centers.dtype)
    return centers.unsqueeze(-2) + stencil_offsets(
        centers.shape[-1], device=centers.device, dtype=centers.dtype
    ) * steps


def finite_difference_derivatives(values, steps):
    """Return first, pure-second and mixed derivatives from an ordered stencil."""
    dimension = values.shape[-1]
    expected = 9 if dimension == 2 else 19 if dimension == 3 else None
    if expected is None or values.shape[-2] != expected:
        raise ValueError("Expected 9-point 2-D or 19-point 3-D stencil")
    step = torch.as_tensor(steps, device=values.device, dtype=values.dtype)
    center = values[..., 0, :]
    plus = values[..., 1 : 1 + 2 * dimension : 2, :]
    minus = values[..., 2 : 1 + 2 * dimension : 2, :]
    first = (plus - minus) / (2 * step[:, None])
    second = (plus - 2 * center.unsqueeze(-2) + minus) / step[:, None].square()
    mixed = []
    start = 1 + 2 * dimension
    for offset, (axis_i, axis_j) in enumerate(combinations(range(dimension), 2)):
        pp, pm, mp, mm = values[
            ..., start + 4 * offset : start + 4 * offset + 4, :
        ].unbind(-2)
        mixed.append((pp - pm - mp + mm) / (4 * step[axis_i] * step[axis_j]))
    return first, second, mixed


def autograd_gradient(value, logical, *, create_graph=True):
    """Differentiate a batched scalar field with respect to logical coordinates."""
    return torch.autograd.grad(
        value,
        logical,
        grad_outputs=torch.ones_like(value),
        create_graph=create_graph,
        retain_graph=True,
    )[0]


def autograd_differential_terms(prediction, logical):
    """Return inverse-elliptic residual, Jacobian, metric and Hessian terms."""
    dimension = prediction.shape[-1]
    if logical.shape[-1] != dimension or dimension not in (2, 3):
        raise ValueError("prediction/logical must describe a 2-D or 3-D mapping")
    first = [autograd_gradient(prediction[..., c], logical) for c in range(dimension)]
    jacobian = torch.stack(first, dim=-2)
    metric = torch.einsum("...ki,...kj->...ij", jacobian, jacobian)
    second = [[None] * dimension for _ in range(dimension)]
    for axis_i in range(dimension):
        for axis_j in range(axis_i, dimension):
            components = []
            for coordinate in range(dimension):
                if first[coordinate][..., axis_i].requires_grad:
                    derivative = autograd_gradient(
                        first[coordinate][..., axis_i], logical
                    )[..., axis_j]
                else:
                    derivative = torch.zeros_like(first[coordinate][..., axis_i])
                if axis_i != axis_j:
                    if first[coordinate][..., axis_j].requires_grad:
                        reverse = autograd_gradient(
                            first[coordinate][..., axis_j], logical
                        )[..., axis_i]
                    else:
                        reverse = torch.zeros_like(first[coordinate][..., axis_j])
                    derivative = 0.5 * (derivative + reverse)
                components.append(derivative)
            second[axis_i][axis_j] = torch.stack(components, -1)
    if dimension == 2:
        cofactor = torch.stack(
            (
                torch.stack((metric[..., 1, 1], -metric[..., 0, 1]), -1),
                torch.stack((-metric[..., 1, 0], metric[..., 0, 0]), -1),
            ),
            -2,
        )
    else:
        g00, g01, g02 = metric[..., 0, 0], metric[..., 0, 1], metric[..., 0, 2]
        g11, g12, g22 = metric[..., 1, 1], metric[..., 1, 2], metric[..., 2, 2]
        c00, c11, c22 = g11*g22-g12*g12, g00*g22-g02*g02, g00*g11-g01*g01
        c01, c02, c12 = g02*g12-g01*g22, g01*g12-g02*g11, g01*g02-g00*g12
        cofactor = torch.stack(
            (
                torch.stack((c00, c01, c02), -1),
                torch.stack((c01, c11, c12), -1),
                torch.stack((c02, c12, c22), -1),
            ),
            -2,
        )
    residual = prediction.new_zeros(prediction.shape)
    for axis_i in range(dimension):
        residual = residual + cofactor[..., axis_i, axis_i, None] * second[axis_i][axis_i]
        for axis_j in range(axis_i + 1, dimension):
            residual = residual + 2 * cofactor[..., axis_i, axis_j, None] * second[axis_i][axis_j]
    return {
        "residual": residual,
        "jacobian": jacobian,
        "metric": metric,
        "second": second,
    }
