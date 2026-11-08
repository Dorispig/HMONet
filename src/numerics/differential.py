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


def boundary_aware_stencil_queries(centers, steps):
    """Build centered stencils internally and inward one-sided stencils near faces.

    Each coordinate direction uses the centered nodes ``(+h, -h)`` when the
    center is at least one step from both faces.  Within one step of a face it
    instead uses the inward nodes ``(+/-h, +/-2h, +/-3h)``.  Mixed derivatives
    use the tensor product of the corresponding second-order first-derivative
    rules.  The fixed query count is ``1 + 3*d + 9*C(d, 2)`` (16 in 2-D),
    which keeps batching simple while ensuring every query remains in [0, 1].
    """
    dimension = centers.shape[-1]
    if dimension not in (2, 3):
        raise ValueError("boundary-aware finite differences require 2-D or 3-D points")
    step = torch.as_tensor(steps, device=centers.device, dtype=centers.dtype)
    if step.shape != (dimension,) or bool((step <= 0).any()) or bool((step > 1 / 3).any()):
        raise ValueError("steps must contain one value in (0, 1/3] per dimension")

    lower = centers < step
    upper = centers > 1 - step
    one_sided = lower | upper
    direction = torch.where(upper, -torch.ones_like(centers), torch.ones_like(centers))
    first_offset = torch.where(one_sided, direction * step, step)
    second_offset = torch.where(one_sided, 2 * direction * step, -step)
    third_offset = torch.where(one_sided, 3 * direction * step, torch.zeros_like(centers))
    axis_offsets = torch.stack((first_offset, second_offset, third_offset), -1)

    eye = torch.eye(dimension, device=centers.device, dtype=centers.dtype)
    pure_offsets = axis_offsets[..., :, :, None] * eye[..., :, None, :]
    query_parts = [centers.unsqueeze(-2), centers.unsqueeze(-2) + pure_offsets.flatten(-3, -2)]

    zero = torch.zeros_like(first_offset[..., 0])
    for axis_i, axis_j in combinations(range(dimension), 2):
        offsets_i = torch.stack(
            (zero, first_offset[..., axis_i], second_offset[..., axis_i]), -1
        )
        offsets_j = torch.stack(
            (zero, first_offset[..., axis_j], second_offset[..., axis_j]), -1
        )
        pair_offsets = centers.new_zeros(*centers.shape[:-1], 3, 3, dimension)
        pair_offsets[..., axis_i] = offsets_i.unsqueeze(-1)
        pair_offsets[..., axis_j] = offsets_j.unsqueeze(-2)
        query_parts.append(centers.unsqueeze(-2) + pair_offsets.flatten(-3, -2))
    queries = torch.cat(query_parts, -2)
    return queries.clamp(0.0, 1.0)


def boundary_aware_finite_difference_derivatives(values, centers, steps):
    """Differentiate values from :func:`boundary_aware_stencil_queries`.

    Centered rules retain their usual second-order formulas. Near a face, the
    first derivative uses ``(-3, 4, -1)/(2h)`` and the pure second derivative
    uses ``(2, -5, 4, -1)/h^2`` in the inward direction.  Mixed derivatives
    are tensor products of the selected first-derivative rules.
    """
    dimension = centers.shape[-1]
    expected = 1 + 3 * dimension + 9 * (dimension * (dimension - 1) // 2)
    if dimension not in (2, 3) or values.shape[-2] != expected:
        raise ValueError(f"Expected {expected}-point boundary-aware stencil")
    step = torch.as_tensor(steps, device=values.device, dtype=values.dtype)
    logical = centers.to(device=values.device, dtype=values.dtype)
    lower = logical < step
    upper = logical > 1 - step
    one_sided = lower | upper
    direction = torch.where(upper, -torch.ones_like(logical), torch.ones_like(logical))

    inverse = torch.ones_like(logical) / step
    centered_first = torch.stack(
        (torch.zeros_like(logical), 0.5 * inverse, -0.5 * inverse), -1
    )
    one_sided_first = torch.stack(
        (-1.5 * direction * inverse, 2.0 * direction * inverse,
         -0.5 * direction * inverse), -1
    )
    first_weights = torch.where(one_sided.unsqueeze(-1), one_sided_first, centered_first)

    inverse_square = inverse.square()
    centered_second = torch.stack(
        (-2.0 * inverse_square, inverse_square, inverse_square,
         torch.zeros_like(inverse_square)), -1
    )
    one_sided_second = torch.stack(
        (2.0 * inverse_square, -5.0 * inverse_square, 4.0 * inverse_square,
         -inverse_square), -1
    )
    second_weights = torch.where(
        one_sided.unsqueeze(-1), one_sided_second, centered_second
    )

    center = values[..., 0, :]
    pure = values[..., 1 : 1 + 3 * dimension, :].reshape(
        *values.shape[:-2], dimension, 3, values.shape[-1]
    )
    first_samples = torch.cat(
        (center.unsqueeze(-2).unsqueeze(-2).expand(*pure.shape[:-2], 1, values.shape[-1]),
         pure[..., :2, :]), -2
    )
    second_samples = torch.cat((center.unsqueeze(-2).unsqueeze(-2).expand(
        *pure.shape[:-2], 1, values.shape[-1]
    ), pure), -2)
    first = (first_weights.unsqueeze(-1) * first_samples).sum(-2)
    second = (second_weights.unsqueeze(-1) * second_samples).sum(-2)

    mixed = []
    start = 1 + 3 * dimension
    for offset, (axis_i, axis_j) in enumerate(combinations(range(dimension), 2)):
        pair = values[..., start + 9 * offset : start + 9 * (offset + 1), :].reshape(
            *values.shape[:-2], 3, 3, values.shape[-1]
        )
        weights = first_weights[..., axis_i, :, None] * first_weights[..., axis_j, None, :]
        mixed.append((weights.unsqueeze(-1) * pair).sum(dim=(-3, -2)))
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
