"""Reference interpolation and downstream field-error measurements."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.data.mesh.sampling import logical_grid

from .q1_fem import integrate_gradient_squared, integrate_nodal_field_squared


def interpolate_logical_field(values, source_shape, target_shape):
    """Interpolate a structured nodal field through the shared logical domain."""
    nx, ny = map(int, source_shape)
    values = np.asarray(values, dtype=float)
    if values.shape != (nx * ny,):
        raise ValueError("Reference field shape does not match source_shape")
    interpolator = RegularGridInterpolator(
        (np.linspace(0.0, 1.0, ny), np.linspace(0.0, 1.0, nx)),
        values.reshape(ny, nx),
        method="linear",
        bounds_error=True,
    )
    queries = logical_grid(target_shape)
    return interpolator(queries[:, [1, 0]])


def compare_to_reference(points, shape, solution, reference, reference_shape):
    """Measure nodal, physical-L2 and physical-H1 errors on one candidate mesh."""
    solution = np.asarray(solution, dtype=float)
    reference_on_grid = interpolate_logical_field(reference, reference_shape, shape)
    error = solution - reference_on_grid
    reference_l2 = integrate_nodal_field_squared(points, shape, reference_on_grid)
    error_l2 = integrate_nodal_field_squared(points, shape, error)
    reference_h1 = integrate_gradient_squared(points, shape, reference_on_grid)
    error_h1 = integrate_gradient_squared(points, shape, error)
    absolute = np.abs(error)
    return {
        "nodal_mean_absolute_error": float(absolute.mean()),
        "nodal_maximum_absolute_error": float(absolute.max()),
        "relative_l2_error": float(np.sqrt(error_l2 / max(reference_l2, 1.0e-30))),
        "relative_h1_seminorm_error": float(
            np.sqrt(error_h1 / max(reference_h1, 1.0e-30))
        ),
        "reference_on_candidate_grid": reference_on_grid,
        "nodal_error": error,
    }


def boundary_heat_flux(points, shape, solution, edge="top", conductivity=1.0):
    """Integrate outward conductive flux on one logical boundary edge.

    The gradient is evaluated at two Gauss points per adjacent Q1 element.
    Local element edges follow counter-clockwise orientation, so rotating each
    tangent clockwise produces the outward normal-times-line-measure vector.
    """
    from .q1_fem import gauss_rule_2x2, shape_functions, structured_connectivity

    nx, ny = map(int, shape)
    points = np.asarray(points, dtype=float)
    solution = np.asarray(solution, dtype=float)
    connectivity = structured_connectivity(shape).reshape(ny - 1, nx - 1, 4)
    if edge == "bottom":
        elements, reference, local_edge = connectivity[0], lambda s: (s, -1.0), (0, 1)
    elif edge == "right":
        elements, reference, local_edge = connectivity[:, -1], lambda s: (1.0, s), (1, 2)
    elif edge == "top":
        elements, reference, local_edge = connectivity[-1, ::-1], lambda s: (-s, 1.0), (2, 3)
    elif edge == "left":
        elements, reference, local_edge = connectivity[::-1, 0], lambda s: (-1.0, -s), (3, 0)
    else:
        raise ValueError("edge must be bottom, right, top or left")
    coordinates = points[elements]
    local_solution = solution[elements]
    one_dimensional = np.unique(gauss_rule_2x2()[0][:, 0])
    total = 0.0
    for s in one_dimensional:
        _, derivatives = shape_functions(*reference(float(s)))
        jacobian = np.einsum("eia,ib->eab", coordinates, derivatives)
        inverse = np.linalg.inv(jacobian)
        gradients = np.einsum("ib,eba->eia", derivatives, inverse)
        gradient = np.einsum("ei,eia->ea", local_solution, gradients)
        a, b = local_edge
        tangent = coordinates[:, b] - coordinates[:, a]
        normal_measure = np.column_stack([tangent[:, 1], -tangent[:, 0]]) / 2.0
        k = conductivity(coordinates.mean(axis=1)) if callable(conductivity) else conductivity
        k = np.broadcast_to(np.asarray(k, dtype=float), (len(elements),))
        total += float(np.sum(-k * np.sum(gradient * normal_measure, axis=1)))
    return total
