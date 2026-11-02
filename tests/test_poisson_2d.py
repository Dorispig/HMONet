"""Manufactured-solution checks for the downstream Poisson solver."""

import numpy as np

from src.data.mesh.sampling import logical_grid
from src.numerics.downstream.boundary_conditions import build_dirichlet_boundary
from src.numerics.downstream.field_comparison import compare_to_reference
from src.numerics.downstream.poisson_2d import solve_poisson_2d


def test_linear_harmonic_solution_is_reproduced_exactly():
    shape = (9, 7)
    points = logical_grid(shape)
    boundary = build_dirichlet_boundary(
        shape,
        {
            "bottom": lambda s: s,
            "right": lambda s: 1.0 + s,
            "top": lambda s: 1.0 + s,
            "left": lambda s: s,
        },
        corner_policy="error",
    )
    result = solve_poisson_2d(points, shape, boundary)
    exact = points[:, 0] + points[:, 1]
    assert np.max(np.abs(result.solution - exact)) < 1.0e-12
    assert result.relative_residual < 1.0e-12


def test_logical_reference_comparison_is_zero_for_same_field():
    source_shape = (17, 17)
    target_shape = (9, 9)
    source_points = logical_grid(source_shape)
    target_points = logical_grid(target_shape)
    reference = source_points[:, 0] + 2.0 * source_points[:, 1]
    candidate = target_points[:, 0] + 2.0 * target_points[:, 1]
    metrics = compare_to_reference(
        target_points, target_shape, candidate, reference, source_shape
    )
    assert metrics["nodal_maximum_absolute_error"] < 1.0e-14
    assert metrics["relative_l2_error"] < 1.0e-14
