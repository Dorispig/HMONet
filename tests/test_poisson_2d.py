"""Manufactured-solution checks for the downstream Poisson solver."""

import numpy as np

from src.data.mesh.sampling import logical_grid
from src.numerics.downstream.boundary_conditions import (
    build_dirichlet_boundary,
    build_exact_dirichlet_boundary,
)
from src.numerics.downstream.field_comparison import (
    compare_to_exact_solution,
    compare_to_reference,
)
from src.numerics.downstream.manufactured_solutions import (
    create_manufactured_solution,
)
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


def test_manufactured_poisson_solution_is_compared_to_the_exact_physical_field():
    shape = (33, 33)
    points = logical_grid(shape)
    manufactured = create_manufactured_solution("sine_product", conductivity=1.0)
    boundary = build_exact_dirichlet_boundary(points, shape, manufactured.value)
    result = solve_poisson_2d(
        points,
        shape,
        boundary,
        conductivity=1.0,
        source=manufactured.source,
    )
    metrics = compare_to_exact_solution(
        points,
        shape,
        result.solution,
        manufactured.value,
        manufactured.gradient,
    )
    assert metrics["relative_l2_error"] < 2.0e-3
    assert metrics["relative_h1_seminorm_error"] < 6.0e-2
    assert metrics["nodal_maximum_absolute_error"] < 1.0e-3


def test_directional_manufactured_solution_has_consistent_source_and_converges():
    manufactured = create_manufactured_solution("exponential_sine", conductivity=1.0)
    point = np.array([[0.37, 0.41]])
    step = 1.0e-5
    offsets = np.eye(2) * step
    laplacian = sum(
        (
            manufactured.value(point + offset)
            - 2.0 * manufactured.value(point)
            + manufactured.value(point - offset)
        )
        / step**2
        for offset in offsets
    )
    assert np.allclose(manufactured.source(point), -laplacian, rtol=2.0e-5)

    shape = (33, 33)
    points = logical_grid(shape)
    boundary = build_exact_dirichlet_boundary(points, shape, manufactured.value)
    result = solve_poisson_2d(points, shape, boundary, source=manufactured.source)
    metrics = compare_to_exact_solution(
        points, shape, result.solution, manufactured.value, manufactured.gradient
    )
    assert metrics["relative_l2_error"] < 8.0e-3
    assert metrics["relative_h1_seminorm_error"] < 1.2e-1
