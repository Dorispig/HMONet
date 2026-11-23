"""Analytic invariants and ordering checks for batched 3-D quality metrics."""
import numpy as np
import pytest

from src.metrics import evaluate_hexahedral_grid_quality
from src.evaluation.quality_3d import surface_quad_quality_fields


def rectangular_grid(nx=4, ny=5, nz=3):
    z, y, x = np.meshgrid(np.arange(nz)*4., np.arange(ny)*3.,
                         np.arange(nx)*2., indexing="ij")
    return np.stack((x, y, z), axis=-1)


@pytest.mark.parametrize("reverse", [False, True])
def test_rectangular_cell_metrics_and_orientation(reverse):
    grid = rectangular_grid()
    if reverse:
        grid = grid[:, :, ::-1]
    fields = evaluate_hexahedral_grid_quality(grid)
    expected = {"minimum_scaled_jacobian": -1 if reverse else 1,
                "center_jacobian": -24 if reverse else 24,
                "center_condition_number": 2, "cell_volume": 24,
                "edge_aspect_ratio": 2, "min_angle": 90, "max_angle": 90,
                "orthogonality": 1, "skewness": 0,
                "flipped_cell": float(reverse), "degenerate_cell": 0}
    for name, value in expected.items():
        np.testing.assert_allclose(fields[name], value, atol=1e-12)


def test_chunk_boundary_keeps_cell_order_and_metric_selection():
    # 33,280 cells crosses the 32,768-cell batch boundary.
    grid = rectangular_grid(65, 66, 9)
    grid[..., 0] += 0.01 * grid[..., 1] * grid[..., 2]
    full = evaluate_hexahedral_grid_quality(grid)
    slices = [evaluate_hexahedral_grid_quality(grid[k:k+2]) for k in range(8)]
    for name, values in full.items():
        np.testing.assert_allclose(values, np.concatenate([part[name] for part in slices]))
        np.testing.assert_allclose(
            evaluate_hexahedral_grid_quality(grid, metrics=[name])[name], values)


def test_surface_metrics_are_rigid_rotation_invariant():
    surface = rectangular_grid()[0]
    rotation, _ = np.linalg.qr(np.random.default_rng(17).normal(size=(3, 3)))
    first = surface_quad_quality_fields(surface)
    second = surface_quad_quality_fields(surface @ rotation + [3, 4, 5])
    for name in first:
        np.testing.assert_allclose(first[name], second[name], atol=1e-12)


def test_collapsed_hex_is_degenerate():
    grid = rectangular_grid(2, 2, 2)
    grid[:, :, 1] = grid[:, :, 0]
    fields = evaluate_hexahedral_grid_quality(grid)
    assert fields["degenerate_cell"][0] == 1
    assert fields["flipped_cell"][0] == 1
    assert fields["center_jacobian"][0] == 0
