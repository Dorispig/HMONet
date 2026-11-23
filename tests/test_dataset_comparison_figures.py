import numpy as np

from scripts.evaluate.generate_dataset_comparison_figures import (
    BOUNDARY_PROCESSING_GROUPS,
    boundary_indices,
    mse_statistics,
    quality_axis_limits,
    selected_metrics,
    worst_key,
)


def test_boundary_indices_2d_are_unique_perimeter_nodes():
    indices = boundary_indices((5, 4))
    assert len(indices) == 2 * 5 + 2 * (4 - 2)
    assert len(np.unique(indices)) == len(indices)
    assert {0, 4, 15, 19}.issubset(set(indices.tolist()))


def test_boundary_indices_3d_are_unique_face_nodes():
    shape = (4, 5, 3)
    indices = boundary_indices(shape)
    interior = (shape[0] - 2) * (shape[1] - 2) * (shape[2] - 2)
    assert len(indices) == np.prod(shape) - interior
    assert len(np.unique(indices)) == len(indices)


def test_quality_groups_produce_six_figures_per_metric():
    assert set(BOUNDARY_PROCESSING_GROUPS) == {"raw", "direct", "idw"}
    assert all(len(group) == 5 for group in BOUNDARY_PROCESSING_GROUPS.values())
    assert BOUNDARY_PROCESSING_GROUPS["raw"] == (
        "TFI", "PDE", "MeshONet raw", "PI-MeshONet raw", "HMONet"
    )
    assert len(BOUNDARY_PROCESSING_GROUPS) * 2 == 6


def test_worst_direction_matches_metric_definition():
    assert worst_key("scaled_jacobian") == "minimum"
    assert worst_key("min_angle") == "minimum"
    assert worst_key("max_angle") == "maximum"
    assert worst_key("aspect_ratio") == "maximum"
    assert worst_key("minimum_scaled_jacobian") == "minimum"
    assert worst_key("center_condition_number") == "maximum"


def test_selected_metrics_excludes_removed_2d_cell_area():
    metrics = selected_metrics(2, ("all",))
    assert "cell_area" not in metrics
    assert "scaled_jacobian" in metrics
    assert "minimum_scaled_jacobian" in selected_metrics(3, ("all",))


def test_quality_axis_limits_zoom_close_positive_values_without_hiding_zero_counts():
    lower, upper = quality_axis_limits("scaled_jacobian", [0.9613, 0.9542, 0.9537, 0.9568, 0.9543])
    assert 0.0 < lower < 0.9537
    assert upper > 0.9613
    assert quality_axis_limits("intersecting_cell", [0.0, 0.0, 1.0]) == (0.0, None)


def test_mse_statistics_matches_evaluation_coordinate_definition():
    reference = np.array([[0.0, 0.0], [1.0, 1.0]])
    meshes = {
        "PDE": reference.copy(),
        "HMONet": np.array([[1.0, 1.0], [1.0, 3.0]]),
    }
    result = mse_statistics(meshes, reference)
    assert result["PDE"]["mse_vs_pde"]["mean"] == 0.0
    assert result["HMONet"]["mse_vs_pde"]["mean"] == 1.5
    assert result["HMONet"]["mse_vs_pde"]["worst"] == 2.0
