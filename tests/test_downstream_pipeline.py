"""Small archive/provider checks for the downstream evaluation pipeline."""

import json

import numpy as np

from src.data.geometry.parametric_curve_2d import BoundaryCurveSet2D, LineSegment
from src.data.mesh.sampling import logical_grid
from src.evaluation.mesh_provider import (
    boundary_geometry_error,
    load_evaluated_meshes,
    replace_with_exact_boundary,
)


def unit_square_boundary():
    return BoundaryCurveSet2D(
        LineSegment((0, 0), (1, 0)),
        LineSegment((1, 0), (1, 1)),
        LineSegment((0, 1), (1, 1)),
        LineSegment((0, 0), (0, 1)),
        metadata={"parameter_order": "computational"},
    )


def test_mesh_provider_loads_standard_archive_and_exact_control(tmp_path):
    shape = (5, 5)
    directory = tmp_path / "2D" / "demo" / "5x5" / "0.5"
    directory.mkdir(parents=True)
    exact = logical_grid(shape)
    perturbed = exact.copy()
    perturbed[:5, 1] += 0.01
    np.savez_compressed(
        directory / "generated_meshes.npz",
        PDE_Winslow=exact,
        MeshONet=perturbed,
    )
    (directory / "results.json").write_text(
        json.dumps({"timings": {"MeshONet": {"total_model_seconds": 0.2}}}),
        encoding="utf-8",
    )
    meshes, timings, _ = load_evaluated_meshes(
        tmp_path, "demo", shape, 0.5, ["pde", "meshonet"]
    )
    assert set(meshes) == {"PDE_Winslow", "MeshONet"}
    assert timings["MeshONet"]["total_model_seconds"] == 0.2
    geometry = unit_square_boundary()
    corrected = replace_with_exact_boundary(meshes["MeshONet"], geometry, shape)
    assert (
        boundary_geometry_error(corrected, geometry, shape)["boundary_max_error"] == 0.0
    )
