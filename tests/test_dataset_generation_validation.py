"""A quality failure is reported only after generated samples are saved."""

import json

import numpy as np
import pytest

from src.data.generation import make_dataset_parametric_2d as gen2d
from src.data.generation import make_dataset_parametric_3d as gen3d


def test_2d_saves_all_samples_and_audits_only_endpoints(tmp_path, monkeypatch):
    calls = []

    def solve(_boundary, nx, ny, *, initial_grid, **_options):
        grid = initial_grid.copy().reshape(ny, nx, 2)
        if len(calls) >= 1:
            grid[2, 2] = [3.0, 3.0]
        calls.append(grid.copy())
        return grid.reshape(-1, 2), {"converged": True, "iterations": 1}

    monkeypatch.setattr(gen2d, "solve_winslow_2d", solve)
    with pytest.warns(RuntimeWarning, match="1/2 checked endpoint grids are invalid"):
        gen2d.generate_explicit_dataset(
            "invalid-endpoint", "rectangle", [{"x_max": x_max} for x_max in (1.0, 1.2, 1.4)],
            grid_shape=(5, 5), output_root=tmp_path, timing_samples=0,
        )

    directory = tmp_path / "invalid-endpoint"
    dataset = np.load(directory / "dataset.npy", allow_pickle=True)
    geometry = np.load(directory / "boundary_geometry.npy", allow_pickle=True)
    audit = json.loads((directory / "quality_audit.json").read_text(encoding="utf-8"))
    assert len(dataset) == len(geometry) == len(calls) == 3
    assert audit["illegal_sample_ids"] == [2]
    assert [sample["sample_id"] for sample in audit["samples"]] == [0, 2]
    assert audit["nonconverged_samples"] == 0
    np.testing.assert_allclose(dataset[1]["mesh_points"], calls[1].reshape(-1, 2))


def test_3d_saves_all_samples_after_breakdown_and_invalid_grid(tmp_path, monkeypatch):
    calls = []

    def solve(_boundary, _nx, _ny, _nz, *, initial_grid, **_options):
        grid = initial_grid.copy()
        calls.append(grid)
        if len(calls) == 1:
            raise FloatingPointError("singular iteration")
        if len(calls) >= 2:
            grid[2, 2, 2] = [10, 10, 10]
        return grid, {"converged": True}

    monkeypatch.setattr(gen3d, "solve_winslow_3d", solve)
    with pytest.warns(RuntimeWarning, match="2/3 checked representative grids are invalid"):
        gen3d.generate_dataset(
            "invalid-volume", "sidewall_deformed_block",
            [{"sidewall_amplitude": amplitude} for amplitude in (0.05, 0.10, 0.15)],
            grid_shape=(5, 5, 5), output_root=tmp_path, timing_samples=0,
        )

    directory = tmp_path / "invalid-volume"
    points = np.load(directory / "pde_mesh.npy")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    audit = json.loads((directory / "quality_audit.json").read_text(encoding="utf-8"))
    assert points.shape == (3, 5, 5, 5, 3)
    np.testing.assert_allclose(points, calls)
    assert audit["illegal_sample_ids"] == [1, 2]
    assert [sample["sample_id"] for sample in audit["samples"]] == [0, 1, 2]
    assert audit["nonconverged_sample_ids"] == [0]
    assert audit["samples"][0]["mesh_source"] == "tfi_fallback"
    assert metadata["size"] == 3


def test_3d_invalid_face_stops_before_solving_or_creating_output(tmp_path, monkeypatch):
    original_sample_faces = gen3d.sample_faces

    def invalid_face(geometry, shape):
        faces, boundary = original_sample_faces(geometry, shape)
        face = faces["x_min"].copy()
        face[1, 1] = face[1, 2]
        return {**faces, "x_min": face}, boundary

    monkeypatch.setattr(gen3d, "sample_faces", invalid_face)
    monkeypatch.setattr(
        gen3d, "solve_winslow_3d",
        lambda *args, **kwargs: pytest.fail("PDE solver ran with an invalid face"),
    )
    with pytest.raises(ValueError, match="invalid quadrilaterals.*x_min"):
        gen3d.generate_dataset(
            "invalid-face", "sidewall_deformed_block", [{"sidewall_amplitude": 0.10}],
            grid_shape=(5, 5, 5), output_root=tmp_path, timing_samples=0,
        )
    assert not (tmp_path / "invalid-face").exists()


def test_rounded_u_notch_extrusion_uses_3d_winslow(monkeypatch):
    spec = gen3d.DATASET_FAMILIES["rounded_u_notch_extrusion"]
    parameters = {**spec["fixed_parameters"], "depth": 0.55}
    calls = []

    def solve(_geometry, nx, ny, nz, *, initial_grid, **_options):
        assert initial_grid.shape == (nz, ny, nx, 3)
        calls.append((nx, ny, nz))
        return initial_grid, {"converged": True}

    monkeypatch.setattr(gen3d, "solve_winslow_3d", solve)
    _, _, _, info, _ = gen3d._solve_dataset_sample_3d(
        (0, spec["function"], parameters, 41, 41, 11, 1000, 1e-6)
    )
    assert calls == [(41, 41, 11)]
    assert info["solver"] == "winslow_3d"
