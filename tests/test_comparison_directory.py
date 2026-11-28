"""Comparison output paths remain compact and collision safe."""

from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate.evaluation_common import comparison_directory


class TestComparisonDirectory:
    def options(self, pimeshonet_projection="none"):
        return {
            "hmonet_boundary_projection": "none",
            "meshonet_boundary_projection": "none",
            "pimeshonet_boundary_projection": pimeshonet_projection,
        }

    def test_default_five_method_path_has_no_extra_pi_layer(self):
        names = ["TFI", "PDE_Winslow", "MeshONet", "PI-MeshONet", "HMONet"]
        actual = comparison_directory(Path("result"), "rational_affine", self.options(), names)
        expected = (
            Path("result")
            / "comparisons"
            / "RAP"
            / "H_raw__M_raw"
            / "TFI_PDE_M_PI_H"
        )
        assert (actual) == (expected)
        assert ("PI_raw") not in (actual.parts)

    def test_nondefault_pi_projection_is_encoded_without_extra_level(self):
        names = ["TFI", "MeshONet", "PI-MeshONet", "HMONet"]
        actual = comparison_directory(
            Path("result"), "rational_affine", self.options("direct"), names
        )
        assert (actual.parent.name) == ("H_raw__M_raw__PI_direct")
        assert (actual.name) == ("TFI_M_PI_H")

    def test_pi_only_comparison_keeps_one_projection_directory(self):
        actual = comparison_directory(
            Path("result"), "rational_affine", self.options(), ["TFI", "PI-MeshONet"]
        )
        assert (actual) == (
            Path("result") / "comparisons" / "PI_raw" / "TFI_PI"
        )


    def test_resmetamesh_single_and_joint_paths(self):
        root = Path("result")
        assert comparison_directory(root, "CCAP", self.options(), ["ResMetaMesh"]) == root / "ResMetaMesh/raw"
        names = ["TFI", "PDE_Winslow", "MeshONet", "PI-MeshONet", "HMONet", "ResMetaMesh"]
        assert comparison_directory(root, "CCAP", self.options(), names) == (
            root / "comparisons/CCAP/H_raw__M_raw/TFI_PDE_M_PI_H_RM"
        )


@pytest.mark.parametrize("dimension,boundary_type,fixed", [
    (2, "four_convex_arcs", {"arc_angle_deg": 60.0, "chord": 1.0}),
    (3, "box", {}),
])
def test_resmetamesh_single_evaluation_publishes_reusable_raw_only(
    tmp_path, monkeypatch, dimension, boundary_type, fixed
):
    import json
    from scripts.evaluate import evaluation_common
    from src.data.mesh.sampling import logical_grid
    from src.evaluation import resmetamesh
    from src.evaluation.configuration import load_configuration

    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"test-checkpoint")
    config_file = tmp_path / "evaluate.yaml"
    config_file.write_text(json.dumps({
        "cases": {"case": {"boundary": boundary_type, "split": "extrapolation",
                           "fixed_parameters": fixed, "grid_shapes": [[5] * dimension]}},
    }), encoding="utf-8")
    root = tmp_path / f"{dimension}D" / "case" / "x".join(["5"] * dimension) / "default"
    comparison = root / "comparisons/CCAP/H_raw__M_raw/TFI_PDE_M_PI_H_RM"
    comparison.mkdir(parents=True)
    joint_result = root / "results.json"
    joint_result.write_text('{"previous":"joint evaluation"}', encoding="utf-8")
    existing = comparison / "evaluation_result.json"
    existing.write_text('{"previous":"comparison"}', encoding="utf-8")
    calls = []

    def generate_mesh(geometry, shape, *args):
        calls.append(shape)
        return logical_grid(shape).astype(np.float32), {"total_seconds": 1.0}

    monkeypatch.setattr(resmetamesh, "generate_resmetamesh", generate_mesh)
    monkeypatch.setattr(evaluation_common, "implementation_version", lambda: "test")
    command = [
        "--config", str(config_file), "--case", "case", "--methods", "resmetamesh",
        "--resmetamesh-checkpoint", str(checkpoint), "--output-root", str(tmp_path),
        "--device", "cpu", "--no-plots", "--inference-batch-size", "2",
    ]

    def evaluate(extra):
        args, config = load_configuration(dimension, command + extra)
        [result] = evaluation_common.run_evaluation(dimension, args, config)
        return result

    first = evaluate([])
    raw = root / "ResMetaMesh/raw"
    assert Path(first["comparison_directory"]) == raw
    assert calls == [(5,) * dimension]
    for file in ("mesh.npy", "evaluation_result.json", "generation_metadata.json",
                 "boundary_error_summary.json", "grid_quality/quality_summary.json"):
        assert (raw / file).is_file()
    if dimension == 3:
        assert (raw / "boundary_faces/z_mid/quality_summary.csv").is_file()
    assert not (root / "comparisons/RM").exists()
    assert not (root / ".cache/ResMetaMesh").exists()
    assert joint_result.read_text(encoding="utf-8") == '{"previous":"joint evaluation"}'
    assert existing.read_text(encoding="utf-8") == '{"previous":"comparison"}'
    assert not (root / "evaluation_parameters.json").exists()

    second = evaluate(["--analysis-only"])
    assert calls == [(5,) * dimension]
    assert second["timings"]["ResMetaMesh"]["cache_hit"] is True
