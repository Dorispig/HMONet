"""End-to-end evaluation tests check numeric truth, method identity, and reuse."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from scripts.evaluate.evaluate_2d import main as evaluate_2d
from scripts.evaluate.evaluate_3d import main as evaluate_3d
from src.data.geometry.boundary_generator_3d import box_boundary, cylindrical_domain_with_airfoil_hole
from src.data.mesh.sampling import logical_grid, sample_boundary, boundary_indices
from scripts.tfi.tfi_3d import tfi_from_surfaces
from scripts.pde.pde_3d import solve_winslow_3d
from src.models.model import HMONet2D, MeshONet2D, HMONet3D, MeshONet3D


class EvaluationPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path.home() / ".hmonet_test_tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="eval_", dir=temporary_root)
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def configuration(self, dimension, methods, shape=None, **options):
        shape = shape or ([5, 4] if dimension == 2 else [4, 3, 5])
        config = {
            "cases": {
                "test": {
                    "boundary_type": "rectangle" if dimension == 2 else "box",
                    "values": [None],
                    "grid_shapes": [shape],
                }
            },
            "evaluation": {
                "default_cases": ["test"],
                "methods": methods,
                "output_root": str(self.root / "results"),
                "device": "cpu",
                "save_plots": False,
                "inference_warmup_runs": 0,
                "inference_timing_repeats": 1,
                "query_batch_size": 7,
                **options,
            },
        }
        path = self.root / f"evaluate_{dimension}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    def run_evaluation(self, dimension, path, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            return (evaluate_2d if dimension == 2 else evaluate_3d)(
                ["--config", str(path), *args]
            )

    def checkpoint(self, dimension, method, mapping="current", bare=False):
        torch.manual_seed(39)
        shape = (5, 4) if dimension == 2 else (4, 3, 5)
        common = dict(
            branch_dim=4,
            trunk_dim=dimension,
            adjust_hidden_dim=4,
            branch_hidden_dim=4,
            trunk_hidden_dim=4,
            adjust_hidden_layer_num=1,
            branch_hidden_layer_num=1,
            trunk_hidden_layer_num=1,
            output_dim=4,
        )
        cls = {
            (2, "hmonet"): HMONet2D,
            (2, "meshonet"): MeshONet2D,
            (3, "hmonet"): HMONet3D,
            (3, "meshonet"): MeshONet3D,
        }[(dimension, method)]
        model = cls(
            *shape,
            **common,
            **({"hmonet_mapping_mode": mapping} if method == "hmonet" else {}),
        )
        grid = dict(zip(("nx", "ny", "nz"), shape))
        config = {
            "grid": grid,
            "model": {
                "name": method,
                "common": common,
                "hmonet": {"hmonet_mapping_mode": mapping},
            },
        }
        folder = self.root / "checkpoints" / str(dimension) / method / mapping
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
        path = folder / ("best_model.pth" if bare else "best.pt")
        payload = (
            model.state_dict()
            if bare
            else dict(
                model_state_dict=model.state_dict(),
                config=config,
                model_name=method,
                grid=grid,
                epoch=1,
            )
        )
        torch.save(payload, path)
        return path

    def test_box_tfi_and_pde_have_known_analytic_answer(self):
        shape = (5, 4, 3)
        q = logical_grid(shape)
        expected = 2 * q - 1
        np.testing.assert_allclose(
            tfi_from_surfaces(box_boundary(), parameter_grid=q), expected, atol=1e-14
        )
        mesh, info = solve_winslow_3d(box_boundary(), *shape, return_info=True)
        self.assertTrue(info["converged"])
        np.testing.assert_allclose(mesh, expected, atol=1e-12)

    def test_curved_tfi_preserves_all_faces_and_offset(self):
        geometry = cylindrical_domain_with_airfoil_hole(z_min=2.0, length=1.3)
        shape = (9, 4, 3)
        points = tfi_from_surfaces(geometry, parameter_grid=logical_grid(shape))
        np.testing.assert_allclose(
            points[boundary_indices(shape)],
            sample_boundary(geometry, shape),
            atol=1e-12,
        )
        self.assertAlmostEqual(points[:, 2].min(), 2.0)
        self.assertAlmostEqual(points[:, 2].max(), 3.3)

    def test_exact_selected_pair_both_dimensions_and_cache_repair(self):
        for dim in (2, 3):
            with self.subTest(dimension=dim):
                config = self.configuration(dim, ["PDE", "TFI"])
                first = self.run_evaluation(dim, config)[0]
                self.assertEqual(first["method_order"], ["TFI", "PDE_Winslow"])
                group = Path(first["comparison_directory"])
                self.assertEqual(group.name, "01_tfi_pde")
                with np.load(group / "comparison_mesh_data.npz") as meshes:
                    self.assertEqual(meshes.files, ["TFI", "PDE_Winslow"])
                    np.testing.assert_allclose(
                        meshes["TFI"], meshes["PDE_Winslow"], atol=1e-12
                    )
                with patch(
                    "src.evaluation.generation.solve_pde",
                    side_effect=AssertionError("must reuse generation"),
                ):
                    again = self.run_evaluation(dim, config, "--analysis-only")[0]
                    self.assertTrue(
                        all(v["cache_hit"] for v in again["timings"].values())
                    )
                    # Missing analysis must be rebuilt from cached generation.
                    (group / "grid_quality/quality_summary.csv").unlink()
                    self.run_evaluation(dim, config, "--analysis-only")
                    self.assertTrue(
                        (group / "grid_quality/quality_summary.csv").is_file()
                    )
                self.assertFalse((group.parents[1] / "HMONet").exists())

    def test_networks_raw_errors_input_resolution_and_projection(self):
        for dim in (2, 3):
            with self.subTest(dimension=dim):
                hp = self.checkpoint(dim, "hmonet")
                mp = self.checkpoint(dim, "meshonet", bare=True)
                query_shape = [7, 6] if dim == 2 else [3, 4, 3]
                config = self.configuration(
                    dim,
                    ["hmonet", "meshonet", "pde", "tfi"],
                    query_shape,
                    hmonet_checkpoint=str(hp),
                    meshonet_checkpoint=str(mp),
                    # This test intentionally exercises the optional direct
                    # projection path; production evaluation defaults to none.
                    meshonet_boundary_projection="direct",
                )
                record = self.run_evaluation(dim, config)[0]
                self.assertEqual(
                    record["method_order"], ["TFI", "PDE_Winslow", "MeshONet", "HMONet"]
                )
                group = Path(record["comparison_directory"])
                root = group.parents[3]
                raw = np.load(root / "MeshONet/raw/mesh.npy")
                processed = np.load(root / "MeshONet/direct/mesh.npy")
                ids = boundary_indices(query_shape)
                exact = 2 * logical_grid(query_shape)[ids] - 1
                self.assertGreater(np.max(np.abs(raw[ids] - exact)), 0.1)
                np.testing.assert_allclose(processed[ids], exact, atol=1e-12)
                expected = np.linalg.norm(raw[ids] - exact, axis=1)
                item = next(
                    r
                    for r in record["boundary_errors"]
                    if r["method"] == "MeshONet" and r["stage"] == "raw"
                )
                self.assertAlmostEqual(item["boundary_max_error"], expected.max())
                self.assertAlmostEqual(item["boundary_mean_error"], expected.mean())
                self.assertTrue(
                    (root / "HMONet/current/raw/boundary_error_data.csv").is_file()
                )
                np.testing.assert_array_equal(
                    np.load(root / "HMONet/current/none/mesh.npy"),
                    np.load(root / "HMONet/current/raw/mesh.npy"),
                )
                self.assertEqual(
                    record["timings"]["HMONet"]["boundary_projection_seconds"], 0.0
                )
                self.assertIn("hmonet_none__meshonet_direct", str(group))
                self.assertEqual(
                    tuple(record["timings"]["HMONet"]["training_grid"]),
                    (5, 4) if dim == 2 else (4, 3, 5),
                )
                with patch(
                    "src.evaluation.generation.predict",
                    side_effect=AssertionError(
                        "must not infer when changing projection"
                    ),
                ):
                    again = self.run_evaluation(
                        dim,
                        config,
                        "--meshonet-boundary-projection",
                        "inverse_distance",
                    )[0]
                self.assertTrue(again["timings"]["MeshONet"]["cache_hit"])
                self.assertNotEqual(
                    again["comparison_directory"], record["comparison_directory"]
                )
                np.testing.assert_array_equal(
                    np.load(root / "MeshONet/raw/mesh.npy"), raw
                )
                self.assertTrue((group / "paper_timing_table.csv").is_file())

    def test_missing_checkpoint_is_not_replaced_with_tfi(self):
        config = self.configuration(2, ["tfi", "meshonet"])
        with patch(
            "scripts.evaluate.evaluation_common.find_checkpoint", return_value=None
        ):
            record = self.run_evaluation(2, config)[0]
        self.assertEqual(record["method_order"], ["TFI"])
        self.assertEqual(record["skipped_methods"][0]["method"], "MeshONet")
        self.assertEqual(Path(record["comparison_directory"]).name, "01_tfi")

    def test_pde_failure_never_becomes_a_tfi_mesh(self):
        config = self.configuration(2, ["pde"])
        with patch(
            "src.evaluation.generation.solve_pde",
            side_effect=RuntimeError("solver failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "solver failed"):
                self.run_evaluation(2, config)
        self.assertFalse(any((self.root / "results").rglob("comparison_mesh_data.npz")))

    def test_checkpoint_model_mismatch_is_rejected(self):
        checkpoint = self.checkpoint(2, "hmonet")
        config = self.configuration(
            2, ["meshonet"], meshonet_checkpoint=str(checkpoint)
        )
        with self.assertRaisesRegex(ValueError, "Checkpoint is hmonet"):
            self.run_evaluation(2, config)

    def test_pde_options_invalidate_cache_and_force_regenerates(self):
        config = self.configuration(2, ["pde"])
        self.run_evaluation(2, config)
        from src.evaluation.generation import solve_pde

        with patch("src.evaluation.generation.solve_pde", wraps=solve_pde) as solver:
            self.run_evaluation(2, config, "--winslow-tolerance", "1e-7")
            self.assertEqual(solver.call_count, 1)
            self.run_evaluation(2, config, "--winslow-tolerance", "1e-7", "--force")
            self.assertEqual(solver.call_count, 2)

    def test_coons_checkpoint_and_local_worst_without_plots(self):
        for dim in (2, 3):
            with self.subTest(dimension=dim):
                hp = self.checkpoint(dim, "hmonet", mapping="coons_residual")
                config = self.configuration(
                    dim,
                    ["tfi", "hmonet"],
                    hmonet_checkpoint=str(hp),
                    hmonet_mapping_mode="coons_residual",
                )
                record = self.run_evaluation(
                    dim, config, "--center-on", "worst", "--metric", "scaled_jacobian"
                )[0]
                self.assertIn("coons_residual", record["comparison_directory"])
                self.assertTrue(
                    any(
                        (Path(record["comparison_directory"]) / "zoom_views").rglob(
                            "*.json"
                        )
                    )
                )

    def test_middle_sections_intersection_detection_is_not_constant_zero(self):
        from src.evaluation.quality_3d import surface_quad_quality_fields

        crossed = np.array(
            [[[0, 0, 0], [1, 1, 0]], [[1, 0, 0], [0, 1, 0]]], dtype=float
        )
        fields = surface_quad_quality_fields(crossed)
        self.assertEqual(fields["self_intersection"].tolist(), [1.0])
        from src.evaluation.quality_output import summarize_fields

        summary = summarize_fields(fields)
        self.assertEqual(summary["self_intersection"]["affected_cell_fraction"], 1.0)

    def test_changed_checkpoint_invalidates_generation(self):
        checkpoint = self.checkpoint(2, "meshonet")
        config = self.configuration(
            2, ["meshonet"], meshonet_checkpoint=str(checkpoint)
        )
        self.run_evaluation(2, config)
        payload = torch.load(checkpoint, weights_only=False)
        parameter = next(iter(payload["model_state_dict"]))
        payload["model_state_dict"][parameter] += 0.03
        torch.save(payload, checkpoint)
        from src.evaluation.generation import predict

        with patch("src.evaluation.generation.predict", wraps=predict) as prediction:
            result = self.run_evaluation(2, config)[0]
            self.assertEqual(prediction.call_count, 1)
        self.assertFalse(result["timings"]["MeshONet"]["cache_hit"])

    def test_auto_checkpoint_selection_filters_model_and_mapping(self):
        from src.evaluation.inference import find_checkpoint

        for model, mapping in (
            ("HMONet", "current"),
            ("HMONet", "coons_residual"),
            ("MeshONet", ""),
        ):
            folder = self.root / "outputs/checkpoints/2D/test" / model
            if mapping:
                folder /= mapping
            folder /= "20260915_2359"
            folder.mkdir(parents=True)
            (folder / "best.pt").write_bytes(b"selection fixture")
        with patch("src.evaluation.inference.ROOT", self.root):
            for model, mapping in (
                ("HMONet", "current"),
                ("HMONet", "coons_residual"),
                ("MeshONet", "current"),
            ):
                selected = find_checkpoint(
                    2, "test", model, mapping, {}, "interpolation"
                )
                self.assertIn(model, selected.parts)
                if model == "HMONet":
                    self.assertIn(mapping, selected.parts)

    def test_binary_colors_are_shared_with_zoom_and_no_false_midpoint(self):
        from src.evaluation.quality_output import quality_plot_style

        _, norm = quality_plot_style("self_intersection", np.zeros(10))
        self.assertEqual((norm.vmin, norm.vmax), (0.0, 1.0))
        self.assertEqual(norm(0), 0)

    def test_generation_only_then_analysis_only_and_parameter_override(self):
        config = self.configuration(2, ["tfi", "pde"])
        first = self.run_evaluation(2, config, "--generation-only")[0]
        self.assertIsNone(first["comparison_directory"])
        second = self.run_evaluation(2, config, "--analysis-only")[0]
        self.assertTrue(all(v["cache_hit"] for v in second["timings"].values()))
        self.assertTrue(Path(second["comparison_directory"]).is_dir())

    def test_refinement_runs_from_hmonet_and_reuses_cache(self):
        checkpoint = self.checkpoint(2, "hmonet")
        config = self.configuration(
            2, ["hmonet"], hmonet_checkpoint=str(checkpoint), hmonet_pde_iterations=2
        )
        first = self.run_evaluation(2, config)[0]
        self.assertEqual(first["method_order"], ["HMONet", "HMONet_PDE_refined"])
        with patch(
            "scripts.evaluate.evaluation_common.solve_pde",
            side_effect=AssertionError("must reuse refinement"),
        ):
            self.run_evaluation(2, config, "--analysis-only")

    def test_force_preserves_unrelated_user_files(self):
        config = self.configuration(2, ["tfi"])
        record = self.run_evaluation(2, config)[0]
        group = Path(record["comparison_directory"])
        note = group / "research_notes.txt"
        note.write_text("user-owned notes", encoding="utf-8")
        self.run_evaluation(2, config)
        self.run_evaluation(2, config, "--force")
        self.assertEqual(note.read_text(encoding="utf-8"), "user-owned notes")

    def test_changed_generated_array_invalidates_analysis(self):
        config = self.configuration(2, ["tfi"])
        first = self.run_evaluation(2, config)[0]
        group = Path(first["comparison_directory"])
        with np.load(group / "comparison_mesh_data.npz") as data:
            before = data["TFI"].copy()
        from src.evaluation.generation import tfi

        def altered(*args, **kwargs):
            result = tfi(*args, **kwargs).copy()
            result[6, 0] += 0.01
            return result

        with patch("src.evaluation.generation.tfi", side_effect=altered):
            self.run_evaluation(2, config, "--force-regenerate")
        with np.load(group / "comparison_mesh_data.npz") as data:
            self.assertAlmostEqual(data["TFI"][6, 0] - before[6, 0], 0.01)

    def test_full_forward_timing_and_generated_result_reuse(self):
        hp = self.checkpoint(2, "hmonet")
        mp = self.checkpoint(2, "meshonet")
        config = self.configuration(
            2,
            ["meshonet", "hmonet"],
            hmonet_checkpoint=str(hp),
            meshonet_checkpoint=str(mp),
        )
        first = self.run_evaluation(2, config)[0]
        htiming = first["timings"]["HMONet"]
        self.assertFalse(any("precompute" in key for key in htiming))
        self.assertEqual(
            htiming["total_model_seconds"],
            htiming["network_inference_seconds"]
            + htiming["boundary_projection_seconds"],
        )
        table = Path(first["comparison_directory"]) / "paper_timing_table.csv"
        self.assertNotIn(
            "Projection precomputation time (s)", table.read_text(encoding="utf-8-sig")
        )
        with patch(
            "src.evaluation.generation.predict",
            side_effect=AssertionError("must reuse generation"),
        ):
            second = self.run_evaluation(2, config, "--analysis-only")[0]
        self.assertTrue(second["timings"]["MeshONet"]["cache_hit"])
        self.assertTrue(second["timings"]["HMONet"]["cache_hit"])

    def test_parameter_batches_save_separate_results_and_reuse_them(self):
        for dim in (2, 3):
            hp = self.checkpoint(dim, "hmonet")
            mp = self.checkpoint(dim, "meshonet")
            path = self.configuration(
                dim,
                ["hmonet", "meshonet"],
                hmonet_checkpoint=str(hp),
                meshonet_checkpoint=str(mp),
                inference_batch_size=2,
            )
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            config["cases"]["test"].update(
                parameter="x_max" if dim == 2 else "xmax", values=[1.0, 1.2, 1.4]
            )
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            records = self.run_evaluation(dim, path)
            self.assertEqual(len(records), 3)
            self.assertEqual(
                [r["timings"]["HMONet"]["inference_batch_size"] for r in records],
                [2, 2, 1],
            )
            self.assertEqual(len({r["comparison_directory"] for r in records}), 3)
            for record in records:
                for method in ("HMONet", "MeshONet"):
                    self.assertFalse(record["timings"][method]["cache_hit"])
                self.assertTrue(
                    (
                        Path(record["comparison_directory"])
                        / "comparison_mesh_data.npz"
                    ).is_file()
                )
            with patch(
                "src.evaluation.inference.predict_batch",
                side_effect=AssertionError("batch must be reused"),
            ):
                cached = self.run_evaluation(dim, path)
            self.assertTrue(all(r["timings"]["HMONet"]["cache_hit"] for r in cached))
            # CLI overrides both execution and cache identity for single-object timing.
            single = self.run_evaluation(dim, path, "--inference-batch-size", "1")
            for before, after in zip(records, single):
                self.assertEqual(after["timings"]["HMONet"]["inference_batch_size"], 1)
                self.assertFalse(after["timings"]["HMONet"]["cache_hit"])

    def test_point_and_cell_mse_with_raw_outputs_and_optional_distribution(self):
        from src.evaluation.analysis import save_point_errors
        from src.data.mesh.topology import HexMeshXYZ, QuadMeshXY

        for dimension, shape in ((2, (3, 3)), (3, (3, 3, 3))):
            with self.subTest(dimension=dimension):
                directory = self.root / f"errors_{dimension}"
                directory.mkdir()
                reference = logical_grid(shape)
                predicted = reference.copy()
                predicted[len(reference) // 2, 0] += 0.3
                raw = predicted + 0.1
                options = dict(
                    save_plots=True,
                    save_png=True,
                    save_pdf=False,
                    save_distribution_plots=False,
                )
                meshes = {"PDE_Winslow": reference, "MeshONet": predicted}
                summary = save_point_errors(
                    directory, meshes, {"MeshONet": raw}, shape, options
                )
                expected = np.mean((predicted - reference) ** 2, axis=1)
                cells = (
                    QuadMeshXY(*shape, reference).faces
                    if dimension == 2
                    else HexMeshXYZ(*shape, reference).cells
                )
                with np.load(directory / "point_loss_data.npz") as data:
                    np.testing.assert_allclose(data["MeshONet"], expected)
                    np.testing.assert_allclose(
                        data["MeshONet_raw"], np.mean((raw - reference) ** 2, axis=1)
                    )
                with np.load(directory / "cell_loss_data.npz") as data:
                    np.testing.assert_allclose(
                        data["MeshONet"], expected[cells].max(axis=1)
                    )
                self.assertAlmostEqual(
                    summary["MeshONet"]["mse_vs_winslow"], expected.mean()
                )
                self.assertTrue((directory / "08_point_loss_values.png").is_file())
                self.assertFalse((directory / "09_point_loss_comparison.png").exists())
                options["save_distribution_plots"] = True
                save_point_errors(directory, meshes, {"MeshONet": raw}, shape, options)
                self.assertTrue((directory / "09_point_loss_comparison.png").is_file())


if __name__ == "__main__":
    unittest.main()

