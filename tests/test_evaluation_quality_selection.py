"""Verify metric gates skip work, preserve values, and invalidate only analysis."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from tests import test_evaluation_pipeline as pipeline
from src.data.mesh.sampling import logical_grid
from src.evaluation.configuration import build_parser, evaluation_options, DEFAULTS
from src.evaluation.inference import summarize_inference_samples
from src.evaluation.quality_output import quad_quality_fields
from src.evaluation.quality_3d import surface_quad_quality_fields
from src.evaluation.zoom_3d import generate_zoom
from src.utils.metrics import evaluate_hexahedral_grid_quality, QuadMeshQualityEvaluator
from src.utils.quality_selection import (
    QUAD_METRICS,
    HEX_METRICS,
    selected_quality_metrics,
)


class QualitySelectionTests(unittest.TestCase):
    setUp = pipeline.EvaluationPipelineTests.setUp
    configuration = pipeline.EvaluationPipelineTests.configuration
    run_evaluation = pipeline.EvaluationPipelineTests.run_evaluation
    checkpoint = pipeline.EvaluationPipelineTests.checkpoint

    def test_defaults_cli_case_switches_and_validation(self):
        for dimension in (2, 3):
            parser = build_parser(dimension)
            defaults = evaluation_options({}, {}, parser.parse_args([]))
            self.assertEqual(defaults["inference_warmup_runs"], 3)
            self.assertEqual(defaults["inference_timing_repeats"], 10)
            names = QUAD_METRICS if dimension == 2 else HEX_METRICS
            self.assertEqual(selected_quality_metrics(defaults, dimension), names)
            config = {"evaluation": {"quality_switches": {"min_angle": False}}}
            args = parser.parse_args(
                [
                    "--quality-metrics",
                    "min_angle,skewness",
                    "--quality-min-angle",
                    "--no-quality-skewness",
                    "--inference-warmup-runs",
                    "2",
                    "--inference-timing-repeats",
                    "7",
                ]
            )
            options = evaluation_options(config, {}, args)
            self.assertEqual(
                selected_quality_metrics(options, dimension), ("min_angle",)
            )
            self.assertEqual(options["inference_timing_repeats"], 7)
            self.assertEqual(options["inference_warmup_runs"], 2)
            disabled = evaluation_options(
                {}, {}, parser.parse_args(["--no-compute-grid-quality"])
            )
            self.assertFalse(selected_quality_metrics(disabled, dimension))
            for bad in ("typo", "both,min_angle"):
                with self.assertRaises(ValueError):
                    evaluation_options(
                        {}, {}, parser.parse_args(["--quality-metrics", bad])
                    )
            with self.assertRaises(ValueError):
                selected_quality_metrics(
                    {"quality_switches": {"min_angle": "false"}}, dimension
                )

    def test_timing_trim_exactly_two_and_short_repeat_fallback(self):
        # Ties at the extremes must not discard every repeated extreme value.
        values = [1, 1, 2, 3, 4, 5, 6, 7, 100, 100]
        mean, info = summarize_inference_samples(values)
        self.assertEqual(mean, 16.0)
        self.assertEqual(info["inference_used_samples"], 8)
        self.assertEqual(info["inference_discarded_samples"], 2)
        for values in ([2], [2, 6]):
            mean, info = summarize_inference_samples(values)
            self.assertEqual(mean, np.mean(values))
            self.assertEqual(info["inference_timing_statistic"], "mean")
        for invalid in ([], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                summarize_inference_samples(invalid)

    def test_each_metric_matches_full_calculation_on_deformed_grid(self):
        rng = np.random.default_rng(59)
        grid = logical_grid((5, 4, 3)).reshape(3, 4, 5, 3)
        grid += rng.normal(0, 0.22, grid.shape)
        surface = grid[1]
        quad = surface[..., :2].reshape(-1, 2)
        calculators = (
            lambda names: quad_quality_fields(quad, 5, 4, metrics=names)[0],
            lambda names: surface_quad_quality_fields(surface, metrics=names),
            lambda names: evaluate_hexahedral_grid_quality(grid, metrics=names),
        )
        for calculate in calculators:
            full = calculate(None)
            for name, expected in full.items():
                single = calculate([name])
                self.assertEqual(list(single), [name])
                np.testing.assert_array_equal(single[name], expected)
            self.assertEqual(calculate([]), {})

    def test_disabled_expensive_operations_are_not_called(self):
        grid = logical_grid((3, 3, 3)).reshape(3, 3, 3, 3)
        with patch.object(
            QuadMeshQualityEvaluator, "IntersectingCells", side_effect=AssertionError
        ), patch.object(
            QuadMeshQualityEvaluator,
            "SelfIntersectingCells",
            side_effect=AssertionError,
        ), patch.object(
            QuadMeshQualityEvaluator, "Angle", side_effect=AssertionError
        ):
            fields, _ = quad_quality_fields(
                grid[0, ..., :2].reshape(-1, 2), 3, 3, metrics=["cell_area"]
            )
            np.testing.assert_allclose(fields["cell_area"], 0.25)
        with patch(
            "src.evaluation.quality_3d._segments_intersect_2d",
            side_effect=AssertionError,
        ), patch("numpy.arccos", side_effect=AssertionError):
            fields = surface_quad_quality_fields(grid[0], metrics=["cell_area"])
            np.testing.assert_allclose(fields["cell_area"], 0.25)
        with patch("numpy.linalg.cond", side_effect=AssertionError), patch(
            "numpy.arccos", side_effect=AssertionError
        ):
            fields = evaluate_hexahedral_grid_quality(grid, metrics=["cell_volume"])
            np.testing.assert_allclose(fields["cell_volume"], 0.125)

    def test_quality_off_keeps_mesh_and_slice_images_without_calling_evaluators(self):
        for dim in (2, 3):
            config = self.configuration(dim, ["tfi"], save_plots=True)
            with patch(
                "src.evaluation.analysis.compute_quad_quality",
                side_effect=AssertionError,
            ), patch(
                "src.evaluation.analysis.evaluate_hexahedral_grid_quality",
                side_effect=AssertionError,
            ), patch(
                "src.evaluation.analysis.surface_quad_quality_fields",
                side_effect=AssertionError,
            ):
                result = self.run_evaluation(dim, config, "--no-compute-grid-quality")[
                    0
                ]
                group = Path(result["comparison_directory"])
                root = group.parents[1]
                self.assertEqual(result["metrics"], {"TFI": {}})
                self.assertTrue((group / "mesh_comparison.png").is_file())
                self.assertTrue((root / "TFI/mesh.png").is_file())
                self.assertFalse(list(root.rglob("quality_*.png")))
                self.assertFalse(list(root.rglob("quality_data.npz")))
                if dim == 3:
                    for section in ("x_mid", "y_mid", "z_mid"):
                        self.assertTrue(
                            (group / "boundary_faces" / section / "mesh.png").is_file()
                        )
                        self.assertTrue(
                            (
                                root / "TFI/boundary_faces" / section / "mesh.png"
                            ).is_file()
                        )
                again = self.run_evaluation(
                    dim, config, "--no-compute-grid-quality", "--analysis-only"
                )[0]
                self.assertTrue(again["timings"]["TFI"]["cache_hit"])

    def test_selection_changes_reuse_mesh_remove_stale_metrics_and_keep_only_requested(
        self,
    ):
        for dim in (2, 3):
            config = self.configuration(dim, ["pde", "tfi"])
            first = self.run_evaluation(dim, config)[0]
            group = Path(first["comparison_directory"])
            with patch(
                "src.evaluation.generation.solve_pde", side_effect=AssertionError
            ):
                selected = self.run_evaluation(
                    dim, config, "--quality-metrics", "min_angle", "--analysis-only"
                )[0]
                self.assertTrue(selected["timings"]["PDE_Winslow"]["cache_hit"])
                with np.load(
                    group / "grid_quality/comparison_quality_data.npz"
                ) as data:
                    self.assertEqual(
                        set(data.files), {"TFI__min_angle", "PDE_Winslow__min_angle"}
                    )
                if dim == 3:
                    for section in ("x_mid", "y_mid", "z_mid"):
                        with np.load(
                            group
                            / "boundary_faces"
                            / section
                            / "comparison_quality_data.npz"
                        ) as data:
                            self.assertEqual(
                                set(data.files),
                                {"TFI__min_angle", "PDE_Winslow__min_angle"},
                            )
                    self.assertFalse((group / "self_intersection_table.csv").exists())
                self.run_evaluation(
                    dim, config, "--quality-metrics", "none", "--analysis-only"
                )
                self.assertFalse(
                    (group / "grid_quality/comparison_quality_data.npz").exists()
                )
                self.assertFalse(
                    list((group.parents[1] / "TFI").rglob("quality_data.npz"))
                )
                self.run_evaluation(
                    dim, config, "--quality-metrics", "both", "--analysis-only"
                )

    def test_3d_volume_only_and_surface_only_selection(self):
        config = self.configuration(3, ["tfi"])
        for metric, volume in (("cell_volume", True), ("cell_area", False)):
            result = self.run_evaluation(3, config, "--quality-metrics", metric)[0]
            group = Path(result["comparison_directory"])
            self.assertEqual(
                (group / "grid_quality/comparison_quality_data.npz").is_file(), volume
            )
            self.assertEqual(
                (group / "boundary_faces/z_mid/comparison_quality_data.npz").is_file(),
                not volume,
            )

    def test_configured_zooms_follow_switches_and_prune_disabled_quality(self):
        for dim in (2, 3):
            config = self.configuration(
                dim,
                ["tfi"],
                quality_metrics=["scaled_jacobian"],
                zoom_views=[
                    dict(metric="scaled_jacobian", xlim=[0.2, 0.8], ylim=[0.2, 0.8])
                ],
            )
            first = self.run_evaluation(dim, config)[0]
            group = Path(first["comparison_directory"])
            self.assertTrue(list((group / "zoom_views").rglob("*zoom_metadata.json")))
            with patch(
                "src.evaluation.zoom_3d.surface_quad_quality_fields",
                side_effect=AssertionError,
            ), patch(
                "src.evaluation.zoom_3d.evaluate_hexahedral_grid_quality",
                side_effect=AssertionError,
            ):
                self.run_evaluation(dim, config, "--no-compute-grid-quality")
            for marker in (group / "zoom_views").rglob("*zoom_metadata.json"):
                record = json.loads(marker.read_text(encoding="utf-8"))
                self.assertEqual(record.get("options", record)["plot"], "mesh")
            self.assertFalse(list((group / "zoom_views").rglob("local_quality*")))
            # A disabled worst search cannot silently run quality computations.
            self.run_evaluation(
                dim, config, "--no-compute-grid-quality", "--center-on", "worst"
            )

    def test_3d_zooms_only_evaluate_requested_metric_and_mesh_zoom_skips_quality(self):
        config = self.configuration(3, ["tfi"])
        result = self.run_evaluation(3, config, "--no-compute-grid-quality")[0]
        group = Path(result["comparison_directory"])
        for mode in ("slice", "volume"):
            with patch(
                "src.evaluation.zoom_3d.surface_quad_quality_fields",
                side_effect=AssertionError,
            ), patch(
                "src.evaluation.zoom_3d.evaluate_hexahedral_grid_quality",
                side_effect=AssertionError,
            ):
                generate_zoom(group, dict(mode=mode, plot="mesh", save_plots=False))
            metric = "cell_area" if mode == "slice" else "cell_volume"
            with patch(
                "src.evaluation.quality_3d._segments_intersect_2d",
                side_effect=AssertionError,
            ), patch("numpy.linalg.cond", side_effect=AssertionError):
                output = generate_zoom(
                    group,
                    dict(mode=mode, metric=metric, center_on="worst", save_plots=False),
                )
                summary = json.loads(
                    (output / "local_quality_summary.json").read_text()
                )
                self.assertEqual(set(summary["TFI"]), {metric})

    def test_network_raw_and_processed_mesh_only_and_recorded_trimmed_timing(self):
        for dim in (2, 3):
            checkpoint = self.checkpoint(dim, "hmonet")
            config = self.configuration(
                dim,
                ["hmonet"],
                hmonet_checkpoint=str(checkpoint),
                inference_warmup_runs=3,
                inference_timing_repeats=10,
                query_batch_size=50000,
                compute_grid_quality=False,
            )
            result = self.run_evaluation(dim, config)[0]
            timing = result["timings"]["HMONet"]
            self.assertEqual(timing["inference_warmup_runs"], 3)
            self.assertEqual(timing["inference_used_samples"], 8)
            expected = np.mean(np.sort(timing["inference_samples_seconds"])[1:-1])
            self.assertAlmostEqual(
                timing["network_inference_seconds"], expected, places=12
            )
            group = Path(result["comparison_directory"])
            root = group.parents[3]
            for stage in ("raw", "none"):
                folder = root / "HMONet/current" / stage
                self.assertTrue((folder / "mesh.npy").is_file())
                self.assertTrue((folder / "boundary_error_summary.json").is_file())
                self.assertFalse((folder / "grid_quality/quality_data.npz").exists())


if __name__ == "__main__":
    unittest.main()
