"""Ablation selection, parameter variants, output profiles and batch-runner tests."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch
from scripts.experiments import run_ablation_2d
from scripts.experiments.run_ablations_2d import (
    ALL_ABLATIONS,
    _forward_common_arguments,
    main,
    resolve_ablations,
)
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from scripts.experiments.run_ablation_2d import (
    plot_training_histories,
    run_initial_learning_rate_experiment,
    run_point_weighting_experiment,
    run_training_sample_count_experiment,
)
from scripts.experiments.run_ablation_2d import (
    evaluate_one,
    write_profiled_metric_outputs,
)


class BatchAblationRunnerTests(unittest.TestCase):
    def test_all_and_selected_experiments(self):
        self.assertEqual(resolve_ablations(["all"]), list(ALL_ABLATIONS))
        selected = ["jacobian_enabled", "training_sample_count"]
        self.assertEqual(resolve_ablations(selected), selected)

    def test_invalid_selections_are_rejected(self):
        for values in (["all", "normal_loss"], ["unknown"], ["normal_loss"] * 2):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    resolve_ablations(values)


    def test_baseline_error_is_deferred_until_all_selected_ablations_finish(self):
        error = run_ablation_2d.BaselineCompatibilityError("baseline mismatch")
        with patch(
            "scripts.experiments.run_ablations_2d.run_ablation_2d.main",
            side_effect=[error, []],
        ) as run:
            with self.assertRaises(run_ablation_2d.BaselineCompatibilityError):
                main(
                    [
                        "--ablations",
                        "jacobian_enabled",
                        "normal_loss",
                        "--dry-run",
                    ]
                )
        self.assertEqual(run.call_count, 2)

    def test_common_options_are_forwarded(self):
        args = SimpleNamespace(
            config="a.yaml",
            train_config="t.yaml",
            evaluate_config="e.yaml",
            experiment_root="experiments",
            run_tag="paper",
            dataset=["one", "two"],
            seeds=[1, 2, 3],
            evaluation_grid_shapes=["101x101"],
            device="cuda",
            dry_run=True,
            train_only=False,
            evaluate_only=False,
            force=False,
            continue_on_error=True,
        )
        command = _forward_common_arguments(args, "normal_loss")
        self.assertIn("normal_loss", command)
        self.assertEqual(command.count("--dataset"), 2)
        self.assertIn("--seeds", command)
        self.assertIn("--evaluation-grid-shapes", command)
        self.assertIn("101x101", command)
        self.assertIn("--dry-run", command)
        self.assertIn("--continue-on-error", command)


class InitialLearningRateAblationTests(unittest.TestCase):
    def test_variants_only_override_initial_learning_rate(self):
        variants = run_initial_learning_rate_experiment(
            {}, [0.0001, 0.0004, 0.001]
        )
        self.assertEqual([item.value for item in variants], [0.0001, 0.0004, 0.001])
        for variant in variants:
            self.assertEqual(variant.overrides, ("--lr", str(variant.value)))

    def test_invalid_or_empty_ranges_are_rejected(self):
        for values in ([], [0.0], [-0.001], [float("nan")]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    run_initial_learning_rate_experiment({}, values)

    def test_training_sample_count_variants(self):
        variants = run_training_sample_count_experiment({})
        self.assertEqual(
            [item.value for item in variants], [50, 100, 200, 400, 800, 1200]
        )
        for variant in variants:
            self.assertEqual(
                variant.overrides,
                ("--global-random-count", str(variant.value)),
            )

    def test_invalid_training_sample_counts_are_rejected(self):
        for values in ([], [0], [-1], [100.5], [100, 100]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    run_training_sample_count_experiment({}, values)

    def test_point_weighting_variants(self):
        variants = run_point_weighting_experiment({})
        self.assertEqual(
            [item.name for item in variants],
            ["position_weighted", "uniform_weight"],
        )
        self.assertEqual(variants[0].overrides, ())
        self.assertEqual(
            variants[1].overrides,
            ("--point-loss-weighting", "uniform"),
        )

    def test_all_variants_share_loss_and_mse_comparison_figures(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "initial_learning_rate"
            for variant, scale in (("lr_1e-4", 1.0), ("lr_0p0004", 0.5)):
                for seed in (1, 2):
                    run = root / "dataset" / variant / f"seed_{seed}" / "paper"
                    training = run / "training"
                    training.mkdir(parents=True)
                    (run / "run_manifest.json").write_text(
                        json.dumps(
                            {
                                "dataset": "dataset",
                                "variant": variant,
                                "seed": seed,
                                "run_tag": "paper",
                            }
                        ),
                        encoding="utf-8",
                    )
                    (training / "training_summary.json").write_text(
                        json.dumps(
                            {
                                "training_loss_history": [scale, scale * 0.8, scale * 0.6],
                                "dataset_mse_epochs": [1, 2, 3],
                                "dataset_mse_history": [scale * 2, scale, scale * 0.5],
                                "completed_epochs": 3,
                                "training_time_seconds": 6.0 * scale + seed,
                            }
                        ),
                        encoding="utf-8",
                    )
            plot_training_histories(root, "paper")
            figures = root / "dataset" / "figures" / "paper" / "training"
            for stem in ("training_loss_comparison", "dataset_mse_comparison"):
                self.assertTrue((figures / f"{stem}.png").is_file())
                csv_text = (figures / f"{stem}.csv").read_text(encoding="utf-8-sig")
                self.assertIn("lr_1e-4", csv_text)
                self.assertIn("lr_0p0004", csv_text)
            self.assertTrue((figures / "parameter_vs_iteration_time.png").is_file())
            self.assertTrue((figures / "parameter_vs_total_training_time.png").is_file())
            self.assertTrue((figures / "parameter_vs_iteration_count.png").is_file())
            statistics = (figures / "parameter_training_statistics.csv").read_text(
                encoding="utf-8-sig"
            )
            self.assertIn("average_epoch_seconds_mean", statistics)
            self.assertIn("completed_epochs_mean", statistics)
            self.assertIn("total_training_seconds_mean", statistics)


class ProfiledAblationOutputTests(unittest.TestCase):
    def test_evaluation_uses_only_default_51_grid(self):
        root = Path(__file__).resolve().parents[1]
        dataset = root / "data/raw/2D/air_t_int_0_1000_21_20"
        with TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                evaluate_config=root / "configs/2D/evaluate.yaml",
                device=None,
            )
            summary = {
                "best_checkpoint": str(Path(temporary) / "best.pt"),
                "hmonet_mapping_mode": "current",
            }
            with patch("scripts.evaluate.evaluate_2d.main", return_value=None) as main:
                command = evaluate_one(args, dataset, summary, Path(temporary) / "eval")
            self.assertEqual(main.call_count, 1)
            index = command.index("--grid-shapes")
            self.assertEqual(command[index + 1], "51x51")

    def test_metric_categories_are_split_inside_each_grid(self):
        rows = []
        for grid in ("51x51",):
            for variant, value in (("jac_off", 0.0), ("jac_on", 1.0)):
                rows.append(
                    {
                        "ablation": "jacobian_enabled",
                        "dataset": "air_t_int_0_1000_21_20",
                        "variant": variant,
                        "variant_value": value,
                        "seed": 1,
                        "run_tag": "paper",
                        "method": "HMONet",
                        "parameter": "t",
                        "value": 0.3,
                        "grid_shape": grid,
                        "result_file": "result.json",
                        "invalid_cell_fraction": value,
                        "self_intersection__affected_cell_fraction": value,
                        "flipping_cell__affected_cell_fraction": value,
                        "jacobian__minimum": 1.0 - value,
                        "scaled_jacobian__minimum": 1.0 - value,
                        "fixed_grid_best_mse": value + 0.1,
                        "orthogonality__mean": 0.9 - value * 0.1,
                    }
                )
        with TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "jacobian_enabled"
            write_profiled_metric_outputs(rows, experiment, "paper")
            for grid in ("51x51",):
                aggregate = (
                    experiment
                    / "air_t_int_0_1000_21_20/aggregate/paper"
                    / grid
                )
                for category in (
                    "important_metrics",
                    "secondary_metrics",
                    "other_metrics",
                ):
                    self.assertTrue((aggregate / f"{category}.csv").is_file())
                figures = (
                    experiment
                    / "air_t_int_0_1000_21_20/figures/paper"
                    / grid
                )
                self.assertTrue(
                    (
                        figures
                        / "important_metrics/invalid_cell_fraction/parameter_0.3.png"
                    ).is_file()
                )
                self.assertTrue(
                    (
                        figures
                        / "secondary_metrics/fixed_grid_best_mse/parameter_0.3.png"
                    ).is_file()
                )
                self.assertTrue(
                    (
                        figures
                        / "other_metrics/orthogonality__mean/parameter_0.3.png"
                    ).is_file()
                )


if __name__ == "__main__":
    unittest.main()
