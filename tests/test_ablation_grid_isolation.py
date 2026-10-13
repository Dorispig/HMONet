"""Regression tests for grid-isolated ablation evaluation outputs."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from scripts.experiments.run_ablation_2d import (
    aggregate_experiment,
    find_reusable_evaluation,
    read_results,
)


class GridIsolationTests(unittest.TestCase):
    def test_per_grid_manifest_entries_are_reused_independently(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "jacobian_enabled/dataset/jac_on/seed_1/paper"
            result_51 = root / "raw/51x51/results.json"
            result_101 = root / "raw/101x101/results.json"
            result_51.parent.mkdir(parents=True)
            result_101.parent.mkdir(parents=True)
            result_51.write_text(json.dumps({"grid_shape": [51, 51]}), encoding="utf-8")
            result_101.write_text(json.dumps({"grid_shape": [101, 101]}), encoding="utf-8")
            run.mkdir(parents=True)
            manifest = {
                "status": "completed",
                "evaluations": {
                    "51x51": {
                        "status": "completed",
                        "fingerprint": "fingerprint-51",
                        "result_files": [str(result_51)],
                    },
                    "101x101": {
                        "status": "completed",
                        "fingerprint": "fingerprint-101",
                        "result_files": [str(result_101)],
                    },
                },
            }
            manifest_path = run / "run_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            reuse_51 = find_reusable_evaluation(root, "fingerprint-51")
            reuse_101 = find_reusable_evaluation(root, "fingerprint-101")

            self.assertEqual(reuse_51["grid_shape"], "51x51")
            self.assertEqual(reuse_51["result_files"], [result_51])
            self.assertEqual(reuse_101["grid_shape"], "101x101")
            self.assertEqual(reuse_101["result_files"], [result_101])
            self.assertEqual(read_results(run, "51x51"), [result_51])
            self.assertEqual(read_results(run, "101x101"), [result_101])

    def test_aggregate_outputs_are_separated_by_grid(self):
        with TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "baseline"
            run = experiment / "dataset/seed_1/paper"
            training = run / "training"
            training.mkdir(parents=True)
            summary_path = training / "training_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "completed_epochs": 10,
                        "training_time_seconds": 1.0,
                        "training_loss_history": [1.0, 0.5],
                        "dataset_mse_epochs": [1, 2],
                        "dataset_mse_history": [1.0, 0.4],
                    }
                ),
                encoding="utf-8",
            )
            evaluations = {}
            for size in (51, 101):
                grid = f"{size}x{size}"
                result = experiment / f"dataset/_evaluation/0.5/seed_1/baseline/paper/{grid}/results.json"
                result.parent.mkdir(parents=True)
                result.write_text(
                    json.dumps(
                        {
                            "parameter": "t",
                            "value": 0.5,
                            "grid_shape": [size, size],
                            "metrics": {
                                "HMONet": {
                                    "invalid_cells": {"count": 0, "fraction": 0.0},
                                    "jacobian": {"minimum": 0.1},
                                }
                            },
                            "timings": {"HMONet": {"network_inference_seconds": 0.01}},
                            "boundary_errors": [],
                        }
                    ),
                    encoding="utf-8",
                )
                evaluations[grid] = {
                    "status": "completed",
                    "fingerprint": f"fingerprint-{grid}",
                    "result_files": [str(result)],
                }
            manifest = {
                "ablation": "baseline",
                "dataset": "dataset",
                "variant": "baseline",
                "variant_value": "baseline",
                "seed": 1,
                "run_tag": "paper",
                "status": "completed",
                "training_summary": str(summary_path),
                "evaluations": evaluations,
            }
            (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            aggregate_experiment(experiment, "paper")

            root = experiment / "dataset/aggregate/paper"
            self.assertTrue((root / "51x51/per_case_metrics.csv").is_file())
            self.assertTrue((root / "101x101/per_case_metrics.csv").is_file())
            self.assertTrue((root / "evaluation_index.json").is_file())
            self.assertFalse((root / "per_case_metrics.csv").exists())
            figures = experiment / "dataset/figures/paper"
            self.assertTrue((figures / "training/training_loss_comparison.png").is_file())
            self.assertTrue((figures / "51x51").is_dir())
            self.assertTrue((figures / "101x101").is_dir())


if __name__ == "__main__":
    unittest.main()
