"""Regression tests for grid-isolated ablation evaluation outputs."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from scripts.experiments.run_ablation_2d import (
    aggregate_experiment,
    compact_evaluation_result_directory,
    find_reusable_evaluation,
    migrate_evaluation_image_paths,
    read_results,
)
from scripts.experiments.evaluation_paths import (
    EvaluationStagingBusyError,
    exclusive_evaluation_staging_root,
)
import pytest


class TestGridIsolation:
    def test_identical_staging_evaluations_are_exclusive(self):
        with TemporaryDirectory() as temporary:
            staging = Path(temporary) / ".eval_work/dataset/e1_1/seed_1/v1/101x101"
            with exclusive_evaluation_staging_root(staging):
                with pytest.raises(EvaluationStagingBusyError, match="already owns"):
                    with exclusive_evaluation_staging_root(staging):
                        pass
            # The normal completion path releases the lock for a later retry.
            with exclusive_evaluation_staging_root(staging):
                pass

    def test_compaction_keeps_summary_and_images_only(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "results.json").write_text("{}", encoding="utf-8")
            (root / "per_cell.csv").write_text("cell,value\n0,1\n", encoding="utf-8")
            (root / "mesh_comparison.png").write_bytes(b"png")
            quality = root / "grid_quality"
            quality.mkdir()
            (quality / "quality_scaled_jacobian.pdf").write_bytes(b"pdf")
            (quality / "quality_data.npz").write_bytes(b"npz")

            result = compact_evaluation_result_directory(root)

            assert (result) == (root / "results.json")
            assert result.is_file()
            assert (root / "plots/mesh_comparison.png").is_file()
            assert (root / "plots/quality/quality_scaled_jacobian.pdf").is_file()
            assert not (root / "per_cell.csv").exists()
            assert not (quality / "quality_data.npz").exists()

    def test_image_only_migration_preserves_nonimage_artifacts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "plots/comparison/TFI_PDE_HMONet/quality"
            legacy.mkdir(parents=True)
            image = legacy / "quality_scaled_jacobian.png"
            image.write_bytes(b"png")
            detail = root / "details.csv"
            detail.write_text("metric,value\n", encoding="utf-8")

            assert migrate_evaluation_image_paths(root) == 1
            assert (
                root
                / "plots/comparison/TFI_PDE_Winslow_HMONet/quality/quality_scaled_jacobian.png"
            ).is_file()
            assert detail.is_file()

    def test_per_grid_manifest_entries_are_reused_independently(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "rational_affine/dataset/rap/seed_1/paper"
            result_51 = root / "raw/51x51/results.json"
            result_101 = root / "raw/101x101/results.json"
            result_51.parent.mkdir(parents=True)
            result_101.parent.mkdir(parents=True)
            result_51.write_text(json.dumps({"grid_shape": [51, 51]}), encoding="utf-8")
            result_101.write_text(
                json.dumps({"grid_shape": [101, 101]}), encoding="utf-8"
            )
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

            assert (reuse_51["grid_shape"]) == ("51x51")
            assert (reuse_51["result_files"]) == ([result_51])
            assert (reuse_101["grid_shape"]) == ("101x101")
            assert (reuse_101["result_files"]) == ([result_101])
            assert (read_results(run, "51x51")) == ([result_51])
            assert (read_results(run, "101x101")) == ([result_101])

    def test_reuse_rejects_stale_geometry_parameter_results(self):
        """A rewritten identity must not relabel old parameter directories."""
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "baseline/dataset/seed_1/v1"
            old_result = root / "eval/101x101/0.62/results.json"
            old_result.parent.mkdir(parents=True)
            old_result.write_text(
                json.dumps({"grid_shape": [101, 101], "value": 0.62}),
                encoding="utf-8",
            )
            identity = run / "evaluation_identities/101x101.json"
            identity.parent.mkdir(parents=True)
            # This models the former bug: the current identity was written
            # before checking a manifest whose result files are still old.
            identity.write_text(
                json.dumps({"values": [1.52, 1.55]}), encoding="utf-8"
            )
            run_manifest = {
                "status": "completed",
                "evaluations": {
                    "101x101": {
                        "status": "completed",
                        "fingerprint": "new-values-fingerprint",
                        "identity_file": str(identity),
                        "result_files": [str(old_result)],
                    }
                },
            }
            (run / "run_manifest.json").write_text(
                json.dumps(run_manifest), encoding="utf-8"
            )

            expected = {"values": [1.52, 1.55]}
            assert (
                find_reusable_evaluation(root, "new-values-fingerprint", expected)
                is None
            )

            new_results = []
            for value in (1.52, 1.55):
                result = root / f"eval/101x101/{value}/results.json"
                result.parent.mkdir(parents=True)
                result.write_text(
                    json.dumps({"grid_shape": [101, 101], "value": value}),
                    encoding="utf-8",
                )
                new_results.append(result)
            run_manifest["evaluations"]["101x101"]["result_files"] = [
                str(old_result),
                *map(str, new_results),
            ]
            (run / "run_manifest.json").write_text(
                json.dumps(run_manifest), encoding="utf-8"
            )

            reusable = find_reusable_evaluation(
                root, "new-values-fingerprint", expected
            )
            assert reusable is not None
            assert reusable["result_files"] == new_results

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
                result = (
                    experiment
                    / f"dataset/_evaluation/0.5/seed_1/baseline/paper/{grid}/results.json"
                )
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
            (run / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            aggregate_experiment(experiment, "paper")

            root = experiment / "dataset/aggregate/paper"
            assert (root / "51x51/per_case_metrics.csv").is_file()
            assert (root / "101x101/per_case_metrics.csv").is_file()
            assert (root / "evaluation_index.json").is_file()
            assert not (root / "per_case_metrics.csv").exists()
            figures = experiment / "dataset/figures/paper"
            assert (figures / "training/training_loss_comparison.png").is_file()
            assert (figures / "51x51").is_dir()
            assert (figures / "101x101").is_dir()
