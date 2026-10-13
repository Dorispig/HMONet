"""Regression tests for exact training reuse across evaluation tags."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.experiments.run_ablation_2d import (
    Variant,
    execute,
    find_reusable_evaluation,
    find_reusable_training,
    organize_evaluation_results,
)
from scripts.experiments.evaluation_paths import evaluation_result_root

TEST_TEMP_ROOT = Path.home() / ".hmonet_test_tmp"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


class AblationTrainingReuseTests(unittest.TestCase):
    def args(self, root, tag):
        project = Path(__file__).resolve().parents[1]
        return SimpleNamespace(
            experiment_root=Path(root),
            ablation="training_sample_count",
            dataset=["air_t_int_0_1000_21_20"],
            seeds=[7],
            run_tag=tag,
            dry_run=False,
            force=False,
            evaluate_only=False,
            train_only=True,
            continue_on_error=False,
            train_config=project / "configs/2D/train.yaml",
            evaluate_config=project / "configs/2D/evaluate.yaml",
            device=None,
        )

    @staticmethod
    def fake_train(args, dataset, variant, seed, destination):
        training = destination / "training"
        checkpoint = destination / "checkpoints/best.pt"
        final_checkpoint = destination / "checkpoints/final.pt"
        training.mkdir(parents=True)
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        final_checkpoint.write_bytes(b"checkpoint")
        summary = {
            "best_checkpoint": str(checkpoint.resolve()),
            "final_checkpoint": str(final_checkpoint.resolve()),
            "completed_epochs": 2,
            "stop_reason": "max_epochs",
            "training_time_seconds": 1.0,
            "training_loss_history": [1.0, 0.5],
            "dataset_mse_epochs": [1, 2],
            "dataset_mse_history": [2.0, 1.0],
            "hmonet_mapping_mode": "current",
        }
        (training / "training_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        return summary, ["fake-train"]

    def test_same_training_is_reused_across_tags_and_changes_retrain(self):
        config = {"datasets": [], "seeds": [7]}
        base = Variant("samples", 100, ("--global-random-count", "100"))
        changed = Variant("samples", 200, ("--global-random-count", "200"))
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary, patch(
            "scripts.experiments.run_ablation_2d.train_one",
            side_effect=self.fake_train,
        ) as train, patch(
            "scripts.experiments.run_ablation_2d.check_trained_baseline_identity",
            return_value=None,
        ):
            execute(self.args(temporary, "first"), config, [base])
            self.assertEqual(train.call_count, 1)
            execute(self.args(temporary, "second"), config, [base])
            self.assertEqual(train.call_count, 1)
            second = (
                Path(temporary)
                / "training_sample_count/air_t_int_0_1000_21_20/samples/seed_7/second/run_manifest.json"
            )
            manifest = json.loads(second.read_text(encoding="utf-8"))
            self.assertTrue(manifest["training_reused"])
            self.assertIn("first", manifest["training_source_directory"])
            execute(self.args(temporary, "second"), config, [changed])
            self.assertEqual(train.call_count, 2)
            replaced = json.loads(second.read_text(encoding="utf-8"))
            self.assertFalse(replaced["training_reused"])

    def test_alias_manifest_resolves_to_physical_training_directory(self):
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            physical = root / "baseline/dataset/seed_7/paper"
            training = physical / "training"
            checkpoint = physical / "checkpoints/best.pt"
            final_checkpoint = physical / "checkpoints/final.pt"
            training.mkdir(parents=True)
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            final_checkpoint.write_bytes(b"checkpoint")
            summary = {
                "best_checkpoint": str(checkpoint.resolve()),
                "final_checkpoint": str(final_checkpoint.resolve()),
                "completed_epochs": 2,
                "stop_reason": "max_epochs",
            }
            summary_path = training / "training_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            physical_manifest = physical / "run_manifest.json"
            physical_manifest.write_text(
                json.dumps(
                    {
                        "training_fingerprint": "training-sha256",
                        "training_summary": str(summary_path.resolve()),
                    }
                ),
                encoding="utf-8",
            )

            alias = root / "jacobian_enabled/dataset/jac_on/seed_7/paper"
            alias.mkdir(parents=True)
            alias_manifest = alias / "run_manifest.json"
            alias_manifest.write_text(
                json.dumps(
                    {
                        "training_fingerprint": "training-sha256",
                        "training_summary": str(summary_path.resolve()),
                    }
                ),
                encoding="utf-8",
            )

            reusable = find_reusable_training(root, "training-sha256")

            self.assertIsNotNone(reusable)
            self.assertEqual(reusable["directory"], physical.resolve())
            self.assertEqual(reusable["manifest_path"], physical_manifest)



    def test_baseline_mismatch_finishes_variant_training_and_evaluation_then_errors(self):
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            dataset = root / "air_t_int_0_1000_21_20"
            dataset.mkdir()
            baseline = root / "baseline/air_t_int_0_1000_21_20/seed_7/paper"
            baseline_training = baseline / "training"
            baseline_checkpoint = baseline / "checkpoints/best.pt"
            baseline_final_checkpoint = baseline / "checkpoints/final.pt"
            baseline_training.mkdir(parents=True)
            baseline_checkpoint.parent.mkdir(parents=True)
            baseline_checkpoint.write_bytes(b"baseline-must-not-change")
            baseline_final_checkpoint.write_bytes(b"baseline-must-not-change")
            baseline_summary = baseline_training / "training_summary.json"
            baseline_summary.write_text(
                json.dumps(
                    {
                        "best_checkpoint": str(baseline_checkpoint.resolve()),
                        "final_checkpoint": str(baseline_final_checkpoint.resolve()),
                        "completed_epochs": 2,
                        "stop_reason": "max_epochs",
                    }
                ),
                encoding="utf-8",
            )
            baseline_identity = baseline / "training_identity.json"
            baseline_identity.write_text(
                json.dumps({"effective_training_config": {"learning_rate": 1e-3}}),
                encoding="utf-8",
            )
            baseline_manifest = baseline / "run_manifest.json"
            baseline_manifest.write_text(
                json.dumps(
                    {
                        "training_fingerprint": "old-baseline",
                        "training_summary": str(baseline_summary.resolve()),
                        "training_identity_file": str(baseline_identity.resolve()),
                    }
                ),
                encoding="utf-8",
            )
            baseline_before = {
                item.relative_to(baseline): item.read_bytes()
                for item in baseline.rglob("*")
                if item.is_file()
            }
            args = self.args(root, "paper")
            args.ablation = "jacobian_enabled"
            args.dataset = [str(dataset)]
            args.train_only = False
            variants = [
                Variant("jac_off", 0.0, ("--no-jacobian-loss-enabled",)),
                Variant("jac_on", 0.01, ("--jacobian-loss-enabled",)),
            ]

            def fake_identity(args, dataset, variant, seed):
                if variant.name == "baseline":
                    return (
                        "new-baseline",
                        {"effective_training_config": {"learning_rate": 2e-3}},
                    )
                return f"training-{variant.name}", {"variant": variant.name}

            def fake_evaluation_identity(args, dataset, summary, fingerprint):
                return f"evaluation-{fingerprint}", {"training_fingerprint": fingerprint}

            def fake_evaluate(args, dataset, summary, evaluation_root):
                output = evaluation_root / "case" / "51x51"
                output.mkdir(parents=True)
                (output / "results.json").write_text(
                    json.dumps({"value": 0.5, "grid_shape": [51, 51]}),
                    encoding="utf-8",
                )
                return ["fake-evaluate"]

            with patch(
                "scripts.experiments.run_ablation_2d.effective_training_identity",
                side_effect=fake_identity,
            ), patch(
                "scripts.experiments.run_ablation_2d.effective_evaluation_identity",
                side_effect=fake_evaluation_identity,
            ), patch(
                "scripts.experiments.run_ablation_2d.train_one",
                side_effect=self.fake_train,
            ) as train, patch(
                "scripts.experiments.run_ablation_2d.evaluate_one",
                side_effect=fake_evaluate,
            ) as evaluate, patch(
                "scripts.experiments.run_ablation_2d.aggregate_experiment"
            ) as aggregate, patch(
                "scripts.experiments.run_ablation_2d.plot_training_histories"
            ) as histories:
                with self.assertRaisesRegex(RuntimeError, "Baseline compatibility"):
                    execute(args, {"datasets": [], "seeds": [7]}, variants)

            self.assertEqual(train.call_count, 2)
            self.assertEqual(evaluate.call_count, 2)
            aggregate.assert_not_called()
            histories.assert_called_once()
            baseline_after = {
                item.relative_to(baseline): item.read_bytes()
                for item in baseline.rglob("*")
                if item.is_file()
            }
            self.assertEqual(baseline_after, baseline_before)
            report = root / "jacobian_enabled/baseline_compatibility_errors.json"
            self.assertTrue(report.is_file())
            issue = json.loads(report.read_text(encoding="utf-8"))[0]
            self.assertEqual(issue["status"], "baseline_parameter_mismatch")
            self.assertTrue(issue["differences"])

    def test_existing_evaluation_is_protected_without_force(self):
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            staging = root / "staging"
            generated = staging / "case" / "51x51"
            generated.mkdir(parents=True)
            (generated / "results.json").write_text(
                json.dumps({"value": 0.5, "grid_shape": [51, 51]}),
                encoding="utf-8",
            )
            target = (
                root
                / "eval/dataset/base/7/paper/51x51/0.5"
            )
            target.mkdir(parents=True)
            sentinel = target / "keep.txt"
            sentinel.write_text("preserve", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                organize_evaluation_results(
                    staging,
                    root / "baseline/dataset",
                    "baseline",
                    7,
                    "paper",
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")


    def test_force_evaluate_only_preserves_baseline_training(self):
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            dataset = root / "air_t_int_0_1000_21_20"
            dataset.mkdir()
            destination = root / "baseline/air_t_int_0_1000_21_20/seed_7/paper"
            training = destination / "training"
            checkpoint = destination / "checkpoints/best.pt"
            final_checkpoint = destination / "checkpoints/final.pt"
            training.mkdir(parents=True)
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"preserve-checkpoint")
            final_checkpoint.write_bytes(b"preserve-checkpoint")
            summary = {
                "best_checkpoint": str(checkpoint.resolve()),
                "final_checkpoint": str(final_checkpoint.resolve()),
                "completed_epochs": 2,
                "stop_reason": "max_epochs",
                "hmonet_mapping_mode": "current",
            }
            summary_path = training / "training_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            (destination / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "training_fingerprint": "training-sha256",
                        "training_summary": str(summary_path.resolve()),
                    }
                ),
                encoding="utf-8",
            )
            args = self.args(root, "paper")
            args.ablation = "baseline"
            args.dataset = [str(dataset)]
            args.train_only = False
            args.evaluate_only = True
            args.force = True

            def fake_evaluate(args, dataset, summary, evaluation_root):
                output = evaluation_root / "case" / "51x51"
                output.mkdir(parents=True)
                (output / "results.json").write_text(
                    json.dumps({"value": 0.5, "grid_shape": [51, 51]}),
                    encoding="utf-8",
                )
                return ["fake-evaluate"]

            with patch(
                "scripts.experiments.run_ablation_2d.effective_training_identity",
                return_value=("training-sha256", {}),
            ), patch(
                "scripts.experiments.run_ablation_2d.effective_evaluation_identity",
                return_value=("evaluation-sha256", {}),
            ), patch(
                "scripts.experiments.run_ablation_2d.evaluate_one",
                side_effect=fake_evaluate,
            ), patch(
                "scripts.experiments.run_ablation_2d.train_one"
            ) as train, patch(
                "scripts.experiments.run_ablation_2d.aggregate_experiment"
            ):
                execute(
                    args,
                    {"datasets": [], "seeds": [7]},
                    [Variant("baseline", "baseline", ())],
                )

            train.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), b"preserve-checkpoint")
            result = (
                evaluation_result_root(
                    root / "baseline" / dataset.name,
                    "baseline",
                    7,
                    "paper",
                    "51x51",
                    0.5,
                )
                / "results.json"
            )
            self.assertTrue(result.is_file())

    def test_existing_baseline_training_is_not_deleted_without_force(self):
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            dataset = root / "air_t_int_0_1000_21_20"
            dataset.mkdir()
            destination = root / "baseline/air_t_int_0_1000_21_20/seed_7/paper"
            destination.mkdir(parents=True)
            sentinel = destination / "keep.pt"
            sentinel.write_bytes(b"preserve")
            args = self.args(root, "paper")
            args.ablation = "baseline"
            args.dataset = [str(dataset)]

            with patch(
                "scripts.experiments.run_ablation_2d.effective_training_identity",
                return_value=("requested-fingerprint", {}),
            ), patch(
                "scripts.experiments.run_ablation_2d.find_reusable_training",
                return_value=None,
            ), patch(
                "scripts.experiments.run_ablation_2d.train_one"
            ) as train:
                with self.assertRaises(FileExistsError):
                    execute(args, {"datasets": [], "seeds": [7]}, [Variant("baseline", "baseline", ())])

            train.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), b"preserve")

    def test_completed_evaluation_is_found_by_fingerprint(self):
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            result = root / "evaluation/case/51x51/results.json"
            result.parent.mkdir(parents=True)
            result.write_text("{}", encoding="utf-8")
            run = root / "jacobian_weight/dataset/jac_w_0p01/seed_7/paper"
            run.mkdir(parents=True)
            manifest = {
                "status": "completed",
                "evaluation_fingerprint": "evaluation-sha256",
                "evaluation_result_files": [str(result)],
            }
            manifest_path = run / "run_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            reusable = find_reusable_evaluation(root, "evaluation-sha256")

            self.assertIsNotNone(reusable)
            self.assertEqual(reusable["manifest_path"], manifest_path)
            self.assertEqual(reusable["result_files"], [result])

    def test_incomplete_or_missing_evaluation_is_not_reused(self):
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            run = root / "sampling_mode/dataset/uniform/seed_7/paper"
            run.mkdir(parents=True)
            (run / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "evaluation_fingerprint": "evaluation-sha256",
                        "evaluation_result_files": [str(root / "missing.json")],
                    }
                ),
                encoding="utf-8",
            )

            self.assertIsNone(
                find_reusable_evaluation(root, "evaluation-sha256")
            )


if __name__ == "__main__":
    unittest.main()
