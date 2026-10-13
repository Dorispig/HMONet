"""Ensure pipeline evaluation uses this run's models and dataset cases."""

import tempfile
import unittest
from pathlib import Path
from scripts.train.train_evaluate import evaluation_commands
from src.evaluation.configuration import load_configuration


class PipelineTests(unittest.TestCase):
    def test_pairing_and_explicit_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            summaries = []
            for name in ("esa_0_1000_20_20", "esa_1_1000_20_20"):
                for model in ("hmonet", "meshonet"):
                    path = Path(directory) / f"{name}_{model}.pt"
                    path.touch()
                    summaries.append(
                        dict(
                            dataset_name=name,
                            model=model,
                            best_checkpoint=str(path),
                            hmonet_mapping_mode="current",
                        )
                    )
            cases = {item["dataset_name"]: {} for item in summaries}
            commands = evaluation_commands(
                summaries, "configs/2D/evaluate.yaml", cases, "best"
            )
            self.assertEqual(len(commands), 2)
            for command in commands:
                args, config = load_configuration(2, command)
                self.assertEqual(
                    args.hmonet_checkpoint.name, f"{args.case[0]}_hmonet.pt"
                )
                self.assertEqual(
                    args.meshonet_checkpoint.name, f"{args.case[0]}_meshonet.pt"
                )
                self.assertEqual(args.methods, "tfi,pde,hmonet,meshonet")

    def test_missing_case_fails(self):
        with self.assertRaises(ValueError):
            evaluation_commands([dict(dataset_name="missing")], "unused", {}, "best")

    def test_missing_checkpoint_fails(self):
        with self.assertRaises(FileNotFoundError):
            evaluation_commands(
                [
                    dict(
                        dataset_name="case",
                        model="hmonet",
                        best_checkpoint="nonexistent.pt",
                    )
                ],
                "unused",
                {"case": {}},
                "best",
            )


if __name__ == "__main__":
    unittest.main()


class EvaluationSubsetTests(unittest.TestCase):
    def test_all_training_with_four_method_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            path.touch()
            summaries = [
                dict(
                    dataset_name="air",
                    model=model,
                    best_checkpoint=str(path),
                    hmonet_mapping_mode="current",
                )
                for model in ("hmonet", "meshonet", "pimeshonet", "pihmonet")
            ]
            command = evaluation_commands(
                summaries, "eval.yaml", {"air": {}}, "best", "tfi,pde,meshonet,hmonet"
            )[0]
            self.assertEqual(
                command[command.index("--methods") + 1], "tfi,pde,meshonet,hmonet"
            )
            self.assertIn("--hmonet-checkpoint", command)
            self.assertIn("--meshonet-checkpoint", command)
            self.assertNotIn("--pimeshonet-checkpoint", command)
            self.assertNotIn("--pihmonet-checkpoint", command)
            unresolved = evaluation_commands(
                summaries[:1], "eval.yaml", {"air": {}}, "best", "pihmonet"
            )[0]
            self.assertEqual(unresolved[unresolved.index("--methods") + 1], "pihmonet")
            self.assertNotIn("--pihmonet-checkpoint", unresolved)


class EvaluationModelSelectorTests(unittest.TestCase):
    def test_model_selector_and_remaining_overrides(self):
        from scripts.train.train_evaluate import extract_evaluation_model

        methods, remaining = extract_evaluation_model(
            ["--model", "both", "--device", "cpu"]
        )
        self.assertEqual(methods, ["tfi", "pde", "meshonet", "hmonet"])
        self.assertEqual(remaining, ["--device", "cpu"])
        self.assertEqual(len(extract_evaluation_model(["--model=all"])[0]), 7)
        with self.assertRaises(ValueError):
            extract_evaluation_model(["--model", "both"], "tfi,pde")
