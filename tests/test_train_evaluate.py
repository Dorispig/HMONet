"""Ensure pipeline evaluation uses this run's models and dataset cases."""

import tempfile
from pathlib import Path
from scripts.train.train_evaluate import (
    _formal_queue_fingerprint,
    _without_value_options,
    evaluation_commands,
    validate_formal_run_tag,
)
from src.evaluation.configuration import load_configuration
import pytest


class TestPipeline:
    def test_formal_run_tag_is_portable_and_bounded(self):
        assert validate_formal_run_tag("physics_v1") == "physics_v1"
        with pytest.raises(ValueError):
            validate_formal_run_tag("bad/tag")
        with pytest.raises(ValueError):
            validate_formal_run_tag("x" * 13)

    def test_formal_queue_fingerprint_is_stable_and_configuration_sensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "dataset.npy").write_bytes(b"dataset")
            train = root / "train.yaml"
            evaluate = root / "evaluate.yaml"
            train.write_text("training: {}\n", encoding="utf-8")
            evaluate.write_text("cases: {}\n", encoding="utf-8")
            arguments = dict(
                dimension=2,
                dataset=dataset,
                models=["hmonet"],
                train_config=train,
                evaluate_config=evaluate,
                train_arguments=["--model", "hmonet"],
                evaluate_arguments=["--device", "cuda:0"],
                evaluate_methods=["tfi", "hmonet"],
                checkpoint="best",
            )
            first = _formal_queue_fingerprint(**arguments)
            assert (first) == (_formal_queue_fingerprint(**arguments))
            changed = dict(arguments, checkpoint="final")
            assert (first) != (_formal_queue_fingerprint(**changed))

    def test_queue_worker_overrides_remove_parent_task_selectors(self):
        assert (
            _without_value_options(
                [
                    "--model=all",
                    "--dataset",
                    "old",
                    "--device",
                    "cuda:3",
                    "--epochs",
                    "10",
                ],
                {"--model", "--dataset", "--device"},
            )
        ) == (["--epochs", "10"])

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
                            hmonet_mapping_mode="rational_affine",
                        )
                    )
            cases = {item["dataset_name"]: {} for item in summaries}
            commands = evaluation_commands(
                summaries, "configs/2D/evaluate.yaml", cases, "best"
            )
            assert (len(commands)) == (2)
            for command in commands:
                args, config = load_configuration(2, command)
                assert (args.hmonet_checkpoint.name) == (f"{args.case[0]}_hmonet.pt")
                assert (args.meshonet_checkpoint.name) == (
                    f"{args.case[0]}_meshonet.pt"
                )
                assert (args.methods) == ("tfi,pde,hmonet,meshonet")

    def test_missing_case_fails(self):
        with pytest.raises(ValueError):
            evaluation_commands([dict(dataset_name="missing")], "unused", {}, "best")

    def test_missing_checkpoint_fails(self):
        with pytest.raises(FileNotFoundError):
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


class TestEvaluationSubset:
    def test_all_training_with_four_method_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            path.touch()
            summaries = [
                dict(
                    dataset_name="air",
                    model=model,
                    best_checkpoint=str(path),
                    hmonet_mapping_mode="rational_affine",
                )
                for model in ("hmonet", "meshonet", "pimeshonet", "qghmonet")
            ]
            command = evaluation_commands(
                summaries, "eval.yaml", {"air": {}}, "best", "tfi,pde,meshonet,hmonet"
            )[0]
            assert (command[command.index("--methods") + 1]) == (
                "tfi,pde,meshonet,hmonet"
            )
            assert ("--hmonet-checkpoint") in (command)
            assert ("--meshonet-checkpoint") in (command)
            assert ("--pimeshonet-checkpoint") not in (command)
            assert ("--qghmonet-checkpoint") not in (command)
            unresolved = evaluation_commands(
                summaries[:1], "eval.yaml", {"air": {}}, "best", "qghmonet"
            )[0]
            assert (unresolved[unresolved.index("--methods") + 1]) == ("qghmonet")
            assert ("--qghmonet-checkpoint") not in (unresolved)


class TestEvaluationModelSelector:
    def test_model_selector_and_remaining_overrides(self):
        from scripts.train.train_evaluate import extract_evaluation_model

        methods, remaining = extract_evaluation_model(
            ["--model", "both", "--device", "cpu"]
        )
        assert (methods) == (["tfi", "pde", "meshonet", "hmonet"])
        assert (remaining) == (["--device", "cpu"])
        assert (len(extract_evaluation_model(["--model=all"])[0])) == (6)
        with pytest.raises(ValueError):
            extract_evaluation_model(["--model", "both"], "tfi,pde")
