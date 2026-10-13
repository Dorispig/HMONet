"""Training output directories, snapshots, TensorBoard mirrors, checkpoints and summaries."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import yaml
from scripts.train.training_state import (
    create_run_directories,
    config_archive_directory,
    save_training_snapshot,
)
import sys
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scripts.train.training_state import (
    TrainingState,
    checkpoint_directory,
    create_run_directories,
    prepare_training_config,
    tensorboard_mirror_directory,
)
import contextlib
import csv
import io
import json
import time
from scripts.train.training_state import (
    TrainingState,
    prepare_training_config,
    print_training_summaries,
)


class ConfigArchiveTests(unittest.TestCase):
    def test_resolved_dataset_and_collision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                output_root=root / "training",
                checkpoint_root=root / "checkpoints",
                config_archive_root=root / "configs/2D",
                dataset_name="sample",
                mode="interpolation",
            )
            config = {
                "datasets": {"interpolation": "old", "extrapolation": "other"},
                "training": {"dataset": "both", "learning_rate": 0.0007},
                "model": {
                    "name": "hmonet",
                    "hmonet": {"hmonet_mapping_mode": "current"},
                },
            }
            with patch("scripts.train.training_state.datetime") as clock:
                clock.now.return_value.strftime.return_value = "20260920_1200"
                for expected in ("20260920_1200", "20260920_1200_02"):
                    run, output, checkpoint = create_run_directories(
                        args, config, "interpolation", "hmonet", "2D"
                    )
                    self.assertEqual(run, expected)
                    directory = config_archive_directory(args, config, "hmonet", run)
                    self.assertEqual(directory, root / "configs/2D/sample/HMONet/current" / run)
                    path = save_training_snapshot(
                        directory, config, root / "data/sample", "interpolation"
                    )
                    restored = yaml.safe_load(path.read_text(encoding="utf-8"))
                    self.assertEqual(
                        restored["datasets"],
                        {"interpolation": str((root / "data/sample").resolve())},
                    )
                    self.assertEqual(restored["hmonet_training"]["learning_rate"], 0.0007)
                    self.assertEqual(restored["training"]["dataset"], "interpolation")
                    with self.assertRaises(FileExistsError):
                        save_training_snapshot(
                            directory, config, root / "data/sample", "interpolation"
                        )
            self.assertEqual(config["training"]["dataset"], "both")
            self.assertEqual(len(config["datasets"]), 2)


ROOT = Path(__file__).resolve().parents[1]


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TrainingRunDirectoryTests(unittest.TestCase):
    def test_separate_tensorboard_runs_share_checkpoint_timestamps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("scripts.train.training_state.datetime") as clock:
                clock.now.return_value.strftime.return_value = "20260915_0807"
                for dimension in (2, 3):
                    for name, mapping in (
                        ("hmonet", "current"),
                        ("hmonet", "coons_residual"),
                        ("meshonet", "current"),
                    ):
                        args = SimpleNamespace(
                            output_root=root / "training" / f"{dimension}D" / "sample",
                            checkpoint_root=root / "checkpoints",
                            dataset_name="sample",
                            mode="interpolation",
                        )
                        locations = []
                        for repeat in range(2):
                            config = prepare_training_config(
                                {
                                    "model": {
                                        "hmonet": {"hmonet_mapping_mode": mapping}
                                    },
                                    "training": {"tensorboard_enabled": True},
                                },
                                name,
                                args,
                            )
                            model = torch.nn.Linear(dimension, dimension)
                            optimizer = torch.optim.Adam(model.parameters())
                            dataset = SimpleNamespace(grid_shape=(3,) * dimension)
                            state = TrainingState(
                                model,
                                optimizer,
                                [dataset],
                                config,
                                "interpolation",
                                args,
                                "cpu",
                            )
                            try:
                                state._measure = lambda: {"mse": 0.125}
                                state.end_epoch(1, {"loss": 0.25}, 0.001)
                                state.writer.flush()
                            finally:
                                state.writer.close()
                            expected = "20260915_0807" + ("_02" if repeat else "")
                            self.assertEqual(state.output.name, expected)
                            self.assertEqual(state.checkpoint.name, expected)
                            self.assertEqual(args.run_id, expected)
                            self.assertEqual(
                                Path(args.training_output_dir), state.output
                            )
                            if name == "hmonet":
                                self.assertEqual(state.output.parent.name, mapping)
                                self.assertEqual(state.checkpoint.parent.name, mapping)
                            event = next(
                                (state.output / "tensorboard").glob("events.out*")
                            )
                            events = EventAccumulator(str(event)).Reload()
                            mirror_dir = state.tensorboard_mirror
                            self.assertEqual(mirror_dir.name, expected)
                            self.assertEqual(
                                mirror_dir.relative_to(root / "tensorboard"),
                                state.output.relative_to(root / "training"),
                            )
                            mirror = EventAccumulator(str(mirror_dir)).Reload()
                            self.assertEqual(events.Tags(), mirror.Tags())
                            for tag in events.Tags()["scalars"]:
                                self.assertEqual(
                                    events.Scalars(tag), mirror.Scalars(tag)
                                )
                            for tag in events.Tags()["tensors"]:
                                self.assertEqual(
                                    events.Tensors(tag), mirror.Tensors(tag)
                                )
                            self.assertAlmostEqual(
                                events.Scalars("loss/training")[0].value, 0.25
                            )
                            self.assertAlmostEqual(
                                events.Scalars("metrics/mse")[0].value, 0.125
                            )
                            self.assertAlmostEqual(
                                events.Scalars("optimizer/learning_rate")[0].value,
                                0.001,
                            )
                            locations.append(event)
                        self.assertNotEqual(locations[0].parent, locations[1].parent)
                        self.assertTrue(all(path.is_file() for path in locations))

    def test_checkpoint_only_collision_does_not_reuse_or_overwrite_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                output_root=Path(temporary) / "training",
                checkpoint_root=Path(temporary) / "checkpoints",
                dataset_name="sample",
                mode="interpolation",
            )
            config = {"model": {"hmonet": {"hmonet_mapping_mode": "current"}}}
            with patch("scripts.train.training_state.datetime") as clock:
                clock.now.return_value.strftime.return_value = "20260915_0807"
                old = checkpoint_directory(args, config, "hmonet", "2D")
                old.mkdir(parents=True)
                marker = old / "keep.txt"
                marker.write_text("existing checkpoint", encoding="utf-8")
                run_id, output, checkpoint = create_run_directories(
                    args, config, "interpolation", "hmonet", "2D"
                )
                self.assertEqual(run_id, "20260915_0807_02")
                self.assertEqual(output.name, checkpoint.name)
                self.assertEqual(
                    marker.read_text(encoding="utf-8"), "existing checkpoint"
                )
                self.assertFalse((output.parent / "20260915_0807").exists())

    def test_mirror_only_collision_and_disabled_tensorboard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                output_root=root / "training" / "2D" / "sample",
                checkpoint_root=root / "checkpoints",
                dataset_name="sample",
                mode="interpolation",
            )
            config = {"training": {"tensorboard_enabled": True}}
            with patch("scripts.train.training_state.datetime") as clock:
                clock.now.return_value.strftime.return_value = "20260915_0807"
                mirror = tensorboard_mirror_directory(
                    args, config, "interpolation", "meshonet", "2D", "20260915_0807"
                )
                mirror.mkdir(parents=True)
                marker = mirror / "keep.txt"
                marker.write_text("old mirror", encoding="utf-8")
                run_id, output, checkpoint = create_run_directories(
                    args, config, "interpolation", "meshonet", "2D"
                )
                self.assertEqual(run_id, "20260915_0807_02")
                self.assertEqual(output.name, checkpoint.name)
                self.assertTrue((mirror.parent / run_id).is_dir())
                self.assertEqual(marker.read_text(encoding="utf-8"), "old mirror")
                self.assertFalse((output.parent / "20260915_0807").exists())
                config["training"]["tensorboard_enabled"] = False
                run_id, output, checkpoint = create_run_directories(
                    args, config, "extrapolation", "meshonet", "2D"
                )
                disabled = tensorboard_mirror_directory(
                    args, config, "extrapolation", "meshonet", "2D", run_id
                )
                self.assertEqual(disabled, mirror)
                self.assertEqual(marker.read_text(encoding="utf-8"), "old mirror")


class TrainingSummaryTests(unittest.TestCase):
    def test_best_checkpoint_loss_and_global_minimum_are_reported_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dimension in (2, 3):
                for name in ("hmonet", "meshonet"):
                    for selection in ("dataset_mse", "training_loss"):
                        with self.subTest(
                            dimension=dimension, model=name, metric=selection
                        ):
                            args = SimpleNamespace(
                                output_root=root / "training" / f"{dimension}D" / name,
                                checkpoint_root=root / "checkpoints",
                                dataset_name="sample",
                                mode="interpolation",
                            )
                            config = prepare_training_config(
                                {
                                    "training": {
                                        "num_epochs": 9,
                                        "best_model_metric": selection,
                                    }
                                },
                                name,
                                args,
                            )
                            model = torch.nn.Linear(dimension, dimension)
                            state = TrainingState(
                                model,
                                torch.optim.Adam(model.parameters()),
                                [
                                    SimpleNamespace(
                                        grid_shape=(3,) * dimension,
                                        root=root / "sample",
                                    )
                                ],
                                config,
                                "interpolation",
                                args,
                                "cpu",
                            )
                            mses = {1: 0.2, 2: 0.1, 3: 0.3}
                            state._measure = lambda: {
                                "mse": mses[int(model.bias[0].item())]
                            }
                            output = io.StringIO()
                            with contextlib.redirect_stdout(output):
                                for epoch, loss in enumerate((0.4, 0.5, 0.1), start=1):
                                    with torch.no_grad():
                                        model.bias.fill_(epoch)
                                    state.end_epoch(epoch, {"loss": loss}, 0.0004)
                                state.reason = "training_loss_convergence"
                                state.started = time.perf_counter() - 65.5
                                with (
                                    patch(
                                        "scripts.train.training_state.write_loss_outputs"
                                    ),
                                    patch(
                                        "scripts.train.training_state.write_mse_outputs"
                                    ),
                                ):
                                    history = state.finish()
                            summary = json.loads(
                                (state.output / "training_summary.json").read_text(
                                    encoding="utf-8"
                                )
                            )
                            expected_epoch = 2 if selection == "dataset_mse" else 3
                            self.assertEqual(summary["completed_epochs"], 3)
                            self.assertEqual(summary["requested_epochs"], 9)
                            self.assertEqual(summary["best_epoch"], expected_epoch)
                            self.assertEqual(
                                summary["best_training_loss"],
                                (0.5 if expected_epoch == 2 else 0.1),
                            )
                            self.assertEqual(summary["minimum_training_loss"], 0.1)
                            self.assertEqual(summary["minimum_training_loss_epoch"], 3)
                            self.assertEqual(
                                summary["best_model_dataset_mse"], mses[expected_epoch]
                            )
                            self.assertEqual(summary["final_dataset_mse"], 0.3)
                            self.assertGreaterEqual(
                                summary["training_time_seconds"], 65.5
                            )
                            self.assertEqual(summary["convergence_epoch"], len(history))
                            self.assertEqual(summary["dimension"], dimension)
                            saved = torch.load(
                                state.checkpoint / "best.pt", weights_only=True
                            )
                            self.assertEqual(saved["epoch"], expected_epoch)
                            self.assertEqual(
                                saved["model_state_dict"]["bias"][0].item(),
                                expected_epoch,
                            )
                            self.assertEqual(model.bias[0].item(), 3)
                            with (state.output / "training_summary.csv").open(
                                newline="", encoding="utf-8"
                            ) as stream:
                                row = next(csv.DictReader(stream))
                            self.assertEqual(int(row["best_epoch"]), expected_epoch)
                            self.assertEqual(
                                float(row["best_training_loss"]),
                                summary["best_training_loss"],
                            )
                            self.assertEqual(
                                json.loads(
                                    (
                                        state.checkpoint / "training_summary.json"
                                    ).read_text(encoding="utf-8")
                                ),
                                summary,
                            )
                            self.assertIn(
                                f"{dimension}-D TRAINING RUN COMPLETED",
                                output.getvalue(),
                            )
                            for label in (
                                "completed epochs",
                                "training time",
                                "best epoch",
                                "best metric",
                                "best model train loss",
                                "best model dataset MSE",
                                "minimum train loss",
                                "stop reason",
                            ):
                                self.assertIn(label, output.getvalue())
                            self.assertIn("(epoch=3)", output.getvalue())
                            batch_output = io.StringIO()
                            with contextlib.redirect_stdout(batch_output):
                                print_training_summaries(
                                    [summary], 70, dimension=dimension
                                )
                            self.assertIn(
                                f"{dimension}-D TRAINING SUMMARY",
                                batch_output.getvalue(),
                            )
                            self.assertIn(
                                "total wall time       : 70.000s",
                                batch_output.getvalue(),
                            )


if __name__ == "__main__":
    unittest.main()
