"""Configuration, convergence monitoring and legacy-format training artifacts.

The model-specific epoch and optimizer loops live in train_2d.py/train_3d.py.
This module records their results; it never performs a training step.
"""

from __future__ import annotations

import copy
import csv
import json
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.train.training_common import (
    FIXED_GRID_MSE_SCALE,
    calculate_fixed_grid_metrics,
    create_summary_writer,
    metric_window_convergence_status,
    simplified_plateau_status,
    write_history,
    write_loss_outputs,
    write_mse_outputs,
)
from src.utils.config_validation import remove_deprecated_projection_config
from src.utils.paths import portable_project_path


def _display_width(value):
    """Return terminal display width for mixed Chinese and ASCII text."""
    width = 0
    for character in str(value):
        if unicodedata.combining(character):
            continue
        width += 2 if unicodedata.east_asian_width(character) in {"W", "F", "A"} else 1
    return width


def _left_align(value, width):
    """Left-align text by terminal display width without leading padding."""
    text = str(value)
    return text + " " * max(0, width - _display_width(text))


def _format_terminal_table(headers, rows):
    """Build a compact left-aligned table; None rows add separators."""
    data_rows = [tuple(map(str, row)) for row in rows if row is not None]
    string_headers = tuple(map(str, headers))
    widths = [
        max(_display_width(string_headers[index]), *(
            _display_width(row[index]) for row in data_rows
        ))
        for index in range(len(string_headers))
    ]
    separator = "+" + "+".join("-" * width for width in widths) + "+"

    def format_row(row):
        return "|" + "|".join(
            _left_align(value, width) for value, width in zip(row, widths)
        ) + "|"

    output = [separator, format_row(string_headers), separator]
    for row in rows:
        output.append(separator if row is None else format_row(tuple(map(str, row))))
    output.append(separator)
    return "\n".join(output)


def prepare_training_config(config, name, args):
    """Resolve per-model flags and CLI overrides before saving effective settings."""
    config = copy.deepcopy(config)
    remove_deprecated_projection_config(config)
    training = config.setdefault("training", {})
    enabled = getattr(args, "plateau_enabled", None)
    if enabled is not None:
        training["plateau_enabled"] = enabled
    specific = config.setdefault(f"{name}_training", {})
    for key in (
        "check_plateau_stability",
        "check_plateau_absolute_stability",
        "check_plateau_near_best",
        "check_plateau_consecutive_stability",
        "check_plateau_best_improvement",
    ):
        value = getattr(args, key, None)
        if value is None:
            value = specific.get(key, training.get(key, False))
        training[key] = bool(value)
        specific[key] = bool(value)
    for argument, key in (("epochs", "num_epochs"), ("batch_size", "batch_size")):
        value = getattr(args, argument, None)
        if value is not None:
            training[key] = value
    if getattr(args, "check_convergence", None) is not None:
        training["check_training_loss_convergence"] = args.check_convergence
    learning_rate_defaults = {
        "learning_rate": 0.0004,
        "lr_warmup_epochs": 1000,
        "lr_update_interval": 1000,
        "lr_decay_gamma": 0.9,
        "lr_stop_below": 1e-5,
    }
    learning_rate_arguments = {
        "learning_rate": "lr",
        "lr_warmup_epochs": "lr_warmup_epochs",
        "lr_update_interval": "lr_update_interval",
        "lr_decay_gamma": "lr_decay_gamma",
        "lr_stop_below": "lr_stop_below",
    }
    for key, default in learning_rate_defaults.items():
        # Old archived configurations remain readable, but new configurations
        # store these values only in the selected model's training section.
        specific.setdefault(key, training.get(key, default))
        override = getattr(args, learning_rate_arguments[key], None)
        if override is not None:
            specific[key] = override

    defaults = {
        "num_epochs": 1,
        "batch_size": 64,
        "seed": 20260905,
        "best_model_metric": "dataset_mse",
        "calculate_dataset_mse": True,
        "dataset_mse_interval": 1,
        "check_mse_convergence": False,
        "mse_tolerance": 1e-6,
        "mse_consecutive_epochs": 100,
        "convergence_start_epoch": 3000,
        "success_plateau_patience": 1000,
        "failure_plateau_enabled": True,
        "failure_plateau_patience": 2000,
        "failure_plateau_metric_ratio": 1.5,
        "failure_plateau_relative_fluctuation": 0.15,
        "failure_plateau_min_lr_updates": 2,
        "save_best_immediately": True,
        "stop_file_enabled": True,
        "check_training_loss_convergence": False,
        "training_loss_tolerance": 1e-6,
        "training_loss_consecutive_epochs": 100,
        "plateau_metric": "mse",
        "plateau_window": 200,
        "plateau_log_interval": 100,
        "fluctuation_quantiles": [0.1, 0.9],
        "fluctuation_relative_tolerance": 0.15,
        "fluctuation_absolute_minimum": 1e-6,
        "fluctuation_absolute_maximum": 1e-5,
        "plateau_near_best_relative_tolerance": 0.2,
        "plateau_near_best_absolute_tolerance": 1e-7,
        "plateau_stable_consecutive_checks": 50,
        "plateau_fluctuation_spread_enabled": True,
        "plateau_fluctuation_spread_activation": 0.4,
        "plateau_fluctuation_spread_start_epoch": 1000,
        "plateau_fluctuation_spread_activation_epoch": 5000,
        "plateau_fluctuation_spread_window": 1000,
        "plateau_fluctuation_spread_quantiles": [0.1, 0.9],
        "plateau_fluctuation_spread_tolerance": 0.15,
        "best_improvement_relative_tolerance": 0.005,
        "gradient_clip_norm": 1.0,
        "skip_nonfinite_loss": True,
        "tensorboard_enabled": False,
    }
    legacy_absolute = training.pop("fluctuation_absolute_tolerance", None)
    if legacy_absolute is not None:
        training.setdefault("fluctuation_absolute_minimum", legacy_absolute)
        training.setdefault("fluctuation_absolute_maximum", max(float(legacy_absolute), 1e-5))
    # The old partial-window parameter is intentionally ignored. Long-term
    # fluctuation stopping now always requires a complete history window.
    training.pop("plateau_fluctuation_spread_min_samples", None)
    # Schema v4 has one success patience counter and one shared convergence window.
    for obsolete in (
        "best_metric_plateau_enabled",
        "best_metric_plateau_require_near_best",
        "best_metric_plateau_patience",
        "plateau_start_epoch",
        "plateau_patience",
        "failure_plateau_start_epoch",
        "failure_plateau_window",
    ):
        training.pop(obsolete, None)
    for key, value in defaults.items():
        training.setdefault(key, value)
    for key in (
        "num_epochs",
        "batch_size",
        "dataset_mse_interval",
        "mse_consecutive_epochs",
        "convergence_start_epoch",
        "success_plateau_patience",
        "failure_plateau_patience",
        "failure_plateau_min_lr_updates",
        "training_loss_consecutive_epochs",
        "plateau_log_interval",
        "plateau_stable_consecutive_checks",
    ):
        if int(training[key]) < 1:
            raise ValueError(f"training.{key} must be positive")
    if training["check_plateau_consecutive_stability"] and not (
        training["check_plateau_stability"]
        or training["check_plateau_absolute_stability"]
    ):
        raise ValueError(
            "Consecutive stability requires relative or absolute stability checking"
        )
    if float(training["failure_plateau_metric_ratio"]) <= 1:
        raise ValueError("failure_plateau_metric_ratio must be greater than one")
    if float(training["failure_plateau_relative_fluctuation"]) < 0:
        raise ValueError("failure_plateau_relative_fluctuation must be nonnegative")
    if training["best_model_metric"] not in {"dataset_mse", "training_loss"}:
        raise ValueError("best_model_metric must be dataset_mse or training_loss")
    if int(training["plateau_fluctuation_spread_window"]) < 2:
        raise ValueError("plateau_fluctuation_spread_window must be at least two")
    spread_start_epoch = int(training["plateau_fluctuation_spread_start_epoch"])
    spread_force_epoch = int(training["plateau_fluctuation_spread_activation_epoch"])
    if spread_start_epoch < 1:
        raise ValueError("plateau_fluctuation_spread_start_epoch must be positive")
    if spread_force_epoch < spread_start_epoch:
        raise ValueError(
            "plateau_fluctuation_spread_activation_epoch must be greater than or "
            "equal to plateau_fluctuation_spread_start_epoch"
        )
    if not 0 <= float(training["plateau_fluctuation_spread_activation"]):
        raise ValueError("plateau_fluctuation_spread_activation must be nonnegative")
    if training["plateau_metric"] not in {"mse", "train_loss"}:
        raise ValueError("plateau_metric must be mse or train_loss")
    if int(training["plateau_window"]) < 2:
        raise ValueError("plateau_window must be at least two")
    quantiles = training["fluctuation_quantiles"]
    if len(quantiles) != 2 or not 0 <= quantiles[0] < quantiles[1] <= 1:
        raise ValueError(
            "fluctuation_quantiles must contain two increasing values in [0, 1]"
        )
    if training["check_mse_convergence"] and not training["calculate_dataset_mse"]:
        raise ValueError("MSE convergence requires calculate_dataset_mse")
    if (
        training["plateau_metric"] == "mse"
        and (
            training.get("plateau_enabled", False)
            or training["check_plateau_stability"]
            or training["check_plateau_absolute_stability"]
            or training["check_plateau_near_best"]
            or training["check_plateau_consecutive_stability"]
            or training["check_plateau_best_improvement"]
        )
        and not training["calculate_dataset_mse"]
    ):
        raise ValueError("MSE plateau convergence requires calculate_dataset_mse")
    for key in (
        "mse_tolerance",
        "training_loss_tolerance",
        "fluctuation_relative_tolerance",
        "fluctuation_absolute_minimum",
        "fluctuation_absolute_maximum",
        "plateau_near_best_relative_tolerance",
        "plateau_near_best_absolute_tolerance",
        "best_improvement_relative_tolerance",
    ):
        if not np.isfinite(float(training[key])) or float(training[key]) < 0:
            raise ValueError(f"training.{key} must be nonnegative")
    if float(training["fluctuation_absolute_minimum"]) > float(
        training["fluctuation_absolute_maximum"]
    ):
        raise ValueError(
            "training.fluctuation_absolute_minimum must not exceed "
            "training.fluctuation_absolute_maximum"
        )
    if int(specific["lr_warmup_epochs"]) < 0:
        raise ValueError(f"{name}_training.lr_warmup_epochs must be nonnegative")
    if int(specific["lr_update_interval"]) < 1:
        raise ValueError(f"{name}_training.lr_update_interval must be positive")
    if float(specific["learning_rate"]) <= 0:
        raise ValueError(f"{name}_training.learning_rate must be positive")
    if not 0 < float(specific["lr_decay_gamma"]) <= 1:
        raise ValueError(f"{name}_training.lr_decay_gamma must be in (0, 1]")
    if float(specific["lr_stop_below"]) <= 0:
        raise ValueError(f"{name}_training.lr_stop_below must be positive")
    clip = training["gradient_clip_norm"]
    if clip is not None and float(clip) <= 0:
        raise ValueError("gradient_clip_norm must be positive or null")
    config.setdefault("model", {})["name"] = name
    return config


def model_specific_training_config(config, name, dataset_path=None, mode=None):
    """Return the reproducible settings used by one selected model only."""
    source = copy.deepcopy(config)
    selected = {
        "datasets": source.get("datasets", {}),
        "grid": source.get("grid", {}),
        "model": {
            "name": name,
            "common": source.get("model", {}).get("common", {}),
            name: source.get("model", {}).get(name, {}),
        },
        "training": source.get("training", {}),
        f"{name}_training": source.get(f"{name}_training", {}),
    }
    learning_rate_keys = (
        "learning_rate",
        "lr_warmup_epochs",
        "lr_update_interval",
        "lr_decay_gamma",
        "lr_stop_below",
    )
    for key in learning_rate_keys:
        if key in selected["training"]:
            selected[f"{name}_training"].setdefault(
                key, selected["training"].pop(key)
            )
    if name == "hmonet" and "hmonet_sampling" in source:
        selected["hmonet_sampling"] = source["hmonet_sampling"]
    if mode is not None:
        selected["training"]["dataset"] = mode
        if dataset_path is not None:
            selected["datasets"] = {mode: str(Path(dataset_path).resolve())}
        elif mode in selected["datasets"]:
            selected["datasets"] = {mode: selected["datasets"][mode]}
    return selected


def checkpoint_directory(args, config, name, dimension, run_id=None):
    """Follow the dataset/model/mapping/date-and-minute hierarchy."""
    root = Path(getattr(args, "checkpoint_root", "outputs/checkpoints"))
    dataset = getattr(args, "dataset_name", "dataset")
    folder = (
        root
        / dimension
        / dataset
        / {
            "hmonet": "HMONet",
            "meshonet": "MeshONet",
            "pimeshonet": "PI-MeshONet",
            "pihmonet": "PI-HMONet",
            "attentionpihmonet": "Attention-PI-HMONet",
        }[name]
    )
    if name in ("hmonet", "pihmonet", "attentionpihmonet"):
        mapping = (
            config.get("model", {}).get(name, {}).get("hmonet_mapping_mode", "current")
        )
        folder = folder / mapping
    return folder / (run_id or datetime.now().strftime("%Y%m%d_%H%M"))


def tensorboard_mirror_directory(args, config, mode, name, dimension, run_id):
    """Mirror the training hierarchy under the outputs/tensorboard root."""
    checkpoint_root = Path(getattr(args, "checkpoint_root", "outputs/checkpoints"))
    dataset = getattr(args, "dataset_name", "dataset")
    folder = (
        checkpoint_root.parent
        / "tensorboard"
        / dimension
        / dataset
        / ({"pimeshonet": "PI-MeshONet", "pihmonet": "PI-HMONet", "attentionpihmonet": "Attention-PI-HMONet"}.get(name, name))
    )
    if name in ("hmonet", "pihmonet", "attentionpihmonet"):
        folder = folder / config.get("model", {}).get(name, {}).get(
            "hmonet_mapping_mode", "current"
        )
    return folder / run_id


def config_archive_directory(args, config, name, run_id):
    """Separate per-dataset snapshots by model, HMONet mapping and run."""
    root = getattr(args, "config_archive_root", None)
    if root is None:
        return None
    directory = (
        Path(root)
        / args.dataset_name
        / {
            "hmonet": "HMONet",
            "meshonet": "MeshONet",
            "pimeshonet": "PI-MeshONet",
            "pihmonet": "PI-HMONet",
            "attentionpihmonet": "Attention-PI-HMONet",
        }[name]
    )
    if name in ("hmonet", "pihmonet", "attentionpihmonet"):
        directory /= (
            config.get("model", {}).get(name, {}).get("hmonet_mapping_mode", "current")
        )
    return directory / run_id


def save_training_snapshot(directory, config, dataset_path, mode):
    """Save resolved options and only the dataset used in this training run."""
    name = config["model"]["name"]
    snapshot = model_specific_training_config(config, name, dataset_path, mode)
    path = Path(directory) / "train.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write("# 首轮训练开始前保存的本次实际生效参数。\n")
        yaml.safe_dump(snapshot, stream, sort_keys=False, allow_unicode=True)
    return path


def create_run_directories(args, config, mode, name, dimension):
    """Reserve matching log/checkpoint directories, including minute collisions.

    A timestamp is captured once so logs and checkpoints cannot disagree when
    setup crosses a minute boundary. Same-minute runs use _02, _03, etc.; a
    fresh run never reuses an existing TensorBoard event directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    flat_root = getattr(args, "flat_run_root", None)
    if flat_root:
        root = Path(flat_root)
        output = root / "training"
        checkpoint = root / "checkpoints"
        archive = root / "effective_config"
        tensorboard = root / "tensorboard"
        for directory in (output, checkpoint, archive):
            directory.mkdir(parents=True, exist_ok=False)
        if config.get("training", {}).get("tensorboard_enabled", False):
            tensorboard.mkdir(parents=True, exist_ok=False)
        args.config_archive_directory = str(archive)
        args.tensorboard_directory = str(tensorboard)
        return root.name, output, checkpoint

    output_parent = (
        Path(args.output_root)
        / ({"pimeshonet": "PI-MeshONet", "pihmonet": "PI-HMONet", "attentionpihmonet": "Attention-PI-HMONet"}.get(name, name))
    )
    if name in ("hmonet", "pihmonet", "attentionpihmonet"):
        mapping = (
            config.get("model", {}).get(name, {}).get("hmonet_mapping_mode", "current")
        )
        output_parent = output_parent / mapping
    checkpoint_parent = checkpoint_directory(
        args, config, name, dimension, run_id=timestamp
    ).parent
    output_parent.mkdir(parents=True, exist_ok=True)
    checkpoint_parent.mkdir(parents=True, exist_ok=True)
    mirror_parent = None
    if config.get("training", {}).get("tensorboard_enabled", False):
        mirror_parent = tensorboard_mirror_directory(
            args, config, mode, name, dimension, timestamp
        ).parent
        mirror_parent.mkdir(parents=True, exist_ok=True)
    sequence = 1
    while True:
        run_id = timestamp if sequence == 1 else f"{timestamp}_{sequence:02d}"
        sequence += 1
        output = output_parent / run_id
        checkpoint = checkpoint_parent / run_id
        directories = [output, checkpoint]
        archive = config_archive_directory(args, config, name, run_id)
        if archive is not None:
            archive.parent.mkdir(parents=True, exist_ok=True)
            directories.append(archive)
        if mirror_parent is not None:
            directories.append(mirror_parent / run_id)
        created = []
        try:
            for directory in directories:
                directory.mkdir()
                created.append(directory)
        except FileExistsError:
            # Only remove empty directories reserved by this call, never an
            # existing run in training, checkpoints or the TensorBoard mirror.
            for directory in reversed(created):
                directory.rmdir()
            continue
        except Exception:
            for directory in reversed(created):
                directory.rmdir()
            raise
        return run_id, output, checkpoint


def _geometry_for(dataset, index):
    return dataset[index].get("boundary")


def _clone_state(model):
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


class TrainingState:
    """Track epoch metrics and early stopping without owning either training loop."""

    def __init__(self, model, optimizer, datasets, config, mode, args, device):
        self.model = model
        self.optimizer = optimizer
        self.datasets = datasets
        self.config = config
        self.training = config["training"]
        self.name = config["model"]["name"]
        self.learning_rate = config[f"{self.name}_training"]
        self.shape = tuple(datasets[0].grid_shape)
        self.dimension = len(self.shape)
        self.mode = mode
        self.device = device
        self.run_id, self.output, self.checkpoint = create_run_directories(
            args, config, mode, self.name, f"{self.dimension}D"
        )
        # Expose the allocated location to the dispatcher without changing the
        # public train_hmonet/train_meshonet history return values.
        args.training_output_dir = str(self.output)
        args.checkpoint_output_dir = str(self.checkpoint)
        args.run_id = self.run_id
        self.history = []
        self.mse_epochs = []
        self.mses = []
        self.best = float("inf")
        self.best_epoch = 0
        self.best_state = None
        self.failure_plateau_best_window = float("inf")
        self.failure_plateau_bad_checks = 0
        self.failure_plateau = None
        self.loss_hits = 0
        self.mse_hits = 0
        self.plateau_history = []
        self.long_fluctuation_history = []
        self.long_fluctuation_active = False
        self.plateau = None
        self.reference_best = None
        self.stale = 0
        self.stable_count = 0
        self.reason = "maximum_epochs"
        self.config["training"]["dataset"] = mode
        self.config["grid"] = dict(zip(("nx", "ny", "nz"), self.shape))
        self.saved_config = model_specific_training_config(
            self.config, self.name, getattr(datasets[0], "root", None), mode
        )
        explicit_tensorboard = getattr(args, "tensorboard_directory", None)
        self.tensorboard_mirror = (
            Path(explicit_tensorboard)
            if explicit_tensorboard and self.training["tensorboard_enabled"]
            else (
                tensorboard_mirror_directory(
                    args, config, mode, self.name, f"{self.dimension}D", self.run_id
                )
                if self.training["tensorboard_enabled"]
                else None
            )
        )
        self.primary_tensorboard_enabled = explicit_tensorboard is None
        self.writer = create_summary_writer(
            self.output,
            self.training["tensorboard_enabled"],
            self.tensorboard_mirror,
            primary_enabled=self.primary_tensorboard_enabled,
        )
        self._save_config()
        explicit_archive = getattr(args, "config_archive_directory", None)
        archive = (
            Path(explicit_archive)
            if explicit_archive
            else config_archive_directory(args, config, self.name, self.run_id)
        )
        if archive is not None:
            self.config_snapshot_path = save_training_snapshot(
                archive, self.config, datasets[0].root, mode
            )
            print(
                f"Training configuration saved to: {self.config_snapshot_path}",
                flush=True,
            )
        if self.writer is not None:
            self.writer.add_text(
                "run/configuration",
                "```json\n" + json.dumps(config, indent=2) + "\n```",
            )
        self.started = time.perf_counter()

    def _save_config(self):
        for directory in (self.output, self.checkpoint):
            (directory / "config.json").write_text(
                json.dumps(self.saved_config, indent=2), encoding="utf-8"
            )
            # Keep training folders identical across dimensions. Checkpoint
            # folders retain YAML for human-readable model reconstruction.
            if directory == self.checkpoint:
                (directory / "config.yaml").write_text(
                    yaml.safe_dump(self.saved_config, sort_keys=False), encoding="utf-8"
                )

    def _measure(self):
        return calculate_fixed_grid_metrics(
            self.model,
            self.datasets,
            self.shape,
            self.device,
            _geometry_for
            if self.name in ("hmonet", "pihmonet", "attentionpihmonet")
            else None,
            int(self.training["batch_size"]),
            bool(
                self.training.get("check_max_point_error", False)
                or self.training.get("check_p99_point_error", False)
            ),
        )

    def _select_best(self, epoch, loss, mse):
        value = mse if self.training["best_model_metric"] == "dataset_mse" else loss
        if value is not None:
            value = float(value)
            if value < self.best:
                self.best = value
                self.best_epoch = epoch
                self.best_state = _clone_state(self.model)
                if self.training.get("save_best_immediately", False):
                    self._save_best_checkpoint()


    @staticmethod
    def _atomic_torch_save(value, path):
        """Write a checkpoint atomically and tolerate short Windows file locks."""
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(value, temporary)
        # Antivirus scanners, indexers, TensorBoard-adjacent tooling, or a
        # checkpoint reader can hold the destination briefly on Windows.
        # The temporary checkpoint is already complete, so retry only the
        # atomic replacement instead of repeating the expensive serialization.
        delays = (0.05, 0.10, 0.20, 0.40, 0.80, 1.00)
        for attempt in range(len(delays) + 1):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == len(delays):
                    raise
                time.sleep(delays[attempt])

    def _save_best_checkpoint(self):
        """Persist each new best state immediately so interruptions cannot lose it."""
        if self.best_state is None:
            return
        payload = {
            "model_state_dict": self.best_state,
            "config": self.saved_config,
            "grid": self.config["grid"],
            "model_name": self.name,
            "epoch": self.best_epoch,
        }
        self._atomic_torch_save(payload, self.checkpoint / "best.pt")
        self._atomic_torch_save(self.best_state, self.checkpoint / "best_model.pth")

    def _failure_plateau_ready(self, epoch):
        """Detect a stable plateau that remains substantially worse than prior progress."""
        t = self.training
        if not t.get("failure_plateau_enabled", False):
            return False
        window = int(t["plateau_window"])
        if len(self.plateau_history) < window:
            return False
        values = np.asarray(self.plateau_history[-window:], dtype=float)
        values = values[np.isfinite(values)]
        if values.size < window:
            return False
        low_q, high_q = map(float, t["fluctuation_quantiles"])
        median = float(np.median(values))
        low = float(np.quantile(values, low_q))
        high = float(np.quantile(values, high_q))
        relative_fluctuation = (high - low) / max(abs(median), 1e-12)
        self.failure_plateau_best_window = min(
            self.failure_plateau_best_window, median
        )
        if epoch < int(t["convergence_start_epoch"]):
            return False
        ratio = median / max(abs(self.failure_plateau_best_window), 1e-12)
        bad = ratio >= float(t["failure_plateau_metric_ratio"])
        stable = relative_fluctuation <= float(
            t["failure_plateau_relative_fluctuation"]
        )
        warmup = int(self.learning_rate["lr_warmup_epochs"])
        interval = int(self.learning_rate["lr_update_interval"])
        lr_updates = max(0, (epoch - warmup) // interval)
        enough_updates = lr_updates >= int(t["failure_plateau_min_lr_updates"])
        qualified = bad and stable and enough_updates
        self.failure_plateau_bad_checks = (
            self.failure_plateau_bad_checks + 1 if qualified else 0
        )
        self.failure_plateau = {
            "median": median,
            "historical_best_window_median": self.failure_plateau_best_window,
            "metric_ratio": ratio,
            "relative_fluctuation": relative_fluctuation,
            "bad_checks": self.failure_plateau_bad_checks,
            "lr_updates": lr_updates,
            "bad": bad,
            "stable": stable,
            "enough_lr_updates": enough_updates,
        }
        return bool(
            bad
            and stable
            and enough_updates
            and self.failure_plateau_bad_checks
            >= int(t["failure_plateau_patience"])
        )

    def _below_tolerance(self, value, tolerance):
        """Retain the legacy 2-D strict and 3-D inclusive threshold conventions."""
        return value < tolerance if self.dimension == 2 else value <= tolerance

    def _print_convergence_table(self, epoch, plateau, long_spread_passes):
        """Print the enabled convergence checks in a compact aligned table."""
        t = self.training
        passed = lambda condition: "通过" if condition else "未通过"
        satisfied = lambda condition: "满足" if condition else "未满足"

        best_metric_name = (
            "固定网格MSE"
            if t["best_model_metric"] == "dataset_mse"
            else "训练损失"
        )
        best_metric_value = (
            f"{self.best:.4e}" if np.isfinite(self.best) else "尚无有效值"
        )
        best_metric_epoch = str(self.best_epoch) if np.isfinite(self.best) else "-"

        rows = [
            (
                "训练信息",
                "当前最佳指标",
                f"{best_metric_name}={best_metric_value}；epoch={best_metric_epoch}",
                "-",
                "-",
            ),
            None,
            (
                "基础检查",
                "窗口波动",
                f"width={plateau['absolute_fluctuation']:.4e}；relative={plateau['relative_fluctuation']:.4f}",
                f"width≤{plateau['fluctuation_limit']:.4e}；relative≤{float(t['fluctuation_relative_tolerance']):.4f}",
                passed(plateau["stability_passes"]),
            ),
            (
                "基础检查",
                "接近历史最佳",
                f"median={plateau['median']:.4e}",
                f"≤{plateau['near_best_limit']:.4e}",
                passed(plateau["near_best_passes"]),
            ),
            (
                "基础检查",
                "是否有显著改善",
                f"累计改善={plateau['relative_best_improvement']:.4f}",
                f">{float(t['best_improvement_relative_tolerance']):.4f}",
                "是" if plateau["significant_improvement"] else "否",
            ),
            (
                "基础检查",
                "无显著改善",
                str(int(plateau["epochs_without_significant_improvement"])),
                f"≥{int(t['success_plateau_patience'])}",
                passed(plateau["no_improvement_passes"]),
            ),
            (
                "短期检查",
                "连续窗口稳定",
                str(int(plateau["consecutive_stable_count"])),
                f"≥{int(t['plateau_stable_consecutive_checks'])}",
                passed(plateau["short_stability_passes"]),
            ),
        ]

        if plateau.get("long_fluctuation_active", False):
            low_q, high_q = map(float, t["plateau_fluctuation_spread_quantiles"])
            low_value = plateau.get("long_fluctuation_low_quantile", float("nan"))
            high_value = plateau.get("long_fluctuation_high_quantile", float("nan"))
            spread = plateau.get("long_fluctuation_spread", float("nan"))
            count = int(plateau.get("long_fluctuation_sample_count", 0))
            ready = count >= int(t["plateau_fluctuation_spread_window"])
            rows.append(
                (
                    "长期检查",
                    "长期波动",
                    f"spread={high_value:.4f}(p{high_q * 100:g})-{low_value:.4f}(p{low_q * 100:g})={spread:.4f}；n={count}",
                    f"≤{float(t['plateau_fluctuation_spread_tolerance']):.4f}；n={int(t['plateau_fluctuation_spread_window'])}",
                    passed(long_spread_passes) if ready else "样本不足",
                )
            )

        rows.append(None)
        stop_conditions = []
        if t.get("check_mse_convergence", False):
            current = int(self.mse_hits)
            required = int(t["mse_consecutive_epochs"])
            stop_conditions.append(("固定网格 MSE", current, required, current >= required))
        if t.get("check_training_loss_convergence", False):
            current = int(self.loss_hits)
            required = int(t["training_loss_consecutive_epochs"])
            stop_conditions.append(("训练损失", current, required, current >= required))
        if t.get("plateau_enabled", False):
            stop_conditions.append(
                (
                    "短窗口平台",
                    int(self.stable_count),
                    int(t["plateau_stable_consecutive_checks"]),
                    bool(plateau.get("short_branch_passes", False)),
                )
            )
            if t.get("plateau_fluctuation_spread_enabled", False):
                stop_conditions.append(
                    (
                        "长期波动平台",
                        int(plateau.get("long_fluctuation_sample_count", 0)),
                        int(t["plateau_fluctuation_spread_window"]),
                        bool(plateau.get("long_branch_passes", False)),
                    )
                )

        for name, current, required, condition in stop_conditions:
            rows.append(("停止检查", name, str(current), str(required), satisfied(condition)))

        triggered = next(
            (name for name, _, _, condition in stop_conditions if condition),
            None,
        )
        rows.extend(
            [
                None,
                (
                    "最终结果",
                    "训练状态",
                    f"{triggered}通过" if triggered else "未触发",
                    "任一停止条件通过",
                    "停止训练" if triggered else "继续训练",
                ),
            ]
        )
        table = _format_terminal_table(
            ("类型", "检查项目", "当前值", "通过要求", "状态"),
            rows,
        )
        print(
            f"[{t['plateau_metric']} convergence | epoch {epoch} | window={t['plateau_window']}]\n{table}",
            flush=True,
        )

    def end_epoch(self, epoch, losses, learning_rate):
        """Measure this epoch, report legacy diagnostics and return whether to stop."""
        t = self.training
        loss = float(losses["loss"])
        if not np.isfinite(loss):
            raise FloatingPointError("Epoch training loss is not finite")
        metrics = {}
        needs_mse = (
            t["calculate_dataset_mse"] or t["best_model_metric"] == "dataset_mse"
        )
        measured_epoch = (
            epoch == 1
            or epoch % t["dataset_mse_interval"] == 0
            or epoch == t["num_epochs"]
        )
        point_checks = t.get("check_max_point_error", False) or t.get(
            "check_p99_point_error", False
        )
        if (needs_mse and measured_epoch) or point_checks:
            metrics = self._measure()
            self.mse_epochs.append(epoch)
            self.mses.append(metrics["mse"])
        mse = metrics.get("mse")
        self._select_best(epoch, loss, mse)
        automatic_stop_allowed = epoch >= int(t["convergence_start_epoch"])
        self.loss_hits = (
            self.loss_hits + 1
            if automatic_stop_allowed
            and t["check_training_loss_convergence"]
            and self._below_tolerance(loss, t["training_loss_tolerance"])
            else 0
        )
        if mse is not None and t["check_mse_convergence"]:
            self.mse_hits = (
                self.mse_hits + 1
                if automatic_stop_allowed
                and self._below_tolerance(mse, t["mse_tolerance"])
                else 0
            )
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "dataset_mse": mse,
            "learning_rate": learning_rate,
            **{key: float(value) for key, value in losses.items() if key != "loss"},
        }
        self.history.append(row)

        metric_text = ", ".join(f"{key}={value:.6e}" for key, value in metrics.items())
        smoothness_text = (
            f", smoothness_loss={losses['grid_smoothness']:.6e}"
            if self.name in ("pihmonet", "attentionpihmonet")
            and "grid_smoothness" in losses
            and self.config.get(f"{self.name}_training", {}).get(
                "smoothness_loss_weight", 0.0
            ) > 0
            else ""
        )
        alpha_text = (
            f", boundary_loss={losses['loss_boundary']:.6e}"
            f", alpha={losses['boundary_alpha']:.8f}"
            f", alpha_penalty={losses['alpha_penalty']:.6e}"
            if self.name == "attentionpihmonet"
            and self.dimension == 2
            and "loss_boundary" in losses
            else ""
        )
        print(
            f"Epoch {epoch}/{t['num_epochs']}: training_loss={loss:.6e}"
            + smoothness_text
            + alpha_text
            + (f", {metric_text}" if metric_text else ""),
            flush=True,
        )
        if self.writer is not None:
            self.writer.add_scalar("loss/training", loss, epoch)
            self.writer.add_scalar("optimizer/learning_rate", learning_rate, epoch)
            for key, value in metrics.items():
                self.writer.add_scalar(f"metrics/{key}", value, epoch)
            if t["check_mse_convergence"]:
                self.writer.add_scalar(
                    "metrics/mse_consecutive_epochs", self.mse_hits, epoch
                )
            if self.name in ("pimeshonet", "pihmonet", "attentionpihmonet"):
                self.writer.add_scalar("loss_components/loss_pde", losses["loss_pde"], epoch)
            if self.name == "pihmonet":
                for key in (
                    "loss_pde_normalized",
                    "jacobian",
                    "anchor",
                    "anchor_weight",
                    "valid_anchor_ratio",
                    "grid_smoothness",
                    "orthogonality",
                ):
                    self.writer.add_scalar(
                        f"loss_components/{key}", losses.get(key, 0.0), epoch
                    )
            if self.name == "attentionpihmonet":
                attention_keys = [
                    "loss_pde_normalized",
                    "jacobian",
                    "anchor",
                    "anchor_weight",
                    "valid_anchor_ratio",
                    "grid_smoothness",
                    "boundary_pde",
                    "boundary_jacobian",
                    "minimum_sampled_jacobian",
                    "minimum_boundary_jacobian",
                ]
                if self.dimension == 2:
                    attention_keys.extend(
                        ("loss_boundary", "alpha_penalty", "boundary_alpha")
                    )
                for key in attention_keys:
                    self.writer.add_scalar(
                        f"loss_components/{key}", losses.get(key, 0.0), epoch
                    )
            if self.name == "pimeshonet":
                self.writer.add_scalar(
                    "loss_components/loss_boundary", losses["loss_boundary"], epoch
                )
            for key in ("jacobian", "boundary_normal"):
                if (
                    self.config.get(f"{self.name}_training", {}).get(
                        f"{key}_loss_enabled", False
                    )
                    and self.name in ("hmonet", "pihmonet")
                ):
                    self.writer.add_scalar(
                        f"loss_components/{key}", losses.get(key, 0.0), epoch
                    )

        checks = []
        if t["check_mse_convergence"]:
            checks.append(
                mse is not None and self.mse_hits >= t["mse_consecutive_epochs"]
            )
        for key in ("max_point_error", "p99_point_error"):
            if t.get(f"check_{key}", False):
                checks.append(
                    self._below_tolerance(
                        metrics.get(key, float("inf")), t.get(f"{key}_tolerance", 1e-3)
                    )
                )
        if automatic_stop_allowed and checks and all(checks):
            self.reason = "metric_convergence"
        elif (
            automatic_stop_allowed
            and t["check_training_loss_convergence"]
            and self.loss_hits >= t["training_loss_consecutive_epochs"]
        ):
            self.reason = "training_loss_convergence"
        else:
            value = mse if t["plateau_metric"] == "mse" else loss
            self._check_plateau(epoch, value)
            if (
                not t.get("plateau_enabled", False)
                and t.get("failure_plateau_enabled", False)
                and value is not None
            ):
                self.plateau_history.append(value)
            if (
                self.reason == "maximum_epochs"
                and value is not None
                and self._failure_plateau_ready(epoch)
            ):
                self.reason = "failed_stable_plateau"
        if (
            self.reason == "maximum_epochs"
            and t.get("stop_file_enabled", True)
            and any(
                (directory / "STOP").is_file()
                for directory in (self.output, self.checkpoint)
            )
        ):
            self.reason = "stop_file_requested"
        if self.reason != "maximum_epochs":
            print(f"Converged at epoch {epoch}: {self.reason}", flush=True)
            return True
        return False

    def _check_plateau(self, epoch, value):
        t = self.training
        if "plateau_enabled" in t:
            if not t["plateau_enabled"] or value is None:
                return
            self.plateau_history.append(value)
            self.plateau = simplified_plateau_status(
                self.plateau_history,
                t["plateau_window"],
                t["success_plateau_patience"],
                t["fluctuation_relative_tolerance"],
                t["fluctuation_absolute_minimum"],
                t["fluctuation_absolute_maximum"],
                t["best_improvement_relative_tolerance"],
                t["plateau_near_best_relative_tolerance"],
                t["plateau_near_best_absolute_tolerance"],
                self.plateau,
                tuple(t["fluctuation_quantiles"]),
                stable_required=t["plateau_stable_consecutive_checks"],
            )
            if self.plateau is None:
                return
            p = self.plateau
            self.reference_best = p["reference_best"]
            if epoch < int(t["convergence_start_epoch"]):
                p["stale"] = 0
                p["epochs_without_significant_improvement"] = 0
                p["no_improvement_passes"] = False
                p["consecutive_stable_count"] = 0
                p["short_stability_passes"] = False
                p["short_branch_passes"] = False
                p["passes"] = False
            self.stale = p["epochs_without_significant_improvement"]
            self.stable_count = p["consecutive_stable_count"]
            # Long-term fluctuation activation is one-way, but cannot begin
            # before the configured start epoch. It starts on a sufficiently low
            # relative fluctuation or is forced on at the activation epoch.
            spread_enabled = bool(t.get("plateau_fluctuation_spread_enabled", False))
            current_relative = float(p.get("relative_fluctuation", float("nan")))
            spread_start_epoch = int(t["plateau_fluctuation_spread_start_epoch"])
            spread_force_epoch = int(t["plateau_fluctuation_spread_activation_epoch"])
            spread_can_activate = epoch >= spread_start_epoch
            if spread_enabled and spread_can_activate and not self.long_fluctuation_active:
                self.long_fluctuation_active = bool(
                    (
                        np.isfinite(current_relative)
                        and current_relative
                        <= float(t["plateau_fluctuation_spread_activation"])
                    )
                    or epoch >= spread_force_epoch
                )
            if spread_enabled and self.long_fluctuation_active and np.isfinite(current_relative):
                self.long_fluctuation_history.append(current_relative)
                self.long_fluctuation_history = self.long_fluctuation_history[
                    -int(t["plateau_fluctuation_spread_window"]):
                ]
            spread = float("inf")
            spread_low = float("nan")
            spread_high = float("nan")
            spread_ready = len(self.long_fluctuation_history) >= int(
                t["plateau_fluctuation_spread_window"]
            )
            if self.long_fluctuation_history:
                low_q, high_q = map(float, t["plateau_fluctuation_spread_quantiles"])
                spread_low = float(np.quantile(self.long_fluctuation_history, low_q))
                spread_high = float(np.quantile(self.long_fluctuation_history, high_q))
                spread = spread_high - spread_low
            long_spread_passes = bool(
                spread_enabled
                and self.long_fluctuation_active
                and spread_ready
                and spread <= float(t["plateau_fluctuation_spread_tolerance"])
            )
            p["long_fluctuation_active"] = self.long_fluctuation_active
            p["long_fluctuation_sample_count"] = len(self.long_fluctuation_history)
            p["long_fluctuation_low_quantile"] = spread_low
            p["long_fluctuation_high_quantile"] = spread_high
            p["long_fluctuation_spread"] = spread
            p["long_fluctuation_passes"] = long_spread_passes
            common_passes = bool(
                epoch >= int(t["convergence_start_epoch"])
                and p["near_best_passes"]
                and p["no_improvement_passes"]
            )
            p["short_branch_passes"] = bool(
                common_passes and p["short_stability_passes"]
            )
            p["long_branch_passes"] = bool(
                common_passes and spread_enabled and long_spread_passes
            )
            p["success_plateau_passes"] = bool(
                p["short_branch_passes"] or p["long_branch_passes"]
            )
            p["passes"] = p["success_plateau_passes"]
            if self.writer is not None:
                for key in (
                    "median",
                    "historical_best",
                    "lower_quantile_value",
                    "reference_best",
                    "relative_fluctuation",
                    "absolute_fluctuation",
                    "relative_fluctuation_limit",
                    "fluctuation_limit",
                    "epochs_without_significant_improvement",
                    "consecutive_stable_count",
                ):
                    self.writer.add_scalar(f"convergence/{key}", p[key], epoch)
            if epoch % t["plateau_log_interval"] == 0 or p["success_plateau_passes"]:
                self._print_convergence_table(epoch, p, long_spread_passes)
            if p["success_plateau_passes"]:
                self.reason = "converged_near_best_plateau"
            return
        if (
            not (
                t["check_plateau_stability"]
                or t["check_plateau_absolute_stability"]
                or t["check_plateau_near_best"]
                or t["check_plateau_consecutive_stability"]
                or t["check_plateau_best_improvement"]
            )
            or value is None
        ):
            return
        self.plateau_history.append(value)
        if epoch < t["convergence_start_epoch"]:
            return
        self.plateau = metric_window_convergence_status(
            self.plateau_history,
            t["plateau_window"],
            t["check_plateau_stability"],
            t["fluctuation_relative_tolerance"],
            t["check_plateau_best_improvement"],
            t["best_improvement_relative_tolerance"],
            self.reference_best,
            self.stale,
            t["success_plateau_patience"],
            tuple(t["fluctuation_quantiles"]),
            check_absolute_stability=t["check_plateau_absolute_stability"],
            absolute_tolerance=t["fluctuation_absolute_maximum"],
            check_near_best=t["check_plateau_near_best"],
            near_best_relative_tolerance=t["plateau_near_best_relative_tolerance"],
            near_best_absolute_tolerance=t["plateau_near_best_absolute_tolerance"],
            check_consecutive_stability=t["check_plateau_consecutive_stability"],
            stable_count=self.stable_count,
            stable_required=t["plateau_stable_consecutive_checks"],
        )
        if self.plateau is None:
            return
        p = self.plateau
        self.reference_best = p["reference_best"]
        self.stale = p["stale"]
        self.stable_count = p["consecutive_stable_count"]
        if self.writer is not None:
            for key in (
                "relative_fluctuation",
                "absolute_fluctuation",
                "relative_best_improvement",
                "epochs_without_significant_improvement",
                "historical_best",
                "near_best_limit",
                "near_best_passes",
                "consecutive_stable_count",
            ):
                self.writer.add_scalar(f"convergence/{key}", p[key], epoch)
        if epoch % t["plateau_log_interval"] == 0 or p["passes"]:
            print(
                f"[{t['plateau_metric']} legacy plateau | epoch {epoch}]\n"
                "  Window statistics:\n"
                f"    median={p['median']:.6e}, q{p['lower_quantile']:.2f}={p['lower_quantile_value']:.6e}, "
                f"q{p['upper_quantile']:.2f}={p['upper_quantile_value']:.6e}\n"
                "  Stability check:\n"
                f"    absolute_width={p['absolute_fluctuation']:.6e}, relative_width={p['relative_fluctuation']:.6e}, "
                f"relative_passes={p['relative_stability_passes']}, absolute_passes={p['absolute_stability_passes']}, "
                f"stable_checks={self.stable_count}/{t['plateau_stable_consecutive_checks']}\n"
                "  Near-best check:\n"
                f"    historical_best={p['historical_best']:.6e}, limit={p['near_best_limit']:.6e}, "
                f"passes={p['near_best_passes']}\n"
                "  Improvement check:\n"
                f"    window_min={p['window_min']:.6e}, anchor={p['reference_best']:.6e}, "
                f"relative_improvement={p['relative_best_improvement']:.6e}\n"
                "  Stop check:\n"
                f"    epochs_without_improvement={self.stale}/{t['success_plateau_patience']}, passes={p['passes']}",
                flush=True,
            )
        if p["passes"]:
            self.reason = "converged_near_best_plateau"

    def finish(self):
        """Save best/final weights, old history schemas, figures and full summaries."""
        elapsed = time.perf_counter() - self.started
        epoch = len(self.history)
        if not epoch:
            raise RuntimeError("Training completed without an epoch")
        if not self.mse_epochs or self.mse_epochs[-1] != epoch:
            final_mse = self._measure()["mse"]
            self.mse_epochs.append(epoch)
            self.mses.append(final_mse)
            self.history[-1]["dataset_mse"] = final_mse
            self._select_best(epoch, self.history[-1]["train_loss"], final_mse)
        final_state = _clone_state(self.model)
        if self.best_state is None:
            raise RuntimeError(
                "Training completed without producing a best model state"
            )
        self.model.load_state_dict(self.best_state)
        try:
            best_mse = self._measure()["mse"]
        finally:
            self.model.load_state_dict(final_state)
        for prefix, state, selected_epoch in (
            ("best", self.best_state, self.best_epoch),
            ("final", final_state, epoch),
        ):
            payload = {
                "model_state_dict": state,
                "config": self.saved_config,
                "grid": self.config["grid"],
                "model_name": self.name,
                "epoch": selected_epoch,
            }
            torch.save(payload, self.checkpoint / f"{prefix}.pt")
            torch.save(state, self.checkpoint / f"{prefix}_model.pth")
        summary = self._write_artifacts(best_mse, elapsed)
        print(
            f"Best model parameters saved to: {self.checkpoint / 'best_model.pth'} "
            f"(epoch={self.best_epoch}, {self.training['best_model_metric']}={self.best:.6e})",
            flush=True,
        )
        print(
            f"Final model parameters saved to: {self.checkpoint / 'final_model.pth'}",
            flush=True,
        )
        print(f"\n{self.dimension}-D TRAINING RUN COMPLETED", flush=True)
        print_training_run_summary(summary)
        return self.history

    def _write_artifacts(self, best_mse, elapsed):
        t = self.training
        h = self.config.get("hmonet_training", {})
        hmodel = self.config.get("model", {}).get(self.name, {})
        mapping = (
            hmodel.get("hmonet_mapping_mode", "current")
            if self.name in ("hmonet", "pihmonet", "attentionpihmonet")
            else None
        )
        title = f"{dict(hmonet='HMONet', meshonet='MeshONet', pimeshonet='PI-MeshONet', pihmonet='PI-HMONet', attentionpihmonet='Attention-PI-HMONet')[self.name]}{self.dimension}D {self.mode} training loss"
        write_loss_outputs(self.output, self.history, title)
        write_mse_outputs(
            self.output, self.mse_epochs, self.mses, f"{title}: full-dataset MSE"
        )
        mse_by_epoch = dict(zip(self.mse_epochs, self.mses))
        rows = []
        for item in self.history:
            row = {"epoch": item["epoch"], "training_loss": item["train_loss"]}

            if self.name in ("pimeshonet", "pihmonet", "attentionpihmonet"):
                row["loss_pde"] = item["loss_pde"]
            if self.name == "pimeshonet":
                row["loss_boundary"] = item["loss_boundary"]
            if self.name == "pihmonet":
                for key in (
                    "loss_pde_normalized",
                    "anchor",
                    "anchor_weight",
                    "valid_anchor_ratio",
                    "grid_smoothness",
                    "orthogonality",
                ):
                    row[key] = item.get(key, "")
            if self.name == "attentionpihmonet":
                for key in (
                    "loss_pde_normalized",
                    "jacobian",
                    "anchor",
                    "anchor_weight",
                    "valid_anchor_ratio",
                    "grid_smoothness",
                    "boundary_pde",
                    "boundary_jacobian",
                    "minimum_sampled_jacobian",
                    "minimum_boundary_jacobian",
                    "loss_boundary",
                    "alpha_penalty",
                    "boundary_alpha",
                ):
                    row[key] = item.get(key, "")
            geometry_training = self.config.get(f"{self.name}_training", {})
            if self.name in ("hmonet", "pihmonet"):
                row["jacobian_loss"] = (
                    item.get("jacobian", "")
                    if geometry_training.get("jacobian_loss_enabled", False)
                    else ""
                )
            row["dataset_mse"] = mse_by_epoch.get(item["epoch"], "")
            rows.append(row)
        write_history(self.output / "training_history.csv", rows)
        minimum_loss_row = min(self.history, key=lambda row: row["train_loss"])
        summary = {
            "model": self.name,
            "physics_training": (
                self.config.get(f"{self.name}_training")
                if self.name in ("pimeshonet", "pihmonet", "attentionpihmonet")
                else None
            ),
            "dimension": self.dimension,
            "dataset_name": self.datasets[0].root.name,
            "grid_shape": list(self.shape),
            "requested_epochs": int(t["num_epochs"]),
            "hmonet_mapping_mode": mapping,
            "residual_regularization_weight": (
                hmodel.get("residual_regularization_weight", 0.0)
                if self.name == "hmonet"
                else None
            ),
            **{
                key: h.get(key, default) if self.name == "hmonet" else None
                for key, default in (
                    ("jacobian_loss_enabled", False),
                    ("jacobian_loss_weight", 0.01),
                    ("jacobian_minimum", 1e-4),
                    ("jacobian_method", "finite_difference"),
                    ("jacobian_fd_step", 1e-3),
                    ("target_interpolation", "curved_boundary_cell"),
                    ("idw_neighbors", -1),
                    ("idw_query_chunk_size", 512),
                    ("idw_support_chunk_size", 2048),
                    ("boundary_normal_loss_enabled", False),
                    ("boundary_normal_loss_weight", 0.01),
                    ("boundary_normal_sample_count", 64),
                    ("boundary_normal_offset", 0.01),
                )
            },
            "dataset": self.mode,
            "training_time_seconds": elapsed,
            "completed_epochs": len(self.history),
            "convergence_epoch": (
                len(self.history) if self.reason != "maximum_epochs" else None
            ),
            "stop_reason": self.reason,
            "best_epoch": self.best_epoch,
            "best_model_metric": t["best_model_metric"],
            "best_metric_value": self.best,
            "convergence_start_epoch": t["convergence_start_epoch"],
            "success_plateau_patience": t["success_plateau_patience"],
            "failure_plateau_enabled": t.get("failure_plateau_enabled", False),
            "shared_plateau_window": t["plateau_window"],
            "failure_plateau_patience": t.get("failure_plateau_patience"),
            "failure_plateau_metric_ratio": t.get("failure_plateau_metric_ratio"),
            "failure_plateau_relative_fluctuation": t.get(
                "failure_plateau_relative_fluctuation"
            ),
            "failure_plateau_min_lr_updates": t.get(
                "failure_plateau_min_lr_updates"
            ),
            "final_failure_plateau": self.failure_plateau,
            "save_best_immediately": t.get("save_best_immediately", True),
            "stop_file_enabled": t.get("stop_file_enabled", True),
            "mse_consecutive_epochs_required": t["mse_consecutive_epochs"],
            "final_mse_consecutive_count": self.mse_hits,
            "check_mse_convergence": t["check_mse_convergence"],
            "check_training_loss_convergence": t["check_training_loss_convergence"],
            "training_loss_tolerance": t["training_loss_tolerance"],
            "training_loss_consecutive_epochs_required": t[
                "training_loss_consecutive_epochs"
            ],
            "final_training_loss_consecutive_count": self.loss_hits,
            "check_plateau_stability": t["check_plateau_stability"],
            "check_plateau_absolute_stability": t["check_plateau_absolute_stability"],
            "check_plateau_near_best": t["check_plateau_near_best"],
            "check_plateau_consecutive_stability": t[
                "check_plateau_consecutive_stability"
            ],
            "plateau_near_best_relative_tolerance": t[
                "plateau_near_best_relative_tolerance"
            ],
            "plateau_near_best_absolute_tolerance": t[
                "plateau_near_best_absolute_tolerance"
            ],
            "plateau_stable_consecutive_checks": t["plateau_stable_consecutive_checks"],
            "final_consecutive_stable_count": self.stable_count,
            "plateau_fluctuation_spread_enabled": t.get("plateau_fluctuation_spread_enabled", False),
            "plateau_fluctuation_spread_activation": t.get("plateau_fluctuation_spread_activation"),
            "plateau_fluctuation_spread_start_epoch": t.get("plateau_fluctuation_spread_start_epoch"),
            "plateau_fluctuation_spread_activation_epoch": t.get("plateau_fluctuation_spread_activation_epoch"),
            "plateau_fluctuation_spread_window": t.get("plateau_fluctuation_spread_window"),
            "plateau_fluctuation_spread_tolerance": t.get("plateau_fluctuation_spread_tolerance"),
            "final_long_fluctuation_active": self.long_fluctuation_active,
            "final_long_fluctuation_sample_count": len(self.long_fluctuation_history),
            "final_long_fluctuation_spread": (self.plateau or {}).get("long_fluctuation_spread"),
            "fluctuation_absolute_minimum": t["fluctuation_absolute_minimum"],
            "fluctuation_absolute_maximum": t["fluctuation_absolute_maximum"],
            "check_plateau_best_improvement": t["check_plateau_best_improvement"],
            "final_plateau_count": self.stale,
            "final_plateau": self.plateau,
            "plateau_metric": t["plateau_metric"],
            "best_training_loss": self.history[self.best_epoch - 1]["train_loss"],
            "minimum_training_loss": minimum_loss_row["train_loss"],
            "minimum_training_loss_epoch": minimum_loss_row["epoch"],
            "final_training_loss": self.history[-1]["train_loss"],
            "final_grid_smoothness_loss": (
                self.history[-1].get("grid_smoothness")
                if self.name in ("pihmonet", "attentionpihmonet")
                else None
            ),
            "final_boundary_alpha": (
                self.history[-1].get("boundary_alpha")
                if self.name == "attentionpihmonet" and self.dimension == 2
                else None
            ),
            "final_alpha_penalty_loss": (
                self.history[-1].get("alpha_penalty")
                if self.name == "attentionpihmonet" and self.dimension == 2
                else None
            ),
            "best_model_dataset_mse": best_mse,
            "final_dataset_mse": self.mses[-1],
            "best_checkpoint": portable_project_path(self.checkpoint / "best.pt"),
            "final_checkpoint": portable_project_path(self.checkpoint / "final.pt"),
            "tensorboard_log_dir": (
                portable_project_path(self.output / "tensorboard")
                if self.writer is not None and self.primary_tensorboard_enabled
                else None
            ),
            "tensorboard_mirror_log_dir": (
                portable_project_path(self.tensorboard_mirror) if self.writer is not None else None
            ),
            "selected_dataset": self.mode,
            "dataset_path": portable_project_path(self.datasets[0].root),
            "seed": t["seed"],
            "best_metric": self.best,
            "initial_learning_rate": self.learning_rate["learning_rate"],
            "final_learning_rate": self.optimizer.param_groups[0]["lr"],
            **{
                key: self.learning_rate[key]
                for key in (
                    "lr_warmup_epochs",
                    "lr_update_interval",
                    "lr_decay_gamma",
                    "lr_stop_below",
                )
            },
            "training_loss_history": [item["train_loss"] for item in self.history],
            "coordinate_loss_history": (
                None
                if self.name in ("pimeshonet", "pihmonet", "attentionpihmonet")
                else [
                    item.get("coordinate_loss", item["train_loss"])
                    for item in self.history
                ]
            ),
            "pde_loss_history": (
                [item["loss_pde"] for item in self.history]
                if self.name in ("pimeshonet", "pihmonet", "attentionpihmonet")
                else None
            ),
            **(
                {
                    "boundary_loss_history": [
                        item["loss_boundary"] for item in self.history
                    ]
                }
                if (
                    self.name == "pimeshonet"
                    or (self.name == "attentionpihmonet" and self.dimension == 2)
                )
                and "loss_boundary" in self.history[0]
                else {}
            ),
            "boundary_alpha_history": (
                [item.get("boundary_alpha") for item in self.history]
                if self.name == "attentionpihmonet" and self.dimension == 2
                else []
            ),
            "alpha_penalty_loss_history": (
                [item.get("alpha_penalty", 0.0) for item in self.history]
                if self.name == "attentionpihmonet" and self.dimension == 2
                else []
            ),
            "normalized_pde_loss_history": (
                [item.get("loss_pde_normalized", 0.0) for item in self.history]
                if self.name in ("pihmonet", "attentionpihmonet")
                else []
            ),
            "anchor_loss_history": (
                [item.get("anchor", 0.0) for item in self.history]
                if self.name in ("pihmonet", "attentionpihmonet")
                else []
            ),
            "anchor_weight_history": (
                [item.get("anchor_weight", 0.0) for item in self.history]
                if self.name in ("pihmonet", "attentionpihmonet")
                else []
            ),
            "grid_smoothness_loss_history": (
                [item.get("grid_smoothness", 0.0) for item in self.history]
                if self.name in ("pihmonet", "attentionpihmonet")
                else []
            ),
            "orthogonality_loss_history": (
                [item.get("orthogonality", 0.0) for item in self.history]
                if self.name in ("pihmonet", "attentionpihmonet")
                else []
            ),
            "jacobian_loss_history": (
                [item.get("jacobian", 0.0) for item in self.history]
                if (
                    self.name == "attentionpihmonet"
                    and self.config[f"{self.name}_training"].get(
                        "jacobian_loss_weight", 0.0
                    ) > 0
                )
                or (
                    self.name in ("hmonet", "pihmonet")
                    and self.config.get(f"{self.name}_training", {}).get(
                        "jacobian_loss_enabled", False
                    )
                )
                else []
            ),
            "boundary_pde_loss_history": (
                [item.get("boundary_pde", 0.0) for item in self.history]
                if self.name == "attentionpihmonet"
                else []
            ),
            "boundary_jacobian_loss_history": (
                [item.get("boundary_jacobian", 0.0) for item in self.history]
                if self.name == "attentionpihmonet"
                else []
            ),
            "boundary_normal_loss_history": (
                [item.get("boundary_normal", 0.0) for item in self.history]
                if h.get("boundary_normal_loss_enabled") and self.name == "hmonet"
                else []
            ),
            "dataset_mse_history": self.mses,
            "dataset_mse_epochs": self.mse_epochs,
            "dataset_mse_interval": t["dataset_mse_interval"],
            "mse_scope": "fixed_interior_nodes",
            "dataset_mse_scale": FIXED_GRID_MSE_SCALE,
            "last_plateau_status": self.plateau,
            "plateau_enabled": t["check_plateau_stability"]
            or t["check_plateau_absolute_stability"]
            or t["check_plateau_near_best"]
            or t["check_plateau_consecutive_stability"]
            or t["check_plateau_best_improvement"],
            "checkpoint_dir": portable_project_path(self.checkpoint),
            "run_id": self.run_id,
            "training_output_dir": str(self.output),
            "checkpoint_files": [
                "best.pt",
                "best_model.pth",
                "final.pt",
                "final_model.pth",
            ],
            "config_json": str(self.output / "config.json"),
            "config_yaml": str(self.checkpoint / "config.yaml"),

        }
        conditions = []
        if t["check_training_loss_convergence"]:
            conditions.append(
                f"training_loss <= {t['training_loss_tolerance']:.6e} for {t['training_loss_consecutive_epochs']} consecutive epochs"
            )
        if t["check_mse_convergence"]:
            conditions.append(
                f"dataset_mse <= {t['mse_tolerance']:.6e} for {t['mse_consecutive_epochs']} consecutive measurements"
            )
        summary["plateau_enabled"] = t.get(
            "plateau_enabled", summary["plateau_enabled"]
        )
        summary["plateau_rule"] = (
            "window_quantile_improvement_and_median_near_best"
            if "plateau_enabled" in t
            else "legacy"
        )
        if summary["plateau_enabled"]:
            conditions.append(
                f"near-best stable {t['plateau_metric']} plateau "
                f"(start={t['convergence_start_epoch']}, "
                f"window={t['plateau_window']}, "
                f"patience={t['success_plateau_patience']})"
            )
        if t.get("failure_plateau_enabled", False):
            conditions.append(
                f"failed stable plateau (start={t['convergence_start_epoch']}, "
                f"window={t['plateau_window']}, "
                f"patience={t['failure_plateau_patience']}, "
                f"ratio>={t['failure_plateau_metric_ratio']})"
            )
        summary["enabled_convergence_conditions"] = conditions
        for directory in (self.output, self.checkpoint):
            (directory / "training_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            with (directory / "training_summary.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(summary.keys())
                writer.writerow(summary.values())
        metadata = {
            "model": self.name,
            "physics_training": (
                self.config.get(f"{self.name}_training")
                if self.name in ("pimeshonet", "pihmonet", "attentionpihmonet")
                else None
            ),
            "model_config": self.saved_config["model"],
            "dataset": self.datasets[0].root.name,
            "sampling_mode": "non_adaptive" if self.name == "hmonet" else None,
            "hmonet_mapping_mode": mapping,
            "run_id": self.checkpoint.name,
            "best_checkpoint": summary["best_checkpoint"],
            "final_checkpoint": summary["final_checkpoint"],
        }
        (self.checkpoint / "checkpoint_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        (self.checkpoint / "model_parameters.json").write_text(
            json.dumps(
                {
                    "model_name": self.name,
                    "grid": self.config["grid"],
                    "model": self.saved_config["model"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if self.writer is not None:

            for key in (
                "best_metric_value",
                "best_model_dataset_mse",
                "final_dataset_mse",
                "final_training_loss",
            ):
                self.writer.add_scalar(
                    f"summary/{key}", summary[key], len(self.history)
                )
            self.writer.add_text(
                "run/summary",
                "```json\n" + json.dumps(summary, indent=2) + "\n```",
                len(self.history),
            )
            self.writer.flush()
            self.writer.close()
        return summary


def print_training_run_summary(summary):
    """Report the saved best checkpoint separately from the minimum train loss.

    Checkpoint selection may minimize fixed-grid MSE, so its training loss need
    not be the smallest training loss seen during the run. Keep the legacy
    best_training_loss meaning and report the global minimum separately.
    """
    model = {
        "hmonet": "HMONet",
        "meshonet": "MeshONet",
        "pimeshonet": "PI-MeshONet",
        "pihmonet": "PI-HMONet",
        "attentionpihmonet": "Attention-PI-HMONet",
    }.get(summary["model"], summary["model"])

    def metric(value):
        return "N/A" if value is None else f"{value:.6e}"

    seconds = float(summary["training_time_seconds"])
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    fields = [
        ("model/dataset", f"{model} / {summary['dataset']}"),
        ("dataset name", summary["dataset_name"]),
        ("run ID", summary["run_id"]),
        ("grid shape", "x".join(map(str, summary["grid_shape"]))),
        ("mapping", summary.get("hmonet_mapping_mode") or "N/A"),
        ("completed epochs", summary["completed_epochs"]),
        ("requested epochs", summary["requested_epochs"]),
        ("training time", f"{seconds:.3f}s ({hours:02d}:{minutes:02d}:{secs:02d})"),
        ("best epoch", summary["best_epoch"]),
        (
            "best metric",
            f"{summary['best_model_metric']} = {metric(summary['best_metric_value'])}",
        ),
        ("best model train loss", metric(summary["best_training_loss"])),
        ("best model dataset MSE", metric(summary["best_model_dataset_mse"])),
        (
            "minimum train loss",
            f"{metric(summary['minimum_training_loss'])} "
            f"(epoch={summary['minimum_training_loss_epoch']})",
        ),
        ("final train loss", metric(summary["final_training_loss"])),
        ("final dataset MSE", metric(summary["final_dataset_mse"])),
        ("stop reason", summary["stop_reason"]),
        (
            "enabled convergence",
            "; ".join(summary["enabled_convergence_conditions"]) or "none",
        ),
        (
            "learning rate",
            f"{summary['initial_learning_rate']:.6e} -> "
            f"{summary['final_learning_rate']:.6e}",
        ),
        ("best checkpoint", summary["best_checkpoint"]),
        ("final checkpoint", summary["final_checkpoint"]),
        ("training output", summary["training_output_dir"]),
    ]
    for label, value in fields:
        print(f"  {label:<22}: {value}", flush=True)


def print_training_summaries(summaries, wall_time_seconds, dimension):
    """Print all model/dataset summaries at the end of either dimension's batch."""
    print("\n" + "=" * 88, flush=True)
    print(f"{dimension}-D TRAINING SUMMARY (all selected runs completed)", flush=True)
    print("=" * 88, flush=True)
    for index, summary in enumerate(summaries, start=1):
        print(f"[{index}/{len(summaries)}]", flush=True)
        print_training_run_summary(summary)
    print("-" * 88, flush=True)
    print(f"completed runs        : {len(summaries)}", flush=True)
    print(f"total wall time       : {wall_time_seconds:.3f}s", flush=True)
    print("=" * 88, flush=True)











