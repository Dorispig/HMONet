"""Run complete 2-D HMONet ablations with isolated training and evaluation outputs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
import sys
import traceback
from types import SimpleNamespace

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import portable_project_path, resolve_project_path
from src.utils.config import load_yaml_mapping
from src.models.mapping_modes import fingerprint_mapping_mode
from scripts.experiments.evaluation_paths import (
    evaluation_experiment_root,
    evaluation_result_root,
    evaluation_staging_root,
    validate_evaluation_destination,
)

ABLATIONS = (
    "rational_affine",
    "coons_residual",
    "uniform_400",
    "adaptive_400",
    "global300_adaptive100",
    "global100_adaptive300",
    "e1_modified_mlp",
    "e2_cross_attention",
    "e3_boundary_decoder",
    "e4_decoupled_weights",
    "e5_qg_hmonet",
    "e6_modified_cross",
    "e7_modified_cross_boundary",
    "e8_full_factorized",
    "e9_meshonet_200k",
    "e10_pimeshonet_200k",
    "e11_raw_pde",
    "e1_fd_full_open_domain",
    "e2_autograd",
    "e3_fd_one_sided_boundary",
    "e1_1_jacobian_w100",
    "e1_2_jacobian_w10",
    "e1_3_jacobian_w1",
    "e1_4_jacobian_w0_1",
    "e1_5_jacobian_w0_001",
    "e1_6_jacobian_w0_0001",
    "e2_1_jacobian_tau0_001",
    "e2_2_jacobian_tau0_005",
    "e2_3_jacobian_tau0_01",
    "e2_4_jacobian_tau0_05",
    "e2_5_jacobian_tau0_1",
    "e2_6_jacobian_tau0_2",
    "e3_1_svd_w100",
    "e3_2_svd_w10",
    "e3_3_svd_w1",
    "e3_4_svd_w0_1",
    "e3_5_svd_w0_01",
    "e3_6_svd_w0_001",
    "e4_1_svd_tau0_001",
    "e4_2_svd_tau0_005",
    "e4_3_svd_tau0_01",
    "e4_4_svd_tau0_1",
    "e4_5_svd_tau0_2",
    "e1_1_fd0_0005",
    "e1_2_fd0_001",
    "e1_3_fd0_005",
    "e1_4_fd0_05",
    "e1_5_fd0_1",
    "e1_1_power1",
    "e1_2_power3",
    "e1_3_power4",
    "e1_4_power5",
    "e1_pde_only",
    "e2_pde_jacobian",
    "e3_pde_svd",
)
BASELINE_ABLATION = "baseline"
EVALUATION_IMAGE_SUFFIXES = {".jpeg", ".jpg", ".pdf", ".png", ".svg", ".webp"}
TRAINING_IDENTITY_SCHEMA_VERSION = 4
DEFAULT_EXPERIMENT_SET = "ex1"
DEFAULT_BASELINE_CONFIG = ROOT / "experiments" / DEFAULT_EXPERIMENT_SET / "baseline/experiment.yaml"
DEFAULT_EVALUATE_CONFIG = ROOT / "experiments" / DEFAULT_EXPERIMENT_SET / "baseline/evaluate.yaml"
EXPERIMENT_SET_PATTERN = re.compile(r"ex[1-9][0-9]*")
RUN_TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
MAX_RUN_TAG_LENGTH = 24


class BaselineCompatibilityError(RuntimeError):
    """Raised after all runnable work when baseline comparison is invalid."""


_SUMMARY_METRICS = {
    "important": (
        "fixed_grid_best_mse",
        "fixed_grid_final_mse",
        "invalid_cell_fraction",
        "self_intersection__affected_cell_fraction",
        "flipping_cell__affected_cell_fraction",
        "scaled_jacobian__minimum",
        "network_inference_seconds",
        "total_model_seconds",
    ),
    "secondary": (
        "jacobian__minimum",
        "skewness__mean",
        "orthogonality__mean",
        "boundary_max_error",
        "boundary_mean_error",
        "training_seconds",
        "average_epoch_seconds",
        "completed_epochs",
    ),
}
METRIC_PROFILES = {
    name: _SUMMARY_METRICS for name in (BASELINE_ABLATION, *ABLATIONS)
}


@dataclass(frozen=True)
class Variant:
    name: str
    value: float | str
    overrides: tuple[str, ...]


def _load_yaml_mapping(path: Path) -> dict:
    """Load one YAML mapping and report malformed experiment files clearly."""
    return load_yaml_mapping(path)


def default_experiment_config(ablation: str, experiment_root: Path | None = None) -> Path:
    """Return the experiment-local definition inside the selected experiment set."""
    root = Path(experiment_root) if experiment_root is not None else ROOT / "experiments" / DEFAULT_EXPERIMENT_SET
    return root / ablation / "experiment.yaml"


def validate_experiment_set_name(value: str) -> str:
    """Accept consecutively numbered experiment suites such as ``ex1``."""
    if not EXPERIMENT_SET_PATTERN.fullmatch(value):
        raise ValueError(
            "--experiment-set must use the form ex1, ex2, ..."
        )
    return value


def validate_run_tag(value: str) -> str:
    """Accept a short portable tag that preserves output-path headroom."""
    value = str(value).strip()
    if not RUN_TAG_PATTERN.fullmatch(value):
        raise ValueError(
            "--run-tag must use letters, digits, '.', '_' or '-', and begin with "
            "a letter or digit"
        )
    if len(value) > MAX_RUN_TAG_LENGTH:
        raise ValueError(
            f"--run-tag must contain at most {MAX_RUN_TAG_LENGTH} characters "
            "to keep Windows and server output paths portable"
        )
    return value


def declared_config_path(definition_path: Path, key: str) -> Path | None:
    """Resolve an optional shared config declared by an experiment definition."""
    definition = _load_yaml_mapping(definition_path)
    configured = definition.get(key)
    if not configured:
        return None
    return (definition_path.parent / configured).resolve()


def baseline_experiment_root(baseline_config: Path, fallback: Path) -> Path:
    """Return the suite that owns a canonical ``baseline/experiment.yaml``."""
    baseline_config = baseline_config.resolve()
    if baseline_config.parent.name == BASELINE_ABLATION:
        return baseline_config.parent.parent.resolve()
    return fallback.resolve()


def variants_from_definition(definition: dict) -> list[Variant]:
    """Build variants from an experiment-local YAML definition."""
    items = definition.get("variants", [])
    variants = []
    for item in items:
        if not isinstance(item, dict) or not item.get("name"):
            raise ValueError("Every experiment variant needs a non-empty name")
        overrides = item.get("overrides", [])
        if not isinstance(overrides, list) or not all(
            isinstance(value, (str, int, float, bool)) for value in overrides
        ):
            raise TypeError(f"Variant {item['name']} overrides must be a scalar list")
        variants.append(
            Variant(
                str(item["name"]),
                item.get("value", item["name"]),
                tuple(str(value) for value in overrides),
            )
        )
    return variants


def reject_embedded_baseline_variants(variants: list[Variant], definition_path: Path):
    """Keep canonical baselines out of experiment-local variant definitions."""
    embedded = [variant.name for variant in variants if variant.name == BASELINE_ABLATION]
    if embedded:
        raise ValueError(
            f"{definition_path} contains an experiment-local baseline variant. "
            "Ablation commands never train or evaluate baselines; keep the canonical "
            "baseline only in the suite baseline/ directory and remove this variant."
        )


def resolve_dataset(value: str) -> Path:
    """Resolve an absolute path, project-relative path, or data/raw/2D name."""
    candidate = Path(value)
    choices = [
        resolve_project_path(value),
        candidate,
        ROOT / candidate,
        ROOT / "data/raw/2D" / candidate,
    ]
    for choice in choices:
        if choice.exists():
            return choice.resolve()
    raise FileNotFoundError(f"2-D dataset not found: {value}")


def dataset_name(path: Path) -> str:
    return path.parent.name if path.is_file() else path.name


def dataset_mode(name: str) -> str:
    candidate = ROOT / "data/raw/2D" / name
    if candidate.is_dir():
        try:
            from src.data.parametric_dataset import read_dataset_metadata

            split = read_dataset_metadata(candidate).get("split")
            if split in {"interpolation", "extrapolation"}:
                return split
        except (FileNotFoundError, KeyError, ValueError):
            pass
    if "_int_" in name:
        return "interpolation"
    if "_ext_" in name:
        return "extrapolation"
    return "custom"


def dataset_grid(path: Path) -> tuple[int, int]:
    from src.data.parametric_dataset import ParametricDataset2D

    data = ParametricDataset2D(path, load_boundary=False)
    shape = tuple(map(int, data.grid_shape))
    del data
    if len(shape) != 2:
        raise ValueError(f"Expected a 2-D dataset, got grid {shape}")
    return shape


def evaluation_grid_shapes(args) -> list[tuple[int, int]]:
    """Parse evaluation grid sizes without changing the training identity."""
    configured = getattr(args, "evaluation_grid_shapes", None)
    if not configured:
        return [(51, 51)]
    shapes = []
    for item in configured:
        for token in str(item).split(","):
            parts = token.strip().lower().split("x")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid evaluation grid shape {token!r}; use NXxNY"
                )
            shape = tuple(int(value) for value in parts)
            if min(shape) < 2:
                raise ValueError("Evaluation grid dimensions must be at least 2")
            if shape not in shapes:
                shapes.append(shape)
    if not shapes:
        raise ValueError("At least one evaluation grid shape is required")
    return shapes


def run_directory(root: Path, ablation: str, dataset: str, variant: str, seed: int, tag: str):
    # Single-variant experiments conventionally give the variant the same
    # short name as the experiment. Avoid storing ``.../e9/e9/...`` while
    # retaining the established layout for true multi-variant ablations.
    if variant == ablation:
        return root / ablation / dataset / f"seed_{seed}" / tag
    return root / ablation / dataset / variant / f"seed_{seed}" / tag


def baseline_directory(root: Path, dataset: str, seed: int, tag: str):
    """Return the canonical location for a separately trained baseline run."""
    return root / BASELINE_ABLATION / dataset / f"seed_{seed}" / tag


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


_DATASET_DIGEST_CACHE = {}
_MIGRATED_DATASET_DIGESTS = {
    # The NumPy payload metadata was renamed without changing any coordinates.
    # Preserve completed experiment identities across that one-time migration.
    "air_t_int_0_1000_41_20": "45b7d46ef7d13787e9dca07b0838c0c2c9fbc7416b2e52fa021520748de5e6e1",
    "air_t_ext_0_1000_21_20": "6dc7441a518358fbb828ff9b730aa13ee4381294929b423bf9ffb501b04ca148",
}


# Training flags that carry their value in the flag itself.  Keep this list in
# one place so identity parsing and command forwarding cannot disagree when a
# new architecture switch is added.
_BOOLEAN_OVERRIDE_FLAGS = frozenset(
    {
        "--jacobian-loss-enabled",
        "--boundary-normal-loss-enabled",
        "--boundary-self-attention-enabled",
        "--pde-loss-enabled",
        "--hmonet-physics-jacobian-enabled",
        "--hmonet-save-phase-best-checkpoints",
        "--hmonet-reset-optimizer-on-phase-transition",
        "--hmonet-physics-scale-invariant-pde-enabled",
        "--modified-mlp-enabled",
        "--query-boundary-cross-attention-enabled",
        "--boundary-layer-decoder-enabled",
        "--decoupled-weight-decoder-enabled",
        "--multiscale-boundary-encoder-enabled",
    }
)


def _override_value_count(flag: str) -> int:
    """Return how many following tokens belong to one controlled override."""
    if flag.startswith("--no-") or flag in _BOOLEAN_OVERRIDE_FLAGS:
        return 0
    if flag == "--pde-fd-step":
        return 2
    return 1


def _dataset_digest(dataset: Path) -> str:
    """Hash the actual NumPy files consumed by training, not plots or audits."""
    dataset = dataset.resolve()
    compatible = _MIGRATED_DATASET_DIGESTS.get(dataset.name)
    if compatible:
        return compatible
    files = sorted(dataset.rglob("*.npy")) if dataset.is_dir() else [dataset]
    signature = tuple((str(item), item.stat().st_size, item.stat().st_mtime_ns) for item in files)
    cached = _DATASET_DIGEST_CACHE.get(signature)
    if cached:
        return cached
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item.relative_to(dataset) if dataset.is_dir() else item.name).encode())
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    value = digest.hexdigest()
    _DATASET_DIGEST_CACHE[signature] = value
    return value


def _variant_options(overrides: tuple[str, ...]) -> dict:
    """Parse the controlled training-only override flags used by ablations."""
    result = {}
    index = 0
    while index < len(overrides):
        flag = overrides[index]
        if not flag.startswith("--"):
            raise ValueError(f"Invalid ablation override flag: {flag!r}")
        if flag.startswith("--no-"):
            result[flag[5:].replace("-", "_")] = False
            index += 1
        elif flag in _BOOLEAN_OVERRIDE_FLAGS:
            result[flag[2:].replace("-", "_")] = True
            index += 1
        else:
            value_count = _override_value_count(flag)
            if index + value_count >= len(overrides):
                raise ValueError(
                    f"Ablation override {flag} requires {value_count} value(s)"
                )
            values = list(overrides[index + 1 : index + 1 + value_count])
            result[flag[2:].replace("-", "_")] = (
                values[0] if value_count == 1 else values
            )
            index += 1 + value_count
    return result


def variant_model(variant: Variant) -> str:
    """Return the one network trained by a variant (HMONet by default)."""
    model = str(_variant_options(variant.overrides).get("model", "hmonet")).lower()
    if model not in {"hmonet", "qghmonet", "meshonet", "pimeshonet"}:
        raise ValueError(f"Unsupported ablation model: {model!r}")
    return model


def variant_training_overrides(variant: Variant) -> list[str]:
    """Remove the runner-owned --model pair from forwarded training flags."""
    values = list(variant.overrides)
    result = []
    index = 0
    while index < len(values):
        flag = values[index]
        if not flag.startswith("--"):
            raise ValueError(f"Invalid ablation override flag: {flag!r}")
        value_count = _override_value_count(flag)
        if index + value_count >= len(values):
            raise ValueError(
                f"Ablation override {flag} requires {value_count} value(s)"
            )
        if flag == "--model":
            index += 1 + value_count
            continue
        result.extend(values[index : index + 1 + value_count])
        index += 1 + value_count
    return result


def _training_config_path(args) -> Path:
    """Return the frozen training baseline, with legacy argument compatibility."""
    configured = getattr(args, "training_config", None)
    if configured is None:
        configured = getattr(args, "train_config", None)
    if configured is None:
        configured = DEFAULT_BASELINE_CONFIG
    return resolve_project_path(configured)


def _normalize_training_snapshot(snapshot: dict, dataset: Path) -> dict:
    """Remove machine-specific paths while retaining every training parameter."""
    normalized = copy.deepcopy(snapshot)
    mode = dataset_mode(dataset_name(dataset))
    normalized["datasets"] = {mode: dataset_name(dataset)}
    # ``uniform_index`` has always been the semantic default. Keeping it explicit
    # preserves completed-run fingerprints created before the config refactor.
    normalized.setdefault("training", {}).setdefault(
        "dataset_subset_mode", "uniform_index"
    )
    # Explicit default Linear is equivalent to legacy snapshots without this option.
    for block in normalized.get("model", {}).values():
        if not isinstance(block, dict):
            continue
        if block.get("weight_head_mlp_enabled") is False:
            block.pop("weight_head_mlp_enabled")
        lift_mode = str(block.get("coordinate_lift_mode", "legacy")).lower()
        if lift_mode == "legacy":
            block.pop("coordinate_lift_mode", None)
            block.pop("coordinate_lift_max_power", None)
            block.pop("coordinate_lift_q", None)
        elif block.get("coordinate_lift_max_power") == 1:
            block.pop("coordinate_lift_max_power")
    return normalized


def _legacy_dataset_identity(value):
    """Use pre-migration names only while hashing completed version-0 datasets.

    Existing checkpoints remain reusable after the directory migration. New
    sizes or parameter-range versions use their canonical names directly.
    """
    if isinstance(value, dict):
        return {
            key: (
                fingerprint_mapping_mode(item)
                if key.endswith("mapping_mode") and item != "both"
                else _legacy_dataset_identity(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_legacy_dataset_identity(item) for item in value]
    if isinstance(value, str):
        match = re.match(
            r"^(.+)_(int|ext)_0_1000_(\d+_\d+)$", value
        )
        if match:
            return f"{match.group(1)}_{match.group(2)}_000_p{match.group(3)}"
    return value


def effective_training_identity(args, dataset: Path, variant: Variant, seed: int):
    """Build the exact effective single-model configuration for reuse checks."""
    from src.engine.state import (
        model_specific_training_config,
        prepare_training_config,
    )

    config = copy.deepcopy(_load_yaml_mapping(_training_config_path(args)))
    nx, ny = dataset_grid(dataset)
    config.setdefault("grid", {}).update(nx=nx, ny=ny)
    training = config.setdefault("training", {})
    training["seed"] = int(seed)
    training["dataset"] = dataset_mode(dataset_name(dataset))
    if args.device:
        training["device"] = args.device
    model_name = variant_model(variant)
    model = config.setdefault("model", {}).setdefault("hmonet", {})
    common_model = config.setdefault("model", {}).setdefault("common", {})
    htraining = config.setdefault("hmonet_training", {})
    sampling = config.setdefault("hmonet_sampling", {})
    options = _variant_options(variant.overrides)
    options.pop("model", None)
    if model_name != "hmonet":
        model_training = config.setdefault(f"{model_name}_training", {})
        model_config = config.setdefault("model", {}).setdefault(model_name, {})
        common_capacity_keys = {
            "branch_dim",
            "trunk_dim",
            "coordinate_lift_mode",
            "coordinate_lift_max_power",
            "coordinate_lift_q",
            "adjust_hidden_dim",
            "adjust_hidden_layer_num",
            "branch_hidden_dim",
            "branch_hidden_layer_num",
            "trunk_hidden_dim",
            "trunk_hidden_layer_num",
            "output_dim",
            "branch_mlp_hidden_dim",
            "branch_mlp_hidden_layer_num",
        }
        if "lr" in options:
            model_training["learning_rate"] = float(options.pop("lr"))
        if "best_model_metric" in options:
            training["best_model_metric"] = str(options.pop("best_model_metric"))
        if "plateau_metric" in options:
            training["plateau_metric"] = str(options.pop("plateau_metric"))
        for key in tuple(options):
            if key in common_capacity_keys:
                value = options.pop(key)
                model_config[key] = (
                    str(value)
                    if key == "coordinate_lift_mode"
                    else int(value)
                )
        if options:
            raise ValueError(
                f"Unsupported {model_name} ablation overrides: {sorted(options)}"
            )
        resolved = prepare_training_config(config, model_name, SimpleNamespace())
        snapshot = model_specific_training_config(
            resolved,
            model_name,
            dataset,
            dataset_mode(dataset_name(dataset)),
        )
        payload = {
            "schema_version": TRAINING_IDENTITY_SCHEMA_VERSION,
            "effective_training_config": _normalize_training_snapshot(snapshot, dataset),
            "dataset_name": dataset_name(dataset),
            "dataset_sha256": _dataset_digest(dataset),
        }
        encoded = json.dumps(
            _legacy_dataset_identity(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), payload
    model_keys = {
        "hmonet_mapping_mode",
        "residual_regularization_weight",
        "weight_head_mlp_enabled",
        "boundary_self_attention_enabled",
        "modified_mlp_enabled",
        "query_boundary_cross_attention_enabled",
        "boundary_layer_decoder_enabled",
        "decoupled_weight_decoder_enabled",
        "multiscale_boundary_encoder_enabled",
        "cross_attention_heads",
        "boundary_layer_width",
        "local_boundary_dim",
    }
    common_model_keys = {
        "branch_dim",
        "trunk_dim",
        "coordinate_lift_mode",
        "coordinate_lift_max_power",
        "coordinate_lift_q",
        "adjust_hidden_dim",
        "adjust_hidden_layer_num",
        "branch_hidden_dim",
        "branch_hidden_layer_num",
        "trunk_hidden_dim",
        "trunk_hidden_layer_num",
        "output_dim",
        "branch_mlp_hidden_dim",
        "branch_mlp_hidden_layer_num",
    }
    sampling_keys = {
        "fixed_grid_count",
        "global_random_count",
        "curvature_adaptive_count",
        "mode",
        "candidate_multiplier",
        "curvature_quantile",
        "curvature_power",
        "distance_sigma_min",
        "distance_sigma_max",
        "uniform_probability",
    }
    aliases = {
        "lr": "learning_rate",
        "sampling_mode": "mode",
        "hmonet_training_strategy": "training_strategy",
        "hmonet_tfi_warmup_epochs": "tfi_warmup_epochs",
        "hmonet_phase1_max_epochs": "phase1_max_epochs",
        "hmonet_phase2_max_epochs": "phase2_max_epochs",
        "hmonet_save_phase_best_checkpoints": "save_phase_best_checkpoints",
        "hmonet_reset_optimizer_on_phase_transition": "reset_optimizer_on_phase_transition",
        "hmonet_fixed_mse_sample_count": "fixed_mse_sample_count",
        "hmonet_fixed_mse_loss_weight": "fixed_mse_loss_weight",
        "hmonet_physics_pde_loss_weight": "physics_pde_loss_weight",
        "hmonet_physics_jacobian_enabled": "physics_jacobian_enabled",
        "hmonet_physics_jacobian_loss_weight": "physics_jacobian_loss_weight",
        "hmonet_random_mse_loss_weight": "random_mse_loss_weight",
        "hmonet_random_loss_sample_count": "random_loss_sample_count",
        "hmonet_random_loss_derivative_backend": "random_loss_derivative_backend",
        "hmonet_random_loss_sampling_mode": "random_loss_sampling_mode",
        "hmonet_random_loss_global_sample_count": "random_loss_global_sample_count",
        "hmonet_random_loss_curvature_sample_count": "random_loss_curvature_sample_count",
        "hmonet_random_loss_boundary_band_sample_count": "random_loss_boundary_band_sample_count",
        "hmonet_random_loss_boundary_band_min_step_multiple": "random_loss_boundary_band_min_step_multiple",
        "hmonet_random_loss_boundary_band_max_step_multiple": "random_loss_boundary_band_max_step_multiple",
        "hmonet_random_loss_boundary_band_min_distance": "random_loss_boundary_band_min_distance",
        "hmonet_random_loss_boundary_band_max_distance": "random_loss_boundary_band_max_distance",
        "hmonet_random_loss_fd_stencil_inside": "random_loss_fd_stencil_inside",
        "hmonet_random_loss_fd_boundary_scheme": "random_loss_fd_boundary_scheme",
        "hmonet_physics_jacobian_minimum": "physics_jacobian_minimum",
        "hmonet_physics_interior_sample_count": "physics_interior_sample_count",
        "hmonet_physics_winslow_loss_weight": "physics_winslow_loss_weight",
        "hmonet_physics_winslow_jacobian_barrier_weight": "physics_winslow_jacobian_barrier_weight",
        "hmonet_physics_winslow_jacobian_minimum": "physics_winslow_jacobian_minimum",
        "hmonet_physics_winslow_epsilon": "physics_winslow_epsilon",
        "hmonet_physics_scale_invariant_pde_enabled": "physics_scale_invariant_pde_enabled",
        "hmonet_physics_normalized_jacobian_loss_weight": "physics_normalized_jacobian_loss_weight",
        "hmonet_physics_normalized_jacobian_minimum": "physics_normalized_jacobian_minimum",
        "hmonet_physics_singular_value_loss_weight": "physics_singular_value_loss_weight",
        "hmonet_physics_singular_value_minimum": "physics_singular_value_minimum",
        "hmonet_physics_barrier_reduction": "physics_barrier_reduction",
    }
    integer_keys = {
        "lr_warmup_epochs",
        "lr_update_interval",
        "boundary_normal_sample_count",
        "fixed_grid_count",
        "global_random_count",
        "curvature_adaptive_count",
        "branch_dim",
        "trunk_dim",
        "coordinate_lift_max_power",
        "coordinate_lift_q",
        "adjust_hidden_dim",
        "adjust_hidden_layer_num",
        "branch_hidden_dim",
        "branch_hidden_layer_num",
        "trunk_hidden_dim",
        "trunk_hidden_layer_num",
        "output_dim",
        "branch_mlp_hidden_dim",
        "branch_mlp_hidden_layer_num",
        "dataset_sample_count",
        "pde_interior_sample_count",
        "tfi_warmup_epochs",
        "phase1_max_epochs",
        "phase2_max_epochs",
        "physics_interior_sample_count",
        "fixed_mse_sample_count",
        "random_loss_sample_count",
        "random_loss_global_sample_count",
        "random_loss_curvature_sample_count",
        "random_loss_boundary_band_sample_count",
    }
    boolean_keys = {
        "jacobian_loss_enabled",
        "boundary_normal_loss_enabled",
        "weight_head_mlp_enabled",
        "boundary_self_attention_enabled",
        "modified_mlp_enabled",
        "query_boundary_cross_attention_enabled",
        "boundary_layer_decoder_enabled",
        "decoupled_weight_decoder_enabled",
        "multiscale_boundary_encoder_enabled",
        "pde_loss_enabled",
        "physics_jacobian_enabled",
        "save_phase_best_checkpoints",
        "reset_optimizer_on_phase_transition",
        "physics_scale_invariant_pde_enabled",
        "random_loss_fd_stencil_inside",
    }
    string_keys = {
        "hmonet_mapping_mode",
        "coordinate_lift_mode",
        "jacobian_method",
        "mode",
        "target_interpolation",
        "point_loss_weighting",
        "training_strategy",
        "random_loss_derivative_backend",
        "random_loss_sampling_mode",
        "random_loss_fd_boundary_scheme",
        "physics_barrier_reduction",
        "best_model_metric",
        "plateau_metric",
    }
    training_keys = {"dataset_sample_count", "best_model_metric", "plateau_metric"}
    for raw_key, raw_value in options.items():
        key = aliases.get(raw_key, raw_key)
        if key == "pde_fd_step":
            value = [float(item) for item in raw_value]
        elif key in boolean_keys:
            value = bool(raw_value)
        elif key in integer_keys:
            value = int(raw_value)
        elif key in string_keys:
            value = str(raw_value)
        else:
            value = float(raw_value)
        if key in training_keys:
            training[key] = value
            training["dataset_subset_mode"] = "uniform_index"
        elif key in common_model_keys:
            common_model[key] = value
        elif key in model_keys:
            model[key] = value
        elif key in sampling_keys:
            sampling[key] = value
        else:
            htraining[key] = value
    resolved = prepare_training_config(config, "hmonet", SimpleNamespace())
    snapshot = model_specific_training_config(
        resolved,
        "hmonet",
        dataset,
        dataset_mode(dataset_name(dataset)),
    )
    payload = {
        "schema_version": TRAINING_IDENTITY_SCHEMA_VERSION,
        "effective_training_config": _normalize_training_snapshot(snapshot, dataset),
        "dataset_name": dataset_name(dataset),
        "dataset_sha256": _dataset_digest(dataset),
    }
    fingerprint_payload = _legacy_dataset_identity(payload)
    encoded = json.dumps(
        fingerprint_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), payload


_FILE_DIGEST_CACHE = {}


def _file_digest(path: Path) -> str:
    """Return a cached SHA-256 digest for one file used by a cache identity."""
    path = path.resolve()
    stat = path.stat()
    signature = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = _FILE_DIGEST_CACHE.get(signature)
    if cached:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    _FILE_DIGEST_CACHE[signature] = value
    return value


def _evaluation_config_digest(path: Path, case_name: str) -> str:
    """Keep migrated case fingerprints stable until relevant settings change."""
    config = _load_yaml_mapping(path)
    compatibility = config.pop("_fingerprint_compatibility", None)
    relevant = {
        "case": config.get("cases", {}).get(case_name),
        "ablation": config.get("ablation", {}),
        "evaluation": config.get("evaluation", {}),
    }
    canonical = hashlib.sha256(
        json.dumps(
            relevant, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    if isinstance(compatibility, dict):
        expected = compatibility.get("case_content_sha256", {}).get(case_name)
        if canonical == expected:
            legacy = compatibility.get("evaluate_config_sha256")
            if isinstance(legacy, str) and legacy:
                return legacy
    return canonical


def ablation_evaluation_settings(args) -> dict:
    """Resolve the small set of evaluation controls owned by experiments."""
    config = _load_yaml_mapping(Path(args.evaluate_config))
    values = config.get("ablation", {})
    configured_shapes = getattr(args, "evaluation_grid_shapes", None)
    if configured_shapes:
        shapes = evaluation_grid_shapes(args)
    else:
        raw_shapes = values.get("grid_shapes", [[51, 51]])
        proxy = SimpleNamespace(
            evaluation_grid_shapes=["x".join(map(str, shape)) for shape in raw_shapes]
        )
        shapes = evaluation_grid_shapes(proxy)
    methods = values.get("methods", ["tfi", "pde", "hmonet"])
    if isinstance(methods, str):
        methods = [item.strip() for item in methods.split(",") if item.strip()]
    return {
        "grid_shapes": shapes,
        "values": getattr(args, "evaluation_values", None),
        "methods": list(methods),
        "quality_metrics": values.get("quality_metrics", "both"),
        "hmonet_boundary_projection": values.get(
            "hmonet_boundary_projection", "none"
        ),
        "qghmonet_boundary_projection": values.get(
            "qghmonet_boundary_projection", "none"
        ),
        "meshonet_boundary_projection": values.get(
            "meshonet_boundary_projection", "none"
        ),
        "pimeshonet_boundary_projection": values.get(
            "pimeshonet_boundary_projection", "none"
        ),
    }


def evaluated_model(summary: dict) -> str:
    model = str(summary.get("model", "hmonet")).lower().replace("-", "")
    aliases = {
        "pimeshonet": "pimeshonet", "meshonet": "meshonet",
        "hmonet": "hmonet", "qghmonet": "qghmonet",
    }
    if model not in aliases:
        raise ValueError(f"Unsupported evaluated model: {summary.get('model')!r}")
    return aliases[model]


def evaluated_methods(settings: dict, summary: dict) -> list[str]:
    """Evaluate the traditional references plus the variant's trained model."""
    model = evaluated_model(summary)
    traditional = [name for name in settings["methods"] if name in {"tfi", "pde"}]
    return list(dict.fromkeys([*traditional, model]))


def _evaluation_source_digest() -> str:
    """Return the explicit numerical-evaluation revision.

    File-content hashes made import-only refactors invalidate every completed
    evaluation even when no numerical algorithm changed. This revision must
    be bumped whenever generation, quality metrics, or intersection semantics
    change; moving code between modules deliberately leaves it unchanged.
    """
    return hashlib.sha256(
        b"hmonet-evaluation:endpoint-contact-v2:no-cell-area:4"
    ).hexdigest()


# Digests written immediately before the package-only 2026-10 refactor. They
# are accepted only after every other identity field matches exactly.
_REFACTOR_COMPATIBLE_EVALUATION_DIGESTS = {
    "afb05603b43e1c34eb608c31b80672a63c03870e0787e4d1ce0935965d8d41e7",
    "db8cb49049a2c6fc03e96727d4da1574ca51146493b48062e84b699949b57887",
    "371663049dc8c2047c09b999946e2908d9d4763cc4d10ef73652260b7e0fd88f",
}


def _refactor_compatible_evaluation_identity(current: dict, saved: dict) -> bool:
    """Accept a legacy identity only for this semantics-preserving refactor."""
    if not isinstance(saved, dict):
        return False
    legacy_digest = saved.get("evaluation_source_sha256")
    if legacy_digest not in _REFACTOR_COMPATIBLE_EVALUATION_DIGESTS:
        return False
    current_without_source = dict(current)
    saved_without_source = dict(saved)
    current_without_source.pop("evaluation_source_sha256", None)
    saved_without_source.pop("evaluation_source_sha256", None)
    return current_without_source == saved_without_source


def effective_evaluation_identity(args, dataset: Path, summary: dict, training_fingerprint: str):
    """Build the exact evaluation identity used to reuse completed results."""
    checkpoint = resolve_project_path(summary["best_checkpoint"])
    evaluate_config = Path(args.evaluate_config).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint}")
    if not evaluate_config.is_file():
        raise FileNotFoundError(f"Evaluation config not found: {evaluate_config}")
    settings = ablation_evaluation_settings(args)
    payload = {
        "schema_version": 1,
        "training_fingerprint": training_fingerprint,
        "checkpoint_sha256": _file_digest(checkpoint),
        "evaluate_config_sha256": _evaluation_config_digest(
            evaluate_config, dataset_name(dataset)
        ),
        "evaluation_source_sha256": _evaluation_source_digest(),
        "dataset": dataset_name(dataset),
        "grid_shapes": [list(shape) for shape in settings["grid_shapes"]],
        "model": evaluated_model(summary),
        "methods": evaluated_methods(settings, summary),
        "hmonet_mapping_mode": summary.get(
            "hmonet_mapping_mode", "rational_affine"
        ),
        "hmonet_boundary_projection": settings["hmonet_boundary_projection"],
        "quality_metrics": settings["quality_metrics"],
        "device": args.device,
    }
    if summary.get("evaluation_checkpoint_stage"):
        payload["checkpoint_stage"] = summary["evaluation_checkpoint_stage"]
    # Preserve fingerprints created before selective parameter evaluation was
    # introduced. A subset receives its own identity; the default full case
    # remains reusable without rewriting completed experiment records.
    if settings["values"] is not None:
        payload["values"] = settings["values"]
    encoded = json.dumps(
        _legacy_dataset_identity(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), payload


def evaluation_checkpoint_stages(summary: dict) -> list[tuple[str | None, dict]]:
    """Return available phase optima and the whole-run optimum."""
    phase1 = summary.get("phase1_best_checkpoint")
    phase2 = summary.get("phase2_best_checkpoint")
    global_best = summary.get("global_best_checkpoint") or summary.get(
        "best_checkpoint"
    )
    if phase1 and global_best and summary.get("phase_best_checkpoints_required"):
        stages = []
        checkpoints = [("phase1", phase1)]
        if phase2:
            checkpoints.append(("phase2", phase2))
        checkpoints.append(("global", global_best))
        for label, checkpoint in checkpoints:
            staged = copy.deepcopy(summary)
            staged["best_checkpoint"] = checkpoint
            staged["best_epoch"] = summary.get(f"{label}_best_epoch")
            staged["best_metric_value"] = summary.get(
                f"{label}_best_metric_value"
            )
            staged["best_model_dataset_mse"] = summary.get(
                f"{label}_best_model_dataset_mse"
            )
            staged["evaluation_checkpoint_stage"] = label
            stages.append((label, staged))
        return stages
    return [(None, summary)]


def tag_evaluation_results(result_files: list[Path], stage: str | None) -> None:
    """Persist the evaluated training stage so aggregate rows remain distinct."""
    if stage is None:
        return
    for result_file in result_files:
        result = json.loads(Path(result_file).read_text(encoding="utf-8"))
        result["experiment_checkpoint_stage"] = stage
        write_json(Path(result_file), result)


def grid_shape_name(shape) -> str:
    """Return the canonical directory key for one evaluation grid."""
    if isinstance(shape, str):
        return shape.lower().replace(" ", "")
    return "x".join(map(str, shape))


def single_grid_evaluation_args(args, shape):
    """Copy parsed arguments and restrict evaluation to exactly one grid."""
    proxy = copy.copy(args)
    proxy.evaluation_grid_shapes = [grid_shape_name(shape)]
    return proxy


def manifest_evaluations(manifest: dict) -> dict:
    """Read per-grid evaluation entries, including legacy one-entry manifests."""
    evaluations = manifest.get("evaluations")
    if isinstance(evaluations, dict):
        return copy.deepcopy(evaluations)
    result_files = [
        portable_project_path(item)
        for item in manifest.get("evaluation_result_files", [])
        if resolve_project_path(item).is_file()
    ]
    if not result_files:
        return {}
    grouped = {}
    for item in result_files:
        result_path = resolve_project_path(item)
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            grid = grid_shape_name(result.get("grid_shape", result_path.parent.name))
        except (OSError, json.JSONDecodeError):
            grid = grid_shape_name(result_path.parent.name)
        grouped.setdefault(grid, []).append(item)
    identity_file = manifest.get("evaluation_identity_file")
    fingerprint = manifest.get("evaluation_fingerprint")
    return {
        grid: {
            "status": "completed" if manifest.get("status") == "completed" else manifest.get("status"),
            "fingerprint": fingerprint,
            "identity_file": identity_file,
            "result_files": files,
            "reused": manifest.get("evaluation_reused"),
        }
        for grid, files in grouped.items()
    }


def _training_summary_path(manifest_path: Path, manifest: dict) -> Path:
    configured = manifest.get("training_summary")
    return resolve_project_path(configured) if configured else manifest_path.parent / "training/training_summary.json"


def _completed_training_summary(summary: dict) -> bool:
    """Accept only a converged/full training state; partial runs are retrained."""
    if not isinstance(summary.get("completed_epochs"), int) or summary["completed_epochs"] <= 0:
        return False
    if not summary.get("stop_reason"):
        return False
    for key in ("best_checkpoint", "final_checkpoint"):
        value = summary.get(key)
        if not value or not resolve_project_path(value).is_file():
            return False
    if summary.get("phase_best_checkpoints_required"):
        for key in ("phase1_best_checkpoint",):
            value = summary.get(key)
            if not value or not resolve_project_path(value).is_file():
                return False
    return True


def _write_resolved_training_config(path: Path, identity: dict) -> None:
    """Persist the resolved training snapshot when the identity owns one."""
    resolved = identity.get("effective_training_config")
    if not isinstance(resolved, dict):
        return
    path.write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def load_training_from_directory(directory: Path):
    """Load one explicitly selected completed training without re-hashing config."""
    manifest_path = directory / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary_path = _training_summary_path(manifest_path, manifest)
        if not summary_path.is_file():
            return None
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not _completed_training_summary(summary):
            return None
    except (OSError, json.JSONDecodeError):
        return None
    return {
        "manifest_path": manifest_path,
        "directory": directory,
        "summary_path": summary_path,
        "summary": summary,
    }


def find_reusable_training(experiment_root: Path, fingerprint: str):
    """Find a completed training with the same fingerprint across all run tags."""
    if not experiment_root.is_dir():
        return None
    candidates = []
    for manifest_path in experiment_root.rglob("run_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("training_fingerprint") != fingerprint:
            continue
        summary_path = _training_summary_path(manifest_path, manifest)
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if _completed_training_summary(summary):
            # Reused-run manifests may point to a summary owned by another run.
            # Always protect and return the directory that physically owns the
            # training summary instead of the alias manifest directory.
            training_directory = summary_path.parent.parent.resolve()
            canonical_manifest = manifest_path.parent.resolve() == training_directory
            candidates.append(
                (
                    canonical_manifest,
                    manifest_path.stat().st_mtime_ns,
                    manifest_path,
                    training_directory,
                    summary_path,
                    summary,
                )
            )
    if not candidates:
        return None
    _, _, manifest_path, training_directory, summary_path, summary = max(
        candidates, key=lambda item: (item[0], item[1])
    )
    return {
        "manifest_path": manifest_path,
        "directory": training_directory,
        "summary_path": summary_path,
        "summary": summary,
    }


def _flatten_identity(value, prefix="") -> dict[str, object]:
    """Flatten a nested identity so configuration differences are readable."""
    if isinstance(value, dict):
        flattened = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_identity(value[key], child))
        return flattened
    return {prefix: value}


def _identity_differences(expected: dict, actual: dict) -> list[dict]:
    """Return every differing baseline identity field."""
    expected_flat = _flatten_identity(expected)
    actual_flat = _flatten_identity(actual)
    differences = []
    for key in sorted(set(expected_flat) | set(actual_flat)):
        expected_value = expected_flat.get(key, "<missing>")
        actual_value = actual_flat.get(key, "<missing>")
        if expected_value != actual_value:
            differences.append(
                {
                    "field": key,
                    "expected_default": expected_value,
                    "trained_baseline": actual_value,
                }
            )
    return differences


def check_trained_baseline_identity(
    experiment_root: Path,
    dataset: str,
    seed: int,
    expected_fingerprint: str,
    expected_identity: dict,
):
    """Check canonical baseline runs without writing anything below baseline/."""
    seed_root = (
        experiment_root.resolve()
        / BASELINE_ABLATION
        / dataset
        / f"seed_{int(seed)}"
    )
    candidates = []
    if seed_root.is_dir():
        for directory in seed_root.iterdir():
            if not directory.is_dir():
                continue
            loaded = load_training_from_directory(directory)
            if loaded is None:
                continue
            manifest_path = loaded["manifest_path"]
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            fingerprint = manifest.get("training_fingerprint")
            if not fingerprint:
                continue
            configured_identity = manifest.get("training_identity_file")
            if configured_identity is None:
                identity_path = directory / "training_identity.json"
            else:
                identity_path = resolve_project_path(configured_identity)
                if not identity_path.is_file() and not Path(configured_identity).is_absolute():
                    identity_path = (directory / configured_identity).resolve()
            try:
                identity = json.loads(identity_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                identity = {}
            candidates.append(
                {
                    "directory": directory.resolve(),
                    "fingerprint": fingerprint,
                    "identity": identity,
                    "modified_time_ns": manifest_path.stat().st_mtime_ns,
                }
            )
    matching = [
        item for item in candidates
        if item["fingerprint"] == expected_fingerprint
    ]
    if matching:
        return None
    if not candidates:
        return {
            "dataset": dataset,
            "seed": int(seed),
            "status": "missing_baseline",
            "expected_fingerprint": expected_fingerprint,
            "message": "No complete canonical baseline training was found.",
            "differences": [],
        }
    trained = max(candidates, key=lambda item: item["modified_time_ns"])
    return {
        "dataset": dataset,
        "seed": int(seed),
        "status": "baseline_parameter_mismatch",
        "expected_fingerprint": expected_fingerprint,
        "trained_baseline_fingerprint": trained["fingerprint"],
        "trained_baseline_directory": str(trained["directory"]),
        "message": "The current default training configuration differs from the trained baseline.",
        "differences": _identity_differences(expected_identity, trained["identity"]),
    }


def _baseline_error_message(issues: list[dict]) -> str:
    lines = [
        "Baseline compatibility check failed after training and per-model evaluation.",
        "Baseline evaluation/comparison and aggregate evaluation were not executed.",
        "No file in the selected experiment set baseline was modified.",
    ]
    for issue in issues:
        lines.append(
            f"- {issue['dataset']} seed={issue['seed']}: {issue['status']}"
        )
        for difference in issue.get("differences", []):
            lines.append(
                "    "
                f"{difference['field']}: default={difference['expected_default']!r}, "
                f"baseline={difference['trained_baseline']!r}"
            )
    return "\n".join(lines)


def train_one(args, dataset: Path, variant: Variant, seed: int, destination: Path):
    from scripts.train import train_2d

    nx, ny = dataset_grid(dataset)
    name = dataset_name(dataset)
    command = [
        "--config",
        str(_training_config_path(args)),
        "--dataset",
        portable_project_path(dataset),
        "--dataset-mode",
        dataset_mode(name),
        "--model",
        variant_model(variant),
        "--nx",
        str(nx),
        "--ny",
        str(ny),
        "--seed",
        str(seed),
        "--flat-run-root",
        str(destination),
    ]
    if args.device:
        command += ["--device", args.device]
    command += variant_training_overrides(variant)
    summaries = train_2d.main(command)
    if len(summaries) != 1:
        raise RuntimeError(f"Expected one training summary, received {len(summaries)}")
    return summaries[0], command


def evaluate_one(args, dataset: Path, summary: dict, evaluation_root: Path):
    from scripts.evaluate import evaluate_2d

    name = dataset_name(dataset)
    checkpoint = resolve_project_path(summary["best_checkpoint"])
    train_shape = dataset_grid(dataset)
    settings = ablation_evaluation_settings(args)
    evaluation_shapes = settings["grid_shapes"]
    grid_argument = ",".join("x".join(map(str, shape)) for shape in evaluation_shapes)
    model = evaluated_model(summary)
    checkpoint_flags = {
        "hmonet": "--hmonet-checkpoint",
        "qghmonet": "--qghmonet-checkpoint",
        "meshonet": "--meshonet-checkpoint",
        "pimeshonet": "--pimeshonet-checkpoint",
    }
    projection = settings[f"{model}_boundary_projection"]
    command = [
        "--config",
        str(args.evaluate_config),
        "--case",
        name,
        "--grid-shapes",
        grid_argument,
        "--methods",
        ",".join(evaluated_methods(settings, summary)),
        checkpoint_flags[model],
        str(checkpoint),
        f"--{model}-boundary-projection",
        projection,
        "--output-root",
        str(evaluation_root),
        "--quality-metrics",
        str(settings["quality_metrics"]),
        "--force",
    ]
    if model in {"hmonet", "qghmonet"}:
        command += [
            "--hmonet-mapping-mode",
            summary.get("hmonet_mapping_mode", "rational_affine"),
        ]
    if args.device:
        command += ["--device", args.device]
    command += [
        "--save-plots" if getattr(args, "evaluation_plots", True) else "--no-plots"
    ]
    if settings["values"]:
        command += ["--values", ",".join(map(str, settings["values"]))]
    evaluate_2d.main(command)
    return command


def compact_evaluation_result_directory(directory: Path) -> Path:
    """Keep the reusable summary and publish images under short readable paths."""
    directory = Path(directory)
    result_file = directory / "results.json"
    if not result_file.is_file():
        raise FileNotFoundError(
            f"Cannot compact evaluation without its summary: {result_file}"
        )
    image_files = [
        child
        for child in directory.rglob("*")
        if child.is_file() and child.suffix.lower() in EVALUATION_IMAGE_SUFFIXES
    ]
    for child in image_files:
        relative = child.relative_to(directory)
        if relative.parts and relative.parts[0] == "p":
            continue
        source_relative = (
            Path(*relative.parts[1:])
            if relative.parts and relative.parts[0] == "plots"
            else relative
        )
        destination = _compact_evaluation_image_path(directory, source_relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            stem, suffix = destination.stem, destination.suffix
            index = 2
            while destination.exists():
                destination = destination.with_name(f"{stem}_{index}{suffix}")
                index += 1
        shutil.move(str(child), str(destination))

    for child in list(directory.rglob("*")):
        if not child.is_file():
            continue
        if child == result_file or child.suffix.lower() in EVALUATION_IMAGE_SUFFIXES:
            continue
        child.unlink()
    for child in sorted(
        (item for item in directory.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        try:
            child.rmdir()
        except OSError:
            pass
    return result_file


def migrate_evaluation_image_paths(directory: Path) -> int:
    """Move only published images into the compact layout, preserving all data.

    This migration intentionally leaves non-image artifacts and empty legacy
    directories untouched.  It is therefore safe for previously completed
    evaluations whose CSV/NPZ details may still be useful to the user.
    """
    directory = Path(directory)
    moved = 0
    for child in list(directory.rglob("*")):
        if not child.is_file() or child.suffix.lower() not in EVALUATION_IMAGE_SUFFIXES:
            continue
        relative = child.relative_to(directory)
        if relative.parts and relative.parts[0] == "p":
            continue
        source_relative = (
            Path(*relative.parts[1:])
            if relative.parts and relative.parts[0] == "plots"
            else relative
        )
        destination = _compact_evaluation_image_path(directory, source_relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            stem, suffix = destination.stem, destination.suffix
            index = 2
            while destination.exists():
                destination = destination.with_name(f"{stem}_{index}{suffix}")
                index += 1
        shutil.move(str(child), str(destination))
        moved += 1
    return moved


def _compact_evaluation_image_path(root: Path, relative: Path) -> Path:
    """Map evaluator-internal folders to the public compact plot hierarchy."""
    parts = list(relative.parts)
    filename = _short_plot_filename(parts[-1])
    lowered = [part.lower() for part in parts]
    if len(parts) == 1:
        return root / "p" / filename
    if parts[0].lower() == "grid_quality":
        return root / "p" / "q" / filename
    if parts[0].lower() in {"comparisons", "comparison"}:
        category = ["p", "c", _comparison_method_group(parts[1:-1])]
        if "grid_quality" in lowered or "quality" in lowered:
            category.append("q")
        return root / Path(*category) / filename

    method_aliases = {
        "pde_winslow": "pde",
        "tfi": "tfi",
        "hmonet": "hm",
        "qg-hmonet": "qg",
        "meshonet": "mo",
        "pi-meshonet": "pi",
    }
    method = method_aliases.get(parts[0].lower(), _safe_plot_component(parts[0]))
    category = ["p", method]
    if "raw" in lowered:
        category.append("r")
    elif method in {"hm", "qg", "mo", "pi"}:
        category.append("p")
    if "grid_quality" in lowered or "quality" in lowered:
        category.append("q")
    return root / Path(*category) / filename


def _comparison_method_group(parts) -> str:
    """Extract a readable method combination from evaluator-internal folders."""
    aliases = {
        "tfi": "t",
        "pde": "p",
        "meshonet": "m",
        "pimeshonet": "pi",
        "hmonet": "h",
        "qghmonet": "qg",
        "pihmonet": "pih",
        "attention": "att",
    }
    ignored = {
        "grid",
        "quality",
        "none",
        "raw",
        "direct",
        "idw",
        "rational",
        "affine",
        "corner",
        "corrected",
        "coons",
        "residual",
    }
    for part in reversed(tuple(parts)):
        tokens = [token for token in re.split(r"[_+]+", part.lower()) if token]
        methods = []
        for token in tokens:
            if token.isdigit() or token in ignored:
                continue
            method = aliases.get(token)
            if method and method not in methods:
                methods.append(method)
        if len(methods) >= 2:
            return "_".join(methods)
    return "methods"


def _short_plot_filename(filename: str) -> str:
    """Shorten published image names while retaining their metric identity."""
    stem, suffix = Path(filename).stem, Path(filename).suffix
    replacements = (
        ("comparison_quality_", "q_"),
        ("quality_", "q_"),
        ("self_intersection", "sx"),
        ("affected_cell_fraction", "acf"),
        ("flipping_cell", "flip"),
        ("invalid_cell", "inv"),
        ("scaled_jacobian", "sj"),
        ("jacobian", "jac"),
        ("singular_value", "sv"),
        ("aspect_ratio", "ar"),
        ("maximum_angle", "maxang"),
        ("minimum_angle", "minang"),
        ("orthogonality", "ortho"),
        ("distribution", "dist"),
    )
    for old, new in replacements:
        stem = stem.replace(old, new)
    return f"{stem}{suffix}"


def _safe_plot_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(value)).strip("._")
    return value or "misc"


def publish_evaluation_result(
    source: Path,
    target: Path,
    *,
    overwrite: bool,
    remove_source: bool = True,
) -> Path:
    """Publish one evaluator case without recreating its deep internal tree."""
    source = Path(source)
    target = Path(target)
    source_result = source / "results.json"
    if not source_result.is_file():
        raise FileNotFoundError(f"Missing evaluation summary: {source_result}")
    if target.exists():
        if not overwrite:
            raise FileExistsError(
                "Evaluation output already exists and is protected: "
                f"{target}. Use --force to replace it or choose another --run-tag."
            )
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_result, target / "results.json")
    result_metadata = json.loads(source_result.read_text(encoding="utf-8"))
    comparison_group = _comparison_group_from_methods(
        result_metadata.get("method_order", result_metadata.get("requested_methods", []))
    )
    for image in source.rglob("*"):
        if not image.is_file() or image.suffix.lower() not in EVALUATION_IMAGE_SUFFIXES:
            continue
        relative = image.relative_to(source)
        if relative.parts and relative.parts[0] == "p":
            parts = list(relative.parts)
            parts[-1] = _short_plot_filename(parts[-1])
            destination = target / Path(*parts)
        elif relative.parts and relative.parts[0] == "plots":
            destination = _compact_evaluation_image_path(
                target, Path(*relative.parts[1:])
            )
        else:
            destination = _compact_evaluation_image_path(target, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            stem, suffix = destination.stem, destination.suffix
            index = 2
            while destination.exists():
                destination = destination.with_name(f"{stem}_{index}{suffix}")
                index += 1
        shutil.copy2(image, destination)
    if remove_source:
        shutil.rmtree(source)
    return target / "results.json"


def _comparison_group_from_methods(methods) -> str:
    aliases = {
        "TFI": "t",
        "PDE_Winslow": "p",
        "PDE": "p",
        "MeshONet": "m",
        "PI-MeshONet": "pi",
        "HMONet": "h",
        "QG-HMONet": "qg",
    }
    names = [aliases.get(str(method), str(method)) for method in methods]
    return "_".join(_safe_plot_component(name) for name in names) or "methods"


def materialize_reused_evaluation_results(
    result_files: list[Path],
    dataset_root: Path,
    variant: str,
    seed: int,
    run_tag: str,
) -> list[Path]:
    """Copy a fingerprint-matched legacy result into the current short layout."""
    materialized = []
    for result_file in result_files:
        result_file = Path(result_file)
        result = json.loads(result_file.read_text(encoding="utf-8"))
        grid_shape = result.get("grid_shape", [])
        grid = (
            "x".join(map(str, grid_shape))
            if grid_shape
            else result_file.parent.parent.name
        )
        target = evaluation_result_root(
            dataset_root,
            variant,
            seed,
            run_tag,
            grid,
            result.get("value", result_file.parent.name),
        )
        target_result = target / "results.json"
        if result_file.resolve() == target_result.resolve():
            materialized.append(result_file)
            continue
        if target_result.is_file():
            if target_result.read_bytes() != result_file.read_bytes():
                raise FileExistsError(
                    "A different evaluation summary already occupies the current "
                    f"short result path: {target_result}. Use --force to replace it."
                )
            materialized.append(target_result)
            continue
        materialized.append(
            publish_evaluation_result(
                result_file.parent,
                target,
                overwrite=False,
                remove_source=False,
            )
        )
    return sorted(materialized)


def evaluation_directory(experiment_root, dataset, variant, seed, run_tag, grid=None):
    """Return the compact, shared staging root for one evaluation grid."""
    return evaluation_staging_root(
        Path(experiment_root), dataset, variant, seed, run_tag, grid
    )


def organize_evaluation_results(
    staging_root: Path,
    dataset_root: Path,
    variant: str,
    seed: int,
    run_tag: str,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Move evaluator output to a compact path without replacing valid results."""
    result_files = sorted(staging_root.rglob("results.json"))
    organized = []
    for result_file in result_files:
        result = json.loads(result_file.read_text(encoding="utf-8"))
        parameter_value = result.get("value", result_file.parent.name)
        grid_shape = result.get("grid_shape", [])
        grid_name = (
            "x".join(map(str, grid_shape)) if grid_shape else result_file.parent.parent.name
        )
        target = evaluation_result_root(
            dataset_root,
            variant,
            seed,
            run_tag,
            grid_name,
            parameter_value,
        )
        published = publish_evaluation_result(
            result_file.parent, target, overwrite=overwrite
        )
        validate_evaluation_destination(target, target)
        organized.append(published)
    shutil.rmtree(staging_root, ignore_errors=True)
    staging_parent = staging_root.parent
    if staging_parent.exists() and not any(staging_parent.iterdir()):
        staging_parent.rmdir()
    return organized


def stored_grid_results(
    dataset_root: Path,
    variant: str,
    seed: int,
    run_tag: str,
    grid: str,
) -> list[Path]:
    """Return all parameter results already stored for one evaluation grid."""
    probe = evaluation_result_root(
        dataset_root, variant, seed, run_tag, grid, "_parameter_"
    )
    grid_root = probe.parent
    return sorted(grid_root.glob("*/results.json")) if grid_root.is_dir() else []


def read_results(destination: Path, grid: str | None = None):
    """Read explicitly registered evaluation results, optionally for one grid."""
    manifest_path = destination / "run_manifest.json"
    if not manifest_path.is_file():
        root = destination / "evaluation"
        results = sorted(root.rglob("results.json")) if root.exists() else []
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest_evaluations(manifest)
        results = []
        for grid_name, entry in entries.items():
            results.extend(
                resolve_project_path(item)
                for item in entry.get("result_files", [])
                if resolve_project_path(item).is_file()
            )
        if not results:
            root = resolve_project_path(manifest.get("evaluation_directory", destination / "evaluation"))
            results = sorted(root.rglob("results.json")) if root.exists() else []
    if grid is None:
        return sorted(set(results))
    selected = []
    for result_path in results:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result_grid = grid_shape_name(result.get("grid_shape", result_path.parent.name))
        except (OSError, json.JSONDecodeError):
            result_grid = grid_shape_name(result_path.parent.name)
        if result_grid == grid_shape_name(grid):
            selected.append(result_path)
    return sorted(set(selected))

def find_reusable_evaluation(
    experiment_root: Path, fingerprint: str, identity: dict | None = None
):
    """Find a completed evaluation by exact or refactor-compatible identity."""
    if not experiment_root.is_dir():
        return None
    candidates = []
    for manifest_path in experiment_root.rglob("run_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for grid, entry in manifest_evaluations(manifest).items():
            exact_match = entry.get("fingerprint") == fingerprint
            compatible_match = False
            identity_file = entry.get("identity_file")
            if not exact_match and identity is not None and identity_file:
                try:
                    saved_identity = json.loads(
                        resolve_project_path(identity_file).read_text(encoding="utf-8")
                    )
                    compatible_match = _refactor_compatible_evaluation_identity(
                        identity, saved_identity
                    )
                except (OSError, json.JSONDecodeError):
                    compatible_match = False
            if not exact_match and not compatible_match:
                continue
            if entry.get("status") != "completed":
                continue
            result_files = [
                resolve_project_path(item)
                for item in entry.get("result_files", [])
                if resolve_project_path(item).is_file()
            ]
            if not result_files:
                continue
            candidates.append(
                (manifest_path.stat().st_mtime_ns, manifest_path, grid, sorted(result_files))
            )
        # Keep direct compatibility with minimal legacy test/manifests whose
        # result files do not contain enough metadata for manifest_evaluations.
        if manifest.get("evaluation_fingerprint") == fingerprint:
            result_files = [
                resolve_project_path(item)
                for item in manifest.get("evaluation_result_files", [])
                if resolve_project_path(item).is_file()
            ]
            if manifest.get("status") == "completed" and result_files:
                candidates.append(
                    (manifest_path.stat().st_mtime_ns, manifest_path, None, sorted(result_files))
                )
    if not candidates:
        return None
    _, manifest_path, grid, result_files = max(candidates, key=lambda item: item[0])
    return {
        "manifest_path": manifest_path,
        "directory": manifest_path.parent,
        "grid_shape": grid,
        "result_files": result_files,
    }

def _numeric_items(prefix, mapping):
    for key, value in mapping.items():
        name = f"{prefix}__{key}" if prefix else key
        if isinstance(value, dict):
            yield from _numeric_items(name, value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield name, float(value)


def result_rows(manifest: dict, summary: dict, result_path: Path):
    result = json.loads(result_path.read_text(encoding="utf-8"))
    method = {
        "hmonet": "HMONet",
        "qghmonet": "QG-HMONet",
        "meshonet": "MeshONet",
        "pimeshonet": "PI-MeshONet",
    }[evaluated_model(summary)]
    checkpoint_stage = result.get("experiment_checkpoint_stage")
    variant = manifest["variant"]
    variant_value = manifest["variant_value"]
    if checkpoint_stage:
        variant = f"{variant}__{checkpoint_stage}"
        variant_value = f"{variant_value} [{checkpoint_stage}]"
    common = {
        "ablation": manifest["ablation"],
        "dataset": manifest["dataset"],
        "variant": variant,
        "variant_value": variant_value,
        "checkpoint_stage": checkpoint_stage or "best",
        "seed": manifest["seed"],
        "run_tag": manifest["run_tag"],
        "method": method,
        "parameter": result.get("parameter"),
        "value": result.get("value"),
        "grid_shape": "x".join(map(str, result.get("grid_shape", []))),
        "best_epoch": (
            summary.get(f"{checkpoint_stage}_best_epoch")
            if checkpoint_stage
            else summary.get("best_epoch")
        ),
        "completed_epochs": summary.get("completed_epochs"),
        "training_seconds": summary.get("training_time_seconds"),
        "average_epoch_seconds": (
            summary.get("training_time_seconds") / summary.get("completed_epochs")
            if isinstance(summary.get("training_time_seconds"), (int, float))
            and isinstance(summary.get("completed_epochs"), (int, float))
            and summary.get("completed_epochs") > 0
            else None
        ),
        "fixed_grid_best_mse": (
            summary.get(f"{checkpoint_stage}_best_model_dataset_mse")
            if checkpoint_stage
            else summary.get("best_model_dataset_mse")
        ),
        "fixed_grid_final_mse": summary.get("final_dataset_mse"),
    }
    metrics = result.get("metrics", {}).get(method, {})
    for metric, statistics in metrics.items():
        if isinstance(statistics, dict):
            for key, number in _numeric_items(metric, statistics):
                common[key] = number
    invalid = metrics.get("invalid_cells", {})
    common["invalid_cell_count"] = invalid.get("count")
    common["invalid_cell_fraction"] = invalid.get("fraction")
    timing = result.get("timings", {}).get(method, {})
    for key in (
        "network_inference_seconds",
        "boundary_projection_seconds",
        "total_model_seconds",
    ):
        common[key] = timing.get(key)
    raw = next(
        (
            item
            for item in result.get("boundary_errors", [])
            if item.get("method") == method and item.get("stage") == "raw"
        ),
        {},
    )
    for key in (
        "boundary_max_error",
        "boundary_mean_error",
        "boundary_rms_error",
        "boundary_mse",
    ):
        common[key] = raw.get(key)
    common["result_file"] = str(result_path)
    return common


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _history_series(summary: dict, metric: str):
    """Return epoch/value arrays for a saved training-history metric."""
    if metric == "training_loss":
        values = summary.get("training_loss_history") or []
        epochs = list(range(1, len(values) + 1))
    elif metric == "dataset_mse":
        values = summary.get("dataset_mse_history") or []
        epochs = summary.get("dataset_mse_epochs") or []
    else:
        raise ValueError(f"Unsupported history metric: {metric}")
    if len(epochs) != len(values):
        raise ValueError(
            f"Mismatched {metric} epochs and values in training summary"
        )
    pairs = [
        (int(epoch), float(value))
        for epoch, value in zip(epochs, values)
        if np.isfinite(value)
    ]
    return dict(pairs)


def _aligned_history_statistics(series: list[dict[int, float]]):
    """Align seeds on epochs shared by every run and compute mean/std."""
    if not series:
        return np.asarray([]), np.asarray([]), np.asarray([])
    common_epochs = sorted(set.intersection(*(set(item) for item in series)))
    if not common_epochs:
        return np.asarray([]), np.asarray([]), np.asarray([])
    values = np.asarray(
        [[item[epoch] for epoch in common_epochs] for item in series], dtype=float
    )
    return (
        np.asarray(common_epochs, dtype=int),
        values.mean(axis=0),
        values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(len(common_epochs)),
    )


def _plot_parameter_training_relationships(
    dataset: str,
    dataset_runs: list[tuple[dict, dict]],
    variants: list[str],
    output: Path,
):
    """Plot ablation parameter against mean epoch time and completed epochs."""
    import matplotlib.pyplot as plt

    rows = []
    for variant in variants:
        members = [item for item in dataset_runs if item[0]["variant"] == variant]
        epoch_times = []
        completed = []
        total_times = []
        parameter = members[0][0].get("variant_value") if members else None
        for _, summary in members:
            epochs = summary.get("completed_epochs")
            seconds = summary.get("training_time_seconds")
            if not isinstance(epochs, (int, float)) or epochs <= 0:
                continue
            if not isinstance(seconds, (int, float)) or not np.isfinite(seconds):
                continue
            completed.append(float(epochs))
            total_times.append(float(seconds))
            epoch_times.append(float(seconds) / float(epochs))
        if not epoch_times:
            continue
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "parameter_value": parameter,
                "seed_count": len(epoch_times),
                "average_epoch_seconds_mean": float(np.mean(epoch_times)),
                "average_epoch_seconds_std": (
                    float(np.std(epoch_times, ddof=1)) if len(epoch_times) > 1 else 0.0
                ),
                "completed_epochs_mean": float(np.mean(completed)),
                "completed_epochs_std": (
                    float(np.std(completed, ddof=1)) if len(completed) > 1 else 0.0
                ),
                "total_training_seconds_mean": float(np.mean(total_times)),
                "total_training_seconds_std": (
                    float(np.std(total_times, ddof=1)) if len(total_times) > 1 else 0.0
                ),
            }
        )
    if not rows:
        return
    numeric_parameters = all(
        isinstance(row["parameter_value"], (int, float))
        and not isinstance(row["parameter_value"], bool)
        for row in rows
    )
    if numeric_parameters:
        rows.sort(key=lambda row: float(row["parameter_value"]))
        x = np.asarray([float(row["parameter_value"]) for row in rows])
        labels = None
        xlabel = "Ablation parameter"
    else:
        x = np.arange(len(rows))
        labels = [str(row["parameter_value"]) for row in rows]
        xlabel = "Ablation parameter"
    write_csv(output / "parameter_training_statistics.csv", rows)
    for mean_key, std_key, ylabel, filename in (
        (
            "average_epoch_seconds_mean",
            "average_epoch_seconds_std",
            "Average time per epoch (s)",
            "parameter_vs_iteration_time.png",
        ),
        (
            "total_training_seconds_mean",
            "total_training_seconds_std",
            "Total training time (s)",
            "parameter_vs_total_training_time.png",
        ),
        (
            "completed_epochs_mean",
            "completed_epochs_std",
            "Completed epochs",
            "parameter_vs_iteration_count.png",
        ),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.errorbar(
            x,
            [row[mean_key] for row in rows],
            yerr=[row[std_key] for row in rows],
            marker="o",
            linewidth=1.5,
            capsize=4,
        )
        if labels is not None:
            axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(f"{dataset}: {ylabel} vs ablation parameter")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=200)
        plt.close(figure)


def plot_training_histories(experiment_root: Path, run_tag: str):
    """Plot all ablation variants together for training loss and fixed-grid MSE."""
    import matplotlib.pyplot as plt

    runs = []
    for manifest_path in experiment_root.rglob("run_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_tag") != run_tag:
            continue
        summary_path = _training_summary_path(manifest_path, manifest)
        if not summary_path.is_file():
            continue
        runs.append(
            (
                manifest,
                json.loads(summary_path.read_text(encoding="utf-8")),
            )
        )
    datasets = list(dict.fromkeys(item[0]["dataset"] for item in runs))
    for dataset in datasets:
        dataset_runs = [item for item in runs if item[0]["dataset"] == dataset]
        variants = list(dict.fromkeys(item[0]["variant"] for item in dataset_runs))
        output = experiment_root / dataset / "figures" / run_tag / "training"
        output.mkdir(parents=True, exist_ok=True)
        for metric, ylabel, filename in (
            ("training_loss", "Training loss", "training_loss_comparison.png"),
            ("dataset_mse", "Fixed-grid MSE", "dataset_mse_comparison.png"),
        ):
            figure, axis = plt.subplots(figsize=(9, 5.5))
            curve_rows = []
            plotted = False
            positive = True
            for variant in variants:
                members = [item for item in dataset_runs if item[0]["variant"] == variant]
                series = [
                    history
                    for _, summary in members
                    if (history := _history_series(summary, metric))
                ]
                epochs, mean, std = _aligned_history_statistics(series)
                if not len(epochs):
                    continue
                plotted = True
                positive = positive and bool(np.all(mean > 0))
                line = axis.plot(epochs, mean, linewidth=1.5, label=variant)[0]
                if len(series) > 1:
                    lower = np.maximum(mean - std, np.finfo(float).tiny)
                    axis.fill_between(
                        epochs,
                        lower,
                        mean + std,
                        color=line.get_color(),
                        alpha=0.18,
                        linewidth=0,
                    )
                for epoch, mean_value, std_value in zip(epochs, mean, std):
                    curve_rows.append(
                        {
                            "dataset": dataset,
                            "variant": variant,
                            "metric": metric,
                            "epoch": int(epoch),
                            "seed_count": len(series),
                            "mean": float(mean_value),
                            "std": float(std_value),
                        }
                    )
            if not plotted:
                plt.close(figure)
                continue
            if positive:
                axis.set_yscale("log")
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.set_title(f"{dataset}: {ylabel} comparison")
            axis.grid(True, which="both", alpha=0.3)
            axis.legend()
            figure.tight_layout()
            figure.savefig(output / filename, dpi=200)
            plt.close(figure)
            write_csv(output / filename.replace(".png", ".csv"), curve_rows)
        _plot_parameter_training_relationships(
            dataset, dataset_runs, variants, output
        )


def _safe_component(value) -> str:
    """Return a path-safe label for a metric, grid, or parameter value."""
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _metric_comparison_plot(rows, metric, output, title):
    """Compare ablation variants for one metric, grid shape and geometry value."""
    import matplotlib.pyplot as plt

    variants = list(dict.fromkeys(row["variant"] for row in rows))
    records = []
    for variant in variants:
        values = np.asarray(
            [
                row[metric]
                for row in rows
                if row["variant"] == variant
                and isinstance(row.get(metric), (int, float))
                and np.isfinite(row[metric])
            ],
            dtype=float,
        )
        if not len(values):
            continue
        records.append(
            (
                variant,
                float(values.mean()),
                float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                len(values),
            )
        )
    if not records:
        return
    figure, axis = plt.subplots(figsize=(max(6, 1.1 * len(records)), 4.8))
    x = np.arange(len(records))
    axis.errorbar(
        x,
        [record[1] for record in records],
        yerr=[record[2] for record in records],
        marker="o",
        capsize=4,
        linewidth=1.4,
    )
    axis.set_xticks(x, [record[0] for record in records], rotation=22, ha="right")
    axis.set_ylabel(metric)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def write_profiled_metric_outputs(rows: list[dict], experiment_root: Path, run_tag: str):
    """Write experiment-specific metric groups under separate grid directories."""
    if not rows:
        return
    ablation = rows[0]["ablation"]
    profile = METRIC_PROFILES.get(ablation, _SUMMARY_METRICS)
    metadata = (
        "ablation",
        "dataset",
        "variant",
        "variant_value",
        "seed",
        "run_tag",
        "method",
        "parameter",
        "value",
        "grid_shape",
        "result_file",
    )
    numeric = list(
        dict.fromkeys(
            key
            for row in rows
            for key, value in row.items()
            if key not in metadata
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
    )
    important = [key for key in profile["important"] if key in numeric]
    secondary = [key for key in profile["secondary"] if key in numeric]
    classified = set(important) | set(secondary)
    categories = {
        "important_metrics": important,
        "secondary_metrics": secondary,
        "other_metrics": [key for key in numeric if key not in classified],
    }
    for dataset in dict.fromkeys(row["dataset"] for row in rows):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        aggregate = experiment_root / dataset / "aggregate" / run_tag
        figures = experiment_root / dataset / "figures" / run_tag
        for grid in dict.fromkeys(row["grid_shape"] for row in dataset_rows):
            grid_rows = [row for row in dataset_rows if row["grid_shape"] == grid]
            grid_name = _safe_component(grid)
            grid_aggregate = aggregate / grid_name
            grid_figures = figures / grid_name
            for category, metrics in categories.items():
                category_rows = [
                    {
                        **{key: row.get(key) for key in metadata},
                        **{metric: row.get(metric) for metric in metrics},
                    }
                    for row in grid_rows
                ]
                write_csv(grid_aggregate / f"{category}.csv", category_rows)
                for metric in metrics:
                    metric_rows = [
                        row
                        for row in grid_rows
                        if isinstance(row.get(metric), (int, float))
                        and np.isfinite(row[metric])
                    ]
                    if not metric_rows:
                        continue
                    metric_dir = grid_figures / category / _safe_component(metric)
                    metric_dir.mkdir(parents=True, exist_ok=True)
                    write_csv(metric_dir / "all_parameters.csv", metric_rows)
                    for value in dict.fromkeys(row.get("value") for row in metric_rows):
                        value_rows = [
                            row for row in metric_rows if row.get("value") == value
                        ]
                        _metric_comparison_plot(
                            value_rows,
                            metric,
                            metric_dir / f"parameter_{_safe_component(value)}.png",
                            f"{dataset} | {grid} | parameter={value} | {metric}",
                        )



def aggregate_experiment(experiment_root: Path, run_tag: str, *, save_plots: bool = True):
    """Create training outputs and completely grid-isolated evaluation summaries."""
    plot_training_histories(experiment_root, run_tag)
    rows = []
    missing = []
    for manifest_path in experiment_root.rglob("run_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_tag") != run_tag:
            continue
        destination = manifest_path.parent
        summary_path = _training_summary_path(manifest_path, manifest)
        if not summary_path.is_file():
            missing.append({**manifest, "missing": "training_summary.json"})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        results = read_results(destination)
        if not results:
            missing.append({**manifest, "missing": "evaluation/results.json"})
            continue
        rows.extend(result_rows(manifest, summary, item) for item in results)

    datasets = list(dict.fromkeys(
        [row["dataset"] for row in rows]
        + [item.get("dataset") for item in missing if item.get("dataset")]
    ))
    all_summaries = []
    identity = {
        "ablation", "dataset", "variant", "variant_value", "run_tag",
        "method", "parameter", "value", "grid_shape",
    }
    group_keys = [
        "ablation", "dataset", "variant", "variant_value", "run_tag",
        "method", "parameter", "value", "grid_shape",
    ]
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        aggregate_root = experiment_root / dataset / "aggregate" / run_tag
        figures_root = experiment_root / dataset / "figures" / run_tag
        grids = list(dict.fromkeys(row["grid_shape"] for row in dataset_rows))
        index = {"dataset": dataset, "run_tag": run_tag, "grids": {}}
        for grid in grids:
            grid_name = _safe_component(grid)
            grid_rows = [row for row in dataset_rows if row["grid_shape"] == grid]
            grid_aggregate = aggregate_root / grid_name
            write_csv(grid_aggregate / "per_case_metrics.csv", grid_rows)
            write_csv(
                grid_aggregate / "missing_runs.csv",
                [item for item in missing if item.get("dataset") == dataset],
            )
            numeric = [
                key for key in grid_rows[0]
                if key not in identity | {"seed", "result_file"}
                and any(isinstance(row.get(key), (int, float)) for row in grid_rows)
            ]
            groups = {}
            for row in grid_rows:
                key = tuple(row.get(name) for name in group_keys)
                groups.setdefault(key, []).append(row)
            grid_summaries = []
            for key, members in groups.items():
                record = dict(zip(group_keys, key))
                record["seed_count"] = len({item["seed"] for item in members})
                for metric in numeric:
                    values = np.asarray(
                        [item[metric] for item in members if isinstance(item.get(metric), (int, float))],
                        dtype=float,
                    )
                    if not len(values):
                        continue
                    record[f"{metric}__mean_across_seeds"] = float(values.mean())
                    record[f"{metric}__variance_across_seeds"] = (
                        float(values.var(ddof=1)) if len(values) > 1 else 0.0
                    )
                    record[f"{metric}__std_across_seeds"] = (
                        float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    )
                grid_summaries.append(record)
            write_csv(grid_aggregate / "mean_variance_by_case.csv", grid_summaries)
            if save_plots:
                plot_metrics(grid_rows, figures_root / grid_name)
            all_summaries.extend(grid_summaries)
            index["grids"][grid_name] = {
                "case_count": len(grid_rows),
                "aggregate_directory": portable_project_path(grid_aggregate),
            }
            if save_plots:
                index["grids"][grid_name]["figures_directory"] = (
                    portable_project_path(figures_root / grid_name)
                )
        write_json(aggregate_root / "evaluation_index.json", index)
    return all_summaries

def plot_metrics(rows: list[dict], output: Path):
    """Plot the principal accuracy, validity, quality and cost metrics."""
    import matplotlib.pyplot as plt

    keys = [
        "fixed_grid_best_mse",
        "invalid_cell_fraction",
        "self_intersection__affected_cell_fraction",
        "intersecting_cell__affected_cell_fraction",
        "flipping_cell__affected_cell_fraction",
        "jacobian__minimum",
        "scaled_jacobian__minimum",
        "skewness__mean",
        "orthogonality__mean",
        "boundary_max_error",
        "network_inference_seconds",
        "training_seconds",
    ]
    output.mkdir(parents=True, exist_ok=True)
    variants = list(dict.fromkeys(row["variant"] for row in rows))
    for metric in keys:
        data = []
        for variant in variants:
            values = np.asarray(
                [
                    row[metric]
                    for row in rows
                    if row["variant"] == variant
                    and isinstance(row.get(metric), (int, float))
                    and np.isfinite(row[metric])
                ],
                dtype=float,
            )
            if len(values):
                data.append(
                    (
                        variant,
                        float(values.mean()),
                        float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    )
                )
        if not data:
            continue
        x = np.arange(len(data))
        fig, ax = plt.subplots(figsize=(max(6, len(data) * 1.2), 4.5))
        ax.errorbar(
            x,
            [item[1] for item in data],
            yerr=[item[2] for item in data],
            marker="o",
            capsize=4,
        )
        ax.set_xticks(x, [item[0] for item in data], rotation=25, ha="right")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output / f"{metric}.png", dpi=180)
        plt.close(fig)


def execute(args, config, variants: list[Variant]):
    experiment_root = args.experiment_root.resolve() / args.ablation
    experiment_root.mkdir(parents=True, exist_ok=True)
    failures = []
    completed = []
    baseline_compatibility_issues = []
    checked_baselines = set()
    for dataset in map(resolve_dataset, args.dataset or config.get("datasets", [])):
        name = dataset_name(dataset)
        for variant in variants:
            for seed in args.seeds or config.get("seeds", []):
                seed = int(seed)
                destination = (
                    baseline_directory(
                        args.experiment_root.resolve(), name, seed, args.run_tag
                    )
                    if args.ablation == BASELINE_ABLATION
                    else run_directory(
                        args.experiment_root.resolve(),
                        args.ablation,
                        name,
                        variant.name,
                        seed,
                        args.run_tag,
                    )
                )
                dataset_root = experiment_root / name
                eval_root = evaluation_directory(
                    experiment_root, name, variant.name, seed, args.run_tag
                )
                fingerprint, identity = effective_training_identity(
                    args, dataset, variant, seed
                )
                variant_value = variant.value
                baseline_fingerprint, baseline_identity = effective_training_identity(
                    args, dataset, Variant("baseline", "baseline", ()), seed
                )
                is_baseline_configuration = fingerprint == baseline_fingerprint
                cache_root = args.experiment_root.resolve()
                baseline_cache_root = Path(
                    getattr(args, "baseline_experiment_root", cache_root)
                ).resolve()
                cache_roots = list(
                    dict.fromkeys((cache_root, baseline_cache_root))
                )
                external_baseline = baseline_cache_root != cache_root
                baseline_key = (name, seed)
                if (
                    args.ablation != BASELINE_ABLATION
                    and not getattr(args, "defer_finalization", False)
                    and baseline_key not in checked_baselines
                ):
                    checked_baselines.add(baseline_key)
                    issue = check_trained_baseline_identity(
                        baseline_cache_root,
                        name,
                        seed,
                        baseline_fingerprint,
                        baseline_identity,
                    )
                    if issue is not None:
                        baseline_compatibility_issues.append(issue)
                if args.ablation != BASELINE_ABLATION and is_baseline_configuration:
                    print(
                        f"Skip canonical-baseline-equivalent variant: {name} | "
                        f"{variant.name} | seed={seed}. Ablation commands never run "
                        "baseline training or evaluation.",
                        flush=True,
                    )
                    continue
                # A variant whose complete training identity equals the frozen
                # baseline shares the suite's single baseline evaluation store.
                # Its ablation manifest still labels the row as that experiment's
                # baseline, but no duplicate ``<ablation>/base`` images are made.
                if is_baseline_configuration:
                    evaluation_dataset_root = (
                        baseline_cache_root / BASELINE_ABLATION / name
                    )
                    evaluation_variant = "baseline"
                else:
                    evaluation_dataset_root = dataset_root
                    evaluation_variant = variant.name
                # --force affects only the phase that is actually requested. This
                # lets ``--evaluate-only --force`` replace evaluation output while
                # preserving and reusing the trained baseline checkpoint.
                immutable_external_baseline = bool(
                    external_baseline and is_baseline_configuration
                )
                force_training = bool(
                    args.force
                    and not args.evaluate_only
                    and not immutable_external_baseline
                )
                force_evaluation = bool(
                    args.force
                    and not args.train_only
                    and not immutable_external_baseline
                )
                reusable = None
                if not force_training:
                    # Prefer the explicitly selected run when it is complete and
                    # has the requested identity. This prevents a newer alias
                    # manifest from displacing a valid canonical baseline.
                    selected = load_training_from_directory(destination)
                    selected_manifest = destination / "run_manifest.json"
                    if selected and selected_manifest.is_file():
                        try:
                            selected_metadata = json.loads(
                                selected_manifest.read_text(encoding="utf-8")
                            )
                        except (OSError, json.JSONDecodeError):
                            selected_metadata = {}
                        if selected_metadata.get("training_fingerprint") == fingerprint:
                            reusable = selected
                    if reusable is None:
                        for search_root in cache_roots:
                            reusable = find_reusable_training(search_root, fingerprint)
                            if reusable is not None:
                                break
                if immutable_external_baseline and reusable is None:
                    raise FileNotFoundError(
                        "The referenced external baseline has no complete training "
                        f"for dataset={name}, seed={seed}, run-tag={args.run_tag}: "
                        f"{baseline_cache_root / BASELINE_ABLATION}. Train the owning "
                        "baseline suite first; the current suite will never create or "
                        "modify it."
                    )
                if args.dry_run:
                    if not reusable:
                        action = "missing-training" if args.evaluate_only else "train+evaluate"
                    elif args.train_only:
                        action = "reuse-training"
                    else:
                        grid_actions = []
                        for shape in ablation_evaluation_settings(args)["grid_shapes"]:
                            grid = grid_shape_name(shape)
                            grid_args = single_grid_evaluation_args(args, shape)
                            for stage, evaluation_summary in evaluation_checkpoint_stages(
                                reusable["summary"]
                            ):
                                evaluation_fingerprint, evaluation_identity = effective_evaluation_identity(
                                    grid_args,
                                    dataset,
                                    evaluation_summary,
                                    fingerprint,
                                )
                                reusable_evaluation = None
                                if not force_evaluation:
                                    for search_root in cache_roots:
                                        reusable_evaluation = find_reusable_evaluation(
                                            search_root,
                                            evaluation_fingerprint,
                                            evaluation_identity,
                                        )
                                        if reusable_evaluation is not None:
                                            break
                                label = f"{grid}/{stage}" if stage else grid
                                grid_actions.append(
                                    f"{label}="
                                    f"{'reuse' if reusable_evaluation else 'evaluate'}"
                                )
                        action = "reuse-training+evaluation[" + ", ".join(grid_actions) + "]"
                    storage = portable_project_path(
                        evaluation_experiment_root(
                            evaluation_dataset_root,
                            evaluation_variant,
                            seed,
                            args.run_tag,
                        )
                    )
                    print(
                        f"{name} | {variant.name} | seed={seed} | {action} "
                        f"| evaluation-storage={storage} -> {destination}"
                    )
                    continue
                if args.evaluate_only and not reusable:
                    raise FileNotFoundError(
                        "The explicitly selected run has no usable training summary "
                        f"or best checkpoint: {destination}"
                    )
                source_is_destination = bool(
                    reusable and reusable["directory"].resolve() == destination.resolve()
                )
                if destination.exists() and not source_is_destination:
                    if args.ablation == BASELINE_ABLATION and not force_training:
                        raise FileExistsError(
                            "The baseline run directory already exists but its training "
                            "identity does not match the requested run: "
                            f"{destination}. Baseline content is protected; use --force "
                            "to retrain and replace it, or choose another --run-tag."
                        )
                    shutil.rmtree(destination)
                # Training and evaluation use separate identities. Evaluation output
                # is refreshed only when no matching completed evaluation can be reused.
                shutil.rmtree(eval_root, ignore_errors=True)
                if reusable:
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and not reusable:
                    for stale in (
                        destination / "completed.json",
                        destination / "training_completed.json",
                    ):
                        stale.unlink(missing_ok=True)
                previous_manifest = None
                previous_manifest_data = {}
                manifest_path = destination / "run_manifest.json"
                if source_is_destination and manifest_path.is_file():
                    previous_manifest = manifest_path.read_text(encoding="utf-8")
                    try:
                        previous_manifest_data = json.loads(previous_manifest)
                    except json.JSONDecodeError:
                        previous_manifest_data = {}
                evaluations = manifest_evaluations(previous_manifest_data)
                manifest = {
                    "ablation": args.ablation,
                    "dataset": name,
                    "dataset_path": portable_project_path(dataset),
                    "variant": variant.name,
                    "variant_value": variant_value,
                    "seed": seed,
                    "run_tag": args.run_tag,
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                    "status": "running",
                    "evaluation_directory": portable_project_path(
                        evaluation_experiment_root(
                            evaluation_dataset_root,
                            evaluation_variant,
                            seed,
                            args.run_tag,
                        )
                    ),
                    "training_fingerprint": fingerprint,
                    "training_identity_file": portable_project_path(destination / "training_identity.json"),
                    "training_identity_schema_version": TRAINING_IDENTITY_SCHEMA_VERSION,
                    "baseline_config": portable_project_path(_training_config_path(args)),
                    "evaluation_config": portable_project_path(
                        resolve_project_path(args.evaluate_config)
                    ),
                    "experiment_config": (
                        portable_project_path(Path(args.experiment_config_resolved))
                        if getattr(args, "experiment_config_resolved", None)
                        else None
                    ),
                    "resume_policy": "restart_incomplete",
                    "is_baseline_configuration": is_baseline_configuration,
                    "evaluation_storage_root": portable_project_path(
                        evaluation_experiment_root(
                            evaluation_dataset_root,
                            evaluation_variant,
                            seed,
                            args.run_tag,
                        )
                    ),
                    "evaluations": evaluations,
                }
                if reusable:
                    write_json(destination / "training_identity.json", identity)
                    _write_resolved_training_config(
                        destination / "resolved_config.yaml", identity
                    )
                try:
                    if reusable:
                        summary = reusable["summary"]
                        summary_path = reusable["summary_path"]
                        train_command = None
                        manifest.update(
                            training_reused=True,
                            training_source_directory=portable_project_path(reusable["directory"]),
                            training_source_manifest=portable_project_path(reusable["manifest_path"]),
                        )
                        print(f"Reuse training: {reusable['directory']}", flush=True)
                    else:
                        # Different effective parameters overwrite this target run.
                        if source_is_destination:
                            raise RuntimeError("Forced retraining cannot preserve its source directory")
                        summary, train_command = train_one(
                            args, dataset, variant, seed, destination
                        )
                        summary_path = destination / "training/training_summary.json"
                        manifest["training_reused"] = False
                        write_json(destination / "training_identity.json", identity)
                        _write_resolved_training_config(
                            destination / "resolved_config.yaml", identity
                        )
                    manifest["train_command"] = train_command
                    manifest["training_summary"] = portable_project_path(summary_path)
                    if not args.train_only:
                        evaluation_reused_by_grid = {}
                        evaluate_commands = {}
                        for shape in ablation_evaluation_settings(args)["grid_shapes"]:
                            grid = grid_shape_name(shape)
                            grid_args = single_grid_evaluation_args(args, shape)
                            for stage, evaluation_summary in evaluation_checkpoint_stages(
                                summary
                            ):
                                entry_key = f"{grid}__{stage}" if stage else grid
                                storage_variant = (
                                    f"{evaluation_variant}__{stage}"
                                    if stage
                                    else evaluation_variant
                                )
                                staging_variant = (
                                    f"{variant.name}__{stage}"
                                    if stage
                                    else variant.name
                                )
                                evaluation_fingerprint, evaluation_identity = effective_evaluation_identity(
                                    grid_args,
                                    dataset,
                                    evaluation_summary,
                                    fingerprint,
                                )
                                if stage:
                                    evaluation_identity["checkpoint_stage"] = stage
                                evaluation_identity_path = (
                                    destination
                                    / "evaluation_identities"
                                    / f"{entry_key}.json"
                                )
                                write_json(evaluation_identity_path, evaluation_identity)
                                reusable_evaluation = None
                                if not force_evaluation:
                                    for search_root in cache_roots:
                                        reusable_evaluation = find_reusable_evaluation(
                                            search_root,
                                            evaluation_fingerprint,
                                            evaluation_identity,
                                        )
                                        if reusable_evaluation is not None:
                                            break
                                if reusable_evaluation:
                                    result_files = reusable_evaluation["result_files"]
                                    reused = True
                                    source_directory = portable_project_path(
                                        reusable_evaluation["directory"]
                                    )
                                    source_manifest = portable_project_path(
                                        reusable_evaluation["manifest_path"]
                                    )
                                    result_files = materialize_reused_evaluation_results(
                                        result_files,
                                        evaluation_dataset_root,
                                        storage_variant,
                                        seed,
                                        args.run_tag,
                                    )
                                    evaluate_command = None
                                    print(
                                        f"Reuse evaluation [{entry_key}]: "
                                        f"{source_directory}",
                                        flush=True,
                                    )
                                else:
                                    reused = False
                                    source_directory = None
                                    source_manifest = None
                                    grid_eval_root = evaluation_directory(
                                        experiment_root,
                                        name,
                                        staging_variant,
                                        seed,
                                        args.run_tag,
                                        grid,
                                    )
                                    shutil.rmtree(grid_eval_root, ignore_errors=True)
                                    evaluate_command = evaluate_one(
                                        grid_args,
                                        dataset,
                                        evaluation_summary,
                                        grid_eval_root,
                                    )
                                    result_files = organize_evaluation_results(
                                        grid_eval_root,
                                        evaluation_dataset_root,
                                        storage_variant,
                                        seed,
                                        args.run_tag,
                                        overwrite=force_evaluation,
                                    )
                                    tag_evaluation_results(result_files, stage)
                                    # A values-only repair run writes just the requested
                                    # parameter directories. Keep previously completed
                                    # parameters registered in the run manifest as well.
                                    if getattr(grid_args, "evaluation_values", None):
                                        result_files = stored_grid_results(
                                            evaluation_dataset_root,
                                            storage_variant,
                                            seed,
                                            args.run_tag,
                                            grid,
                                        )
                                evaluations[entry_key] = {
                                    "status": "completed",
                                    "grid_shape": grid,
                                    "checkpoint_stage": stage or "best",
                                    "fingerprint": evaluation_fingerprint,
                                    "identity_file": portable_project_path(
                                        evaluation_identity_path
                                    ),
                                    "result_files": [
                                        portable_project_path(item)
                                        for item in result_files
                                    ],
                                    "reused": reused,
                                    "source_directory": source_directory,
                                    "source_manifest": source_manifest,
                                    "evaluate_command": evaluate_command,
                                    "finished_at": datetime.now().isoformat(
                                        timespec="seconds"
                                    ),
                                }
                                evaluation_reused_by_grid[entry_key] = reused
                                evaluate_commands[entry_key] = evaluate_command
                        manifest["evaluation_directory"] = portable_project_path(
                            evaluation_experiment_root(
                                evaluation_dataset_root,
                                evaluation_variant,
                                seed,
                                args.run_tag,
                            )
                        )
                        manifest["evaluations"] = evaluations
                        manifest["evaluation_reused_by_grid"] = evaluation_reused_by_grid
                        manifest["evaluate_commands"] = evaluate_commands
                    write_json(destination / "run_manifest.json", manifest)
                    status = (
                        "training_reused"
                        if args.train_only and reusable
                        else ("training_completed" if args.train_only else "completed")
                    )
                    manifest.update(
                        status=status,
                        finished_at=datetime.now().isoformat(timespec="seconds"),
                        best_checkpoint=summary.get("best_checkpoint"),
                        phase1_best_checkpoint=summary.get(
                            "phase1_best_checkpoint"
                        ),
                        phase2_best_checkpoint=summary.get(
                            "phase2_best_checkpoint"
                        ),
                        global_best_checkpoint=summary.get(
                            "global_best_checkpoint", summary.get("best_checkpoint")
                        ),
                    )
                    write_json(destination / "run_manifest.json", manifest)
                    marker_name = "training_completed.json" if args.train_only else "completed.json"
                    write_json(
                        destination / marker_name,
                        {
                            "status": status,
                            "training_reused": bool(reusable),
                            "evaluation_reused_by_grid": manifest.get(
                                "evaluation_reused_by_grid", {}
                            ),
                        },
                    )
                    completed.append(destination)
                except Exception as error:
                    failure_details = {
                        "status": "failed",
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                        "failed_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    manifest.update(**failure_details)
                    destination.mkdir(parents=True, exist_ok=True)
                    # Keep the complete diagnostic independently from the run
                    # manifest.  A reusable/completed manifest may be restored
                    # below, but the most recent failure must remain inspectable.
                    write_json(destination / "failure.json", failure_details)
                    if previous_manifest is not None and not force_training:
                        manifest_path.write_text(previous_manifest, encoding="utf-8")
                    else:
                        write_json(manifest_path, manifest)
                    failures.append(manifest)
                    if not args.continue_on_error:
                        raise
    if not args.dry_run and not getattr(args, "defer_finalization", False):
        if args.train_only or baseline_compatibility_issues:
            # Training curves remain useful, but a baseline mismatch makes any
            # baseline comparison or aggregate evaluation scientifically invalid.
            plot_training_histories(experiment_root, args.run_tag)
        else:
            aggregate_experiment(
                experiment_root,
                args.run_tag,
                save_plots=getattr(args, "evaluation_plots", True),
            )
    if not args.dry_run and not getattr(args, "defer_finalization", False):
        for dataset in map(resolve_dataset, args.dataset or config.get("datasets", [])):
            name = dataset_name(dataset)
            write_json(
                experiment_root / name / "failures.json",
                [item for item in failures if item.get("dataset") == name],
            )
        # Preserve any existing baseline audit file unless the user explicitly
        # requested replacement. Other ablations retain the legacy cleanup behavior.
        if args.ablation != BASELINE_ABLATION or args.force:
            (experiment_root / "failures.json").unlink(missing_ok=True)
    if getattr(args, "defer_finalization", False):
        return completed
    compatibility_report = experiment_root / "baseline_compatibility_errors.json"
    if baseline_compatibility_issues:
        message = _baseline_error_message(baseline_compatibility_issues)
        if args.dry_run:
            print(
                "Dry-run note: the baseline is not written, so later selected "
                "experiments cannot validate it until the real batch run.",
                flush=True,
            )
        else:
            write_json(compatibility_report, baseline_compatibility_issues)
            print(message, file=sys.stderr, flush=True)
            raise BaselineCompatibilityError(message)
    if not args.dry_run and args.ablation != BASELINE_ABLATION:
        compatibility_report.unlink(missing_ok=True)
    return completed


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablation", required=True,
        help="Canonical experiment directory name or 'baseline'.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset name/path; repeat for multiple datasets. Config defaults apply when omitted.",
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--evaluation-grid-shapes",
        "--grid-shapes",
        dest="evaluation_grid_shapes",
        nargs="+",
        metavar="NXxNY",
        help="Evaluation grids, for example 101x101 or 51x51 101x101.",
    )
    parser.add_argument(
        "--evaluation-values",
        nargs="+",
        type=float,
        help=(
            "Evaluate only these geometry parameter values. Existing results for "
            "other values remain untouched."
        ),
    )
    parser.add_argument(
        "--evaluation-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate pure-grid, quality, and comparison figures during ablation "
            "evaluation (default: enabled). Use --no-evaluation-plots to disable."
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        help="Run only the named YAML variants for the selected experiment.",
    )
    parser.add_argument(
        "--experiment-set", default=DEFAULT_EXPERIMENT_SET,
        help="Experiment suite directory below experiments, for example ex1, ex2 or ex3.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--baseline-config", type=Path,
        help="Frozen training baseline shared by every ablation.",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        help="Experiment-local variant definition; defaults to experiments/<experiment-set>/<ablation>/experiment.yaml.",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        help="Deprecated alias for --baseline-config; it must name the same file.",
    )
    parser.add_argument(
        "--evaluate-config", type=Path
    )
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--run-tag", default="default")
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--defer-finalization",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_experiment_set_name(args.experiment_set)
    args.run_tag = validate_run_tag(args.run_tag)
    suite_root = (ROOT / "experiments" / args.experiment_set).resolve()
    args.experiment_root = (args.experiment_root or suite_root).resolve()
    args.config = (args.config or suite_root / "configs/ablation_2d.yaml").resolve()
    if args.ablation != BASELINE_ABLATION and not default_experiment_config(
        args.ablation, args.experiment_root
    ).is_file():
        raise ValueError(
            f"Unknown ablation {args.ablation!r} in {args.experiment_root}; "
            "expected experiments/<suite>/<name>/experiment.yaml"
        )
    definition = {}
    definition_path = None
    if args.ablation != BASELINE_ABLATION:
        definition_path = (
            args.experiment_config.resolve()
            if args.experiment_config
            else default_experiment_config(args.ablation, args.experiment_root).resolve()
        )
        definition = _load_yaml_mapping(definition_path)
        args.experiment_config_resolved = definition_path
    declared_baseline = (
        declared_config_path(definition_path, "baseline_config")
        if definition_path is not None
        else None
    )
    declared_evaluation = (
        declared_config_path(definition_path, "evaluation_config")
        if definition_path is not None
        else None
    )
    args.baseline_config = (
        args.baseline_config
        or declared_baseline
        or suite_root / "baseline/experiment.yaml"
    ).resolve()
    args.evaluate_config = (
        args.evaluate_config
        or declared_evaluation
        or (
            suite_root / "baseline/evaluate.yaml"
            if (suite_root / "baseline/evaluate.yaml").is_file()
            else ROOT / "configs/2D/evaluate.yaml"
        )
    ).resolve()
    args.baseline_experiment_root = baseline_experiment_root(
        args.baseline_config, args.experiment_root
    )
    if args.train_only and args.evaluate_only:
        raise ValueError("--train-only and --evaluate-only are mutually exclusive")
    os.chdir(ROOT)
    config = _load_yaml_mapping(args.config)
    if args.train_config is not None and args.train_config.resolve() != args.baseline_config.resolve():
        raise ValueError(
            "All ablations use one frozen baseline. --train-config must equal "
            "--baseline-config; update the selected experiment set baseline/experiment.yaml instead."
        )
    args.training_config = args.baseline_config.resolve()
    if args.ablation != BASELINE_ABLATION:
        if declared_baseline and declared_baseline != args.training_config:
            raise ValueError(
                f"{definition_path} declares baseline {declared_baseline}, but the "
                f"active baseline is {args.training_config}."
            )
        if declared_evaluation and declared_evaluation != args.evaluate_config.resolve():
            raise ValueError(
                f"{definition_path} declares evaluation config {declared_evaluation}, "
                f"but the active config is {args.evaluate_config.resolve()}."
            )
    else:
        args.experiment_config_resolved = args.training_config
    datasets = args.dataset or config.get("datasets", [])
    seeds = args.seeds or config.get("seeds", [])
    if not datasets:
        raise ValueError("Specify --dataset or configure datasets")
    if len(seeds) < 1:
        raise ValueError("At least one random seed is required")
    args.dataset = datasets
    args.seeds = list(map(int, seeds))
    configured_variants = variants_from_definition(definition)
    if args.ablation != BASELINE_ABLATION and definition_path is not None:
        reject_embedded_baseline_variants(configured_variants, definition_path)
    if args.ablation == BASELINE_ABLATION:
        variants = [Variant("baseline", "baseline", ())]
    elif configured_variants:
        variants = configured_variants
    else:
        raise ValueError(
            f"{definition_path} does not declare any variants for {args.ablation!r}"
        )
    if args.variants:
        requested = set(args.variants)
        available = {variant.name for variant in variants}
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(
                f"Unknown variants for {args.ablation}: {unknown}; "
                f"available variants: {sorted(available)}"
            )
        variants = [variant for variant in variants if variant.name in requested]
    return execute(args, config, variants)


if __name__ == "__main__":
    main()






