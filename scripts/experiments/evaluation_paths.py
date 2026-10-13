"""Compact, cross-platform storage paths for ablation evaluation artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re


WINDOWS_CLASSIC_PATH_LIMIT = 260
EVALUATION_PATH_LIMIT = int(WINDOWS_CLASSIC_PATH_LIMIT * 0.9)
# Longest current TFI/PDE/HMONet ablation artifact below one parameter folder.
# Final publication also validates every generated file, so this preflight may
# use the measured ablation layout instead of the longer four-method layout.
EVALUATION_RELATIVE_PATH_RESERVE = 120

ABLATION_ALIASES = {
    "baseline": "base",
    "boundary_self_attention": "attn",
    "dataset_size": "data",
    "initial_learning_rate": "lr",
    "jacobian_enabled": "jac",
    "jacobian_interpolation": "ji",
    "jacobian_weight": "jac_w",
    "mapping_mode": "mapping",
    "model_capacity": "cap",
    "normal_loss": "normal",
    "point_weighting": "point_w",
    "projection_activation": "act",
    "sampling_mode": "samp",
    "target_interpolation": "tgt",
    "training_sample_count": "samples",
    "trunk_geometry_coefficients": "trunk",
}

VARIANT_ALIASES = {
    "adaptive": "adapt",
    "attention_off": "off",
    "attention_on": "on",
    "baseline": "base",
    "coons_residual": "coons",
    "curved_boundary_cell": "curved",
    "jac_off": "off",
    "jac_on": "on",
    "jacobian_only": "jac",
    "curved_only": "curve",
    "compact_16k": "16k",
    "compact_40k": "40k",
    "compact_80k": "80k",
    "compact_178k": "178k",
    "full_552k": "552k",
    "jacobian_curved": "both",
    "normal_off": "off",
    "normal_on": "on",
    "position_weighted": "position",
    "sigmoid": "sig",
    "softplus": "sp",
    "trunk_off": "off",
    "trunk_on": "on",
    "uniform_weight": "uniform",
}


def _safe_text(value) -> str:
    text = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(value)).strip("._")
    return text or "item"


def compact_component(value, maximum: int = 24) -> str:
    """Return a readable bounded component, adding a stable suffix if shortened."""
    text = _safe_text(value)
    if len(text) <= maximum:
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:6]
    return f"{text[: maximum - 7]}-{digest}"


def compact_variant(ablation: str, variant: str) -> str:
    """Use concise labels without losing the meaning of known experiment variants."""
    value = str(variant)
    if value in VARIANT_ALIASES:
        return VARIANT_ALIASES[value]
    prefixes = {
        "dataset_size": "samples_",
        "initial_learning_rate": "lr_",
        "jacobian_weight": "jac_w_",
        "training_sample_count": "samples_",
    }
    prefix = prefixes.get(str(ablation))
    if prefix and value.startswith(prefix):
        value = value[len(prefix) :]
    return compact_component(value, 10)


def experiments_root_from_dataset_root(dataset_root: Path) -> Path:
    """Resolve ``.../experiments/<ablation>/<dataset>`` to ``.../experiments``."""
    dataset_root = Path(dataset_root).resolve()
    if len(dataset_root.parents) < 2:
        raise ValueError(f"Invalid ablation dataset root: {dataset_root}")
    return dataset_root.parents[1]


def evaluation_collection_root(dataset_root: Path) -> Path:
    """Return the readable shared root for one dataset and ablation."""
    dataset_root = Path(dataset_root).resolve()
    experiments_root = experiments_root_from_dataset_root(dataset_root)
    ablation = dataset_root.parent.name
    return (
        experiments_root
        / "eval"
        # Current dataset names are already bounded by the dataset generator.
        # Preserve them verbatim so the evaluation tree remains readable and
        # maps one-to-one to the corresponding ablation dataset directory.
        / compact_component(dataset_root.name, 22)
        / ABLATION_ALIASES.get(ablation, compact_component(ablation, 8))
    )


def evaluation_result_root(
    dataset_root: Path,
    variant: str,
    seed: int,
    run_tag: str,
    grid: str,
    parameter_value,
) -> Path:
    """Return the final directory for one parameter value and grid evaluation."""
    dataset_root = Path(dataset_root).resolve()
    ablation = dataset_root.parent.name
    result = evaluation_collection_root(dataset_root)
    if not (ablation == "baseline" and str(variant) == "baseline"):
        result /= compact_variant(ablation, variant)
    result = (
        result
        / str(int(seed))
        / compact_component(run_tag, 8)
        / compact_component(grid, 12)
        / compact_component(parameter_value, 12)
    )
    projected = len(str(result)) + 1 + EVALUATION_RELATIVE_PATH_RESERVE
    if projected > EVALUATION_PATH_LIMIT:
        raise OSError(
            "Evaluation output root leaves insufficient Windows path headroom: "
            f"projected={projected}, limit={EVALUATION_PATH_LIMIT}, root={result}"
        )
    return result


def evaluation_staging_root(
    experiment_root: Path,
    dataset,
    variant,
    seed,
    run_tag,
    grid=None,
) -> Path:
    """Return the suite-local temporary root used by serial evaluation.

    The ablation runner evaluates one grid at a time and clears this directory
    before starting the next grid.  A single deterministic directory therefore
    avoids opaque hash folders while retaining the short path required by the
    evaluator's deeply nested plot layout.
    """
    experiment_root = Path(experiment_root).resolve()
    result = experiment_root.parent / ".eval_work"
    projected = len(str(result)) + 1 + EVALUATION_RELATIVE_PATH_RESERVE
    if projected > EVALUATION_PATH_LIMIT:
        raise OSError(
            "Evaluation staging root leaves insufficient Windows path headroom: "
            f"projected={projected}, limit={EVALUATION_PATH_LIMIT}, root={result}"
        )
    return result


def validate_evaluation_destination(source_root: Path, target_root: Path) -> None:
    """Validate every generated artifact against the 90% Windows path budget."""
    source_root = Path(source_root)
    target_root = Path(target_root)
    violations = []
    for source in source_root.rglob("*"):
        relative = source.relative_to(source_root)
        destination = target_root / relative
        if len(str(destination.resolve())) > EVALUATION_PATH_LIMIT:
            violations.append((len(str(destination.resolve())), destination))
    if violations:
        length, path = max(violations, key=lambda item: item[0])
        raise OSError(
            "Evaluation artifact exceeds the safe path budget: "
            f"length={length}, limit={EVALUATION_PATH_LIMIT}, path={path}"
        )

