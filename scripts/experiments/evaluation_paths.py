"""Compact, cross-platform storage paths for ablation evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
import re


WINDOWS_CLASSIC_PATH_LIMIT = 260
EVALUATION_PATH_LIMIT = int(WINDOWS_CLASSIC_PATH_LIMIT * 0.9)
# Longest current TFI/PDE/HMONet ablation artifact below one parameter folder.
# Final publication also validates every generated file, so this preflight may
# use the measured ablation layout instead of the longer four-method layout.
# Compact publication keeps the deepest retained image below
# ``plots/<method>/<stage>/quality/<file>``.  Reserve that measured maximum,
# not the evaluator's much deeper transient directory tree.
EVALUATION_RELATIVE_PATH_RESERVE = 72

def _safe_text(value) -> str:
    text = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(value)).strip("._")
    return text or "item"


def evaluation_collection_root(dataset_root: Path) -> Path:
    """Return ``<suite>/eval/<dataset>`` for one experiment dataset."""
    dataset_root = Path(dataset_root).resolve()
    if len(dataset_root.parents) < 2:
        raise ValueError(f"Invalid experiment dataset root: {dataset_root}")
    suite_root = dataset_root.parents[1]
    return suite_root / "eval" / dataset_root.name


def _evaluation_run_root(
    dataset_root: Path,
    variant: str,
    seed: int,
    run_tag: str,
) -> Path:
    """Return the compact public root selected by dataset, run, and experiment."""
    dataset_root = Path(dataset_root).resolve()
    variant_text = str(variant)
    checkpoint_stage = None
    if "__" in variant_text:
        variant_text, checkpoint_stage = variant_text.split("__", 1)

    experiment = _safe_text(variant_text)
    if checkpoint_stage:
        experiment += f"__{_safe_text(checkpoint_stage)}"
    return (
        evaluation_collection_root(dataset_root)
        / f"seed_{int(seed)}"
        / _safe_text(run_tag)
        / experiment
    )


def evaluation_experiment_root(
    dataset_root: Path,
    variant: str,
    seed: int,
    run_tag: str,
) -> Path:
    """Public experiment-level directory containing all grids and parameters."""
    return _evaluation_run_root(dataset_root, variant, seed, run_tag)


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
    result = (
        _evaluation_run_root(dataset_root, variant, seed, run_tag)
        / _safe_text(grid)
        / _safe_text(parameter_value)
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

