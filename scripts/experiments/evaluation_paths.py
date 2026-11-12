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
# ``p/<method>/<stage>/q/<file>``.  The public plot layout is deliberately
# abbreviated so that a server checkout nested below a home directory and a
# classic Windows checkout both retain ample headroom.  Reserve more than the
# measured deepest retained artifact rather than the evaluator's transient tree.
EVALUATION_RELATIVE_PATH_RESERVE = 96

def _safe_text(value) -> str:
    text = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(value)).strip("._")
    return text or "item"


def _declared_cli_alias(experiment_root: Path, variant: str) -> str | None:
    """Read the optional short alias without importing the runner module.

    ``evaluation_paths`` is intentionally dependency-light because it is used
    by both the one-run and queued multi-GPU entry points.  A small YAML-line
    reader is sufficient here: experiment definitions place ``cli_alias`` at
    the top level and aliases are already restricted to portable path text.
    """
    definition = Path(experiment_root) / variant / "experiment.yaml"
    try:
        text = definition.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"(?m)^\s*cli_alias\s*:\s*([^#\r\n]+)", text)
    if not match:
        return None
    return _safe_text(match.group(1).strip().strip("'\""))


def staging_experiment_name(experiment_root: Path, variant: str) -> str:
    """Return the readable short experiment name used by temporary work.

    The final evaluation directory preserves the canonical experiment name for
    backwards-compatible result reuse.  Temporary work is deliberately more
    compact: ``rational_affine`` becomes its declared ``e1_1`` alias and
    ``e1_6_fd0_008`` becomes ``e1_6``.  A checkpoint suffix, if present, is
    retained because it selects a distinct model to evaluate.
    """
    canonical, separator, checkpoint_stage = str(variant).partition("__")
    alias = _declared_cli_alias(experiment_root, canonical)
    if not alias:
        match = re.match(r"^(e\d+(?:_\d+)?)(?:_|$)", canonical)
        alias = match.group(1) if match else _safe_text(canonical)
    if separator and checkpoint_stage:
        alias = f"{alias}__{_safe_text(checkpoint_stage)}"
    return alias


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
    # Keep permanent results as compact as the temporary hierarchy.  The
    # experiment definition remains the canonical identity in manifests and
    # fingerprints; its readable short alias is sufficient for the directory
    # name because the surrounding suite already scopes it uniquely.
    experiment = staging_experiment_name(dataset_root.parents[1], str(variant))
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
    *,
    validate: bool = True,
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
    if validate and projected > EVALUATION_PATH_LIMIT:
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
    *,
    validate: bool = True,
) -> Path:
    """Return the task-isolated temporary root for one ablation evaluation.

    Evaluations can run concurrently on different GPUs.  The staging path is
    therefore keyed by dataset, short experiment alias, seed, tag, and grid.
    This is the only temporary hierarchy::

        .eval_work/<dataset>/<e1_1>/seed_<seed>/<tag>/<grid>/

    The suite name and canonical long experiment name are intentionally not
    repeated here: the latter is only used for the permanent, reusable result
    directory.  Every component is readable and deterministic; no hash
    directory is needed.
    """
    experiment_root = Path(experiment_root).resolve()
    result = (
        experiment_root.parent
        / ".eval_work"
        / _safe_text(dataset)
        / staging_experiment_name(experiment_root, str(variant))
        / f"seed_{int(seed)}"
        / _safe_text(run_tag)
    )
    if grid is not None:
        result = result / _safe_text(grid)
    projected = len(str(result)) + 1 + EVALUATION_RELATIVE_PATH_RESERVE
    if validate and projected > EVALUATION_PATH_LIMIT:
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

