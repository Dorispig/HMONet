"""Compact, cross-platform storage paths for ablation evaluation artifacts."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import uuid
import warnings


WINDOWS_CLASSIC_PATH_LIMIT = 260
LINUX_PATH_LIMIT = 4096
# Reserve headroom is a warning threshold only.  It must never make an
# otherwise valid experiment fail before the operating system would fail it.
EVALUATION_RESERVE_WARNING_LIMIT = int(WINDOWS_CLASSIC_PATH_LIMIT * 0.9)
# Backwards-compatible public name used by older tests and scripts.
EVALUATION_PATH_LIMIT = EVALUATION_RESERVE_WARNING_LIMIT
# Retained public results are shallow after publication.  This reserve is used
# for their advisory dry-run warning only.
EVALUATION_RELATIVE_PATH_RESERVE = 96
# The evaluator creates a substantially deeper internal hierarchy before
# publication.  Its deepest current 2-D quality artifact is about 161
# characters below the task root; reserve 180 for staging checks.  This is
# warning-only, never an artificial failure below the real system limit.
STAGING_RELATIVE_PATH_RESERVE = 180
STAGING_WORKSPACE_NAME = ".w"


class EvaluationStagingBusyError(RuntimeError):
    """Raised when a second evaluator targets the same temporary task root."""


def evaluation_staging_lock_path(staging_root: Path) -> Path:
    """Return a short sibling lock path without changing the public layout."""
    staging_root = Path(staging_root)
    return staging_root.parent / f".{staging_root.name}.lock"


@contextmanager
def exclusive_evaluation_staging_root(staging_root: Path):
    """Prevent two processes from deleting or writing one staging evaluation.

    The public temporary hierarchy intentionally has no random/hash component.
    Therefore a duplicate request for the exact same dataset, experiment,
    seed, tag, and grid must be rejected rather than silently sharing files.
    The lock is a sibling of the work directory so cleanup of the work
    directory cannot remove the lock held by the active evaluator.
    """
    staging_root = Path(staging_root)
    lock_path = evaluation_staging_lock_path(staging_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    payload = {
        "token": token,
        "pid": os.getpid(),
        "staging_root": str(staging_root),
    }
    try:
        with lock_path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)
    except FileExistsError as error:
        try:
            owner = json.loads(lock_path.read_text(encoding="utf-8"))
            owner_hint = f" pid={owner.get('pid', 'unknown')}"
        except (OSError, ValueError, TypeError):
            owner_hint = ""
        raise EvaluationStagingBusyError(
            "Another evaluation already owns this temporary output: "
            f"{staging_root}.{owner_hint} Do not launch the same "
            "dataset/experiment/seed/tag/grid twice simultaneously. "
            f"If no such process is running, remove the stale lock: {lock_path}"
        ) from error
    try:
        yield staging_root
    finally:
        # Do not delete a lock that was somehow replaced by another owner.
        try:
            owner = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            owner = {}
        if owner.get("token") == token:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def system_path_limit() -> int:
    """Return the active platform's real total-path threshold."""
    return WINDOWS_CLASSIC_PATH_LIMIT if os.name == "nt" else LINUX_PATH_LIMIT


def _actual_path_length(path: Path) -> int:
    return len(str(Path(path).resolve()))


def validate_runtime_path(
    path: Path, *, context: str, warn_reserved: bool = True
) -> int:
    """Warn past reserved headroom; fail only at the actual OS limit.

    Windows classic paths reserve one character for the terminating null, so
    a 260-character path is rejected.  A path component is separately limited
    to 255 characters/bytes by the common Windows and POSIX filesystems; 255
    itself remains valid, while longer names are rejected.
    """
    path = Path(path)
    length = _actual_path_length(path)
    limit = system_path_limit()
    oversized = [part for part in path.parts if len(part) > 255]
    if length >= limit or oversized:
        detail = (
            f"components above 255 characters: {oversized}"
            if oversized
            else f"length={length}, system_limit={limit}"
        )
        raise OSError(f"{context} exceeds the real system path limit: {detail}; path={path}")
    if warn_reserved and length > EVALUATION_RESERVE_WARNING_LIMIT:
        warnings.warn(
            f"{context} exceeds the reserved path headroom "
            f"({length}>{EVALUATION_RESERVE_WARNING_LIMIT}) but remains valid "
            f"for this system ({limit - 1} maximum usable characters): {path}",
            RuntimeWarning,
            stacklevel=2,
        )
    return length


def warn_for_projected_path(path: Path, *, context: str, reserve: int = 0) -> int:
    """Warn for a projected deep artifact without treating it as a hard error."""
    length = _actual_path_length(path) + int(reserve)
    if length > EVALUATION_RESERVE_WARNING_LIMIT:
        warnings.warn(
            f"{context} projected length={length} exceeds reserved headroom "
            f"{EVALUATION_RESERVE_WARNING_LIMIT}; actual files remain allowed until "
            f"the system limit ({system_path_limit() - 1}). Root={path}",
            RuntimeWarning,
            stacklevel=2,
        )
    return length

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


def staging_workspace_root(experiment_root: Path) -> Path:
    """Return the short hidden workspace used only while evaluating.

    The evaluator itself adds the case, grid, parameter, mapping, comparison,
    and quality-file levels.  Keeping scratch work below the experiment suite
    would make valid Windows paths reach MAX_PATH before publication.  The
    permanent ``experiments/<suite>/eval`` tree remains fully reader-facing;
    only this disposable workspace uses short components.
    """
    experiment_root = Path(experiment_root).resolve()
    # Callers normally pass a concrete variant directory such as
    # ``experiments/ex8/e2_pde_jacobian``.  The workspace label belongs to the
    # experiment *suite* (``e8``), while the next component below it identifies
    # the variant.  Looking only at ``experiment_root.name`` accidentally made
    # the workspace itself ``.w/e2_pde_jacobian`` and duplicated that meaning
    # in every temporary path.
    suite_root = next(
        (
            candidate
            for candidate in (experiment_root, *experiment_root.parents)
            if re.fullmatch(r"ex\d+", candidate.name, flags=re.IGNORECASE)
        ),
        None,
    )
    if suite_root is not None:
        match = re.fullmatch(r"ex(\d+)", suite_root.name, flags=re.IGNORECASE)
        suite = f"e{match.group(1)}"
        try:
            project_root = suite_root.parents[1]
            workspace_parent = project_root.parent
        except IndexError:
            workspace_parent = suite_root.parent
    else:
        try:
            project_root = experiment_root.parents[1]
            workspace_parent = project_root.parent
        except IndexError:
            workspace_parent = experiment_root.parent
        suite = _safe_text(experiment_root.name)
    return workspace_parent / STAGING_WORKSPACE_NAME / suite


def _staging_variant_component(experiment_root: Path, variant: str) -> str:
    """Use ``b`` only in disposable scratch work for the baseline."""
    alias = staging_experiment_name(experiment_root, variant)
    if alias == "baseline":
        return "b"
    if alias.startswith("baseline__"):
        return "b" + alias.removeprefix("baseline")
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
    if validate:
        validate_runtime_path(result, context="Evaluation output root")
        warn_for_projected_path(
            result,
            context="Evaluation output root",
            reserve=1 + EVALUATION_RELATIVE_PATH_RESERVE,
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
    The evaluator adds several deep implementation folders beneath this root.
    Therefore the disposable hierarchy uses compact but deterministic names::

        <project-parent>/.w/e8/<dataset>/<b-or-e1_1>/<seed>/<tag>/<grid>/

    The permanent ``experiments/ex8/eval`` hierarchy remains fully readable.
    No hash or random component is used, so task isolation and reuse remain
    deterministic across local and server runs.
    """
    experiment_root = Path(experiment_root).resolve()
    result = (
        staging_workspace_root(experiment_root)
        / _safe_text(dataset)
        / _staging_variant_component(experiment_root, str(variant))
        / str(int(seed))
        / _safe_text(run_tag)
    )
    if grid is not None:
        result = result / _safe_text(grid)
    if validate:
        validate_runtime_path(result, context="Evaluation staging root")
        warn_for_projected_path(
            result,
            context="Evaluation staging root",
            reserve=1 + STAGING_RELATIVE_PATH_RESERVE,
        )
    return result


def validate_evaluation_destination(source_root: Path, target_root: Path) -> None:
    """Validate every generated artifact against the 90% Windows path budget."""
    source_root = Path(source_root)
    target_root = Path(target_root)
    violations = []
    reserve_warning = None
    for source in source_root.rglob("*"):
        relative = source.relative_to(source_root)
        destination = target_root / relative
        length = validate_runtime_path(
            destination,
            context="Published evaluation artifact",
            warn_reserved=False,
        )
        if length > EVALUATION_RESERVE_WARNING_LIMIT:
            reserve_warning = max(
                reserve_warning or (length, destination),
                (length, destination),
                key=lambda item: item[0],
            )
    if violations:
        length, path = max(violations, key=lambda item: item[0])
        raise OSError(
            "Evaluation artifact exceeds the real system path limit: "
            f"length={length}, system_limit={system_path_limit()}, path={path}"
        )
    if reserve_warning is not None:
        length, path = reserve_warning
        warnings.warn(
            f"Published evaluation artifact exceeds reserved path headroom "
            f"({length}>{EVALUATION_RESERVE_WARNING_LIMIT}) but is still valid: {path}",
            RuntimeWarning,
            stacklevel=2,
        )

