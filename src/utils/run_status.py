"""Readable completion summaries for formal and ablation batch runs.

The training/evaluation runners persist detailed manifests for reproducibility,
but a user running several serial or queued jobs also needs one concise answer
at the end: which task trained, which task evaluated, and why a task failed.
This module derives that terminal summary without changing the manifest schema.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Mapping

from src.utils.progress import wrapped_print


_TERMINAL_STATUSES = {"completed", "failed", "interrupted", "cancelled", "dry_run", "skipped"}


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _compact_error(value, limit: int = 360) -> str | None:
    """Extract one informative, bounded error line for the final report."""
    if value is None:
        return None
    lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    if not lines:
        return None
    preferred = next(
        (
            line
            for line in reversed(lines)
            if re.search(r"(?:error|exception|traceback|failed)", line, re.IGNORECASE)
        ),
        lines[-1],
    )
    return preferred if len(preferred) <= limit else preferred[: limit - 3] + "..."


def log_error(path: str | Path | None) -> str | None:
    """Read only the tail of a queue log and return its most useful error line."""
    if not path:
        return None
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - 65536))
            text = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    return _compact_error(text)


def ablation_process_status(record: Mapping[str, object]) -> dict:
    """Resolve one expected ablation task from its run manifest and queue record.

    ``run_manifest.json`` tells us separately whether training and every grid
    evaluation completed.  ``failure.json`` is deliberately checked when the
    current subprocess failed, because a preserved old manifest can still be
    valid while a new attempt has failed.
    """
    metadata = dict(record.get("metadata") or {})
    manifest_path = Path(str(metadata["manifest_path"])) if metadata.get("manifest_path") else None
    manifest = _read_json(manifest_path) if manifest_path else {}
    failure_path = (
        Path(str(metadata["failure_path"]))
        if metadata.get("failure_path")
        else (manifest_path.parent / "failure.json" if manifest_path else None)
    )
    queue_status = str(record.get("status", "unknown"))
    manifest_status = str(manifest.get("status", ""))
    if queue_status in {"dry_run", "skipped"}:
        return {
            "task": str(record.get("task") or metadata.get("label") or "task"),
            "status": queue_status,
            "dataset": metadata.get("dataset"),
            "ablation": metadata.get("ablation"),
            "variant": metadata.get("experiment"),
            "seed": metadata.get("seed"),
            "training": queue_status,
            "evaluation": queue_status,
            "error": None,
            "log": record.get("log"),
            "manifest": str(manifest_path) if manifest_path else None,
        }
    failure = _read_json(failure_path) if failure_path else {}
    failed = (
        queue_status in {"failed", "interrupted", "cancelled"}
        or manifest_status == "failed"
        or bool(failure)
    )

    if failed:
        error = (
            _compact_error(failure.get("error"))
            or _compact_error(record.get("error"))
            or _compact_error(failure.get("traceback"))
            or log_error(record.get("log"))
        )
    else:
        error = None

    train_requested = bool(metadata.get("train_requested", True))
    evaluation_requested = bool(metadata.get("evaluation_requested", True))
    has_training = bool(manifest.get("training_summary"))
    if not train_requested:
        training = "not-requested"
    elif has_training:
        training = "reused" if manifest.get("training_reused") else "completed"
    elif failed:
        training = "failed"
    elif queue_status in {"skipped", "dry_run"}:
        training = queue_status
    else:
        training = "not-started"

    evaluations = manifest.get("evaluations")
    evaluations = evaluations if isinstance(evaluations, dict) else {}
    completed_evaluations = [
        item
        for item in evaluations.values()
        if isinstance(item, dict) and item.get("status") == "completed"
    ]
    if not evaluation_requested:
        evaluation = "not-requested"
    elif evaluations and len(completed_evaluations) == len(evaluations) and not failed:
        evaluation = (
            "reused"
            if all(item.get("reused") for item in completed_evaluations)
            else "completed"
        )
    elif failed and has_training:
        suffix = (
            f" ({len(completed_evaluations)}/{len(evaluations)} grids completed)"
            if evaluations
            else ""
        )
        evaluation = "failed" + suffix
    elif failed:
        evaluation = "not-started"
    elif queue_status in {"skipped", "dry_run"}:
        evaluation = queue_status
    else:
        evaluation = "not-started"

    status = queue_status
    if status == "completed" and failed:
        status = "failed"
    if status not in _TERMINAL_STATUSES:
        status = "completed" if training in {"completed", "reused"} else status
    label = str(record.get("task") or metadata.get("label") or "task")
    return {
        "task": label,
        "status": status,
        "dataset": metadata.get("dataset"),
        "ablation": metadata.get("ablation"),
        "variant": metadata.get("experiment"),
        "seed": metadata.get("seed"),
        "training": training,
        "evaluation": evaluation,
        "error": error,
        "log": record.get("log"),
        "manifest": str(manifest_path) if manifest_path else None,
    }


def print_completion_summary(rows, *, title: str) -> None:
    """Print one terminal-width-aware status report after a command finishes."""
    rows = list(rows)
    counts = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    rendered_counts = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    wrapped_print(f"\n=== {title} ===")
    wrapped_print(f"tasks={len(rows)} | {rendered_counts or 'no tasks'}")
    for row in rows:
        task = str(row.get("task", "task"))
        pieces = [
            f"[{row.get('status', 'unknown')}] {task}",
            f"train={row.get('training', 'unknown')}",
            f"evaluate={row.get('evaluation', 'unknown')}",
        ]
        if row.get("dataset"):
            pieces.insert(1, f"dataset={row['dataset']}")
        if row.get("variant"):
            pieces.insert(2, f"variant={row['variant']}")
        if row.get("seed") is not None:
            pieces.insert(3, f"seed={row['seed']}")
        if row.get("error"):
            pieces.append(f"error={row['error']}")
        wrapped_print(" | ".join(pieces), indent="  ")

