"""Dataset-level training comparisons shared by 2-D and 3-D training.

The individual run directories remain the authoritative records.  This module
creates a small, readable comparison folder beside them only after the three
formal neural models have all finished a run for the same dataset.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory


MODEL_LAYOUT = (
    ("hmonet", "HMONet"),
    ("meshonet", "MeshONet"),
    ("pimeshonet", "PI-MeshONet"),
)
REQUIRED_FILES = (
    "mse_comparison.png",
    "relative_mse_comparison.png",
    "training_cost_comparison.png",
    "training_comparison.csv",
    "training_comparison.json",
)


def _summary_signature(path: Path) -> str:
    """Return a content signature, so changed/retrained runs rebuild figures."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed_training_runs(dataset_root: Path, model: str):
    """Return completed records newest first, including legacy model folders."""
    if not dataset_root.is_dir():
        return []
    candidates: list[tuple[int, Path, dict]] = []
    for root in dataset_root.iterdir():
        if not root.is_dir() or root.name.casefold().replace("-", "") != model:
            continue
        for path in root.rglob("training_summary.json"):
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
                completed = int(summary.get("completed_epochs") or 0)
                reason = str(summary.get("stop_reason", ""))
                if not completed or reason.startswith("failed") or reason == "stop_file_requested":
                    continue
                if reason == "maximum_epochs" and completed < int(summary.get("requested_epochs") or completed):
                    continue
                candidates.append((path.stat().st_mtime_ns, path.parent, summary))
            except (AttributeError, OSError, TypeError, ValueError):
                continue
    return [(directory, summary) for _, directory, summary in sorted(candidates, reverse=True)]


def _history(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                rows.append((int(float(row["epoch"])), float(row["dataset_mse"])))
            except (KeyError, TypeError, ValueError):
                continue
    return [(epoch, value) for epoch, value in rows if math.isfinite(value) and value >= 0.0]


def _records(dataset_root: Path, selected_runs: dict) -> tuple[list[dict], list[str]]:
    records, missing = [], []
    for key, label in MODEL_LAYOUT:
        candidates = completed_training_runs(dataset_root, key)
        selected = selected_runs.get(key)
        found = next(
            (run for run in candidates if selected is None or run[0].resolve() == selected),
            None,
        )
        if found is None:
            missing.append(label)
            continue
        directory, summary = found
        history = {}
        for epoch, value in zip(summary.get("dataset_mse_epochs", []), summary.get("dataset_mse_history", [])):
            if value is not None and math.isfinite(value) and value >= 0.0:
                history[int(epoch)] = float(value)
        for filename in ("training_history.csv", "mse_history.csv"):
            history_path = directory / filename
            if history_path.is_file():
                history.update(_history(history_path))
        if not history:
            missing.append(label)
            continue
        records.append(
            {
                "key": key,
                "model": label,
                "directory": directory,
                "summary": summary,
                "history": sorted(history.items()),
                "summary_signature": _summary_signature(directory / "training_summary.json"),
                "mse_signature": hashlib.sha256(
                    json.dumps(sorted(history.items())).encode("utf-8")
                ).hexdigest(),
            }
        )
    return records, missing


def _manifest_matches(path: Path, records: list[dict]) -> bool:
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    previous = manifest.get("sources", [])
    wanted = [
        {
            "model": item["model"],
            "directory": str(item["directory"]),
            "summary_signature": item["summary_signature"],
            "mse_signature": item["mse_signature"],
        }
        for item in records
    ]
    return previous == wanted


def _write_table(folder: Path, records: list[dict]) -> None:
    fields = (
        "model", "run_directory", "completed_epochs", "training_time_seconds",
        "best_epoch", "best_model_dataset_mse", "final_dataset_mse",
    )
    rows = []
    for item in records:
        summary = item["summary"]
        rows.append(
            {
                "model": item["model"],
                "run_directory": str(item["directory"]),
                "completed_epochs": summary.get("completed_epochs"),
                "training_time_seconds": summary.get("training_time_seconds"),
                "best_epoch": summary.get("best_epoch"),
                "best_model_dataset_mse": summary.get("best_model_dataset_mse"),
                "final_dataset_mse": summary.get("final_dataset_mse"),
            }
        )
    with (folder / "training_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(folder: Path, records: list[dict], title: str) -> None:
    """Write comparable MSE/convergence/cost figures without mixing objectives."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    for item in records:
        xs, ys = zip(*item["history"])
        axis.plot(xs, ys, label=item["model"], linewidth=1.5)
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Fixed-grid interior MSE")
    axis.set_title(f"{title}: fixed-grid MSE")
    axis.grid(True, alpha=0.28)
    axis.legend()
    figure.tight_layout()
    figure.savefig(folder / "mse_comparison.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    for item in records:
        xs, ys = zip(*item["history"])
        initial = max(float(ys[0]), 1.0e-30)
        axis.plot(xs, [max(float(value) / initial, 1.0e-30) for value in ys], label=item["model"], linewidth=1.5)
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE / first recorded MSE")
    axis.set_title(f"{title}: relative MSE reduction")
    axis.grid(True, alpha=0.28)
    axis.legend()
    figure.tight_layout()
    figure.savefig(folder / "relative_mse_comparison.png", dpi=220)
    plt.close(figure)

    labels = [item["model"] for item in records]
    epochs = [float(item["summary"].get("completed_epochs") or 0.0) for item in records]
    seconds = [float(item["summary"].get("training_time_seconds") or 0.0) for item in records]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.5))
    axes[0].bar(labels, epochs, color=(0.23, 0.47, 0.72))
    axes[0].set_ylabel("Completed epochs")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, axis="y", alpha=0.28)
    axes[1].bar(labels, seconds, color=(0.35, 0.66, 0.42))
    axes[1].set_ylabel("Training wall time (s)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, axis="y", alpha=0.28)
    figure.suptitle(f"{title}: training cost")
    figure.tight_layout()
    figure.savefig(folder / "training_cost_comparison.png", dpi=220)
    plt.close(figure)


def ensure_training_comparison_in_dataset(
    dataset_root: str | Path, dimension: int, *, selected_runs=None
) -> dict:
    """Create or reuse one comparison when the concrete dataset root is known."""
    dataset_root = Path(dataset_root)
    dataset_name = dataset_root.name
    folder = dataset_root / "comparison"
    records, missing = _records(dataset_root, selected_runs or {})
    if missing:
        return {"status": "waiting_for_models", "directory": str(folder), "missing_models": missing}
    manifest_path = folder / "training_comparison.json"
    if _manifest_matches(manifest_path, records) and all((folder / name).is_file() for name in REQUIRED_FILES):
        return {"status": "reused", "directory": str(folder), "missing_models": []}
    try:
        folder.mkdir(parents=True, exist_ok=True)
        sources = [
            {
                "model": item["model"],
                "directory": str(item["directory"]),
                "summary_signature": item["summary_signature"],
                "mse_signature": item["mse_signature"],
            }
            for item in records
        ]
        with TemporaryDirectory(prefix=".comparison-", dir=dataset_root) as temporary:
            staging = Path(temporary)
            _plot(staging, records, f"{dataset_name} ({int(dimension)}D)")
            _write_table(staging, records)
            (staging / "training_comparison.json").write_text(
                json.dumps(
                    {"dataset": dataset_name, "dimension": int(dimension), "sources": sources},
                    ensure_ascii=False, indent=2,
                ), encoding="utf-8",
            )
            for name in REQUIRED_FILES:
                (staging / name).replace(folder / name)
    except ImportError:
        return {"status": "matplotlib_unavailable", "directory": str(folder), "missing_models": []}
    return {"status": "created", "directory": str(folder), "missing_models": []}


def ensure_training_comparison(output_root: str | Path, dimension: int, dataset_name: str) -> dict:
    """Create or reuse one dataset-level comparison of the three neural models.

    Returns a status record.  A missing model is not an error: an individual
    training command should still succeed, and the third completed model will
    create the comparison automatically.
    """
    return ensure_training_comparison_in_dataset(
        Path(output_root) / f"{int(dimension)}D" / str(dataset_name), dimension
    )
