"""Compact stored ablation evaluations to reusable parameter summaries.

Usage:
    python scripts/experiments/compact_ablation_evaluations.py --dry-run
    python scripts/experiments/compact_ablation_evaluations.py
    python scripts/experiments/compact_ablation_evaluations.py --experiment-sets ex3 ex4

Training outputs, checkpoints, TensorBoard logs, and training figures are never
removed. Evaluation fingerprints remain reusable because every results.json and
run_manifest.json is retained.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = ROOT / "experiments"
TEMPORARY_EVALUATION_DIRS = (".eval_tmp", ".eval_work", ".staging")
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".pdf", ".png", ".svg", ".webp"}
AGGREGATE_FILES = {
    "evaluation_index.json",
    "mean_variance_by_case.csv",
    "missing_runs.csv",
    "per_case_metrics.csv",
}


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def remove_path(path: Path, dry_run: bool) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_file() or path.is_symlink():
        size = file_size(path)
        if not dry_run:
            path.unlink()
        return size, 1
    files = [item for item in path.rglob("*") if item.is_file()]
    size = sum(file_size(item) for item in files)
    if not dry_run:
        shutil.rmtree(path)
    return size, len(files)


def remove_empty_directories(root: Path) -> None:
    if not root.is_dir():
        return
    for directory in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass


def compact_suite(suite: Path, dry_run: bool = False) -> dict:
    removed_bytes = 0
    removed_files = 0
    retained_results = 0
    retained_images = 0

    evaluation_roots = sorted(
        path for path in suite.rglob("eval") if path.is_dir()
    )
    for evaluation_root in evaluation_roots:
        for path in list(evaluation_root.rglob("*")):
            if not path.is_file():
                continue
            if path.name == "results.json":
                retained_results += 1
                continue
            if path.suffix.lower() in IMAGE_SUFFIXES:
                continue
            size, count = remove_path(path, dry_run)
            removed_bytes += size
            removed_files += count
        if not dry_run:
            remove_empty_directories(evaluation_root)

    for name in TEMPORARY_EVALUATION_DIRS:
        temporary_root = suite / name
        if not temporary_root.is_dir():
            continue
        for path in list(temporary_root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in IMAGE_SUFFIXES:
                continue
            size, count = remove_path(path, dry_run)
            removed_bytes += size
            removed_files += count
        if not dry_run:
            remove_empty_directories(temporary_root)

    # Keep all images plus the canonical summaries and hand-written reports.
    for aggregate_root in list(suite.rglob("aggregate")):
        if not aggregate_root.is_dir():
            continue
        for path in list(aggregate_root.rglob("*")):
            if not path.is_file():
                continue
            if (
                path.name in AGGREGATE_FILES
                or path.suffix.lower() == ".md"
                or path.suffix.lower() in IMAGE_SUFFIXES
            ):
                continue
            size, count = remove_path(path, dry_run)
            removed_bytes += size
            removed_files += count
        if not dry_run:
            remove_empty_directories(aggregate_root)

    retained_images = sum(
        1
        for path in suite.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    return {
        "experiment_set": suite.name,
        "removed_bytes": removed_bytes,
        "removed_files": removed_files,
        "retained_results": retained_results,
        "retained_images": retained_images,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-sets",
        nargs="*",
        help="Experiment sets to compact (default: every ex* directory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Measure removable files without deleting them.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if EXPERIMENTS_ROOT.name != "experiments" or not EXPERIMENTS_ROOT.is_dir():
        raise RuntimeError(f"Unsafe or missing experiments root: {EXPERIMENTS_ROOT}")
    if args.experiment_sets:
        suites = [EXPERIMENTS_ROOT / name for name in args.experiment_sets]
    else:
        suites = sorted(
            path
            for path in EXPERIMENTS_ROOT.iterdir()
            if path.is_dir() and path.name.startswith("ex")
        )
    missing = [path for path in suites if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Missing experiment set(s): " + ", ".join(map(str, missing))
        )
    reports = [compact_suite(suite, args.dry_run) for suite in suites]
    total = {
        "removed_bytes": sum(item["removed_bytes"] for item in reports),
        "removed_files": sum(item["removed_files"] for item in reports),
        "retained_results": sum(item["retained_results"] for item in reports),
        "retained_images": sum(item["retained_images"] for item in reports),
    }
    print(json.dumps({"dry_run": args.dry_run, "sets": reports, "total": total}, indent=2))
    return reports


if __name__ == "__main__":
    main()
