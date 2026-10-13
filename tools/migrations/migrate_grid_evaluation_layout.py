"""Migrate ablation evaluation metadata and summaries to per-grid layout.

Run from the project root:
    python tools/migrations/migrate_grid_evaluation_layout.py

The migration never modifies checkpoints or raw ``results.json`` files.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_ablation_2d import (  # noqa: E402
    aggregate_experiment,
    grid_shape_name,
    write_json,
)


LEGACY_FIELDS = (
    "evaluation_fingerprint",
    "evaluation_identity_file",
    "evaluation_result_files",
    "evaluation_reused",
    "evaluation_source_directory",
    "evaluation_source_manifest",
    "evaluate_command",
)
MIXED_AGGREGATE_FILES = (
    "per_case_metrics.csv",
    "mean_variance_by_case.csv",
    "missing_runs.csv",
    "important_metrics.csv",
    "secondary_metrics.csv",
    "other_metrics.csv",
)


def _fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _group_result_files(files):
    groups = {}
    for value in files:
        path = Path(value)
        if not path.is_file():
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            grid = grid_shape_name(result.get("grid_shape", path.parent.name))
        except (OSError, json.JSONDecodeError):
            grid = grid_shape_name(path.parent.name)
        groups.setdefault(grid, []).append(str(path))
    return groups


def migrate_manifest(path: Path, dry_run: bool = False) -> tuple[bool, list[str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(manifest.get("evaluations"), dict):
        return False, sorted(manifest["evaluations"])
    groups = _group_result_files(manifest.get("evaluation_result_files", []))
    if not groups:
        return False, []
    old_identity_path = Path(manifest.get("evaluation_identity_file", ""))
    old_identity = None
    if old_identity_path.is_file():
        old_identity = json.loads(old_identity_path.read_text(encoding="utf-8"))
    evaluations = {}
    for grid, files in sorted(groups.items()):
        identity = copy.deepcopy(old_identity) if isinstance(old_identity, dict) else None
        fingerprint = manifest.get("evaluation_fingerprint")
        if identity is not None:
            dimensions = [int(item) for item in grid.split("x")]
            identity["grid_shapes"] = [dimensions]
            # Preserve the exact historical fingerprint for the common one-grid
            # case; recompute only when an old manifest mixed several grids.
            if len(groups) > 1:
                fingerprint = _fingerprint(identity)
        identity_path = path.parent / "evaluation_identities" / f"{grid}.json"
        if identity is not None and not dry_run:
            write_json(identity_path, identity)
        evaluations[grid] = {
            "status": "completed",
            "fingerprint": fingerprint,
            "identity_file": str(identity_path) if identity is not None else None,
            "result_files": files,
            "reused": manifest.get("evaluation_reused"),
            "source_directory": manifest.get("evaluation_source_directory"),
            "source_manifest": manifest.get("evaluation_source_manifest"),
            "evaluate_command": manifest.get("evaluate_command"),
            "finished_at": manifest.get("finished_at"),
            "migrated_from_legacy_manifest": True,
        }
    legacy = {key: manifest.get(key) for key in LEGACY_FIELDS if key in manifest}
    manifest["evaluations"] = evaluations
    manifest["legacy_evaluation"] = legacy
    for key in LEGACY_FIELDS:
        manifest.pop(key, None)
    if not dry_run:
        write_json(path, manifest)
    return True, sorted(groups)


def cleanup_mixed_outputs(dataset_root: Path, run_tag: str, dry_run: bool = False):
    aggregate_root = dataset_root / "aggregate" / run_tag
    for name in MIXED_AGGREGATE_FILES:
        path = aggregate_root / name
        if path.is_file() and not dry_run:
            path.unlink()
    figures_root = dataset_root / "figures" / run_tag
    if figures_root.is_dir():
        for path in figures_root.iterdir():
            if path.is_file() and not dry_run:
                path.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=ROOT / "experiments")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifests = sorted(args.experiments_root.rglob("run_manifest.json"))
    migrated = 0
    grids = {}
    affected = set()
    for manifest_path in manifests:
        changed, names = migrate_manifest(manifest_path, args.dry_run)
        if changed:
            migrated += 1
        if names:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            experiment_root = manifest_path
            while experiment_root.parent != args.experiments_root and experiment_root.parent != experiment_root:
                experiment_root = experiment_root.parent
            if experiment_root.parent != args.experiments_root:
                continue
            dataset = manifest.get("dataset")
            run_tag = manifest.get("run_tag")
            if dataset and run_tag:
                affected.add((experiment_root, dataset, run_tag))
            for name in names:
                grids[name] = grids.get(name, 0) + 1
    if not args.dry_run:
        for experiment_root, dataset, run_tag in sorted(affected, key=lambda x: tuple(map(str, x))):
            aggregate_experiment(experiment_root, run_tag)
            cleanup_mixed_outputs(experiment_root / dataset, run_tag)
    report = {
        "manifest_count": len(manifests),
        "migrated_manifest_count": migrated,
        "grid_entry_counts": grids,
        "affected_dataset_tags": [
            {"experiment": str(root), "dataset": dataset, "run_tag": tag}
            for root, dataset, tag in sorted(affected, key=lambda x: tuple(map(str, x)))
        ],
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        write_json(args.experiments_root / "grid_evaluation_layout_migration_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
