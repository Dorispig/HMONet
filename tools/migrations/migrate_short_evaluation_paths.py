"""Migrate all completed ablation evaluations to the compact shared layout.

Run from the project root:

    python tools/migrations/migrate_short_evaluation_paths.py --dry-run
    python tools/migrations/migrate_short_evaluation_paths.py

The migration moves only completed result directories. Incomplete staging
directories are removed after their failed cases have been recorded in the
migration report. Evaluation fingerprints are preserved byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
from datetime import datetime


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.evaluation_paths import (
    EVALUATION_PATH_LIMIT,
    evaluation_collection_root,
    evaluation_result_root,
    validate_evaluation_destination,
)
from src.utils.paths import portable_project_path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_preserving_time(path: Path, payload) -> None:
    stat = path.stat() if path.exists() else None
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if stat is not None:
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def fingerprint_snapshot(experiments_root: Path) -> dict[str, dict[str, str]]:
    snapshot = {}
    for path in sorted(experiments_root.rglob("run_manifest.json")):
        try:
            data = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        values = {"training": str(data.get("training_fingerprint", ""))}
        for grid, entry in data.get("evaluations", {}).items():
            values[f"evaluation:{grid}"] = str(entry.get("fingerprint", ""))
        values["legacy_evaluation"] = str(data.get("evaluation_fingerprint", ""))
        snapshot[str(path.resolve())] = values
    return snapshot


def old_result_directories(experiments_root: Path):
    for result in sorted(experiments_root.rglob("results.json")):
        parts = result.relative_to(experiments_root).parts
        if "_evaluation" not in parts or ".staging" in parts:
            continue
        index = parts.index("_evaluation")
        if len(parts) != index + 7:
            continue
        ablation, dataset = parts[0], parts[1]
        parameter, seed_part, variant, run_tag, grid = parts[index + 1 : index + 6]
        if not seed_part.startswith("seed_"):
            continue
        yield {
            "source": result.parent.resolve(),
            "ablation": ablation,
            "dataset": dataset,
            "parameter": parameter,
            "seed": int(seed_part.removeprefix("seed_")),
            "variant": variant,
            "run_tag": run_tag,
            "grid": grid,
        }


def build_plan(experiments_root: Path):
    moves = []
    targets = set()
    for item in old_result_directories(experiments_root):
        dataset_root = experiments_root / item["ablation"] / item["dataset"]
        target = evaluation_result_root(
            dataset_root,
            item["variant"],
            item["seed"],
            item["run_tag"],
            item["grid"],
            item["parameter"],
        ).resolve()
        validate_evaluation_destination(item["source"], target)
        if target in targets:
            raise RuntimeError(f"Compact evaluation path collision: {target}")
        if target.exists() and target != item["source"]:
            raise FileExistsError(f"Migration target already exists: {target}")
        targets.add(target)
        item["target"] = target
        moves.append(item)
    return moves


def replace_path_text(text: str, replacements) -> str:
    for source, target in replacements:
        source_abs = str(source)
        target_abs = str(target)
        source_rel = portable_project_path(source)
        target_rel = portable_project_path(target)
        pairs = (
            (source_abs, target_abs),
            (source_abs.replace("\\", "/"), target_abs.replace("\\", "/")),
            (source_rel, target_rel),
            (source_rel.replace("/", "\\"), target_rel.replace("/", "\\")),
        )
        for old, new in pairs:
            text = text.replace(old, new)
    # Historical commands may contain the former per-dataset staging root.
    text = re.sub(
        r"experiments[\\/][^\\/]+[\\/][^\\/]+[\\/]_evaluation[\\/]\.staging[\\/]([0-9a-f]{12})",
        r"experiments/.eval_tmp/\1",
        text,
    )
    return text


def update_generated_references(experiments_root: Path, replacements) -> int:
    changed = 0
    extensions = {".json", ".csv", ".md", ".yaml", ".yml"}
    for path in sorted(experiments_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if "eval" in path.relative_to(experiments_root).parts:
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        updated = replace_path_text(original, replacements)
        if updated != original:
            stat = path.stat()
            path.write_text(updated, encoding="utf-8")
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            changed += 1
    return changed


def update_manifests(experiments_root: Path) -> int:
    changed = 0
    for path in sorted(experiments_root.rglob("run_manifest.json")):
        try:
            data = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        ablation = data.get("ablation")
        dataset = data.get("dataset")
        if not ablation or not dataset:
            continue
        dataset_root = experiments_root / str(ablation) / str(dataset)
        collection = portable_project_path(evaluation_collection_root(dataset_root))
        before = json.dumps(data, sort_keys=True, ensure_ascii=False)
        data["evaluation_directory"] = collection
        data["evaluation_storage_root"] = collection
        data["evaluation_path_layout"] = {
            "schema": "compact_v1",
            "path_limit": EVALUATION_PATH_LIMIT,
            "migrated_at": datetime.now().isoformat(timespec="seconds"),
        }
        after = json.dumps(data, sort_keys=True, ensure_ascii=False)
        if after != before:
            write_json_preserving_time(path, data)
            changed += 1
    return changed


def failure_inventory(experiments_root: Path):
    failures = []
    for path in sorted(experiments_root.rglob("failures.json")):
        try:
            entries = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        for entry in entries if isinstance(entries, list) else []:
            if entry:
                failures.append(
                    {
                        key: entry.get(key)
                        for key in ("ablation", "dataset", "variant", "seed", "error")
                    }
                )
    return failures


def remove_incomplete_staging(experiments_root: Path) -> list[str]:
    removed = []
    roots = list(experiments_root.rglob(".staging")) + [
        experiments_root / ".eval_tmp"
    ]
    resolved_experiments = experiments_root.resolve()
    for root in roots:
        if not root.exists():
            continue
        resolved = root.resolve()
        if resolved_experiments not in resolved.parents:
            raise RuntimeError(f"Refusing to remove staging outside experiments: {resolved}")
        shutil.rmtree(resolved)
        removed.append(str(resolved))
    return removed


def remove_empty_legacy_directories(experiments_root: Path) -> None:
    roots = sorted(
        [path for path in experiments_root.rglob("_evaluation") if path.is_dir()],
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for root in roots:
        for directory in sorted(
            [item for item in root.rglob("*") if item.is_dir()],
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            if not any(directory.iterdir()):
                directory.rmdir()
        if root.exists() and not any(root.iterdir()):
            root.rmdir()


def verify(experiments_root: Path, fingerprints_before) -> dict:
    fingerprints_after = fingerprint_snapshot(experiments_root)
    if fingerprints_after != fingerprints_before:
        raise RuntimeError("Evaluation or training fingerprints changed during migration")
    missing = []
    completed_entries = 0
    for manifest in experiments_root.rglob("run_manifest.json"):
        try:
            data = read_json(manifest)
        except (OSError, json.JSONDecodeError):
            continue
        for grid, entry in data.get("evaluations", {}).items():
            if entry.get("status") != "completed":
                continue
            completed_entries += 1
            for value in entry.get("result_files", []):
                path = Path(value)
                if not path.is_absolute():
                    path = ROOT / path
                if not path.is_file():
                    missing.append(f"{manifest}:{grid}:{value}")
    if missing:
        raise RuntimeError("Missing migrated result references:\n" + "\n".join(missing[:20]))
    evaluation_files = [
        item for item in (experiments_root / "eval").rglob("*") if item.is_file()
    ]
    longest = max(evaluation_files, key=lambda item: len(str(item.resolve())), default=None)
    maximum = len(str(longest.resolve())) if longest else 0
    if maximum > EVALUATION_PATH_LIMIT:
        raise RuntimeError(
            f"Migrated path exceeds {EVALUATION_PATH_LIMIT}: {maximum}, {longest}"
        )
    return {
        "completed_manifest_entries": completed_entries,
        "evaluation_file_count": len(evaluation_files),
        "maximum_path_length": maximum,
        "longest_path": str(longest.resolve()) if longest else None,
        "missing_result_references": len(missing),
        "fingerprints_preserved": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--experiments-root", type=Path, default=ROOT / "experiments"
    )
    args = parser.parse_args(argv)
    experiments_root = args.experiments_root.resolve()
    plan = build_plan(experiments_root)
    failures = failure_inventory(experiments_root)
    projected_max = 0
    for item in plan:
        for source in item["source"].rglob("*"):
            projected = item["target"] / source.relative_to(item["source"])
            projected_max = max(projected_max, len(str(projected.resolve())))
    summary = {
        "schema": "compact_evaluation_paths_v1",
        "dry_run": bool(args.dry_run),
        "path_limit": EVALUATION_PATH_LIMIT,
        "completed_result_directories": len(plan),
        "failed_cases_pending": len(failures),
        "projected_maximum_path_length": projected_max,
        "failures": failures,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return summary

    report_path = experiments_root / "evaluation_path_migration_report.json"
    fingerprints = fingerprint_snapshot(experiments_root)
    report = dict(summary)
    report["started_at"] = datetime.now().isoformat(timespec="seconds")
    write_json_preserving_time(report_path, report)

    replacements = []
    for item in plan:
        source, target = item["source"], item["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        replacements.append((source, target))

    changed_references = update_generated_references(experiments_root, replacements)
    changed_manifests = update_manifests(experiments_root)
    removed_staging = remove_incomplete_staging(experiments_root)
    remove_empty_legacy_directories(experiments_root)
    verification = verify(experiments_root, fingerprints)
    report.update(
        {
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "changed_reference_files": changed_references,
            "changed_manifests": changed_manifests,
            "removed_staging_roots": removed_staging,
            "verification": verification,
        }
    )
    write_json_preserving_time(report_path, report)
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()

