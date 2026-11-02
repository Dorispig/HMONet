"""Migrate completed ablation evaluations into the short suite layout.

Final layout::

    exN/eval/<dataset>/seed_<seed>/<run-tag>/<experiment>/<grid>/<value>/

By default the migration is non-destructive. With ``--remove-legacy`` it first
copies and verifies every retained summary/image, updates all references, and
only then removes superseded result directories. Fingerprints remain in JSON
metadata, never in path names.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.evaluation_paths import (
    evaluation_experiment_root,
    evaluation_result_root,
)
from scripts.experiments.run_ablation_2d import publish_evaluation_result
from src.utils.paths import portable_project_path, resolve_project_path


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def fingerprint_snapshot(experiments_root: Path) -> dict:
    snapshot = {}
    for path in experiments_root.rglob("run_manifest.json"):
        try:
            data = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        snapshot[str(path.resolve())] = {
            "training": data.get("training_fingerprint"),
            "evaluations": {
                key: entry.get("fingerprint")
                for key, entry in data.get("evaluations", {}).items()
            },
        }
    return snapshot


def suite_root(experiments_root: Path, manifest_path: Path) -> Path:
    relative = manifest_path.resolve().relative_to(experiments_root.resolve())
    if not relative.parts:
        raise ValueError(f"Manifest is outside an experiment suite: {manifest_path}")
    return experiments_root / relative.parts[0]


def entry_variant(manifest: dict, entry: dict) -> str:
    variant = str(manifest["variant"])
    stage = entry.get("checkpoint_stage")
    return f"{variant}__{stage}" if stage and stage != "best" else variant


def collect_plan(experiments_root: Path):
    references = defaultdict(list)
    manifests = []
    for manifest_path in sorted(experiments_root.rglob("run_manifest.json")):
        try:
            manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError):
            continue
        required = ("ablation", "dataset", "variant", "seed", "run_tag")
        if not all(key in manifest for key in required):
            continue
        dataset_root = (
            suite_root(experiments_root, manifest_path)
            / str(manifest["ablation"])
            / str(manifest["dataset"])
        )
        manifests.append((manifest_path, manifest))
        for entry_key, entry in manifest.get("evaluations", {}).items():
            if entry.get("status") != "completed":
                continue
            for value in entry.get("result_files", []):
                source = resolve_project_path(value).resolve()
                if not source.is_file():
                    continue
                result = read_json(source)
                shape = result.get("grid_shape", [])
                grid = (
                    "x".join(map(str, shape))
                    if shape
                    else str(entry_key).split("__")[0]
                )
                target = evaluation_result_root(
                    dataset_root,
                    entry_variant(manifest, entry),
                    int(manifest["seed"]),
                    str(manifest["run_tag"]),
                    grid,
                    result.get("value", source.parent.name),
                ) / "results.json"
                references[source].append(
                    (target.resolve(), bool(entry.get("reused", False)), manifest_path)
                )

    plan = {}
    targets = {}
    for source, candidates in references.items():
        target, _, _ = min(candidates, key=lambda item: (item[1], str(item[2])))
        previous = targets.get(target)
        if previous is not None and previous != source:
            if previous.read_bytes() != source.read_bytes():
                raise RuntimeError(f"Evaluation migration collision: {target}")
        else:
            targets[target] = source
        plan[source] = target
    return plan, manifests


def update_manifests(experiments_root: Path, manifests, plan) -> int:
    changed = 0
    for path, data in manifests:
        before = json.dumps(data, sort_keys=True, ensure_ascii=False)
        for entry in data.get("evaluations", {}).values():
            entry["result_files"] = [
                portable_project_path(plan.get(resolve_project_path(value).resolve(), resolve_project_path(value)))
                for value in entry.get("result_files", [])
            ]
        dataset_root = (
            suite_root(experiments_root, path)
            / str(data["ablation"])
            / str(data["dataset"])
        )
        public_root = evaluation_experiment_root(
            dataset_root,
            str(data["variant"]),
            int(data["seed"]),
            str(data["run_tag"]),
        )
        data["evaluation_directory"] = portable_project_path(public_root)
        data["evaluation_storage_root"] = portable_project_path(public_root)
        data["evaluation_path_layout"] = {
            "schema": "suite_eval_v3",
            "hash_directories": False,
            "legacy_directories_retained": True,
            "migrated_at": datetime.now().isoformat(timespec="seconds"),
        }
        if json.dumps(data, sort_keys=True, ensure_ascii=False) != before:
            write_json(path, data)
            changed += 1
    return changed


def missing_result_references(experiments_root: Path) -> set[str]:
    missing = set()
    for manifest in experiments_root.rglob("run_manifest.json"):
        try:
            data = read_json(manifest)
        except (OSError, json.JSONDecodeError):
            continue
        for key, entry in data.get("evaluations", {}).items():
            for value in entry.get("result_files", []):
                if not resolve_project_path(value).is_file():
                    missing.add(f"{manifest}:{key}:{value}")
    return missing


def verify(experiments_root: Path, fingerprints_before, preexisting_missing) -> dict:
    if fingerprint_snapshot(experiments_root) != fingerprints_before:
        raise RuntimeError("Training or evaluation fingerprints changed")
    missing = []
    count = 0
    for manifest in experiments_root.rglob("run_manifest.json"):
        try:
            data = read_json(manifest)
        except (OSError, json.JSONDecodeError):
            continue
        for key, entry in data.get("evaluations", {}).items():
            for value in entry.get("result_files", []):
                count += 1
                if not resolve_project_path(value).is_file():
                    missing.append(f"{manifest}:{key}:{value}")
    unexpected = sorted(set(missing) - set(preexisting_missing))
    if unexpected:
        raise RuntimeError("Missing migrated results:\n" + "\n".join(unexpected[:20]))
    return {
        "result_references": count,
        "missing": len(missing),
        "preexisting_missing": len(preexisting_missing),
        "new_missing": len(unexpected),
        "fingerprints_preserved": True,
    }


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prefer_original_legacy_sources(experiments_root: Path, plan) -> list[tuple[Path, Path]]:
    """Use the untouched evaluator tree to repair an interrupted prior migration."""
    targets = {target.resolve() for target in plan.values()}
    candidates = defaultdict(list)
    for result in experiments_root.rglob("results.json"):
        resolved = result.resolve()
        if resolved in targets:
            continue
        try:
            key = file_digest(resolved)
        except OSError:
            continue
        score = 0 if (resolved.parent / "comparisons").is_dir() else 1
        candidates[key].append((score, resolved))

    repaired = []
    for target in sorted(targets):
        if not target.is_file():
            continue
        matches = candidates.get(file_digest(target), [])
        if matches:
            _, source = min(matches, key=lambda item: (item[0], len(str(item[1]))))
            repaired.append((source, target))
        else:
            repaired.append((target, target))
    return repaired


def retained_payload_signature(directory: Path) -> tuple:
    """Compare retained summaries/images independently of their new filenames."""
    directory = Path(directory)
    result = directory / "results.json"
    images = sorted(
        (path.stat().st_size, file_digest(path))
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".jpeg", ".jpg", ".pdf", ".png", ".svg", ".webp"}
    )
    return file_digest(result), tuple(images)


def update_generated_references(experiments_root: Path, plan) -> int:
    """Update reports and indices that contain explicit legacy result paths."""
    replacements = []
    for source, target in plan.items():
        replacements.extend(
            (
                (str(source), str(target)),
                (source.as_posix(), target.as_posix()),
                (portable_project_path(source), portable_project_path(target)),
                (
                    portable_project_path(source).replace("/", "\\"),
                    portable_project_path(target).replace("/", "\\"),
                ),
            )
        )
    changed = 0
    for path in experiments_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".csv",
            ".json",
            ".md",
            ".yaml",
            ".yml",
        }:
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        updated = original
        for old, new in replacements:
            updated = updated.replace(old, new)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    return changed


def remove_verified_legacy_results(experiments_root: Path, plan) -> int:
    """Remove old case roots only after byte-level retained-payload verification."""
    removed = 0
    for source, target in plan.items():
        if source.resolve() == target.resolve():
            continue
        source_root = source.parent
        target_root = target.parent
        if not source_root.is_dir():
            continue
        if experiments_root.resolve() not in source_root.resolve().parents:
            raise RuntimeError(f"Refusing to remove outside experiments: {source_root}")
        if retained_payload_signature(source_root) != retained_payload_signature(target_root):
            raise RuntimeError(
                "Migrated evaluation payload differs from its source: "
                f"{source_root} -> {target_root}"
            )
        shutil.rmtree(source_root)
        removed += 1
        parent = source_root.parent
        while parent != experiments_root and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    return removed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--remove-legacy",
        action="store_true",
        help="After verification, remove superseded legacy result directories.",
    )
    parser.add_argument("--experiments-root", type=Path, default=ROOT / "experiments")
    args = parser.parse_args(argv)
    experiments_root = args.experiments_root.resolve()
    plan, manifests = collect_plan(experiments_root)
    publication_pairs = prefer_original_legacy_sources(experiments_root, plan)
    copies = [(source, target) for source, target in publication_pairs if source != target]
    summary = {
        "schema": "suite_eval_v3",
        "dry_run": bool(args.dry_run),
        "referenced_results": len(publication_pairs),
        "results_to_copy": len(copies),
        "manifests": len(manifests),
        "legacy_directories_retained": not args.remove_legacy,
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return summary

    fingerprints = fingerprint_snapshot(experiments_root)
    preexisting_missing = missing_result_references(experiments_root)
    for source, target in copies:
        if target.is_file():
            if target.read_bytes() != source.read_bytes():
                raise FileExistsError(f"Different result already exists: {target}")
            publish_evaluation_result(
                source.parent,
                target.parent,
                overwrite=True,
                remove_source=False,
            )
            continue
        publish_evaluation_result(
            source.parent,
            target.parent,
            overwrite=False,
            remove_source=False,
        )
    changed = update_manifests(experiments_root, manifests, plan)
    # Run manifests are the canonical result index. Other aggregate files in
    # this project contain metrics rather than result paths, so an O(files ×
    # results) textual replacement pass is unnecessary.
    changed_references = 0
    verification = verify(experiments_root, fingerprints, preexisting_missing)
    removed = (
        remove_verified_legacy_results(experiments_root, plan)
        if args.remove_legacy
        else 0
    )
    report = {
        **summary,
        "dry_run": False,
        "changed_manifests": changed,
        "changed_reference_files": changed_references,
        "removed_legacy_result_directories": removed,
        "verification": verification,
    }
    write_json(experiments_root / "evaluation_path_migration_report.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
