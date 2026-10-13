"""Migrate completed ablation runs to failure-plateau identity schema v3.

Only metadata belonging to complete, loadable training runs is changed. Model
weights and optimizer results remain byte-for-byte untouched.
"""

from __future__ import annotations

import csv
import copy
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import run_ablation_2d as ab
try:
    from . import migrate_ablation_metadata as legacy
except ImportError:
    import migrate_ablation_metadata as legacy


FIELDS = (
    "failure_plateau_enabled",
    "failure_plateau_start_epoch",
    "failure_plateau_window",
    "failure_plateau_patience",
    "failure_plateau_metric_ratio",
    "failure_plateau_relative_fluctuation",
    "failure_plateau_min_lr_updates",
    "save_best_immediately",
    "stop_file_enabled",
)
SCHEMA = "failure_plateau_identity_v3"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_csv(path, summary):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(summary.keys())
        writer.writerow(summary.values())


def update_config_file(path, values):
    path = Path(path)
    if not path.is_file():
        return False
    if path.suffix.lower() == ".json":
        config = read_json(path)
        config.setdefault("training", {}).update(values)
        write_json(path, config)
    else:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config.setdefault("training", {}).update(values)
        path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return True


def config_candidates(run_dir, summary):
    candidates = {
        run_dir / "training/config.json",
        run_dir / "training/config.yaml",
        run_dir / "checkpoints/config.json",
        run_dir / "checkpoints/config.yaml",
        run_dir / "effective_config/train.yaml",
        run_dir / "resolved_config.yaml",
    }
    for key in ("training_output_dir", "checkpoint_dir"):
        value = summary.get(key)
        if value:
            directory = Path(value)
            candidates.update(
                {
                    directory / "config.json",
                    directory / "config.yaml",
                    directory / "train.yaml",
                }
            )
    config_json = summary.get("config_json")
    config_yaml = summary.get("config_yaml")
    if config_json:
        candidates.add(Path(config_json))
    if config_yaml:
        candidates.add(Path(config_yaml))
    return sorted(candidates, key=str)


def update_summary_files(summary_path, summary, values, migrated_at):
    audit = {
        "schema": SCHEMA,
        "migrated_at": migrated_at,
        "reason": "preserve completed-run reuse after adding failure-only stopping controls",
        "weights_modified": False,
        "original_stop_reason": summary.get("stop_reason"),
        "original_completed_epochs": summary.get("completed_epochs"),
    }
    summary.update(values)
    summary["failure_plateau_metadata_migration"] = audit
    locations = {Path(summary_path)}
    checkpoint_dir = summary.get("checkpoint_dir")
    if checkpoint_dir:
        locations.add(Path(checkpoint_dir) / "training_summary.json")
    output_dir = summary.get("training_output_dir")
    if output_dir:
        locations.add(Path(output_dir) / "training_summary.json")
    written = []
    for path in sorted(locations, key=str):
        if path.is_file() or path == Path(summary_path):
            write_json(path, summary)
            write_summary_csv(path.with_suffix(".csv"), summary)
            written.append(str(path))
    return written


def migrate_manifest(manifest_path, variants, evaluate_config):
    manifest = read_json(manifest_path)
    experiment = str(manifest.get("ablation", ""))
    dataset_name = str(manifest.get("dataset", ""))
    variant_name = str(manifest.get("variant", ""))
    try:
        seed = int(manifest["seed"])
    except (KeyError, TypeError, ValueError):
        return {"status": "skipped", "reason": "invalid_seed", "manifest": str(manifest_path)}
    variant = variants.get(variant_name)
    if variant is None:
        return {"status": "skipped", "reason": "unknown_variant", "manifest": str(manifest_path)}

    summary_path = ab._training_summary_path(manifest_path, manifest)
    if not summary_path.is_file():
        return {"status": "skipped", "reason": "missing_summary", "manifest": str(manifest_path)}
    summary = read_json(summary_path)
    if not ab._completed_training_summary(summary):
        return {"status": "skipped", "reason": "incomplete_training", "manifest": str(manifest_path)}

    dataset = ab.resolve_dataset(dataset_name)
    args = SimpleNamespace(
        training_config=ab.DEFAULT_BASELINE_CONFIG,
        train_config=None,
        device=None,
        evaluate_config=evaluate_config,
    )
    fingerprint, identity = ab.effective_training_identity(args, dataset, variant, seed)
    expected = identity["effective_training_config"]
    values = {key: copy.deepcopy(expected["training"][key]) for key in FIELDS}

    saved, saved_path = legacy.saved_effective_config(manifest_path.parent, summary_path)
    if saved is None:
        return {"status": "skipped", "reason": "missing_saved_config", "manifest": str(manifest_path)}
    actual = legacy.normalized_saved_config(saved, dataset)
    actual.setdefault("training", {}).update(values)
    differences = ab._identity_differences(expected, actual)
    if differences:
        return {
            "status": "skipped",
            "reason": "non_migration_parameter_mismatch",
            "manifest": str(manifest_path),
            "saved_config": str(saved_path),
            "differences": differences,
        }

    migrated_at = datetime.now().isoformat(timespec="seconds")
    run_dir = manifest_path.parent
    updated_configs = [
        str(path)
        for path in config_candidates(run_dir, summary)
        if update_config_file(path, values)
    ]
    summary_files = update_summary_files(summary_path, summary, values, migrated_at)

    checkpoint_metadata_path = Path(summary["checkpoint_dir"]) / "checkpoint_metadata.json"
    if checkpoint_metadata_path.is_file():
        checkpoint_metadata = read_json(checkpoint_metadata_path)
        checkpoint_metadata["training_control"] = values
        checkpoint_metadata["metadata_migration"] = {
            "schema": SCHEMA,
            "migrated_at": migrated_at,
            "weights_modified": False,
        }
        write_json(checkpoint_metadata_path, checkpoint_metadata)

    ab.write_json(run_dir / "training_identity.json", identity)
    ab._write_resolved_training_config(run_dir / "resolved_config.yaml", identity)
    migration = {
        "schema": SCHEMA,
        "migrated_at": migrated_at,
        "weights_modified": False,
        "original_training_provenance_preserved": True,
        "added_training_fields": values,
    }
    prior = manifest.get("migration")
    if prior:
        migration["previous_migration"] = prior
    manifest.update(
        training_fingerprint=fingerprint,
        training_identity_file=str(run_dir / "training_identity.json"),
        training_identity_schema_version=ab.TRAINING_IDENTITY_SCHEMA_VERSION,
        migration=migration,
    )

    result_files = legacy.evaluation_files(
        experiment,
        dataset_name,
        variant_name,
        seed,
        str(manifest.get("run_tag", run_dir.name)),
    )
    if result_files:
        evaluation_fingerprint, evaluation_identity = ab.effective_evaluation_identity(
            args, dataset, summary, fingerprint
        )
        ab.write_json(run_dir / "evaluation_identity.json", evaluation_identity)
        manifest.update(
            evaluation_fingerprint=evaluation_fingerprint,
            evaluation_identity_file=str(run_dir / "evaluation_identity.json"),
            evaluation_result_files=[str(path) for path in result_files],
        )

    for marker_name in ("training_completed.json", "completed.json"):
        marker = run_dir / marker_name
        if marker.is_file():
            value = read_json(marker)
            value["metadata_migration"] = migration
            write_json(marker, value)
    write_json(manifest_path, manifest)
    return {
        "status": "migrated",
        "manifest": str(manifest_path),
        "training_fingerprint": fingerprint,
        "updated_configs": updated_configs,
        "updated_summaries": summary_files,
        "evaluation_count": len(result_files),
    }


def main():
    evaluate_config = ab.DEFAULT_EVALUATE_CONFIG.resolve()
    records = []
    for experiment in (ab.BASELINE_ABLATION,) + ab.ABLATIONS:
        root = ROOT / "experiments" / experiment
        if not root.is_dir():
            continue
        variants = legacy.load_variants(experiment)
        for manifest_path in sorted(root.rglob("run_manifest.json")):
            records.append(migrate_manifest(manifest_path, variants, evaluate_config))
    summary = {
        "schema": SCHEMA,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "migrated": sum(item["status"] == "migrated" for item in records),
        "skipped": sum(item["status"] != "migrated" for item in records),
        "skipped_by_reason": {},
        "records": records,
    }
    for item in records:
        if item["status"] != "migrated":
            reason = item["reason"]
            summary["skipped_by_reason"][reason] = summary["skipped_by_reason"].get(reason, 0) + 1
    output = ROOT / "experiments/failure_plateau_migration_report.json"
    write_json(output, summary)
    print(f"Migrated: {summary['migrated']}; skipped: {summary['skipped']}")
    print(json.dumps(summary["skipped_by_reason"], ensure_ascii=False, indent=2))
    print(output)


if __name__ == "__main__":
    main()

