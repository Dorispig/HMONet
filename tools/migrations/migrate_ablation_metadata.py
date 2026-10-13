"""Migrate completed 2-D ablations to frozen, path-independent identities."""

from __future__ import annotations

import argparse
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


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_variants(name: str):
    if name == ab.BASELINE_ABLATION:
        return {"baseline": ab.Variant("baseline", "baseline", ())}
    definition = ab._load_yaml_mapping(ab.default_experiment_config(name))
    return {item.name: item for item in ab.variants_from_definition(definition)}


def saved_effective_config(run_dir: Path, summary_path: Path):
    candidates = [
        run_dir / "training/config.json",
        run_dir / "checkpoints/config.yaml",
        summary_path.parent / "config.json",
        summary_path.parent.parent / "checkpoints/config.yaml",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix == ".json":
            return load_json(path), path
        return yaml.safe_load(path.read_text(encoding="utf-8")), path
    return None, None


def normalized_saved_config(config: dict, dataset: Path):
    value = json.loads(json.dumps(config))
    value["datasets"] = {ab.dataset_mode(ab.dataset_name(dataset)): ab.dataset_name(dataset)}
    return value


def evaluation_files(experiment: str, dataset: str, variant: str, seed: int, tag: str):
    root = ROOT / "experiments" / experiment / dataset / "_evaluation"
    if not root.is_dir():
        return []
    pattern = f"*/seed_{seed}/{variant}/{tag}/*/results.json"
    return sorted(root.glob(pattern))


def migrate_manifest(manifest_path: Path, variants: dict, evaluate_config: Path):
    manifest = load_json(manifest_path)
    experiment = str(manifest.get("ablation", ""))
    dataset_name = str(manifest.get("dataset", ""))
    variant_name = str(manifest.get("variant", ""))
    seed = int(manifest.get("seed"))
    tag = str(manifest.get("run_tag", manifest_path.parent.name))
    variant = variants.get(variant_name)
    if variant is None:
        return {"status": "skipped", "reason": "unknown_variant", "manifest": str(manifest_path)}
    dataset = ab.resolve_dataset(dataset_name)
    summary_path = ab._training_summary_path(manifest_path, manifest)
    if not summary_path.is_file():
        return {"status": "skipped", "reason": "missing_summary", "manifest": str(manifest_path)}
    summary = load_json(summary_path)
    if not ab._completed_training_summary(summary):
        return {"status": "skipped", "reason": "incomplete_training", "manifest": str(manifest_path)}
    args = SimpleNamespace(
        training_config=ab.DEFAULT_BASELINE_CONFIG,
        train_config=None,
        device=None,
        evaluate_config=evaluate_config,
    )
    fingerprint, identity = ab.effective_training_identity(args, dataset, variant, seed)
    saved, saved_path = saved_effective_config(manifest_path.parent, summary_path)
    if saved is None:
        return {"status": "skipped", "reason": "missing_saved_config", "manifest": str(manifest_path)}
    expected = identity["effective_training_config"]
    actual = normalized_saved_config(saved, dataset)
    differences = ab._identity_differences(expected, actual)
    if differences:
        return {
            "status": "skipped",
            "reason": "parameter_mismatch",
            "manifest": str(manifest_path),
            "saved_config": str(saved_path),
            "differences": differences,
        }
    run_dir = manifest_path.parent
    ab.write_json(run_dir / "training_identity.json", identity)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(expected, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    manifest.update(
        dataset_path=f"data/raw/2D/{dataset_name}",
        training_fingerprint=fingerprint,
        training_identity_file="training_identity.json",
        training_identity_schema_version=ab.TRAINING_IDENTITY_SCHEMA_VERSION,
        baseline_config="experiments/baseline/experiment.yaml",
        evaluation_config="experiments/baseline/evaluate.yaml",
        experiment_config=(
            "experiments/baseline/experiment.yaml"
            if experiment == ab.BASELINE_ABLATION
            else f"experiments/{experiment}/experiment.yaml"
        ),
        resume_policy="restart_incomplete",
        migration={
            "schema": "frozen_ablation_config_v1",
            "migrated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    command = manifest.get("train_command")
    if isinstance(command, list) and "--config" in command:
        index = command.index("--config") + 1
        if index < len(command):
            command[index] = "experiments/baseline/experiment.yaml"
    ab.write_json(
        run_dir / "training_completed.json",
        {
            "status": "training_completed",
            "training_reused": bool(manifest.get("training_reused", False)),
            "migrated": True,
        },
    )
    results = evaluation_files(experiment, dataset_name, variant_name, seed, tag)
    if results:
        evaluation_fingerprint, evaluation_identity = ab.effective_evaluation_identity(
            args, dataset, summary, fingerprint
        )
        ab.write_json(run_dir / "evaluation_identity.json", evaluation_identity)
        manifest.update(
            status="completed",
            evaluation_fingerprint=evaluation_fingerprint,
            evaluation_identity_file="evaluation_identity.json",
            evaluation_directory=(
                ROOT / "experiments" / experiment / dataset_name / "_evaluation"
            ).relative_to(ROOT).as_posix(),
            evaluation_result_files=[str(path) for path in results],
        )
        ab.write_json(
            run_dir / "completed.json",
            {"status": "completed", "training_reused": True, "evaluation_reused": True},
        )
    else:
        manifest["status"] = "training_completed"
        ab.write_json(
            run_dir / "training_completed.json",
            {"status": "training_completed", "training_reused": True},
        )
    ab.write_json(manifest_path, manifest)
    return {
        "status": "migrated",
        "manifest": str(manifest_path),
        "training_fingerprint": fingerprint,
        "evaluation_count": len(results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluate-config",
        type=Path,
        default=ab.DEFAULT_EVALUATE_CONFIG,
    )
    args = parser.parse_args()
    records = []
    for experiment in (ab.BASELINE_ABLATION,) + ab.ABLATIONS:
        root = ROOT / "experiments" / experiment
        if not root.is_dir():
            continue
        variants = load_variants(experiment)
        for manifest_path in sorted(root.rglob("run_manifest.json")):
            records.append(migrate_manifest(manifest_path, variants, args.evaluate_config.resolve()))
    summary = {
        "schema": "frozen_ablation_config_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "migrated": sum(item["status"] == "migrated" for item in records),
        "skipped": sum(item["status"] != "migrated" for item in records),
        "records": records,
    }
    output = ROOT / "experiments/config_migration_report.json"
    write_json(output, summary)
    print(f"Migrated: {summary['migrated']}; skipped: {summary['skipped']}")
    print(output)


if __name__ == "__main__":
    main()
