"""Regression tests for concise batch completion status reporting."""

import json
from pathlib import Path

from src.utils.run_status import ablation_process_status


def _record(root: Path, *, status="completed"):
    return {
        "task": "e1__sample__e1_1__seed_7",
        "status": status,
        "metadata": {
            "dataset": "sample",
            "ablation": "ex1",
            "experiment": "e1_1",
            "seed": 7,
            "manifest_path": str(root / "run_manifest.json"),
            "failure_path": str(root / "failure.json"),
            "train_requested": True,
            "evaluation_requested": True,
        },
    }


def test_completed_ablation_process_reports_training_and_all_evaluations(tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "training_summary": "training/training_summary.json",
                "training_reused": True,
                "evaluations": {
                    "101x101": {"status": "completed", "reused": True},
                    "201x201": {"status": "completed", "reused": True},
                },
            }
        ),
        encoding="utf-8",
    )
    row = ablation_process_status(_record(tmp_path))
    assert row["status"] == "completed"
    assert row["training"] == "reused"
    assert row["evaluation"] == "reused"


def test_failed_evaluation_keeps_completed_training_and_failure_reason(tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "training_summary": "training/training_summary.json",
                "evaluations": {"101x101": {"status": "completed"}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "failure.json").write_text(
        json.dumps({"error": "RuntimeError: evaluator failed"}), encoding="utf-8"
    )
    row = ablation_process_status(_record(tmp_path, status="failed"))
    assert row["training"] == "completed"
    assert row["evaluation"] == "failed (1/1 grids completed)"
    assert row["error"] == "RuntimeError: evaluator failed"


def test_current_failure_marker_overrides_a_preserved_old_completed_manifest(tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "training_summary": "training/training_summary.json",
                "evaluations": {"101x101": {"status": "completed"}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "failure.json").write_text(
        json.dumps({"error": "FileNotFoundError: fresh evaluation failed"}),
        encoding="utf-8",
    )
    row = ablation_process_status(_record(tmp_path))
    assert row["status"] == "failed"
    assert row["training"] == "completed"
    assert row["evaluation"].startswith("failed")


def test_dry_run_never_reports_old_manifest_as_an_executed_task(tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"status": "completed", "training_summary": "old.json"}),
        encoding="utf-8",
    )
    row = ablation_process_status(_record(tmp_path, status="dry_run"))
    assert row["training"] == "dry_run"
    assert row["evaluation"] == "dry_run"
