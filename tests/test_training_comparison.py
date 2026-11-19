"""Regression checks for dataset-level training comparison artifacts."""

import csv
import json

from src.engine.training_comparison import ensure_training_comparison


def _run(root, model, epoch=10):
    path = root / "2D" / "sample" / model / "run"
    path.mkdir(parents=True)
    (path / "training_summary.json").write_text(json.dumps({
        "completed_epochs": epoch,
        "training_time_seconds": float(epoch),
        "best_epoch": epoch - 1,
        "best_model_dataset_mse": 1.0e-3,
        "final_dataset_mse": 1.2e-3,
    }), encoding="utf-8")
    with (path / "mse_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "dataset_mse"))
        writer.writerows(((1, 1.0e-2), (epoch, 1.0e-3)))


def test_training_comparison_waits_then_reuses(tmp_path):
    _run(tmp_path, "hmonet")
    assert ensure_training_comparison(tmp_path, 2, "sample")["status"] == "waiting_for_models"
    _run(tmp_path, "meshonet")
    _run(tmp_path, "pimeshonet")
    first = ensure_training_comparison(tmp_path, 2, "sample")
    assert first["status"] == "created"
    root = tmp_path / "2D" / "sample" / "comparison"
    assert (root / "mse_comparison.png").is_file()
    assert (root / "relative_mse_comparison.png").is_file()
    assert (root / "training_cost_comparison.png").is_file()
    assert ensure_training_comparison(tmp_path, 2, "sample")["status"] == "reused"
