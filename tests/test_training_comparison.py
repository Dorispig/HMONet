"""Regression checks for dataset-level training comparison artifacts."""

import csv
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.engine.state import (
    checkpoint_directory,
    model_specific_training_config,
    prepare_pimeshonet_training_config,
    prepare_training_config,
)
from src.engine.training_comparison import ensure_training_comparison


def _run(root, model, epoch=10, *, dimension=2, dataset="sample"):
    path = root / f"{dimension}D" / dataset / model / "run"
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


def test_comparison_cli_handles_multiple_datasets(tmp_path, capsys):
    from scripts.train.compare_training import main

    for model in ("hmonet", "meshonet", "PI-MeshONet"):
        _run(tmp_path, model, dimension=3, dataset="complete")
    args = ["--dimension", "3", "--training-root", str(tmp_path), "--dataset"]

    assert main([*args, "missing", "complete"]) == 1
    assert (tmp_path / "3D" / "complete" / "comparison" / "mse_comparison.png").is_file()
    output = capsys.readouterr().out
    assert "missing completed MSE history for MeshONet, PI-MeshONet, HMONet" in output
    assert "complete | created" in output
    assert main([*args, "complete"]) == 0
    assert "complete | reused" in capsys.readouterr().out


@pytest.mark.parametrize("dimension", [2, 3])
def test_formal_training_backfills_comparison_without_retraining(tmp_path, monkeypatch, dimension):
    module = importlib.import_module(f"scripts.train.train_{dimension}d")
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / f"{dimension}D" / "train.yaml").read_text(encoding="utf-8"))
    dataset = tmp_path / "sample_ext"
    dataset.mkdir()
    output = tmp_path / "outputs" / "training" / f"{dimension}D"
    checkpoint_root = tmp_path / "outputs" / "checkpoints"
    args = SimpleNamespace(output_root=str(output))
    shape = (5,) * dimension
    config["grid"] = dict(zip(("nx", "ny", "nz"), shape))
    monkeypatch.setattr(
        module, f"ParametricDataset{dimension}D",
        lambda *a, **kw: SimpleNamespace(root=dataset, grid_shape=shape),
    )
    monkeypatch.setattr(module, "print_training_summaries", lambda *a, **kw: None)

    def unexpected_training(*args, **kwargs):
        pytest.fail("A compatible completed run must not train again")

    names = ("hmonet", "meshonet", "pimeshonet")
    for name, directory in zip(names, ("hmonet/CCAP", "meshonet", "PI-MeshONet")):
        monkeypatch.setattr(module, "train_" + name, unexpected_training)
        run = output / dataset.name / directory / "run"
        run.mkdir(parents=True)
        effective = (
            prepare_pimeshonet_training_config(config, args, shape)
            if name == "pimeshonet" else prepare_training_config(config, name, args)
        )
        saved = model_specific_training_config(effective, name, dataset, "extrapolation")
        saved["training"]["device"] = "cuda:3"
        (run / "config.json").write_text(json.dumps(saved), encoding="utf-8")
        (run / "training_summary.json").write_text(json.dumps({
            "model": name,
            "dataset_name": dataset.name,
            "completed_epochs": 3,
            "stop_reason": "converged_near_best_plateau",
            "dataset_mse_epochs": [1, 2, 3],
            "dataset_mse_history": [0.1, 0.02, 0.003],
            "training_time_seconds": 10,
        }), encoding="utf-8")
        checkpoint = checkpoint_directory(
            SimpleNamespace(checkpoint_root=checkpoint_root, dataset_name=dataset.name),
            config, name, f"{dimension}D", "run",
        )
        checkpoint.mkdir(parents=True)
        for filename in ("best.pt", "final.pt"):
            (checkpoint / filename).write_bytes(b"checkpoint fixture")

    comparison = output / dataset.name / "comparison"
    for _ in range(2):
        summaries = module.train_models(names, [dataset], config, "extrapolation", args, "cpu")
        assert all(summary["training_reused"] for summary in summaries)
        assert all(Path(summary["best_checkpoint"]).is_file() for summary in summaries)
        assert (comparison / "mse_comparison.png").is_file()
        assert (comparison / "training_comparison.json").is_file()
        (comparison / "mse_comparison.png").unlink()
