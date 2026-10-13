"""Train Attention-PI-HMONet; 2-D uses PDE, boundary and alpha losses."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from scripts.train.attention_pihmonet_common import (
    autograd_physics_losses,
    autograd_normalized_pde_loss,
    build_boundary_inputs,
    finite_difference_physics_losses,
    finite_difference_normalized_pde_loss,
    predict_attention,
    sample_boundary_logical_points,
)
from scripts.train.pi_anchor import (
    central_anchor_queries,
    masked_anchor_mse,
    tfi_anchor_targets,
)
from scripts.train.training_common import make_data_loader, seed_everything
from src.data.dataset import ParametricDataset2D, ParametricDataset3D
from src.models.model import AttentionHMONet2D, AttentionHMONet3D


ROOT = Path(__file__).resolve().parents[2]
MODEL_KEY = "attention_pihmonet"
MODEL_FOLDER = "Attention-PI-HMONet"


def _anchor_weight(initial, iteration, decay_end):
    """Linearly decay the TFI continuation term to zero."""
    if initial <= 0 or iteration >= decay_end:
        return 0.0
    return initial * (decay_end - iteration) / max(decay_end - 1, 1)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _logical_grid(shape, device, dtype):
    axes = [torch.linspace(0, 1, n, device=device, dtype=dtype) for n in reversed(shape)]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(tuple(reversed(mesh)), -1).reshape(1, -1, len(shape))


@torch.no_grad()
def _dataset_mse(model, dataset, sample_count, boundary_counts, device):
    indices = np.linspace(0, len(dataset) - 1, min(sample_count, len(dataset)), dtype=int)
    logical = _logical_grid(dataset.grid_shape, device, torch.float32)
    squared, points = 0.0, 0
    model.eval()
    for index in indices:
        sample = dataset[int(index)]
        geometry = [sample["boundary"]]
        boundary = build_boundary_inputs(
            geometry, dataset.grid_shape, device, torch.float32, boundary_counts
        )
        prediction = predict_attention(model, boundary, logical, geometry)
        target = torch.as_tensor(sample["mesh_points"], device=device, dtype=torch.float32)[None]
        squared += float((prediction - target).square().sum())
        points += target.numel()
    model.train()
    return squared / max(points, 1)


def _save_checkpoint(path, model, optimizer, config, shape, iteration, history,
                     model_state_dict=None):
    torch.save(
        {
            "model_state_dict": model_state_dict or model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "grid": dict(zip(("nx", "ny", "nz"), shape)),
            "model_name": MODEL_KEY,
            "iteration": iteration,
            "history": history,
        },
        path,
    )


def train_dataset(dataset_name, config, dimension, overrides=None):
    """Train one stored dataset and return the generated artifact paths."""
    overrides = overrides or {}
    training = {**config["training"], **overrides}
    model_options = {
        **config.get("model", {}).get("common", {}),
        **config.get("model", {}).get(MODEL_KEY, {}),
    }
    pde = {**config.get("pde_loss", {})}
    loss_defaults = {
        "backend": "finite_difference",
        "interior_sample_count": 64 if dimension == 2 else 24,
        "interior_margin": 0.0,
        "fd_step": None,
        "normalized_pde_loss_weight": 1.0,
        "jacobian_loss_weight": 0.01,
        "jacobian_minimum": 1e-4,
        "anchor_loss_weight": 0.0,
        "anchor_decay_end_iteration": 1000,
        "anchor_points_per_axis": 3,
        "anchor_margin": 0.2,
        "smoothness_loss_weight": 1e-4,
        "boundary_loss_weight": 1.0,
        "alpha_penalty_weight": 0.01,
        "boundary_loss_sample_count": 64,
    }
    for key, value in loss_defaults.items():
        pde.setdefault(key, value)
    dataset_root = _resolve(training.get("data_root", f"data/raw/{dimension}D")) / dataset_name
    dataset_cls = ParametricDataset2D if dimension == 2 else ParametricDataset3D
    dataset = dataset_cls(dataset_root, load_boundary=True)
    shape = tuple(dataset.grid_shape)
    device = _device(training.get("device", "auto"))
    seed_everything(int(training["seed"]))
    model_cls = AttentionHMONet2D if dimension == 2 else AttentionHMONet3D
    model = model_cls(*shape, **model_options).to(device)
    if dimension == 2 and model.hmonet_mapping_mode != "current":
        raise ValueError("The 2-D boundary-alpha strategy requires current mapping")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    loader = make_data_loader(dataset if isinstance(dataset, list) else [dataset],
                              int(training["batch_size"]), int(training["seed"]), True)
    iterator = iter(loader)
    iterations = int(training["num_iterations"])
    backend = str(pde["backend"]).lower()
    if backend not in {"autograd", "finite_difference"}:
        raise ValueError("pde_loss.backend must be autograd or finite_difference")
    count = int(pde["interior_sample_count"])
    if count < 1:
        raise ValueError("interior_sample_count must be positive")
    steps = pde.get("fd_step") or [1 / (n - 1) for n in shape]
    if np.isscalar(steps):
        steps = [float(steps)] * dimension
    boundary_counts = config.get("boundary_encoder", {}).get("sample_counts")
    weights = {
        "pde": float(pde["normalized_pde_loss_weight"]),
        "jacobian": float(pde["jacobian_loss_weight"]),
        "anchor": float(pde["anchor_loss_weight"]),
        "smoothness": float(pde["smoothness_loss_weight"]),
        "boundary": float(pde["boundary_loss_weight"]),
        "alpha_penalty": float(pde["alpha_penalty_weight"]),
    }
    if dimension == 2:
        if any(weights[key] != 0.0 for key in ("jacobian", "anchor", "smoothness")):
            raise ValueError(
                "The 2-D boundary-alpha strategy requires Jacobian, anchor and "
                "smoothness weights to be zero"
            )
        weights.update(jacobian=0.0, anchor=0.0, smoothness=0.0)
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("all four loss weights must be finite and nonnegative")
    if weights["pde"] <= 0:
        raise ValueError("normalized_pde_loss_weight must be positive")
    jacobian_minimum = float(pde["jacobian_minimum"])
    anchor_end = int(pde["anchor_decay_end_iteration"])
    anchor_count = int(pde["anchor_points_per_axis"])
    anchor_margin = float(pde["anchor_margin"])
    anchor_queries = central_anchor_queries(
        dimension, anchor_count, anchor_margin,
        device=device, dtype=torch.float32,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = _resolve(training.get("output_root", "outputs"))
    relative = Path(f"{dimension}D") / dataset_name / MODEL_FOLDER / "current" / timestamp
    training_dir = base / "training" / relative
    checkpoint_dir = base / "checkpoints" / relative
    training_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    effective = {
        "dimension": dimension,
        "dataset": dataset_name,
        "grid_shape": list(shape),
        "model": {"name": MODEL_KEY, **model_options},
        "boundary_encoder": {"sample_counts": boundary_counts},
        "pde_loss": {**pde, "backend": backend, "fd_step": list(map(float, steps))},
        "training": training,
    }
    for directory in (training_dir, checkpoint_dir):
        (directory / "config.yaml").write_text(
            yaml.safe_dump(effective, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        (directory / "config.json").write_text(json.dumps(effective, indent=2), encoding="utf-8")
    history, best, best_iteration, best_state = [], float("inf"), 0, None
    started = time.perf_counter()
    model.train()
    for iteration in range(1, iterations + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        geometries = batch["boundary"]
        batch_size = len(geometries)
        boundary_inputs = build_boundary_inputs(
            geometries, shape, device, torch.float32, boundary_counts
        )
        optimizer.zero_grad(set_to_none=True)
        if backend == "autograd":
            margin = float(pde["interior_margin"])
            if margin != 0.0:
                raise ValueError("autograd sampling requires interior_margin=0")
            margin = torch.finfo(torch.float32).eps
            logical = margin + (1 - 2 * margin) * torch.rand(
                batch_size, count, dimension, device=device, dtype=torch.float32
            )
            logical.requires_grad_(True)
            prediction = predict_attention(
                model, boundary_inputs, logical, geometries
            )
            if dimension == 2:
                pde_only = autograd_normalized_pde_loss(prediction, logical)
                zero = pde_only.new_zeros(())
                physics = {
                    "pde_normalized": pde_only,
                    "jacobian": zero,
                    "smoothness": zero,
                    "minimum_jacobian": zero.detach(),
                }
            else:
                physics = autograd_physics_losses(
                    prediction, logical, jacobian_minimum
                )
        else:
            if dimension == 2:
                pde_only = finite_difference_normalized_pde_loss(
                    model, boundary_inputs, geometries, batch_size, count,
                    steps, device, torch.float32,
                )
                zero = pde_only.new_zeros(())
                physics = {
                    "pde_normalized": pde_only,
                    "jacobian": zero,
                    "smoothness": zero,
                    "minimum_jacobian": zero.detach(),
                }
            else:
                physics = finite_difference_physics_losses(
                    model, boundary_inputs, geometries, batch_size, count,
                    steps, device, torch.float32, jacobian_minimum,
                )
        boundary_loss = physics["pde_normalized"].new_zeros(())
        alpha_penalty = boundary_loss
        if dimension == 2:
            boundary_logical = sample_boundary_logical_points(
                batch_size,
                int(pde["boundary_loss_sample_count"]),
                2,
                device,
                torch.float32,
                offset=0.0,
                tangential_margin=0.0,
            ).detach()
            boundary_prediction = predict_attention(
                model, boundary_inputs, boundary_logical, geometries
            )
            with torch.no_grad():
                boundary_target = model.evaluate_coons_points_from_edges(
                    *boundary_inputs,
                    boundary_logical[..., 0],
                    boundary_logical[..., 1],
                    boundary_geometry=geometries,
                )
            boundary_loss = (
                boundary_prediction - boundary_target
            ).square().sum(-1).mean()
            alpha_penalty = (model.boundary_alpha - 1.0).square()
        current_anchor_weight = _anchor_weight(
            weights["anchor"], iteration, anchor_end
        )
        anchor = physics["pde_normalized"].new_zeros(())
        valid_anchor_ratio = anchor.detach()
        if current_anchor_weight > 0:
            anchor_logical = anchor_queries[None].expand(batch_size, -1, -1)
            anchor_prediction = predict_attention(
                model, boundary_inputs, anchor_logical, geometries
            )
            with torch.no_grad():
                target, valid = tfi_anchor_targets(
                    batch["boundary_points"].to(device), anchor_logical, shape
                )
            anchor = masked_anchor_mse(anchor_prediction, target, valid)
            valid_anchor_ratio = valid.float().mean()
        if dimension == 2:
            loss = (
                weights["pde"] * physics["pde_normalized"]
                + weights["boundary"] * boundary_loss
                + weights["alpha_penalty"] * alpha_penalty
            )
        else:
            loss = (
                weights["pde"] * physics["pde_normalized"]
                + weights["jacobian"] * physics["jacobian"]
                + current_anchor_weight * anchor
                + weights["smoothness"] * physics["smoothness"]
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite PDE loss at iteration {iteration}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip_norm", 1.0)))
        optimizer.step()
        value = float(loss.detach())
        row = {
            "iteration": iteration,
            "loss": value,
            "loss_pde_normalized": float(physics["pde_normalized"].detach()),
            "loss_jacobian": float(physics["jacobian"].detach()),
            "loss_anchor": float(anchor.detach()),
            "anchor_weight": current_anchor_weight,
            "valid_anchor_ratio": float(valid_anchor_ratio),
            "loss_smoothness": float(physics["smoothness"].detach()),
            "loss_boundary": float(boundary_loss.detach()),
            "loss_alpha_penalty": float(alpha_penalty.detach()),
            "boundary_alpha": (
                float(model.boundary_alpha.detach()) if dimension == 2 else None
            ),
            "minimum_sampled_jacobian": float(physics["minimum_jacobian"]),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        mse_interval = int(training.get("dataset_mse_interval", 0))
        if mse_interval > 0 and (iteration % mse_interval == 0 or iteration == iterations):
            row["dataset_mse"] = _dataset_mse(
                model, dataset, int(training.get("dataset_mse_samples", 3)),
                boundary_counts, device,
            )
        history.append(row)
        if value < best:
            best, best_iteration = value, iteration
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
        interval = int(training.get("log_interval", 100))
        if iteration == 1 or iteration % interval == 0 or iteration == iterations:
            mse_text = f", mse={row['dataset_mse']:.6e}" if "dataset_mse" in row else ""
            if dimension == 2:
                print(
                    f"[{dataset_name}] iteration {iteration}/{iterations}: "
                    f"loss={value:.6e}, pde={row['loss_pde_normalized']:.6e}, "
                    f"boundary={row['loss_boundary']:.6e}, "
                    f"alpha={row['boundary_alpha']:.8f}, "
                    f"alpha_penalty={row['loss_alpha_penalty']:.6e}{mse_text}",
                    flush=True,
                )
            else:
                print(
                    f"[{dataset_name}] iteration {iteration}/{iterations}: "
                    f"loss={value:.6e}, pde={row['loss_pde_normalized']:.6e}, "
                    f"jac={row['loss_jacobian']:.6e}, anchor={row['loss_anchor']:.6e}, "
                    f"smooth={row['loss_smoothness']:.6e}{mse_text}", flush=True,
                )
    elapsed = time.perf_counter() - started
    _save_checkpoint(checkpoint_dir / "final.pt", model, optimizer,
                     effective, shape, iterations, history)
    torch.save(model.state_dict(), checkpoint_dir / "final_model.pth")
    _save_checkpoint(
        checkpoint_dir / "best.pt", model, optimizer, effective, shape,
        best_iteration, history, best_state,
    )
    torch.save(best_state, checkpoint_dir / "best_model.pth")
    fields = sorted({key for row in history for key in row})
    with (training_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(history)
    (training_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.semilogy([row["iteration"] for row in history], [row["loss"] for row in history], label="total")
    axis.semilogy([row["iteration"] for row in history], [row["loss_pde_normalized"] for row in history], label="normalized PDE")
    axis.set(xlabel="Iteration", ylabel="Loss", title=f"{MODEL_FOLDER}: {dataset_name}")
    axis.legend()
    axis.grid(True, alpha=0.25); figure.tight_layout()
    figure.savefig(training_dir / "loss.png", dpi=180); plt.close(figure)
    summary = {
        "model": MODEL_KEY, "dataset": dataset_name, "dimension": dimension,
        "grid_shape": list(shape), "iterations": iterations,
        "best_iteration": best_iteration, "best_loss": best,
        "final_loss": history[-1]["loss"],
        "final_normalized_pde_loss": history[-1]["loss_pde_normalized"],
        "final_jacobian_loss": history[-1]["loss_jacobian"],
        "final_anchor_loss": history[-1]["loss_anchor"],
        "final_smoothness_loss": history[-1]["loss_smoothness"],
        "final_boundary_loss": history[-1]["loss_boundary"],
        "final_boundary_alpha": history[-1]["boundary_alpha"],
        "final_alpha_penalty_loss": history[-1]["loss_alpha_penalty"],
        "training_seconds": elapsed,
        "seconds_per_iteration": elapsed / iterations,
        "pde_backend": backend, "checkpoint": str(checkpoint_dir / "best.pt"),
    }
    for directory in (training_dir, checkpoint_dir):
        (directory / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return {"training_dir": training_dir, "checkpoint_dir": checkpoint_dir,
            "checkpoint": checkpoint_dir / "best.pt", "summary": summary}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--pde-backend", choices=("autograd", "finite_difference"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    path = args.config or ROOT / f"configs/{args.dimension}D/attention_pihmonet_train.yaml"
    config = yaml.safe_load(_resolve(path).read_text(encoding="utf-8-sig"))
    datasets = args.dataset or config["training"].get("datasets", [])
    if not datasets:
        raise ValueError("Specify at least one dataset in YAML or with --dataset")
    overrides = {
        key: value for key, value in {
            "num_iterations": args.iterations, "batch_size": args.batch_size,
            "seed": args.seed, "device": args.device,
        }.items() if value is not None
    }
    if args.pde_backend:
        config.setdefault("pde_loss", {})["backend"] = args.pde_backend
    return [train_dataset(name, config, args.dimension, overrides) for name in datasets]


if __name__ == "__main__":
    main()
