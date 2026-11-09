"""Measure pre-clipping gradient norms for the formal 2-D HMONet objective.

The diagnostic follows the active ``random_loss_combo_from_start`` branch
without creating checkpoints, TensorBoard files, or experiment outputs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from scripts.train.train_2d import _random_loss_combo
from src.data.geometry.parametric_curve_2d import BoundaryCurveSet2D
from src.data.mesh.sampling import sample_rectangle_boundary_uniform
from src.data.parametric_dataset import ParametricDataset2D
from src.engine.training import make_data_loader, seed_everything
from src.models import HMONet2D
from src.utils.config import load_yaml_mapping
from src.utils.math import compute_discrete_boundary_curvature


ROOT = Path(__file__).resolve().parents[2]


def _dataset_path(value: str) -> Path:
    path = Path(value)
    if path.is_dir():
        return path.resolve()
    candidate = ROOT / "data" / "raw" / "2D" / value
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f"Dataset not found: {value}")


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        key: float(np.quantile(values, percentile))
        for key, percentile in (("p50", 0.50), ("p75", 0.75), ("p90", 0.90),
                                ("p95", 0.95), ("p98", 0.98), ("p99", 0.99))
    }


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/2D/train.yaml")
    parser.add_argument("--dataset", default="air_t_ext_1_500_21_21")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--state",
        type=Path,
        help="Optional resumable diagnostic state.  Repeated invocations continue the same trajectory.",
    )
    args = parser.parse_args(argv)
    if args.steps < 1:
        raise ValueError("--steps must be positive")

    config = load_yaml_mapping(args.config)
    training = config["training"]
    options = config["hmonet_training"]
    sampling = config["hmonet_sampling"]
    device = torch.device(args.device)
    seed_everything(int(training["seed"]))
    data = ParametricDataset2D(_dataset_path(args.dataset), load_boundary=True)
    shape = tuple(data.grid_shape)
    model_options = dict(config["model"].get("common", {}))
    model_options.update(config["model"].get("hmonet", {}))
    model_options.pop("residual_regularization_weight", None)
    model_options.pop("projection_softplus_enabled", None)
    model_options["trunk_projection_coefficients_enabled"] = True
    model = HMONet2D(*shape, **model_options).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(options["learning_rate"]))
    loader = make_data_loader([data], int(training["batch_size"]), training["seed"], shuffle=True)
    pde_steps = torch.as_tensor(options["pde_fd_step"], device=device, dtype=torch.float32)
    computational_boundary = torch.as_tensor(
        sample_rectangle_boundary_uniform(*shape)[0], dtype=torch.float32, device=device
    )
    norms: list[float] = []
    losses: list[float] = []
    completed_steps = 0

    def save_state() -> None:
        if not args.state:
            return
        args.state.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "norms": norms, "losses": losses, "completed_steps": len(norms),
                "torch_rng_state": torch.get_rng_state(), "numpy_rng_state": np.random.get_state(),
            },
            args.state,
        )
    if args.state and args.state.is_file():
        state = torch.load(args.state, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        norms = list(state["norms"])
        losses = list(state["losses"])
        completed_steps = int(state["completed_steps"])
        torch.set_rng_state(state["torch_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        print(f"resuming from step={completed_steps}", flush=True)
    iterator = iter(loader)
    # The loader has eight batches for the formal 500-case/64-batch setup.
    # Skip only the cyclic batch position; RNG restoration preserves the sampled points.
    for _ in range(completed_steps % len(loader)):
        next(iterator)
    started = time.perf_counter()

    for local_step in range(1, args.steps + 1):
        step = completed_steps + local_step
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        boundary = batch["boundary_points"].to(device)
        mesh = batch["mesh_points"].to(device)
        geometries = BoundaryCurveSet2D.batch(batch["boundary"], device=device, dtype=boundary.dtype)
        curvature = compute_discrete_boundary_curvature(
            boundary, quantile=float(sampling.get("curvature_quantile", 0.95))
        )
        loss, _ = _random_loss_combo(
            model=model, boundary=boundary, mesh=mesh, geometries=geometries,
            shape=shape, device=device, steps=pde_steps,
            sample_count=int(options["random_loss_sample_count"]),
            mse_weight=float(options.get("random_mse_loss_weight", 0.0)),
            pde_weight=float(options["physics_pde_loss_weight"]),
            jacobian_weight=float(options.get("physics_jacobian_loss_weight", 0.0)),
            jacobian_minimum=float(options.get("physics_jacobian_minimum", 1e-3)),
            winslow_weight=float(options.get("physics_winslow_loss_weight", 0.0)),
            winslow_barrier_weight=float(options.get("physics_winslow_jacobian_barrier_weight", 0.0)),
            winslow_minimum=float(options.get("physics_winslow_jacobian_minimum", 1e-4)),
            winslow_epsilon=float(options.get("physics_winslow_epsilon", 1e-6)),
            winslow_fd_step=float(options.get("jacobian_fd_step", 1e-4)),
            scale_invariant_pde=bool(options["physics_scale_invariant_pde_enabled"]),
            normalized_jacobian_weight=float(options["physics_normalized_jacobian_loss_weight"]),
            normalized_jacobian_minimum=float(options["physics_normalized_jacobian_minimum"]),
            singular_value_weight=float(options["physics_singular_value_loss_weight"]),
            singular_value_minimum=float(options["physics_singular_value_minimum"]),
            barrier_reduction=str(options["physics_barrier_reduction"]),
            derivative_backend=str(options["random_loss_derivative_backend"]),
            mixed_sampling=str(options["random_loss_sampling_mode"]) == "mixed_boundary",
            global_count=int(options["random_loss_global_sample_count"]),
            curvature_count=int(options["random_loss_curvature_sample_count"]),
            boundary_band_count=int(options["random_loss_boundary_band_sample_count"]),
            boundary_curvature=curvature,
            computational_boundary=computational_boundary,
            candidate_multiplier=int(sampling.get("candidate_multiplier", 10)),
            curvature_power=float(sampling.get("curvature_power", 1.0)),
            sigma_min=float(sampling.get("distance_sigma_min", 0.02)),
            sigma_max=float(sampling.get("distance_sigma_max", 0.10)),
            uniform_probability=float(sampling.get("uniform_probability", 0.05)),
            boundary_min_step_multiple=float(options.get("random_loss_boundary_band_min_step_multiple", 1.0)),
            boundary_max_step_multiple=float(options.get("random_loss_boundary_band_max_step_multiple", 4.0)),
            boundary_min_distance=float(options.get("random_loss_boundary_min_distance", 0.0)),
            boundary_max_distance=float(options.get("random_loss_boundary_max_distance", 4e-4)),
            fd_stencil_inside=bool(options["random_loss_fd_stencil_inside"]),
            fd_boundary_scheme=str(options["random_loss_fd_boundary_scheme"]),
            target_interpolation=str(options.get("target_interpolation", "bilinear")),
            point_loss_weighting=str(options.get("point_loss_weighting", "uniform")),
            idw_query_chunk_size=int(options.get("idw_query_chunk_size", 128)),
            idw_support_chunk_size=int(options.get("idw_support_chunk_size", 2048)),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        norms.append(float(raw_norm.detach().cpu()))
        losses.append(float(loss.detach().cpu()))
        # Match the formal training trajectory after recording the pre-clipping norm.
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
        optimizer.step()
        # Emit a heartbeat every 25 iterations so long CPU-only diagnostics
        # remain observable without changing the numerical trajectory.
        if step == 1 or step % 25 == 0 or local_step == args.steps:
            print(f"step={step} raw_grad_norm={norms[-1]:.4e} loss={losses[-1]:.4e}", flush=True)
            save_state()

    array = np.asarray(norms, dtype=float)
    clip = float(training["gradient_clip_norm"])
    report = {
        "config": str(Path(args.config).resolve()),
        "dataset": str(_dataset_path(args.dataset)),
        "device": str(device),
        "steps": int(len(norms)),
        "formal_clip_threshold": clip,
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "quantiles": _quantiles(array),
        "clip_activation_fraction": float(np.mean(array > clip)),
        "final_loss": losses[-1],
        "elapsed_seconds": time.perf_counter() - started,
        "seconds_per_step": (time.perf_counter() - started) / args.steps,
        "recommended_threshold": float(np.quantile(array, 0.95)),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text, flush=True)
    save_state()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
