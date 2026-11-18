"""Time comparable 3-D HMONet and PI-MeshONet physics iterations.

The HMONet reference path deliberately restores the former dense six-face edge
projection.  It is used only for a controlled before/after timing comparison;
production training always uses the sparse exact-edge evaluator.
"""

from __future__ import annotations

import argparse
import copy
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.models.model as hmonet_module
from src.data.backend.parametric_torch import evaluate_boundary_surfaces_torch
from src.data.mesh.sampling import boundary_indices, logical_grid
from src.data.parametric_dataset import ParametricDataset3D
from src.engine.training import predict_coordinates, safe_optimizer_step
from src.losses.physics import (
    elliptic_grid_pde_residual,
    elliptic_grid_scale_invariant_pde_loss,
    stencil_queries,
)
from src.models.model import HMONet3D
from src.models.pi_meshonet import PIMeshONet3D
from src.numerics.differential import _device_stencil_offsets
from src.utils.config import load_yaml_mapping


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _dense_selected_faces(boundaries, xi, eta, zeta, face_ids):
    """Former CCAP edge path: compute six faces then discard five."""
    values = evaluate_boundary_surfaces_torch(boundaries, xi, eta, zeta)
    face_ids = torch.as_tensor(face_ids, device=xi.device, dtype=torch.long)
    if face_ids.ndim == 1:
        face_ids = face_ids[None].expand(xi.shape[0], -1)
    return values.gather(
        2, face_ids[..., None, None].expand(-1, -1, 1, 3)
    ).squeeze(2)


def _timed(label, step, device, warmups, repeats):
    for _ in range(warmups):
        step()
    _sync(device)
    times = []
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        step()
        _sync(device)
        times.append((time.perf_counter() - started) * 1000.0)
    print(
        f"{label}: mean={statistics.fmean(times):.2f} ms, "
        f"std={statistics.stdev(times) if len(times) > 1 else 0.0:.2f} ms, "
        f"samples={[round(value, 2) for value in times]}",
        flush=True,
    )
    return statistics.fmean(times)


def _model_options(config, name):
    options = {
        **config["model"].get("common", {}),
        **config["model"].get(name, {}),
    }
    options.pop("architecture", None)
    return options


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/3D/train.yaml")
    parser.add_argument("--dataset", default="swb_a_ext_1_500_11_21_11")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--points", type=int, default=400)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config = load_yaml_mapping(ROOT / args.config)
    dataset = ParametricDataset3D(ROOT / "data" / "raw" / "3D" / args.dataset)
    size = min(args.batch_size, len(dataset))
    shape = tuple(dataset.grid_shape)
    boundary = torch.as_tensor(dataset.boundary_points[:size], dtype=torch.float32, device=device)
    geometries = dataset.boundary[:size]
    steps = torch.as_tensor(
        config["hmonet_training"]["pde_fd_step"], dtype=torch.float32, device=device
    )
    torch.manual_seed(args.seed)
    centers = steps + torch.rand(size, args.points, 3, device=device) * (1.0 - 2.0 * steps)
    stencil = stencil_queries(centers, steps)

    hmonet = HMONet3D(*shape, **_model_options(config, "hmonet")).to(device).train()
    h_optimizer = torch.optim.Adam(hmonet.parameters(), lr=config["hmonet_training"]["learning_rate"])

    def hmonet_step():
        values = predict_coordinates(
            hmonet, boundary, stencil.flatten(-3, -2), geometries
        ).reshape(*stencil.shape)
        loss = elliptic_grid_scale_invariant_pde_loss(values, steps)
        safe_optimizer_step(loss, hmonet, h_optimizer, 1.0, False)

    original_selected = hmonet_module.evaluate_selected_boundary_surfaces_torch
    try:
        hmonet_module.evaluate_selected_boundary_surfaces_torch = _dense_selected_faces
        dense = _timed("HMONet dense-edge reference", hmonet_step, device, args.warmups, args.repeats)
    finally:
        hmonet_module.evaluate_selected_boundary_surfaces_torch = original_selected
    sparse = _timed("HMONet optimized sparse-edge", hmonet_step, device, args.warmups, args.repeats)

    pi = config["pimeshonet_training"]
    pimeshonet = PIMeshONet3D(*shape, **_model_options(config, "pimeshonet")).to(device).train()
    pi_optimizer = torch.optim.Adam(pimeshonet.parameters(), lr=pi["learning_rate"])
    boundary_query = torch.as_tensor(
        logical_grid(shape)[boundary_indices(shape)], dtype=torch.float32, device=device
    )
    pi_steps = torch.as_tensor(pi["fd_step"], dtype=torch.float32, device=device)
    torch.manual_seed(args.seed)
    pi_centers = pi_steps + torch.rand(args.points, 3, device=device) * (1.0 - 2.0 * pi_steps)
    pi_stencil = stencil_queries(pi_centers, pi_steps)
    pi_query = torch.cat((boundary_query, pi_stencil.reshape(-1, 3)), dim=0)[None].expand(size, -1, -1)

    def pimeshonet_step():
        prediction = predict_coordinates(pimeshonet, boundary, pi_query)
        boundary_loss = (prediction[:, :len(boundary_query)] - boundary).square().sum(-1).mean()
        values = prediction[:, len(boundary_query):].reshape(size, args.points, pi_stencil.shape[-2], 3)
        pde_loss = elliptic_grid_pde_residual(values, pi_steps).square().sum(-1).mean()
        loss = float(pi["pde_loss_weight"]) * pde_loss + float(pi["boundary_loss_weight"]) * boundary_loss
        safe_optimizer_step(loss, pimeshonet, pi_optimizer, 1.0, False)

    # The existing PI path already flattens all 19-point stencils into one
    # forward.  Its only new shared optimization is the cached tiny template.
    # PI-MeshONet already uses the fully flattened batch route. Its stencil is
    # built before the timed forward/backward section, so device-side template
    # caching has no meaningful per-iteration effect. We still time the same
    # training iteration for a fair HMONet/PI-MeshONet comparison.
    _device_stencil_offsets.cache_clear()
    pi_reference = _timed("PI-MeshONet batched stencil", pimeshonet_step, device, args.warmups, args.repeats)

    print(
        "\nSummary: "
        f"HMONet {dense:.2f} -> {sparse:.2f} ms "
        f"({(dense - sparse) / dense * 100.0:.1f}% lower); "
        f"PI-MeshONet batched stencil={pi_reference:.2f} ms.",
        flush=True,
    )


if __name__ == "__main__":
    main()
