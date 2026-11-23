"""Compare the frozen per-cap interpolation with batched run_e interpolation.

Uses the same physical surfaces, weights and queries for both implementations.
Training timing includes forward + normalized PDE loss + backward, excluding
optimizer updates, data loading and the periodic full-grid MSE diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from src.data.geometry import pde_cap_3d as cap_module
from src.data.geometry import pde_surface_3d as surface_module
from src.data.geometry.boundary_generator_3d import rounded_u_notch_extrusion_boundary
from src.data.backend.parametric_torch import evaluate_boundary_surfaces_torch
from src.data.mesh.sampling import boundary_indices, sample_faces
from src.losses.physics import elliptic_grid_scale_invariant_pde_loss
from src.models.model import HMONet3D
from src.numerics.differential import stencil_queries


def reference_loop(caps, u, v):
    """Original interpolation algebra, benchmark-only; never used by training."""
    values = []
    for index, cap in enumerate(caps):
        tu, tv, c = cap.tensors(u)
        iu, bu = cap_module._cubic_basis(u[index], tu)
        iv, bv = cap_module._cubic_basis(v[index], tv)
        local = c[iu[..., :, None], iv[..., None, :], :]
        values.append((local * bu[..., :, None, None] * bv[..., None, :, None]).sum((-3, -2)))
    return torch.stack(values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--train-repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/3D/run_e_pde_caps/batch_interpolation_timing.json")
    args = parser.parse_args(argv)
    if min(args.repeats, args.train_repeats) < 1 or args.warmup < 1:
        parser.error("repeats and warmup must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(20261030)
    device = torch.device(args.device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    boundaries = [rounded_u_notch_extrusion_boundary(depth=d) for d in (.55, 1., 1.5, 1.55)]
    shape = (61, 21, 11)
    config = yaml.safe_load((ROOT / "configs/3D/train.yaml").read_text(encoding="utf-8"))
    model = HMONet3D(*shape, **dict(config["model"]["common"], **config["model"]["hmonet"])).to(device)
    points = []
    for boundary in boundaries:
        _, grid = sample_faces(boundary, shape)
        points.append(grid.reshape(-1, 3)[boundary_indices(shape)])
    points = torch.tensor(np.stack(points), device=device, dtype=torch.float32)
    implementations = {"before_loop": reference_loop, "after_batch": surface_module._batch_spline_residual}
    result = dict(device=str(device), hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                  torch_version=torch.__version__, dtype="float32", training_grid=shape,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  depths=[.55, 1., 1.5, 1.55], batch_content="cycle four different PDE caps",
                  timing_scope="warm synchronous wall time; no optimizer, loading, full-grid MSE or cap construction",
                  measurements=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = [("projection", 64, 400), ("ccap", 1, 400), ("ccap", 64, 400),
             ("forward", 64, 400), ("forward", 1, 14091),
             ("pde_backward", 8, 400), ("pde_backward", 64, 400)]
    for mode, batch, count in cases:
        selection = torch.arange(batch, device=device) % len(boundaries)
        coordinates = points.index_select(0, selection).unbind(-1)
        geometries = [boundaries[i % len(boundaries)] for i in range(batch)]
        centers = .02 + .96 * torch.rand(batch, count, 3, device=device)
        training = mode == "pde_backward"
        queries = stencil_queries(centers, [.01]*3).flatten(1, 2) if training else centers
        raw = torch.rand(batch, queries.shape[1], 3, device=device)
        coefficients = torch.sigmoid(torch.rand(batch, queries.shape[1], 6, device=device))
        model.train(training)

        def operation():
            if mode == "projection":
                return evaluate_boundary_surfaces_torch(geometries, *queries.unbind(-1))
            if mode == "ccap":
                return model.evaluate_corner_corrected_points(
                    *coordinates, *queries.unbind(-1), raw, coefficients, boundary_geometry=geometries)
            prediction = torch.cat(model(*coordinates, *queries.unbind(-1), boundary_geometry=geometries), -1)
            if training:
                loss = elliptic_grid_scale_invariant_pde_loss(prediction.reshape(batch, count, 19, 3), [.01]*3)
                loss.backward()
                return prediction.detach(), loss.detach()
            return prediction

        # Compare complete outputs, loss and every network parameter gradient.
        checks = []
        for implementation in implementations.values():
            model.zero_grad(set_to_none=True)
            with patch.object(surface_module, "_batch_spline_residual", implementation), torch.set_grad_enabled(training):
                value = operation()
            outputs = value if training else (value,)
            gradients = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
            checks.append(([v.detach().clone() for v in outputs], gradients))
        output_error = max(float((a-b).abs().max()) for a, b in zip(checks[0][0], checks[1][0]))
        gradient_error = max((float((a-b).abs().max()) for a, b in zip(checks[0][1], checks[1][1])), default=0.)
        for expected, actual in zip(checks[0][0] + checks[0][1], checks[1][0] + checks[1][1]):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        del checks, outputs, gradients, value
        repeats = args.train_repeats if training else args.repeats
        samples = {key: [] for key in implementations}
        peak_memory = {}
        for repetition in range(-args.warmup, repeats):
            order = list(implementations) if repetition % 2 == 0 else list(reversed(implementations))
            for name in order:
                model.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                with patch.object(surface_module, "_batch_spline_residual", implementations[name]), torch.set_grad_enabled(training):
                    synchronize()
                    started = time.perf_counter()
                    value = operation()
                    synchronize()
                    elapsed = 1000 * (time.perf_counter() - started)
                if device.type == "cuda":
                    peak_memory[name] = max(peak_memory.get(name, 0), torch.cuda.max_memory_allocated(device))
                del value
                if repetition >= 0:
                    samples[name].append(elapsed)
        times = {k: dict(mean_ms=statistics.mean(v), median_ms=statistics.median(v),
                         min_ms=min(v), max_ms=max(v)) for k, v in samples.items()}
        record = dict(mode=mode, batch=batch, centers=count, actual_queries=queries.shape[1],
                      repeats=repeats, output_max_abs_error=output_error,
                      parameter_gradient_max_abs_error=gradient_error,
                      times=times, peak_allocated_bytes=peak_memory,
                      speedup=times["before_loop"]["mean_ms"]/times["after_batch"]["mean_ms"])
        result["measurements"].append(record)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
        model.zero_grad(set_to_none=True)
    return result


if __name__ == "__main__":
    main()
