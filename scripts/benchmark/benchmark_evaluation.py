"""Compare ordinary full forward and batch throughput without setup adapters.

All ordinary variants run interleaved after equal warmups. The baseline source
argument accepts an archived model.py with checkpoint-compatible parameters.
"""

import argparse
import copy
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.geometry.boundary_generator_2d import circle_airfoil_boundary
from src.data.geometry.boundary_generator_3d import cylindrical_domain_with_airfoil_hole
from src.data.generation.make_dataset_parametric_2d import to_computational_boundary
from src.data.mesh.sampling import logical_grid, sample_boundary
from src.evaluation.inference import load_network, predict_batch
from src.models.model import HMONet3D, MeshONet3D


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def interleaved(functions, device, warmups, repeats):
    for _ in range(warmups):
        for function in functions.values():
            function()
    synchronize(device)
    names = list(functions)
    samples = {name: [] for name in names}
    for iteration in range(repeats):
        shift = iteration % len(names)
        for name in names[shift:] + names[:shift]:
            synchronize(device)
            start = time.perf_counter()
            functions[name]()
            synchronize(device)
            samples[name].append((time.perf_counter() - start) * 1000)
    return {
        name: dict(median_ms=float(np.median(values)), samples_ms=values)
        for name, values in samples.items()
    }


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", type=Path)
    parser.add_argument("--dataset", default="air_t_int_0_1000_21_20")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/benchmarks"
        / datetime.now().strftime("evaluation_speed_%Y%m%d_%H%M%S"),
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    baseline = None
    if args.baseline_model:
        spec = importlib.util.spec_from_file_location(
            "evaluation_baseline", args.baseline_model
        )
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
    report = dict(
        torch=torch.__version__,
        device=str(device),
        warmups=args.warmups,
        repeats=args.repeats,
        timing="synchronized wall time, resident inputs, interleaved variants",
        runs=[],
        batch_runs=[],
    )
    if device.type == "cuda":
        report["gpu"] = torch.cuda.get_device_name(device)
    for dimension in (2, 3):
        if dimension == 2:
            models = {}
            for name in ("HMONet", "MeshONet"):
                path = max(
                    (ROOT / "outputs/checkpoints/2D" / args.dataset).rglob("best.pt"),
                    key=lambda p: p.stat().st_mtime if name in p.parts else -1,
                )
                model, shape, metadata = load_network(path, name, 2, "current", device)
                models[name] = model
                report.setdefault("checkpoints", {})[name] = metadata
            geometry = to_computational_boundary(circle_airfoil_boundary(t=0.3))
            grids = ((21, 20), (51, 50), (101, 100))
        else:
            shape = (11, 11, 11)
            common = dict(
                branch_dim=64,
                trunk_dim=24,
                adjust_hidden_dim=128,
                branch_hidden_dim=128,
                trunk_hidden_dim=128,
                output_dim=128,
                adjust_hidden_layer_num=4,
                branch_hidden_layer_num=4,
                trunk_hidden_layer_num=4,
            )
            torch.manual_seed(51)
            models = {
                "HMONet": HMONet3D(*shape, **common).to(device).eval(),
                "MeshONet": MeshONet3D(*shape, **common).to(device).eval(),
            }
            geometry = cylindrical_domain_with_airfoil_hole(thickness=0.3)
            grids = ((11, 11, 11), (21, 21, 21))
        h, m = models["HMONet"], models["MeshONet"]
        old = copy.deepcopy(h)
        if baseline:
            old.__class__ = getattr(baseline, f"HMONet{dimension}D")
        for grid in grids:
            q = logical_grid(grid)
            b = torch.as_tensor(
                sample_boundary(geometry, shape), dtype=torch.float32, device=device
            )
            query = torch.as_tensor(q, dtype=torch.float32, device=device)
            arguments = (*b.unbind(-1), *query.unbind(-1))
            calls = {
                "MeshONet": lambda: m(*arguments),
                "HMONet_baseline": lambda: old(*arguments, boundary_geometry=geometry),
                "HMONet_direct": lambda: h(*arguments, boundary_geometry=geometry),
            }
            expected = torch.cat(calls["HMONet_baseline"](), -1)
            actual = torch.cat(calls["HMONet_direct"](), -1)
            error = float((expected - actual).abs().max())
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
            measurements = interleaved(calls, device, args.warmups, args.repeats)
            row = dict(
                dimension=dimension,
                shape=grid,
                fixture=(
                    "actual_air_t_weights" if dimension == 2 else "synthetic_3D_weights"
                ),
                max_output_difference=error,
                measurements=measurements,
            )
            report["runs"].append(row)
            print(
                json.dumps(
                    dict(
                        dimension=dimension,
                        shape=grid,
                        medians={n: v["median_ms"] for n, v in measurements.items()},
                    )
                ),
                flush=True,
            )
            (args.output / "benchmark.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
        geometries = (
            [
                to_computational_boundary(circle_airfoil_boundary(t=t))
                for t in (0.15, 0.2, 0.25, 0.3)
            ]
            if dimension == 2
            else [cylindrical_domain_with_airfoil_hole(thickness=t) for t in (0.15, 0.2, 0.25, 0.3)]
        )
        for name, model in models.items():
            _, timing = predict_batch(
                model,
                name,
                geometries,
                shape,
                logical_grid(grids[0]),
                device,
                dict(
                    query_batch_size=50000,
                    inference_warmup_runs=args.warmups,
                    inference_timing_repeats=args.repeats,
                ),
            )
            report["batch_runs"].append(
                dict(dimension=dimension, method=name, **timing)
            )
        (args.output / "benchmark.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()

