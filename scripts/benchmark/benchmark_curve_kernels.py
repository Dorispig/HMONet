"""Compare exact fused curve evaluation with the reference Torch formulas.

Uses actual 2-D checkpoints and same-size synthetic 3-D networks. Coons weights
are copied from current solely for timing, not to evaluate prediction quality.
"""

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.data.geometry.boundary_generator_2d import circle_airfoil_boundary
from src.data.geometry.boundary_generator_3d import cylindrical_domain_with_airfoil_hole
from src.data.backend.curve_kernels import curve_kernel_mode, kernel_status
from src.data.generation.make_dataset_parametric_2d import to_computational_boundary
from src.data.mesh.sampling import logical_grid, sample_boundary
from src.evaluation.inference import load_network, predict
from src.models.model import HMONet3D, MeshONet3D


def stats(values):
    return dict(
        median_ms=float(np.median(values)),
        mean_ms=float(np.mean(values)),
        samples_ms=values,
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/benchmarks"
        / ("curve_kernels_" + datetime.now().strftime("%Y%m%d_%H%M")),
    )
    parser.add_argument("--warmups", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    source = (
        ROOT
        / "outputs/results/2D/air_t_int_0_1000_21_20/51x50/0.3/evaluation_parameters.json"
    )
    parameters = json.loads(source.read_text("utf-8"))
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "runs": [],
    }
    for dimension in (2, 3):
        if dimension == 2:
            current, shape, _ = load_network(
                Path(parameters["checkpoints"]["HMONet"]),
                "HMONet",
                2,
                "current",
                device,
            )
            meshonet, _, _ = load_network(
                Path(parameters["checkpoints"]["MeshONet"]),
                "MeshONet",
                2,
                "current",
                device,
            )
            geometry = to_computational_boundary(
                circle_airfoil_boundary(**parameters["parameters"])
            )
            grid = (51, 50)
        else:
            shape = (11, 11, 11)
            kwargs = dict(
                branch_dim=64,
                trunk_dim=24,
                output_dim=128,
                adjust_hidden_dim=128,
                branch_hidden_dim=128,
                trunk_hidden_dim=128,
                adjust_hidden_layer_num=4,
                branch_hidden_layer_num=4,
                trunk_hidden_layer_num=4,
            )
            torch.manual_seed(23)
            current = HMONet3D(*shape, **kwargs).to(device).eval()
            meshonet = MeshONet3D(*shape, **kwargs).to(device).eval()
            geometry = cylindrical_domain_with_airfoil_hole(thickness=0.3)
            grid = (21, 21, 21)
        coons = copy.deepcopy(current)
        coons.hmonet_mapping_mode = "coons_residual"
        queries = logical_grid(grid)
        boundary = torch.as_tensor(
            sample_boundary(geometry, shape), dtype=torch.float32, device=device
        )
        query = torch.as_tensor(queries, dtype=torch.float32, device=device)
        arguments = (*boundary.unbind(-1), *query.unbind(-1))
        # Report one-time kernel initialization separately from repeated forward.
        _, initialization = predict(
            current,
            "HMONet",
            geometry,
            shape,
            queries,
            device,
            dict(
                query_batch_size=50000,
                inference_warmup_runs=1,
                inference_timing_repeats=3,
                curve_kernel_backend="auto",
            ),
        )
        calls = {"MeshONet": lambda: meshonet(*arguments)}
        differences = {}
        for name, function in (
            (
                "projection",
                lambda: current.project(*arguments, boundary_geometry=geometry),
            ),
            (
                "current",
                lambda: torch.cat(current(*arguments, boundary_geometry=geometry), -1),
            ),
            (
                "coons_residual",
                lambda: torch.cat(coons(*arguments, boundary_geometry=geometry), -1),
            ),
        ):

            def call(enabled, function=function):
                with curve_kernel_mode(enabled):
                    return function()

            calls[name + "_torch"] = lambda call=call: call(False)
            calls[name + "_auto"] = lambda call=call: call(True)
            expected = call(False)
            actual = call(True)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            differences[name] = float((actual - expected).abs().max())
        for _ in range(args.warmups):
            for function in calls.values():
                function()
        torch.cuda.synchronize()
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        samples = {name: [] for name in calls}
        names = list(calls)
        for i in range(args.repeats):
            offset = i % len(names)
            for name in names[offset:] + names[:offset]:
                start.record()
                calls[name]()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end))
        row = dict(
            dimension=dimension,
            grid=grid,
            fixture=(
                "actual_air_t_weights" if dimension == 2 else "synthetic_3D_weights"
            ),
            measurements={name: stats(values) for name, values in samples.items()},
            maximum_coordinate_differences=differences,
            initialization=initialization,
        )
        report["runs"].append(row)
        print(
            json.dumps(
                dict(
                    dimension=dimension,
                    medians={
                        name: stats(values)["median_ms"]
                        for name, values in samples.items()
                    },
                    differences=differences,
                )
            ),
            flush=True,
        )
    report["kernel_status"] = kernel_status()
    (args.output / "benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

