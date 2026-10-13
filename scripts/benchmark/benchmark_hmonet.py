"""Compare HMONet implementations with identical inputs and synchronized timing.

The old model uses its original projection and network modules. The two new
variants use exact boundary objects. Dataset I/O, checkpoint I/O and plots are
excluded. Parameter packing and CPU/GPU transfers inside forward are included.
The 3-D box is a timing fixture, not a claim about production training quality.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import importlib.util
import json
import platform
import statistics
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.models.model as current_models
from scripts.train.training_common import (
    calculate_fixed_grid_metrics,
    hmonet_main_loss,
    interpolate_pde_mesh,
    logical_grid_and_interior_indices,
    make_data_loader,
    predict_coordinates,
    safe_optimizer_step,
    sample_shared_random_points,
)
from src.data.geometry.boundary_generator_3d import box_boundary
from src.data.dataset import ParametricDataset2D


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class BoxTimingDataset(torch.utils.data.Dataset):
    """Known exact affine meshes for a controlled 3-D model timing fixture."""

    def __init__(self, shape, count=64):
        self.grid_shape = tuple(shape)
        self.root = Path("synthetic_box_timing_fixture")
        q, interior = logical_grid_and_interior_indices(shape, "cpu")
        mask = torch.ones(len(q), dtype=torch.bool)
        mask[interior] = False
        meshes, self.boundary = [], []
        for index in range(count):
            scale = 1 + index / (2 * count)
            mesh = (2 * q.numpy() - 1) * scale
            meshes.append(mesh)
            self.boundary.append(
                box_boundary(
                    xmin=-scale,
                    xmax=scale,
                    ymin=-scale,
                    ymax=scale,
                    zmin=-scale,
                    zmax=scale,
                )
            )
        self.mesh_points = np.stack(meshes)
        self.boundary_points = self.mesh_points[:, mask.numpy()]

    def __len__(self):
        return len(self.boundary)

    def __getitem__(self, index):
        return {
            "boundary_points": self.boundary_points[index],
            "mesh_points": self.mesh_points[index],
            "boundary": self.boundary[index],
        }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_samples(function, device, repeats, warmups):
    for _ in range(warmups):
        function()
    synchronize(device)
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        synchronize(device)
        values.append((time.perf_counter() - start) * 1000)
    return values


def run_variant(label, model, dataset, use_geometry, config, args, output_rows):
    dimension = len(dataset.grid_shape)
    device = torch.device(args.device)
    training = config["training"]
    batch_size = min(args.batch_size, len(dataset))
    boundary = torch.as_tensor(
        dataset.boundary_points[:batch_size], dtype=torch.float32, device=device
    )
    mesh = torch.as_tensor(
        dataset.mesh_points[:batch_size], dtype=torch.float32, device=device
    )
    geometries = dataset.boundary[:batch_size] if use_geometry else None
    torch.manual_seed(args.seed)
    query = sample_shared_random_points(batch_size, args.points, dimension, device)
    target = interpolate_pde_mesh(mesh, query, dataset.grid_shape)
    grid, _ = logical_grid_and_interior_indices(dataset.grid_shape, device)
    query_grid = grid[None]
    saved_weights = copy.deepcopy(model.state_dict())

    def record(metric, values, **extra):
        row = dict(
            dimension=dimension,
            dataset=dataset.root.name,
            variant=label,
            mapping=args.mapping,
            metric=metric,
            median_ms=statistics.median(values),
            min_ms=min(values),
            max_ms=max(values),
            repetitions=len(values),
            samples_ms=values,
            **extra,
        )
        output_rows.append(row)
        print(
            json.dumps({k: v for k, v in row.items() if k != "samples_ms"}), flush=True
        )

    model.eval()

    @torch.inference_mode()
    def projection():
        kwargs = {"boundary_geometry": geometries} if use_geometry else {}
        return model.project(*boundary.unbind(-1), *query.unbind(-1), **kwargs)

    @torch.inference_mode()
    def single_prediction():
        geometry = geometries[:1] if use_geometry else None
        return predict_coordinates(model, boundary[:1], query_grid, geometry)

    @torch.inference_mode()
    def batch_prediction():
        return predict_coordinates(model, boundary, query, geometries)

    record(
        "projection_batch",
        timed_samples(projection, device, args.repeats, args.warmups),
        batch_size=batch_size,
        point_count=args.points,
    )
    record(
        "prediction_single_full_grid",
        timed_samples(single_prediction, device, args.repeats, args.warmups),
        batch_size=1,
        point_count=len(grid),
    )
    record(
        "prediction_batch_random_points",
        timed_samples(batch_prediction, device, args.repeats, args.warmups),
        batch_size=batch_size,
        point_count=args.points,
    )

    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["hmonet_training"]["learning_rate"]
    )

    def train_step():
        prediction = predict_coordinates(model, boundary, query, geometries)
        loss = hmonet_main_loss(prediction, target, query)
        safe_optimizer_step(
            loss, model, optimizer, training.get("gradient_clip_norm", 1.0), False
        )

    record(
        "train_batch_with_adam",
        timed_samples(train_step, device, args.repeats, args.warmups),
        batch_size=batch_size,
        point_count=args.points,
    )

    # Each implementation starts whole-epoch timing from identical weights,
    # random seeds and shuffling. Optimizer/geometry behavior is not stubbed.
    model.load_state_dict(saved_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["hmonet_training"]["learning_rate"]
    )
    loader = make_data_loader([dataset], args.batch_size, args.seed, shuffle=True)
    torch.manual_seed(args.seed)
    train_times, mse_times, total_times = [], [], []
    for epoch in range(args.epochs + 1):
        model.train()
        synchronize(device)
        start = time.perf_counter()
        loss_sum = 0.0
        for batch in loader:
            b = batch["boundary_points"].to(device)
            meshes = batch["mesh_points"].to(device)
            q = sample_shared_random_points(len(b), args.points, dimension, device)
            target = interpolate_pde_mesh(meshes, q, dataset.grid_shape)
            geometry = batch["boundary"] if use_geometry else None
            prediction = predict_coordinates(model, b, q, geometry)
            loss = hmonet_main_loss(prediction, target, q)
            safe_optimizer_step(
                loss, model, optimizer, training.get("gradient_clip_norm", 1.0), False
            )
            loss_sum += loss.item() * len(b)
        synchronize(device)
        train_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        metrics = calculate_fixed_grid_metrics(
            model,
            [dataset],
            dataset.grid_shape,
            device,
            geometry_getter=(lambda d, i: d.boundary[i]) if use_geometry else None,
            batch_size=args.batch_size,
        )
        synchronize(device)
        mse_ms = (time.perf_counter() - start) * 1000
        if epoch:
            train_times.append(train_ms)
            mse_times.append(mse_ms)
            total_times.append(train_ms + mse_ms)
        print(
            f"{dimension}D {label} epoch {epoch}: train={train_ms:.2f} ms, MSE={mse_ms:.2f} ms",
            flush=True,
        )
    record(
        "epoch_training",
        train_times,
        batch_size=args.batch_size,
        point_count=args.points,
    )
    record(
        "epoch_fixed_grid_mse",
        mse_times,
        batch_size=args.batch_size,
        point_count=int(np.prod(np.asarray(dataset.grid_shape) - 2)),
    )
    record(
        "epoch_training_plus_mse",
        total_times,
        batch_size=args.batch_size,
        point_count=args.points,
        last_training_loss=loss_sum / len(dataset),
        last_metrics=metrics,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-root",
        type=Path,
        required=True,
        help="Path to a separate legacy HMONet checkout used for comparison",
    )
    parser.add_argument(
        "--baseline-root", type=Path, help="Optional unmodified source backup"
    )
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/raw/2D/air_t_int_0_1000_21_20"
    )
    parser.add_argument("--dimension", choices=["2D", "3D", "both"], default="both")
    parser.add_argument(
        "--mapping", choices=["current", "coons_residual"], default="current"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--points", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if (
        min(args.batch_size, args.points, args.epochs, args.repeats) < 1
        or args.warmups < 0
    ):
        parser.error("Counts must be positive; warmups must be nonnegative")
    legacy_model_path = args.legacy_root / "src/models/model.py"
    legacy_modules_path = args.legacy_root / "src/models/modules.py"
    if not legacy_model_path.is_file() or not legacy_modules_path.is_file():
        parser.error(
            "--legacy-root must point to a separate legacy checkout containing "
            "src/models/model.py and src/models/modules.py"
        )
    if args.legacy_root.resolve() == ROOT.resolve():
        parser.error("--legacy-root must not point to the current project")
    device = torch.device(args.device)
    legacy = load_module("benchmark_legacy_models", legacy_model_path)
    legacy.modules = load_module(
        "benchmark_legacy_modules", legacy_modules_path
    )
    baseline = baseline_curves = baseline_surfaces = None
    if args.baseline_root:
        baseline = load_module(
            "benchmark_before_models", args.baseline_root / "src/models/model.py"
        )
        package = types.ModuleType("benchmark_before_data")
        package.__path__ = [str(args.baseline_root / "src/data")]
        sys.modules[package.__name__] = package
        baseline_curves = importlib.import_module(
            "benchmark_before_data.parametric_curve_2d"
        )
        baseline_surfaces = importlib.import_module(
            "benchmark_before_data.parametric_geometry_3d"
        )
    dimensions = [2, 3] if args.dimension == "both" else [int(args.dimension[0])]
    output = args.output_dir or ROOT / "outputs/benchmarks" / datetime.now().strftime(
        "hmonet_%Y%m%d_%H%M%S"
    )
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "method": "Synchronized wall-clock milliseconds; warmed kernels; same weights/data/training driver; no file I/O or plots. Each full-epoch series discards its first epoch. 3D uses a 64-sample affine box fixture, not a production dataset.",
        "rows": rows,
        "configurations": {},
    }
    for dimension in dimensions:
        config = yaml.safe_load(
            (ROOT / f"configs/{dimension}D/train.yaml").read_text(encoding="utf-8-sig")
        )
        report["configurations"][f"{dimension}D"] = config
        dataset = (
            ParametricDataset2D(args.dataset)
            if dimension == 2
            else BoxTimingDataset(tuple(config["grid"][k] for k in ("nx", "ny", "nz")))
        )
        constructor = f"HMONet{dimension}D"
        options = dict(config["model"]["common"], hmonet_mapping_mode=args.mapping)
        torch.manual_seed(args.seed)
        initial = getattr(current_models, constructor)(
            *dataset.grid_shape, **options
        ).state_dict()
        variants = [
            ("old_project", legacy, False),
            ("new_optimized", current_models, True),
        ]
        if baseline:
            variants.append(("new_before_optimization", baseline, True))
        for label, module, exact in variants:
            variant_options = dict(options)
            if dimension == 2 and label == "old_project":
                variant_options["boundary_curve_interpolation"] = "fitted_curve"
            model = getattr(module, constructor)(
                *dataset.grid_shape, **variant_options
            ).to(device)
            model.load_state_dict(initial, strict=True)
            data = dataset
            if label == "new_before_optimization":
                data = copy.copy(dataset)
                klass = (
                    baseline_curves.BoundaryCurveSet2D
                    if dimension == 2
                    else baseline_surfaces.BoundarySurfaceSet3D
                )
                data.boundary = [klass.from_dict(g.to_dict()) for g in dataset.boundary]
            run_variant(label, model, data, exact, config, args, rows)
            (output / "results.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            del model
    columns = [
        "dimension",
        "dataset",
        "variant",
        "mapping",
        "metric",
        "median_ms",
        "min_ms",
        "max_ms",
        "repetitions",
        "batch_size",
        "point_count",
    ]
    with (output / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# HMONet timing comparison",
        "",
        report["method"],
        "",
        "All values are medians in milliseconds.",
        "",
        "| Dimension | Metric | Old project | New before optimization | New optimized |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for dimension in dimensions:
        metrics = dict.fromkeys(
            row["metric"] for row in rows if row["dimension"] == dimension
        )
        for metric in metrics:
            values = {
                row["variant"]: row["median_ms"]
                for row in rows
                if row["dimension"] == dimension and row["metric"] == metric
            }
            cells = [
                f"{values[key]:.3f}" if key in values else "-"
                for key in ("old_project", "new_before_optimization", "new_optimized")
            ]
            lines.append(f"| {dimension}D | {metric} | " + " | ".join(cells) + " |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {output}", flush=True)


if __name__ == "__main__":
    main()
