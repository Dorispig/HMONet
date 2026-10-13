"""Evaluate Attention-PI-HMONet checkpoints on any stored parametric dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from scripts.train.attention_pihmonet_common import build_boundary_inputs, predict_attention
from src.data.dataset import ParametricDataset2D, ParametricDataset3D
from src.data.mesh.sampling import boundary_indices, sample_boundary
from src.evaluation.analysis import save_method
from src.models.model import AttentionHMONet2D, AttentionHMONet3D


ROOT = Path(__file__).resolve().parents[2]
MODEL_KEY = "attention_pihmonet"
MODEL_FOLDER = "Attention-PI-HMONet"


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def _logical_grid(shape):
    axes = [np.linspace(0, 1, n) for n in reversed(shape)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack(tuple(reversed(mesh)), -1).reshape(-1, len(shape))


def _checkpoint(dataset, dimension, explicit, root):
    if explicit:
        path = _resolve(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    base = _resolve(root) / f"{dimension}D" / dataset / MODEL_FOLDER
    candidates = list(base.rglob("best.pt"))
    if not candidates:
        raise FileNotFoundError(f"No {MODEL_FOLDER} checkpoint under {base}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def load_model(path, dimension, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    shape = tuple(config["grid_shape"])
    options = {key: value for key, value in config["model"].items() if key != "name"}
    cls = AttentionHMONet2D if dimension == 2 else AttentionHMONet3D
    model = cls(*shape, **options)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), config


@torch.inference_mode()
def infer(model, geometry, training_shape, output_shape, boundary_counts,
          device, chunk_size, warmup, repeats):
    boundary = build_boundary_inputs([geometry], training_shape, device,
                                     torch.float32, boundary_counts)
    logical = torch.as_tensor(_logical_grid(output_shape), device=device,
                              dtype=torch.float32)[None]
    chunks = logical.split(int(chunk_size), 1)
    def run():
        return torch.cat([
            predict_attention(model, boundary, part, [geometry]) for part in chunks
        ], 1)
    for _ in range(int(warmup)):
        run()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples, result = [], None
    for _ in range(int(repeats)):
        start = time.perf_counter(); result = run()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - start)
    used = np.sort(samples)[1:-1] if len(samples) >= 3 else np.asarray(samples)
    return result[0].cpu().numpy(), float(np.mean(used)), samples


def _plot_mesh(points, shape, path):
    if len(shape) == 2:
        nx, ny = shape; grid = points.reshape(ny, nx, 2)
        figure, axis = plt.subplots(figsize=(7, 6))
        axis.plot(grid[..., 0], grid[..., 1], color="black", linewidth=0.45)
        axis.plot(grid[..., 0].T, grid[..., 1].T, color="black", linewidth=0.45)
        axis.set_aspect("equal", adjustable="box")
    else:
        nx, ny, nz = shape; grid = points.reshape(nz, ny, nx, 3)
        figure = plt.figure(figsize=(8, 7)); axis = figure.add_subplot(111, projection="3d")
        for slab in (grid[0], grid[-1]):
            for line in slab: axis.plot(*line.T, color="black", linewidth=0.3)
            for line in slab.transpose(1, 0, 2): axis.plot(*line.T, color="black", linewidth=0.3)
        for j in (0, ny - 1):
            for i in range(nx): axis.plot(*grid[:, j, i].T, color="black", linewidth=0.3)
        axis.set_box_aspect(np.ptp(points, axis=0).clip(1e-6))
    axis.set_title(MODEL_FOLDER); figure.tight_layout()
    figure.savefig(path, dpi=180); plt.close(figure)


def evaluate_dataset(dataset_name, config, dimension, checkpoint=None):
    evaluation = config["evaluation"]
    device = _device(evaluation.get("device", "auto"))
    dataset_root = _resolve(evaluation.get("data_root", f"data/raw/{dimension}D")) / dataset_name
    dataset_cls = ParametricDataset2D if dimension == 2 else ParametricDataset3D
    dataset = dataset_cls(dataset_root, load_boundary=True)
    checkpoint_path = _checkpoint(
        dataset_name, dimension, checkpoint or evaluation.get("checkpoint"),
        evaluation.get("checkpoint_root", "outputs/checkpoints"),
    )
    model, train_config = load_model(checkpoint_path, dimension, device)
    training_shape = tuple(train_config["grid_shape"])
    boundary_counts = train_config.get("boundary_encoder", {}).get("sample_counts")
    shapes = [tuple(map(int, item)) for item in evaluation.get("grid_shapes", [dataset.grid_shape])]
    indices = [int(index) for index in evaluation.get("sample_indices", [0])]
    options = {
        "compute_grid_quality": True, "quality_metrics": "both", "quality_switches": {},
        "save_plots": bool(evaluation.get("save_plots", True)),
        "save_png": True, "save_pdf": False, "save_distribution_plots": False,
    }
    records = []
    for index in indices:
        sample = dataset[index]
        for shape in shapes:
            points, seconds, samples = infer(
                model, sample["boundary"], training_shape, shape, boundary_counts,
                device, evaluation.get("query_batch_size", 50000),
                evaluation.get("warmup_runs", 3), evaluation.get("timing_repeats", 10),
            )
            output = _resolve(evaluation.get("output_root", "outputs/results")) / \
                f"{dimension}D" / dataset_name / f"sample_{index:04d}" / \
                "x".join(map(str, shape)) / MODEL_FOLDER
            quality = save_method(output, MODEL_FOLDER, points, shape, options)
            _plot_mesh(points, shape, output / "mesh.png")
            mse = None
            if tuple(shape) == tuple(dataset.grid_shape):
                mse = float(np.mean((points - sample["mesh_points"]) ** 2))
            exact_boundary = sample_boundary(sample["boundary"], shape)
            boundary_error = np.linalg.norm(
                points[boundary_indices(shape)] - exact_boundary, axis=-1
            )
            record = {
                "dataset": dataset_name, "sample_index": index,
                "grid_shape": list(shape), "checkpoint": str(checkpoint_path),
                "network_inference_seconds": seconds,
                "timing_samples_seconds": samples, "mesh_mse": mse,
                "boundary_error_max": float(boundary_error.max(initial=0.0)),
                "boundary_error_mean": float(boundary_error.mean()),
                "quality_summary": quality.get("summary", {}),
            }
            (output / "evaluation_summary.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
            records.append(record)
            print(f"Evaluated {dataset_name} sample={index} shape={shape}: "
                  f"time={seconds:.6f}s, mse={mse}", flush=True)
    return records


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--grid-shapes", help="Comma-separated, e.g. 51x51,101x101")
    parser.add_argument("--sample-indices", help="Comma-separated stored sample indices")
    parser.add_argument("--device")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    path = args.config or ROOT / f"configs/{args.dimension}D/attention_pihmonet_evaluate.yaml"
    config = yaml.safe_load(_resolve(path).read_text(encoding="utf-8-sig"))
    datasets = args.dataset or config["evaluation"].get("datasets", [])
    if args.grid_shapes:
        config["evaluation"]["grid_shapes"] = [
            [int(value) for value in item.split("x")] for item in args.grid_shapes.split(",")
        ]
    if args.sample_indices:
        config["evaluation"]["sample_indices"] = [int(x) for x in args.sample_indices.split(",")]
    if args.device:
        config["evaluation"]["device"] = args.device
    return [evaluate_dataset(name, config, args.dimension, args.checkpoint) for name in datasets]


if __name__ == "__main__":
    main()
