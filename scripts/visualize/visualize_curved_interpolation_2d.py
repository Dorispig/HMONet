"""Visualize bilinear and curved-boundary corrected training targets.

Example
-------
python scripts/visualize/visualize_curved_interpolation_2d.py \
    --dataset data/raw/2D/air_t_int_0_1000_41_20 \
    --parameter t --value 0.20 --points-per-side 600
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.training_common import interpolate_pde_mesh
from src.data.dataset import ParametricDataset2D


def select_sample(dataset, parameter, requested):
    values = np.asarray(
        [float(dataset[index]["boundary"].metadata[parameter]) for index in range(len(dataset))]
    )
    index = int(np.argmin(np.abs(values - float(requested))))
    return index, float(values[index]), dataset[index]


def boundary_biased_queries(points_per_side, seed):
    """Use identical random tangential coordinates and cubic near-edge distances."""
    generator = np.random.default_rng(seed)
    tangential = generator.random((4, points_per_side))
    # Most samples lie in the first few percent of the computational domain;
    # a smaller fraction reaches roughly two logical grid layers inward.
    distance = 0.002 + 0.14 * generator.random((4, points_per_side)) ** 3
    parts = (
        np.column_stack((tangential[0], distance[0])),
        np.column_stack((1.0 - distance[1], tangential[1])),
        np.column_stack((tangential[2], 1.0 - distance[2])),
        np.column_stack((distance[3], tangential[3])),
    )
    queries = np.concatenate(parts, axis=0)
    generator.shuffle(queries)
    return queries.astype(np.float32)


def plot_mesh(axis, mesh, shape):
    nx, ny = shape
    grid = mesh.reshape(ny, nx, 2)
    for row in grid:
        axis.plot(row[:, 0], row[:, 1], color="0.72", lw=0.42, alpha=0.52, zorder=1)
    for column in grid.transpose(1, 0, 2):
        axis.plot(
            column[:, 0], column[:, 1], color="0.72", lw=0.42, alpha=0.52, zorder=1
        )


def decorate(axis, boundary, mesh, shape, limits, title):
    plot_mesh(axis, mesh, shape)
    curve = boundary.sample_closed(
        {"bottom": 600, "right": 300, "top": 600, "left": 900}
    )
    axis.plot(curve[:, 0], curve[:, 1], color="black", lw=1.65, zorder=4,
              label="Exact parametric boundary")
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(title)
    axis.grid(False)


def save_figure(path, boundary, mesh, shape, limits, title, draw):
    figure, axis = plt.subplots(figsize=(8.0, 7.0), constrained_layout=True)
    decorate(axis, boundary, mesh, shape, limits, title)
    draw(axis)
    handles, labels = axis.get_legend_handles_labels()
    order = dict(zip(labels, handles))
    axis.legend(order.values(), order.keys(), loc="upper right", frameon=True)
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="data/raw/2D/air_t_int_0_1000_41_20"
    )
    parser.add_argument("--parameter", default="t")
    parser.add_argument("--value", type=float, default=0.20)
    parser.add_argument("--points-per-side", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    dataset = ParametricDataset2D(dataset_path, load_boundary=True)
    index, value, sample = select_sample(dataset, args.parameter, args.value)
    boundary = sample["boundary"]
    mesh = np.asarray(sample["mesh_points"], dtype=np.float32)
    queries = boundary_biased_queries(args.points_per_side, args.seed)

    mesh_tensor = torch.from_numpy(mesh).unsqueeze(0)
    query_tensor = torch.from_numpy(queries).unsqueeze(0)
    with torch.no_grad():
        bilinear = interpolate_pde_mesh(
            mesh_tensor, query_tensor, dataset.grid_shape, target_interpolation="bilinear"
        )[0].cpu().numpy()
        corrected = interpolate_pde_mesh(
            mesh_tensor,
            query_tensor,
            dataset.grid_shape,
            target_interpolation="curved_boundary_cell",
            boundary_geometry=boundary,
            idw_neighbors=-1,
            idw_query_chunk_size=256,
            idw_support_chunk_size=2048,
        )[0].cpu().numpy()

    output = (
        Path(args.output)
        if args.output
        else ROOT
        / "outputs"
        / "interpolation_illustrations"
        / f"air_t_{args.parameter}{value:.4f}_{dataset.grid_shape[0]}x{dataset.grid_shape[1]}"
    )
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    all_points = np.vstack((mesh, bilinear, corrected))
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    padding = 0.06 * np.maximum(maximum - minimum, 1e-6)
    limits = ((minimum[0] - padding[0], maximum[0] + padding[0]),
              (minimum[1] - padding[1], maximum[1] + padding[1]))
    suffix = f"air_t, {args.parameter}={value:.4f}, sample {index + 1}/{len(dataset)}"

    save_figure(
        output / "01_bilinear_targets.png",
        boundary,
        mesh,
        dataset.grid_shape,
        limits,
        "Bilinear interpolation targets\n" + suffix,
        lambda axis: axis.scatter(
            bilinear[:, 0], bilinear[:, 1], s=7, c="#d62728", alpha=0.62,
            edgecolors="none", zorder=3, label="Bilinear targets"
        ),
    )
    save_figure(
        output / "02_curved_corrected_targets.png",
        boundary,
        mesh,
        dataset.grid_shape,
        limits,
        "Curved-boundary IDW-corrected targets\n" + suffix,
        lambda axis: axis.scatter(
            corrected[:, 0], corrected[:, 1], s=7, c="#1f77b4", alpha=0.62,
            edgecolors="none", zorder=3, label="Corrected targets"
        ),
    )

    def draw_overlay(axis):
        stride = max(1, len(bilinear) // 240)
        selected = np.arange(0, len(bilinear), stride)
        segments = np.stack((bilinear[selected], corrected[selected]), axis=1)
        from matplotlib.collections import LineCollection

        axis.add_collection(
            LineCollection(segments, colors="0.25", linewidths=0.35, alpha=0.22, zorder=2)
        )
        axis.scatter(
            bilinear[:, 0], bilinear[:, 1], s=9, c="#d62728", marker="x",
            linewidths=0.45, alpha=0.52, zorder=3, label="Bilinear targets"
        )
        axis.scatter(
            corrected[:, 0], corrected[:, 1], s=6, c="#1f77b4", alpha=0.58,
            edgecolors="none", zorder=3, label="Corrected targets"
        )

    save_figure(
        output / "03_overlay.png",
        boundary,
        mesh,
        dataset.grid_shape,
        limits,
        "Bilinear and corrected targets\n" + suffix,
        draw_overlay,
    )

    displacement = np.linalg.norm(corrected - bilinear, axis=1)
    print(f"dataset={dataset_path}")
    print(f"sample={index}, {args.parameter}={value:.12g}, points={len(queries)}")
    print(
        "displacement: "
        f"mean={displacement.mean():.6e}, median={np.median(displacement):.6e}, "
        f"max={displacement.max():.6e}"
    )
    for path in sorted(output.glob("*.png")):
        print(path)
    return output


if __name__ == "__main__":
    main()
