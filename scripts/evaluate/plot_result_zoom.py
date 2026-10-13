"""Create traceable local mesh/quality views from saved evaluation results.

The script never reruns a model or recomputes quality.  It reads one method
directory, one comparison directory, or every existing comparison combination
below a parameter directory and writes zoom figures plus JSON metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mesh.topology import QuadMeshXY
from src.evaluation.method_order import ordered_methods
from src.evaluation.quality_output import (
    DISPLAY_NAMES,
    HIGHER_IS_BETTER,
    add_shared_quality_colorbar,
    plot_quality_field,
    save_figure_formats,
)
from src.utils.visualize import plot_quad_meshXY


def pair(value: str) -> tuple[float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 2 or parts[0] >= parts[1]:
        raise argparse.ArgumentTypeError("expected increasing MIN,MAX")
    return parts[0], parts[1]


def integer_pair(value: str) -> tuple[int, int]:
    result = pair(value)
    if any(item != int(item) for item in result):
        raise argparse.ArgumentTypeError("expected integer MIN,MAX")
    return int(result[0]), int(result[1])


def method_title(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "adp" in parts or "adaptive" in parts:
        return "HMONet (Adaptive)"
    if "non_adp" in parts or "non_adaptive" in parts:
        return "HMONet (Non-adaptive)"
    if "pi-hmonet" in parts:
        return "PI-HMONet"
    if "pi-meshonet" in parts:
        return "PI-MeshONet"
    if "meshonet" in parts:
        return "MeshONet"
    return path.name


def load_single(directory: Path, metric: str | None):
    mesh_path = directory / "mesh_data.npz"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing saved mesh: {mesh_path}")
    with np.load(mesh_path) as data:
        vertices = np.asarray(data["vertices"], dtype=float)
        nx, ny = int(data["nx"]), int(data["ny"])
    fields = {}
    if metric:
        with np.load(directory / "quality_data.npz") as data:
            if metric not in data.files:
                raise KeyError(f"Metric {metric!r} is unavailable: {data.files}")
            fields[method_title(directory)] = np.asarray(data[metric], dtype=float)
    return {method_title(directory): vertices}, fields, nx, ny


def load_comparison(directory: Path, metric: str | None):
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    nx, ny = int(metadata["nx"]), int(metadata["ny"])
    with np.load(directory / "comparison_mesh_data.npz") as data:
        meshes = {
            DISPLAY_NAMES.get(name, name): np.asarray(data[name], dtype=float)
            for name in ordered_methods(data.files)
        }
    fields = {}
    if metric:
        quality_path = directory / "grid_quality" / "comparison_quality_data.npz"
        if not quality_path.exists():
            quality_path = directory / "comparison_quality_data.npz"
        with np.load(quality_path) as data:
            for raw_name in ordered_methods(metadata["method_order"]):
                key = f"{raw_name}__{metric}"
                if key not in data.files:
                    raise KeyError(f"Metric {metric!r} is unavailable in {directory}")
                fields[DISPLAY_NAMES.get(raw_name, raw_name)] = np.asarray(
                    data[key], dtype=float
                )
    return meshes, fields, nx, ny


def worst_cell(fields: dict[str, np.ndarray], metric: str) -> int:
    if not fields:
        raise ValueError("--center-on worst requires a quality metric")
    candidates = []
    for values in fields.values():
        finite = np.asarray(values, dtype=float)
        if metric == "cell_area":
            score = np.abs(finite - np.nanmedian(finite))
            candidates.append((float(np.nanmax(score)), int(np.nanargmax(score))))
        elif metric in HIGHER_IS_BETTER:
            candidates.append((-float(np.nanmin(finite)), int(np.nanargmin(finite))))
        else:
            candidates.append((float(np.nanmax(finite)), int(np.nanargmax(finite))))
    return max(candidates)[1]


def logical_bounds(meshes, nx, ny, i_range, j_range, padding=0.04):
    points = []
    for vertices in meshes.values():
        grid = np.asarray(vertices).reshape(ny, nx, 2)
        points.append(
            grid[j_range[0] : j_range[1] + 1, i_range[0] : i_range[1] + 1].reshape(
                -1, 2
            )
        )
    points = np.concatenate(points)
    lower, upper = points.min(axis=0), points.max(axis=0)
    span = np.maximum(upper - lower, 1e-12)
    return (
        (float(lower[0] - padding * span[0]), float(upper[0] + padding * span[0])),
        (float(lower[1] - padding * span[1]), float(upper[1] + padding * span[1])),
    )


def resolve_bounds(meshes, fields, nx, ny, metric, args):
    if args.xlim or args.ylim:
        all_points = np.concatenate(list(meshes.values()))
        xlim = args.xlim or (
            float(all_points[:, 0].min()),
            float(all_points[:, 0].max()),
        )
        ylim = args.ylim or (
            float(all_points[:, 1].min()),
            float(all_points[:, 1].max()),
        )
        return xlim, ylim, {"selection": "physical_coordinates"}
    i_range, j_range = args.i_range, args.j_range
    selected_cell = args.cell_index
    if args.center_on == "worst":
        selected_cell = worst_cell(fields, metric)
    if selected_cell is not None:
        if not 0 <= selected_cell < (nx - 1) * (ny - 1):
            raise ValueError(f"cell index must be in [0, {(nx - 1) * (ny - 1) - 1}]")
        cell_i, cell_j = selected_cell % (nx - 1), selected_cell // (nx - 1)
        pad = int(args.cell_padding)
        i_range = (max(0, cell_i - pad), min(nx - 1, cell_i + 1 + pad))
        j_range = (max(0, cell_j - pad), min(ny - 1, cell_j + 1 + pad))
    if i_range or j_range:
        i_range = i_range or (0, nx - 1)
        j_range = j_range or (0, ny - 1)
        if not (0 <= i_range[0] < i_range[1] < nx):
            raise ValueError(f"i range must satisfy 0 <= min < max < {nx}")
        if not (0 <= j_range[0] < j_range[1] < ny):
            raise ValueError(f"j range must satisfy 0 <= min < max < {ny}")
        xlim, ylim = logical_bounds(meshes, nx, ny, i_range, j_range)
        return (
            xlim,
            ylim,
            {
                "selection": "logical_indices",
                "i_range": i_range,
                "j_range": j_range,
                "center_cell_index": selected_cell,
            },
        )
    raise ValueError(
        "Specify --xlim/--ylim, --i-range/--j-range, --cell-index, or --center-on worst"
    )


def visible_cell_indices(vertices, nx, ny, xlim, ylim):
    mesh = QuadMeshXY(nx, ny, vertices)
    centers = mesh.vertices[mesh.faces].mean(axis=1)
    mask = (
        (centers[:, 0] >= xlim[0])
        & (centers[:, 0] <= xlim[1])
        & (centers[:, 1] >= ylim[0])
        & (centers[:, 1] <= ylim[1])
    )
    return np.flatnonzero(mask)


def quality_limits(fields, meshes, nx, ny, xlim, ylim, scale):
    arrays = []
    for name, values in fields.items():
        if scale == "local":
            indices = visible_cell_indices(meshes[name], nx, ny, xlim, ylim)
            arrays.append(values[indices])
        else:
            arrays.append(values)
    finite = np.concatenate(arrays)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 0.0, 1.0
    return float(finite.min()), float(finite.max())


def draw_axis(axis, mesh, values, metric, limits=None, center=None):
    if values is None:
        plot_quad_meshXY(mesh, ax=axis, show_node_id=False)
        return None
    _, colored = plot_quality_field(
        mesh,
        metric,
        values,
        axis,
        vmin=limits[0],
        vmax=limits[1],
        center=center,
    )
    return colored


def save_zoom(
    directory, meshes, fields, nx, ny, plot_type, metric, xlim, ylim, selection, args
):
    output = args.output_dir or directory / "zoom_views"
    if args.output_dir and args.parameter_dir:
        output = args.output_dir / directory.name
    output.mkdir(parents=True, exist_ok=True)
    names = ordered_methods(meshes)
    mesh_objects = {name: QuadMeshXY(nx, ny, meshes[name]) for name in names}
    limits = (
        quality_limits(fields, meshes, nx, ny, xlim, ylim, args.color_scale)
        if plot_type == "quality"
        else None
    )
    shared_center = (
        float(np.nanmedian(np.concatenate(list(fields.values()))))
        if metric == "cell_area"
        else None
    )
    stem = (
        f"{metric}_comparison"
        if len(names) > 1 and plot_type == "quality"
        else (
            "mesh_comparison"
            if len(names) > 1
            else metric if plot_type == "quality" else "mesh"
        )
    )

    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5), squeeze=False)
    colored = None
    for axis, name in zip(axes[0], names):
        current_colored = draw_axis(
            axis,
            mesh_objects[name],
            fields.get(name) if fields else None,
            metric,
            limits,
            shared_center,
        )
        if current_colored is not None:
            colored = current_colored
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        axis.set_title(name)
    fig.subplots_adjust(left=0.05, right=0.84, bottom=0.08, top=0.90, wspace=0.18)
    if colored is not None:
        add_shared_quality_colorbar(fig, axes[0], colored, metric)
    save_figure_formats(
        fig,
        output / f"{stem}_zoom",
        save_png=args.save_png,
        save_pdf=args.save_pdf,
        dpi=240,
    )
    plt.close(fig)

    fig, axes = plt.subplots(2, len(names), figsize=(5 * len(names), 9), squeeze=False)
    colored = None
    for column, name in enumerate(names):
        current_colored = draw_axis(
            axes[0, column],
            mesh_objects[name],
            fields.get(name) if fields else None,
            metric,
            limits,
            shared_center,
        )
        if current_colored is not None:
            colored = current_colored
        axes[0, column].set_title(name)
        rectangle = plt.Rectangle(
            (xlim[0], ylim[0]),
            xlim[1] - xlim[0],
            ylim[1] - ylim[0],
            fill=False,
            edgecolor="red",
            linewidth=1.8,
        )
        axes[0, column].add_patch(rectangle)
        draw_axis(
            axes[1, column],
            mesh_objects[name],
            fields.get(name) if fields else None,
            metric,
            limits,
            shared_center,
        )
        axes[1, column].set_xlim(xlim)
        axes[1, column].set_ylim(ylim)
        axes[1, column].set_title("Zoom")
    fig.subplots_adjust(
        left=0.05, right=0.84, bottom=0.06, top=0.94, hspace=0.22, wspace=0.18
    )
    if colored is not None:
        add_shared_quality_colorbar(fig, axes, colored, metric)
    save_figure_formats(
        fig,
        output / f"{stem}_overview_and_zoom",
        save_png=args.save_png,
        save_pdf=args.save_pdf,
        dpi=240,
    )
    plt.close(fig)

    visible = {
        name: visible_cell_indices(meshes[name], nx, ny, xlim, ylim).tolist()
        for name in names
    }
    metadata = {
        "source_directory": str(directory.resolve()),
        "plot": plot_type,
        "metric": metric,
        "xlim": xlim,
        "ylim": ylim,
        "color_scale": args.color_scale if metric else None,
        "color_limits": limits,
        "method_order": names,
        "visible_cell_indices": visible,
        **selection,
    }
    (output / f"{stem}_zoom_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved zoom views: {output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--result-dir", type=Path)
    source.add_argument("--comparison-dir", type=Path)
    source.add_argument(
        "--parameter-dir",
        type=Path,
        help="Update every saved comparison combination recursively with one shared ROI.",
    )
    parser.add_argument("--plot", choices=["mesh", "quality"], required=True)
    parser.add_argument("--metric")
    parser.add_argument("--xlim", type=pair)
    parser.add_argument("--ylim", type=pair)
    parser.add_argument("--i-range", type=integer_pair)
    parser.add_argument("--j-range", type=integer_pair)
    parser.add_argument("--cell-index", type=int)
    parser.add_argument("--center-on", choices=["worst"])
    parser.add_argument("--cell-padding", type=int, default=5)
    parser.add_argument("--color-scale", choices=["global", "local"], default="global")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--save-png", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save-pdf", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    if args.plot == "quality" and not args.metric:
        parser.error("--plot quality requires --metric")
    if args.plot == "mesh" and args.center_on:
        parser.error("--center-on worst requires --plot quality")
    if args.cell_padding < 0:
        parser.error("--cell-padding must be non-negative")
    return args


def main():
    args = parse_args()
    metric = args.metric if args.plot == "quality" else None
    if args.result_dir:
        directory = args.result_dir.resolve()
        meshes, fields, nx, ny = load_single(directory, metric)
        xlim, ylim, selection = resolve_bounds(meshes, fields, nx, ny, metric, args)
        save_zoom(
            directory,
            meshes,
            fields,
            nx,
            ny,
            args.plot,
            metric,
            xlim,
            ylim,
            selection,
            args,
        )
        return
    if args.comparison_dir:
        directory = args.comparison_dir.resolve()
        meshes, fields, nx, ny = load_comparison(directory, metric)
        xlim, ylim, selection = resolve_bounds(meshes, fields, nx, ny, metric, args)
        save_zoom(
            directory,
            meshes,
            fields,
            nx,
            ny,
            args.plot,
            metric,
            xlim,
            ylim,
            selection,
            args,
        )
        return

    comparisons = args.parameter_dir.resolve() / "comparisons"
    directories = sorted(
        {
            path.parent
            for path in comparisons.rglob("comparison_mesh_data.npz")
            if path.is_file()
        }
    )
    if not directories:
        raise FileNotFoundError(f"No saved comparison combinations: {comparisons}")
    reference = next(
        (path for path in directories if path.name == "08_all_five_methods"),
        directories[0],
    )
    ref_meshes, ref_fields, ref_nx, ref_ny = load_comparison(reference, metric)
    xlim, ylim, selection = resolve_bounds(
        ref_meshes, ref_fields, ref_nx, ref_ny, metric, args
    )
    for directory in directories:
        meshes, fields, nx, ny = load_comparison(directory, metric)
        if (nx, ny) != (ref_nx, ref_ny):
            raise ValueError(f"Comparison grid mismatch: {directory}")
        save_zoom(
            directory,
            meshes,
            fields,
            nx,
            ny,
            args.plot,
            metric,
            xlim,
            ylim,
            selection,
            args,
        )


if __name__ == "__main__":
    main()
