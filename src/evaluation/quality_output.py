"""Documentation string."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import numpy as np

from src.data.mesh.topology import QuadMeshXY
from .method_order import ordered_methods
from src.utils.metrics import QuadMeshQualityEvaluator
from src.utils.quality_selection import QUAD_METRICS, metric_names
from src.utils.visualize import (
    plot_quad_meshXY,
    plot_quad_mesh_colored,
    plot_quad_mesh_colored_smooth,
)

FLAG_FIELDS = {
    "concave_vertex_count",
    "self_intersection",
    "intersecting_cell",
    "flipping_cell",
    "degenerate_cell",
    "flipped_cell",
}
BINARY_FLAG_FIELDS = {
    "self_intersection",
    "intersecting_cell",
    "flipping_cell",
    "degenerate_cell",
    "flipped_cell",
}

DISPLAY_NAMES = {
    "TFI": "TFI",
    "PDE_Winslow": "PDE",
    "MeshONet2D": "MeshONet",
    "HMONet2D_nonadaptive": "HMONet (Non-adaptive)",
    "HMONet2D_adaptive": "HMONet (Adaptive)",
    "HMONet2D": "HMONet",
}

LOWER_IS_BETTER = {
    "max_angle",
    "skewness",
    "aspect_ratio",
    "concave_vertex_count",
    "self_intersection",
    "intersecting_cell",
    "flipping_cell",
    "degenerate_cell",
}
HIGHER_IS_BETTER = {
    "min_angle",
    "jacobian",
    "scaled_jacobian",
    "orthogonality",
    "minimum_scaled_jacobian",
    "center_jacobian",
}
SMOOTH_FIELDS = {
    "max_angle",
    "min_angle",
    "jacobian",
    "scaled_jacobian",
    "skewness",
    "aspect_ratio",
    "orthogonality",
    "cell_area",
}
AREA_QUALITY_CMAP = LinearSegmentedColormap.from_list(
    "area_quality_red_blue_red", ["#d7191c", "#2166ac", "#d7191c"]
)


def _annotate_count_bars(axis, bars, counts, totals=None) -> None:
    """Label discrete-quality bars with exact counts and optional percentages."""
    labels = []
    for index, count in enumerate(counts):
        if totals is None:
            labels.append(f"{int(count)}")
        else:
            total = max(int(totals[index]), 1)
            labels.append(f"{int(count)}\n({100.0 * count / total:.2f}%)")
    axis.bar_label(bars, labels=labels, padding=3, fontsize=9)


def _use_imbalance_scale(axis, counts) -> None:
    """Keep rare nonzero events visible without hiding an exact zero label."""
    positive = np.asarray(counts, dtype=float)
    positive = positive[positive > 0]
    if positive.size >= 2 and positive.max() / positive.min() >= 20.0:
        axis.set_yscale("symlog", linthresh=1.0)


def plot_single_quality_distribution(axis, metric: str, raw_values) -> None:
    """Choose a readable distribution plot for continuous or discrete fields."""
    values = np.asarray(raw_values, dtype=float)
    values = values[np.isfinite(values)]
    if metric in BINARY_FLAG_FIELDS:
        affected = int(np.count_nonzero(values > 0))
        counts = [int(values.size) - affected, affected]
        bars = axis.bar(["Normal", "Affected"], counts, color=["#2166ac", "#d7191c"])
        _annotate_count_bars(axis, bars, counts, [values.size, values.size])
        _use_imbalance_scale(axis, counts)
        axis.set_title(f"{metric}: normal / affected cells")
        axis.set_ylabel("Cell count")
    elif metric == "concave_vertex_count":
        integer_values = np.maximum(np.rint(values).astype(int), 0)
        maximum = int(integer_values.max()) if integer_values.size else 0
        categories = np.arange(maximum + 1)
        counts = np.bincount(integer_values, minlength=maximum + 1)
        colors = ["#2166ac", *(["#d7191c"] * maximum)]
        bars = axis.bar(categories, counts, color=colors, width=0.75)
        _annotate_count_bars(axis, bars, counts, [values.size] * len(counts))
        _use_imbalance_scale(axis, counts)
        axis.set_xticks(categories)
        axis.set_xlabel("Concave vertices in one cell")
        axis.set_ylabel("Cell count")
        axis.set_title("concave_vertex_count: discrete cell counts")
    else:
        if values.size:
            axis.hist(values, bins=30, edgecolor="black")
        axis.set(title=f"{metric} distribution", xlabel=metric, ylabel="Cell count")
    axis.grid(axis="y", which="both", linestyle=":", alpha=0.35)


def plot_comparison_distribution(axis, metric: str, method_values: dict) -> None:
    """Plot method comparisons without histogramming sparse discrete flags."""
    names = ordered_methods(method_values)
    display_names = [DISPLAY_NAMES.get(name, name) for name in names]
    finite_values = {
        name: np.asarray(method_values[name], dtype=float)[
            np.isfinite(np.asarray(method_values[name], dtype=float))
        ]
        for name in names
    }
    if metric in BINARY_FLAG_FIELDS:
        counts = [int(np.count_nonzero(finite_values[name] > 0)) for name in names]
        totals = [int(finite_values[name].size) for name in names]
        bars = axis.bar(display_names, counts, color="#d7191c")
        _annotate_count_bars(axis, bars, counts, totals)
        _use_imbalance_scale(axis, counts)
        axis.set_ylabel("Affected cell count")
        axis.set_title(f"{metric}: affected cells by method")
        axis.tick_params(axis="x", rotation=15)
    elif metric == "concave_vertex_count":
        maximum = max(
            (
                int(np.max(np.rint(values)))
                for values in finite_values.values()
                if values.size
            ),
            default=0,
        )
        categories = np.arange(maximum + 1)
        width = 0.8 / max(len(names), 1)
        all_counts = []
        for index, name in enumerate(names):
            integers = np.maximum(np.rint(finite_values[name]).astype(int), 0)
            counts = np.bincount(integers, minlength=maximum + 1)
            all_counts.extend(counts.tolist())
            positions = categories + (index - (len(names) - 1) / 2) * width
            bars = axis.bar(
                positions,
                counts,
                width=width,
                label=DISPLAY_NAMES.get(name, name),
            )
            _annotate_count_bars(
                axis, bars, counts, [finite_values[name].size] * len(counts)
            )
        _use_imbalance_scale(axis, all_counts)
        axis.set_xticks(categories)
        axis.set_xlabel("Concave vertices in one cell")
        axis.set_ylabel("Cell count")
        axis.set_title("concave_vertex_count: discrete comparison")
        axis.legend()
    else:
        combined = np.concatenate(list(finite_values.values()))
        if combined.size:
            lower, upper = float(combined.min()), float(combined.max())
            if lower == upper:
                padding = max(abs(lower) * 0.01, 0.5)
                lower -= padding
                upper += padding
            bins = np.linspace(lower, upper, 31)
            for name in names:
                if finite_values[name].size:
                    axis.hist(
                        finite_values[name],
                        bins=bins,
                        histtype="step",
                        linewidth=2,
                        label=DISPLAY_NAMES.get(name, name),
                    )
        axis.set_xlabel(metric)
        axis.set_ylabel("Cell count")
        axis.set_title(f"{metric} distribution")
        axis.legend()
    axis.grid(axis="y", which="both", linestyle=":", alpha=0.35)


def save_figure_formats(
    fig, path: Path, *, save_png: bool, save_pdf: bool, dpi: int
) -> None:
    """Save a figure in the enabled formats using a suffix-free path."""
    if save_png:
        fig.savefig(path.with_suffix(".png"), dpi=dpi)
    if save_pdf:
        fig.savefig(path.with_suffix(".pdf"))


def quality_plot_style(
    name: str, values: np.ndarray, vmin=None, vmax=None, center=None
):
    """Return a colormap/norm where blue means good and red means poor."""
    if name in BINARY_FLAG_FIELDS:
        return plt.get_cmap("jet"), Normalize(vmin=0.0, vmax=1.0)
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if vmin is None:
        vmin = float(finite.min()) if finite.size else 0.0
    if vmax is None:
        vmax = float(finite.max()) if finite.size else 1.0
    if vmin == vmax:
        padding = max(abs(vmin) * 0.01, 0.5)
        vmin -= padding
        vmax += padding
    if name == "cell_area":
        center = (
            float(np.mean(finite))
            if center is None and finite.size
            else ((vmin + vmax) / 2 if center is None else float(center))
        )
        center = min(max(center, np.nextafter(vmin, vmax)), np.nextafter(vmax, vmin))
        return AREA_QUALITY_CMAP, TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    cmap = plt.get_cmap("jet_r" if name in HIGHER_IS_BETTER else "jet")
    return cmap, Normalize(vmin=vmin, vmax=vmax)


def add_quality_direction_to_colorbar(colorbar, metric: str) -> None:
    """Annotate a vertical colorbar with the direction of improving quality."""
    axis = colorbar.ax
    if metric == "cell_area":
        axis.annotate(
            "Better near mean",
            xy=(-2.2, 0.5),
            xytext=(-2.2, 0.88),
            xycoords="axes fraction",
            textcoords="axes fraction",
            ha="center",
            va="center",
            rotation=90,
            arrowprops={"arrowstyle": "->", "linewidth": 1.2},
        )
        axis.annotate(
            "",
            xy=(-2.2, 0.5),
            xytext=(-2.2, 0.12),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "linewidth": 1.2},
        )
        return
    better_up = metric in HIGHER_IS_BETTER
    axis.annotate(
        "Better quality",
        xy=(-2.2, 0.82 if better_up else 0.18),
        xytext=(-2.2, 0.18 if better_up else 0.82),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        rotation=90,
        arrowprops={"arrowstyle": "->", "linewidth": 1.4},
    )


def add_shared_quality_colorbar(fig, axes, colored, metric: str):
    """Add a shared colorbar in a dedicated right margin without overlap."""
    axes = np.asarray(axes, dtype=object).ravel().tolist()
    # ``fig.colorbar(..., ax=axes)`` first shrinks the axes, but a later
    # ``subplots_adjust`` can expand the last subplot back underneath the
    # colorbar.  Reserve the margin here and use an explicit colorbar axis so
    # its position is independent of the number and aspect of method panels.
    fig.subplots_adjust(right=0.84)
    positions = [axis.get_position() for axis in axes]
    bottom = min(position.y0 for position in positions)
    top = max(position.y1 for position in positions)
    colorbar_axis = fig.add_axes([0.90, bottom, 0.012, top - bottom])
    colorbar = fig.colorbar(
        colored,
        cax=colorbar_axis,
        orientation="vertical",
        label=metric,
    )
    add_quality_direction_to_colorbar(colorbar, metric)
    return colorbar


def plot_quality_field(
    mesh, name: str, values: np.ndarray, axis, vmin=None, vmax=None, center=None
):
    cmap, norm = quality_plot_style(name, values, vmin=vmin, vmax=vmax, center=center)
    plotter = (
        plot_quad_mesh_colored_smooth
        if name in SMOOTH_FIELDS
        else plot_quad_mesh_colored
    )
    return plotter(
        mesh.vertices,
        mesh.faces,
        values,
        ax=axis,
        cmap=cmap,
        norm=norm,
    )


def quad_quality_fields(vertices: np.ndarray, nx: int, ny: int, metrics=None):
    """Evaluate requested quad metrics; dependencies retain their old definitions."""
    mesh = QuadMeshXY(nx, ny, np.asarray(vertices))
    selected = metric_names(metrics, QUAD_METRICS)
    if not selected:
        return {}, mesh
    evaluator = QuadMeshQualityEvaluator(mesh.vertices, mesh.faces)

    def maximum_angle():
        angles = evaluator.Angle()
        return np.max(np.minimum(angles, 360.0 - angles), axis=1)

    def scaled_jacobian():
        forward = np.roll(evaluator.cells, -1, axis=1) - evaluator.cells
        backward = evaluator.cells - np.roll(evaluator.cells, 1, axis=1)
        denominator = np.linalg.norm(forward, axis=2) * np.linalg.norm(backward, axis=2)
        scaled = evaluator.NodeJacobian() / np.maximum(denominator, np.finfo(float).eps)
        return np.min(scaled, axis=1)

    calculations = {
        "max_angle": maximum_angle,
        "min_angle": evaluator.MinAngle,
        "jacobian": lambda: evaluator.Jacobian()[0],
        "scaled_jacobian": scaled_jacobian,
        "skewness": evaluator.Skewness,
        "aspect_ratio": evaluator.AspectRatio,
        "orthogonality": evaluator.Orthogonality,
        "cell_area": evaluator.Area,
        "concave_vertex_count": evaluator.NumConcaveVertex,
        "self_intersection": evaluator.SelfIntersectingCells,
        "intersecting_cell": evaluator.IntersectingCells,
        "flipping_cell": evaluator.FlippingCells,
        "degenerate_cell": evaluator.DegenerateCells,
    }
    fields = {name: np.asarray(calculations[name](), dtype=float) for name in selected}
    return fields, mesh


def summarize_fields(fields: dict[str, np.ndarray]) -> dict:
    """Keep only the mean and the direction-aware worst value per metric."""
    summary = {}
    for name, raw in fields.items():
        values = np.asarray(raw, dtype=float)
        finite = values[np.isfinite(values)]
        lower_is_better = name in LOWER_IS_BETTER or name in FLAG_FIELDS
        item = {
            "mean": float(finite.mean()) if finite.size else None,
            "nonfinite_count": int(values.size - finite.size),
        }
        worst_key = "maximum" if lower_is_better else "minimum"
        item[worst_key] = (
            float(finite.max() if lower_is_better else finite.min())
            if finite.size
            else None
        )
        if name in FLAG_FIELDS:
            affected = int(np.count_nonzero(np.isfinite(values) & (values > 0)))
            item.update(
                total=float(np.nansum(values)),
                affected_cell_count=affected,
                cell_count=int(values.size),
                affected_cell_fraction=float(affected / max(values.size, 1)),
            )
        if name in {"jacobian", "scaled_jacobian"}:
            item["non_positive_cell_count"] = int(np.count_nonzero(values <= 0))
        summary[name] = item
    return summary


def invalid_quad_cells(
    fields: dict[str, np.ndarray], mesh: QuadMeshXY, tolerance: float = 1e-12
) -> list[dict]:
    """Return traceable diagnostics for every invalid or malformed quad."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    records = []
    for index, face in enumerate(np.asarray(mesh.faces, dtype=int)):
        reasons = []
        bad_values = {}
        indices_valid = bool(np.all((face >= 0) & (face < len(vertices))))
        if not indices_valid:
            reasons.append("node_index_out_of_range")
            coordinates = []
        else:
            points = vertices[face]
            coordinates = points.tolist()
            if not np.all(np.isfinite(points)):
                reasons.append("nonfinite_coordinate")
            if len(set(map(int, face))) < 4:
                reasons.append("duplicate_node_index")
            if "degenerate_cell" in fields and np.all(np.isfinite(points)):
                edge_lengths = np.linalg.norm(
                    np.roll(points, -1, axis=0) - points, axis=1
                )
                if np.any(edge_lengths <= tolerance):
                    reasons.append("zero_or_near_zero_edge")
        for name, values in fields.items():
            value = float(np.asarray(values)[index])
            if not np.isfinite(value):
                bad_values[name] = value
        if "self_intersection" in fields and fields["self_intersection"][index] > 0:
            reasons.append("self_intersection")
        if "intersecting_cell" in fields and fields["intersecting_cell"][index] > 0:
            reasons.append("intersection_with_another_cell")
        if (
            "concave_vertex_count" in fields
            and fields["concave_vertex_count"][index] > 0
        ):
            reasons.append("concave_cell")
        if "degenerate_cell" in fields and fields["degenerate_cell"][index] > 0:
            reasons.append("degenerate_cell")
        if "flipping_cell" in fields and fields["flipping_cell"][index] > 0:
            reasons.append("flipping_cell")
        if "jacobian" in fields and fields["jacobian"][index] <= tolerance:
            reasons.append("non_positive_or_near_zero_jacobian")
        if bad_values:
            reasons.append("nonfinite_quality_value")
        if reasons:
            records.append(
                {
                    "cell_index": int(index),
                    "node_indices": [int(item) for item in face],
                    "node_coordinates": coordinates,
                    "reasons": sorted(set(reasons)),
                    "quality_values": {
                        name: float(np.asarray(values)[index])
                        for name, values in fields.items()
                    },
                    "invalid_quality_values": bad_values,
                }
            )
    return records


def compute_quad_quality(vertices: np.ndarray, nx: int, ny: int, metrics=None) -> dict:
    """Compute reusable quality fields, topology, summaries, and diagnostics."""
    fields, mesh = quad_quality_fields(vertices, nx, ny, metrics=metrics)
    summary = summarize_fields(fields)
    if "orthogonality" in fields:
        orthogonality = np.asarray(fields["orthogonality"], dtype=float).reshape(
            ny - 1, nx - 1
        )
        boundary_mask = np.zeros_like(orthogonality, dtype=bool)
        boundary_mask[[0, -1], :] = True
        boundary_mask[:, [0, -1]] = True
        summary.update(
            summarize_fields(
                {"boundary_layer_orthogonality": orthogonality[boundary_mask]}
            )
        )
    invalid = invalid_quad_cells(fields, mesh)
    summary["invalid_cells"] = {
        "count": len(invalid),
        "fraction": len(invalid) / max(len(mesh.faces), 1),
        "assessed_metrics": list(fields),
        "reason_counts": {
            reason: sum(reason in item["reasons"] for item in invalid)
            for reason in sorted(
                {reason for item in invalid for reason in item["reasons"]}
            )
        },
    }
    return {
        "fields": fields,
        "mesh": mesh,
        "summary": summary,
        "invalid_cells": invalid,
    }


def save_invalid_cells(
    out_dir: Path,
    invalid: list[dict],
    mesh: QuadMeshXY,
    *,
    save_png: bool = True,
    save_pdf: bool = True,
) -> None:
    (out_dir / "invalid_cells.json").write_text(
        json.dumps(invalid, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    with (out_dir / "invalid_cells.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "cell_index",
                "node_indices",
                "node_coordinates",
                "reasons",
                "quality_values",
                "invalid_quality_values",
            ]
        )
        for item in invalid:
            writer.writerow(
                [
                    item["cell_index"],
                    json.dumps(item["node_indices"]),
                    json.dumps(item["node_coordinates"], allow_nan=True),
                    ";".join(item["reasons"]),
                    json.dumps(item["quality_values"], allow_nan=True),
                    json.dumps(item["invalid_quality_values"], allow_nan=True),
                ]
            )
    fig, axis = plt.subplots(figsize=(8, 7))
    plot_quad_meshXY(mesh, ax=axis, show_node_id=False)
    for item in invalid:
        coordinates = np.asarray(item["node_coordinates"], dtype=float)
        if coordinates.shape != (4, 2) or not np.all(np.isfinite(coordinates)):
            continue
        polygon = np.vstack((coordinates, coordinates[0]))
        axis.fill(coordinates[:, 0], coordinates[:, 1], color="red", alpha=0.35)
        axis.plot(polygon[:, 0], polygon[:, 1], color="red", linewidth=1.5)
        center = coordinates.mean(axis=0)
        axis.text(
            center[0],
            center[1],
            str(item["cell_index"]),
            color="darkred",
            fontsize=7,
            ha="center",
            va="center",
        )
    axis.set_title(f"Invalid cells ({len(invalid)})")
    fig.tight_layout()
    save_figure_formats(
        fig,
        out_dir / "invalid_cells_visualization",
        save_png=save_png,
        save_pdf=save_pdf,
        dpi=180,
    )
    plt.close(fig)


def save_boundary_error(
    out_dir: Path,
    predicted_boundary: np.ndarray,
    target_boundary: np.ndarray,
    *,
    make_plots: bool = True,
    save_png: bool = True,
    save_pdf: bool = True,
) -> dict:
    """Save pre-projection boundary errors for one learned model output."""
    predicted = np.asarray(predicted_boundary, dtype=float)
    target = np.asarray(target_boundary, dtype=float)
    if (
        predicted.shape != target.shape
        or predicted.ndim != 2
        or predicted.shape[1] != 2
    ):
        raise ValueError(
            "predicted_boundary and target_boundary must have matching shape (N, 2)"
        )
    delta = predicted - target
    absolute = np.abs(delta)
    distance = np.linalg.norm(delta, axis=1)
    finite = distance[np.isfinite(distance)]
    finite_components = delta[np.isfinite(delta)]
    maximum_index = int(np.nanargmax(distance)) if finite.size else None
    summary = {
        "point_count": int(len(distance)),
        "boundary_mse": (
            float(np.mean(finite_components**2)) if finite_components.size else None
        ),
        "mean_euclidean_error": float(np.mean(finite)) if finite.size else None,
        "maximum_euclidean_error": float(np.max(finite)) if finite.size else None,
        "maximum_error_boundary_index": maximum_index,
        "nonfinite_count": int(distance.size - finite.size),
    }
    (out_dir / "boundary_error_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (out_dir / "boundary_error_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "boundary_index",
                "predicted_x",
                "predicted_y",
                "target_x",
                "target_y",
                "error_x",
                "error_y",
                "absolute_error_x",
                "absolute_error_y",
                "euclidean_error",
            ]
        )
        for index in range(len(distance)):
            writer.writerow(
                [
                    index,
                    *predicted[index],
                    *target[index],
                    *delta[index],
                    *absolute[index],
                    distance[index],
                ]
            )
    if make_plots and (save_png or save_pdf):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].plot(np.arange(len(distance)), distance, linewidth=1.3)
        if maximum_index is not None:
            axes[0].scatter(
                [maximum_index],
                [distance[maximum_index]],
                color="red",
                zorder=3,
                label=f"maximum: index {maximum_index}",
            )
            axes[0].legend()
        axes[0].set(
            title="Raw boundary error by boundary index",
            xlabel="Boundary point index",
            ylabel="Euclidean error",
        )
        axes[0].grid(linestyle=":", alpha=0.35)
        scatter = axes[1].scatter(
            target[:, 0],
            target[:, 1],
            c=distance,
            cmap="jet",
            s=20,
        )
        axes[1].plot(target[:, 0], target[:, 1], color="black", linewidth=0.5)
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].set_title("Raw boundary error location")
        fig.colorbar(scatter, ax=axes[1], label="Euclidean error")
        fig.tight_layout()
        save_figure_formats(
            fig,
            out_dir / "boundary_error",
            save_png=save_png,
            save_pdf=save_pdf,
            dpi=200,
        )
        plt.close(fig)
    return summary


def save_quad_quality(
    out_dir: Path,
    vertices: np.ndarray,
    nx: int,
    ny: int,
    make_plots: bool = True,
    quality: dict | None = None,
    *,
    save_png: bool = True,
    save_pdf: bool = True,
    save_distribution_plots: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Prevent disabled formats from leaving stale figures from an older run.
    if not save_png:
        for path in out_dir.glob("*.png"):
            path.unlink()
    if not save_pdf:
        for path in out_dir.glob("*.pdf"):
            path.unlink()
    quality = quality or compute_quad_quality(vertices, nx, ny)
    fields, mesh = quality["fields"], quality["mesh"]
    np.savez_compressed(
        out_dir / "mesh_data.npz",
        vertices=mesh.vertices,
        faces=mesh.faces,
        nx=nx,
        ny=ny,
        indexing="index=j*nx+i",
    )
    np.savez_compressed(out_dir / "quality_data.npz", **fields)
    names = list(fields)
    with (out_dir / "quality_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["cell_index", *names])
        for index in range(len(mesh.faces)):
            writer.writerow([index, *(fields[name][index] for name in names)])
    summary = quality["summary"]
    invalid = quality["invalid_cells"]
    (out_dir / "quality_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if make_plots and (save_png or save_pdf):
        fig, axis = plt.subplots(figsize=(7, 6))
        plot_quad_meshXY(mesh, ax=axis, show_node_id=False)
        axis.set_title("mesh")
        fig.tight_layout()
        save_figure_formats(
            fig, out_dir / "mesh", save_png=save_png, save_pdf=save_pdf, dpi=180
        )
        plt.close(fig)
        for name, values in fields.items():
            fig, axes = plt.subplots(
                1,
                2 if save_distribution_plots else 1,
                figsize=(12 if save_distribution_plots else 7, 5),
                squeeze=False,
            )
            axis = axes[0, -1]
            if save_distribution_plots:
                plot_single_quality_distribution(axes[0, 0], name, values)
            _, colored = plot_quality_field(mesh, name, values, axis)
            axis.set_title(name)
            colorbar = fig.colorbar(colored, ax=axis, label=name)
            add_quality_direction_to_colorbar(colorbar, name)
            fig.tight_layout()
            save_figure_formats(
                fig,
                out_dir / f"quality_{name}",
                save_png=save_png,
                save_pdf=save_pdf,
                dpi=180,
            )
            plt.close(fig)
        save_invalid_cells(out_dir, invalid, mesh, save_png=save_png, save_pdf=save_pdf)
    else:
        (out_dir / "invalid_cells.json").write_text(
            json.dumps(invalid, indent=2, ensure_ascii=False, allow_nan=True),
            encoding="utf-8",
        )
        with (out_dir / "invalid_cells.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "cell_index",
                    "node_indices",
                    "node_coordinates",
                    "reasons",
                    "quality_values",
                    "invalid_quality_values",
                ]
            )
            for item in invalid:
                writer.writerow(
                    [
                        item["cell_index"],
                        json.dumps(item["node_indices"]),
                        json.dumps(item["node_coordinates"], allow_nan=True),
                        ";".join(item["reasons"]),
                        json.dumps(item["quality_values"], allow_nan=True),
                        json.dumps(item["invalid_quality_values"], allow_nan=True),
                    ]
                )
    return summary


def save_quality_comparison(
    out_dir: Path,
    meshes: dict[str, np.ndarray],
    nx: int,
    ny: int,
    metadata: dict,
    make_plots: bool = True,
    quality_cache: dict[str, dict] | None = None,
    *,
    save_png: bool = True,
    save_pdf: bool = True,
    save_distribution_plots: bool = True,
) -> dict:
    """Save paper-oriented same-scale quality comparisons for available methods."""
    meshes = {name: meshes[name] for name in ordered_methods(meshes)}
    out_dir.mkdir(parents=True, exist_ok=True)
    method_fields = {}
    method_meshes = {}
    summaries = {}
    for method, vertices in meshes.items():
        quality = (
            quality_cache[method]
            if quality_cache is not None and method in quality_cache
            else compute_quad_quality(vertices, nx, ny)
        )
        fields, mesh = quality["fields"], quality["mesh"]
        method_fields[method], method_meshes[method] = fields, mesh
        summaries[method] = {
            **quality["summary"],
            "invalid_cells": {
                "count": len(quality["invalid_cells"]),
                "fraction": len(quality["invalid_cells"]) / max(len(mesh.faces), 1),
                "assessed_metrics": list(fields),
            },
        }

    np.savez_compressed(out_dir / "comparison_mesh_data.npz", **meshes)
    np.savez_compressed(
        out_dir / "comparison_quality_data.npz",
        **{
            f"{method}__{metric}": values
            for method, fields in method_fields.items()
            for metric, values in fields.items()
        },
    )
    metrics = list(next(iter(method_fields.values())))
    with (out_dir / "comparison_quality_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "cell_index",
                *[f"{method}__{metric}" for method in meshes for metric in metrics],
            ]
        )
        for index in range((nx - 1) * (ny - 1)):
            writer.writerow(
                [
                    index,
                    *[
                        method_fields[method][metric][index]
                        for method in meshes
                        for metric in metrics
                    ],
                ]
            )

    metadata = {
        **metadata,
        "method_order": list(meshes),
        "method_display_names": [DISPLAY_NAMES.get(name, name) for name in meshes],
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary_payload = {"metadata": metadata, "methods": summaries}
    (out_dir / "quality_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (out_dir / "quality_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "method",
                "boundary_projection",
                "metric",
                "worst",
                "worst_direction",
                "mean",
                "nonfinite_count",
            ]
        )
        for method, summary in summaries.items():
            if method.startswith("HMONet"):
                projection = metadata["boundary_projection"].get("HMONet", "")
            elif method in {"MeshONet2D", "MeshONet"}:
                projection = metadata["boundary_projection"].get("MeshONet", "")
            elif method == "PI-HMONet":
                projection = metadata["boundary_projection"].get("PI-HMONet", "")
            elif method == "PI-MeshONet":
                projection = metadata["boundary_projection"].get("PI-MeshONet", "")
            else:
                projection = "not_applicable"
            for metric in metrics:
                item = summary[metric]
                writer.writerow(
                    [
                        DISPLAY_NAMES.get(method, method),
                        projection,
                        metric,
                        item.get("maximum", item.get("minimum")),
                        "maximum" if "maximum" in item else "minimum",
                        item.get("mean"),
                        item.get("nonfinite_count"),
                    ]
                )

    pairwise = []
    for metric in metrics:
        for left_index, left in enumerate(meshes):
            for right in list(meshes)[left_index + 1 :]:
                left_values = np.asarray(method_fields[left][metric], dtype=float)
                right_values = np.asarray(method_fields[right][metric], dtype=float)
                difference = left_values - right_values
                finite = difference[np.isfinite(difference)]
                pairwise.append(
                    {
                        "metric": metric,
                        "left_method": DISPLAY_NAMES.get(left, left),
                        "right_method": DISPLAY_NAMES.get(right, right),
                        "mean_left_minus_right": (
                            float(finite.mean()) if finite.size else None
                        ),
                        "maximum_absolute_difference": (
                            float(np.max(np.abs(finite))) if finite.size else None
                        ),
                    }
                )
    with (out_dir / "pairwise_quality_differences.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        columns = list(pairwise[0]) if pairwise else []
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(pairwise)
    (out_dir / "pairwise_quality_differences.json").write_text(
        json.dumps(pairwise, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not make_plots or not (save_png or save_pdf):
        return summaries
    names = list(meshes)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5), squeeze=False)
    for axis, name in zip(axes[0], names):
        plot_quad_meshXY(method_meshes[name], ax=axis, show_node_id=False)
        axis.set_title(DISPLAY_NAMES.get(name, name))
    fig.tight_layout()
    save_figure_formats(
        fig, out_dir / "mesh_comparison", save_png=save_png, save_pdf=save_pdf, dpi=220
    )
    plt.close(fig)
    for metric in metrics:
        combined = np.concatenate(
            [np.asarray(method_fields[name][metric]) for name in names]
        )
        finite = combined[np.isfinite(combined)]
        vmin, vmax = (
            (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        )
        if vmin == vmax:
            pad = max(abs(vmin) * 0.01, 0.5)
            vmin -= pad
            vmax += pad
        shared_center = (
            float(np.mean(finite)) if metric == "cell_area" and finite.size else None
        )
        fig, axes = plt.subplots(
            1, len(names), figsize=(5 * len(names), 5), squeeze=False
        )
        colored = None
        for axis, name in zip(axes[0], names):
            _, colored = plot_quality_field(
                method_meshes[name],
                metric,
                method_fields[name][metric],
                axis,
                vmin=vmin,
                vmax=vmax,
                center=shared_center,
            )
            axis.set_title(DISPLAY_NAMES.get(name, name))
        fig.subplots_adjust(left=0.04, right=0.84, bottom=0.08, top=0.90, wspace=0.18)
        add_shared_quality_colorbar(fig, axes[0], colored, metric)
        save_figure_formats(
            fig,
            out_dir / f"quality_{metric}",
            save_png=save_png,
            save_pdf=save_pdf,
            dpi=220,
        )
        plt.close(fig)
        if not save_distribution_plots:
            continue
        fig, axis = plt.subplots(figsize=(9, 6))
        plot_comparison_distribution(
            axis,
            metric,
            {name: method_fields[name][metric] for name in names},
        )
        fig.tight_layout()
        save_figure_formats(
            fig,
            out_dir / f"distribution_{metric}",
            save_png=save_png,
            save_pdf=save_pdf,
            dpi=220,
        )
        plt.close(fig)
    return summaries


def save_hex_quality(
    out_dir: Path,
    grid: np.ndarray,
    make_plots: bool = True,
    save_png: bool = True,
    save_pdf: bool = False,
    save_distribution_plots: bool = True,
    quality_fields: dict | None = None,
) -> dict:
    """Documentation string."""
    from src.utils.metrics import evaluate_hexahedral_grid_quality
    from src.data.mesh.topology import HexMeshXYZ

    grid = np.asarray(grid, dtype=float)
    nz, ny, nx, coordinate_dim = grid.shape
    if coordinate_dim != 3:
        raise ValueError("grid must have shape (Nz, Ny, Nx, 3)")
    mesh = HexMeshXYZ(nx, ny, nz, grid.reshape(-1, 3))
    fields = (
        quality_fields
        if quality_fields is not None
        else evaluate_hexahedral_grid_quality(grid)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "mesh_data.npz",
        vertices=mesh.vertices,
        cells=mesh.cells,
        nx=nx,
        ny=ny,
        nz=nz,
        indexing="index=(k*ny+j)*nx+i",
    )
    np.savez_compressed(out_dir / "quality_data.npz", **fields)
    summary = summarize_fields(fields)
    metric_names = list(fields)
    with (out_dir / "quality_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["cell_index", *metric_names])
        for cell_index in range(len(mesh.cells)):
            writer.writerow(
                [cell_index, *[fields[name][cell_index] for name in metric_names]]
            )
    (out_dir / "quality_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (out_dir / "quality_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "metric",
                "worst",
                "worst_direction",
                "mean",
                "nonfinite_count",
            ]
        )
        for metric_name in metric_names:
            item = summary[metric_name]
            writer.writerow(
                [
                    metric_name,
                    item.get("maximum", item.get("minimum")),
                    "maximum" if "maximum" in item else "minimum",
                    item.get("mean"),
                    item.get("nonfinite_count"),
                ]
            )

    invalid_mask = np.zeros(len(mesh.cells), dtype=bool)
    for name in ("flipped_cell", "degenerate_cell"):
        if name in fields:
            invalid_mask |= np.asarray(fields[name], dtype=bool)
    invalid_records = [
        {
            "cell_index": int(cell_index),
            "node_indices": mesh.cells[cell_index].tolist(),
            "quality_values": {
                name: float(fields[name][cell_index]) for name in metric_names
            },
        }
        for cell_index in np.flatnonzero(invalid_mask)
    ]
    (out_dir / "invalid_cells.json").write_text(
        json.dumps(invalid_records, indent=2), encoding="utf-8"
    )
    with (out_dir / "invalid_cells.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["cell_index", "node_indices", *metric_names])
        for record in invalid_records:
            writer.writerow(
                [
                    record["cell_index"],
                    json.dumps(record["node_indices"]),
                    *[record["quality_values"][name] for name in metric_names],
                ]
            )
    if make_plots and save_distribution_plots:
        for name, values in fields.items():
            fig, axis = plt.subplots(figsize=(7, 5))
            axis.hist(values[np.isfinite(values)], bins=30, edgecolor="black")
            axis.set(title=f"3D {name}", xlabel=name, ylabel="cell count")
            fig.tight_layout()
            if save_png:
                fig.savefig(out_dir / f"quality_{name}.png", dpi=180)
            if save_pdf:
                fig.savefig(out_dir / f"quality_{name}.pdf")
            plt.close(fig)
    return summary
