"""Mesh quality, true method comparisons, middle sections, and point errors."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
import numpy as np

from src.data.mesh.topology import HexMeshXYZ, QuadMeshXY
from .artifacts import read_json, write_json, write_csv, save_quality_table
from .method_order import ordered_methods, method_label
from .quality_output import (
    compute_quad_quality,
    invalid_quad_cells,
    save_quad_quality,
    save_hex_quality,
    save_quality_comparison,
    summarize_fields,
    quality_plot_style,
    save_figure_formats,
)
from .quality_3d import middle_surface_grids, surface_quad_quality_fields, HIGHER
from .zoom_3d import quad_cells, draw_volume, DEFAULTS as ZOOM_DEFAULTS
from src.utils.metrics import evaluate_hexahedral_grid_quality
from src.utils.quality_selection import selected_quality_metrics


def plot_enabled(options):
    return options["save_plots"] and (options["save_png"] or options["save_pdf"])


def save_figure(fig, path, options):
    save_figure_formats(
        fig,
        Path(path),
        save_png=options["save_png"],
        save_pdf=options["save_pdf"],
        dpi=180,
    )
    plt.close(fig)


def load_quality(directory, points, shape, options=None):
    """Load existing numeric quality for comparisons, without rerunning detection."""
    directory = Path(directory)
    if options is not None and not selected_quality_metrics(options, len(shape)):
        return empty_quality(points, shape)
    with np.load(directory / "quality_data.npz") as archive:
        fields = {key: archive[key].copy() for key in archive.files}
    summary = read_json(directory / "quality_summary.json")
    if len(shape) == 2:
        mesh = QuadMeshXY(*shape, points)
        return {
            "fields": fields,
            "summary": summary,
            "mesh": mesh,
            "invalid_cells": read_json(directory / "invalid_cells.json", []),
        }
    return {"fields": fields, "summary": summary}


def empty_quality(points, shape):
    """Carry mesh topology without evaluating any cell-quality indicators."""
    quality = {"fields": {}, "summary": {}, "invalid_cells": []}
    if len(shape) == 2:
        quality["mesh"] = QuadMeshXY(*shape, points)
    return quality


def save_method(directory, method, points, shape, options):
    """Save mesh geometry and only the selected quality calculations."""
    directory = Path(directory)
    quality_dir = directory / "grid_quality"
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "mesh.npy", points)
    selected = selected_quality_metrics(options, len(shape))
    quality = empty_quality(points, shape)
    if len(shape) == 2:
        if selected:
            quality = compute_quad_quality(points, *shape, metrics=selected)
            save_quad_quality(
                quality_dir,
                points,
                *shape,
                quality=quality,
                make_plots=options["save_plots"],
                save_png=options["save_png"],
                save_pdf=options["save_pdf"],
                save_distribution_plots=options["save_distribution_plots"],
            )
        np.savez_compressed(
            directory / "mesh_data.npz",
            vertices=points,
            faces=quality["mesh"].faces,
            nx=shape[0],
            ny=shape[1],
        )
    else:
        grid = points.reshape(*shape[::-1], 3)
        if selected:
            fields = evaluate_hexahedral_grid_quality(grid, metrics=selected)
            summary = save_hex_quality(
                quality_dir,
                grid,
                quality_fields=fields,
                make_plots=options["save_plots"],
                save_png=options["save_png"],
                save_pdf=options["save_pdf"],
                save_distribution_plots=options["save_distribution_plots"],
            )
            quality = {"fields": fields, "summary": summary}
            flags = [
                name for name in ("flipped_cell", "degenerate_cell") if name in fields
            ]
            if flags:
                invalid = np.logical_or.reduce([fields[name] > 0 for name in flags])
                summary["invalid_cells"] = {
                    "count": int(np.count_nonzero(invalid)),
                    "fraction": float(np.mean(invalid)),
                    "assessed_metrics": flags,
                }
                write_json(quality_dir / "quality_summary.json", summary)
            if plot_enabled(options):
                plot_volumes(
                    quality_dir, {method: points}, {method: fields}, shape, options
                )
        mesh = HexMeshXYZ(*shape, points)
        np.savez_compressed(
            directory / "mesh_data.npz",
            vertices=points,
            cells=mesh.cells,
            nx=shape[0],
            ny=shape[1],
            nz=shape[2],
        )
        save_sections(directory, {method: points}, shape, options)
    if selected:
        save_quality_table(
            quality_dir / "quality_summary.csv", {method: quality["summary"]}
        )
    if plot_enabled(options):
        if len(shape) == 2:
            from src.utils.visualize import plot_quad_meshXY

            fig, ax = plt.subplots(figsize=(7, 6))
            plot_quad_meshXY(quality["mesh"], ax=ax, show_node_id=False)
        else:
            from src.utils.visualize import plot_hex_mesh_paper

            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
            plot_hex_mesh_paper(points, *shape, ax=ax)
        ax.set_title(method_label(method))
        save_figure(fig, directory / "mesh", options)
    return quality


def save_sections(directory, meshes, shape, options):
    """Intrinsic 3-D section metrics, drawn with the same axes as local views."""
    meshes = {name: meshes[name] for name in ordered_methods(meshes)}
    directory = Path(directory)
    sections = {
        m: middle_surface_grids(v.reshape(*shape[::-1], 3)) for m, v in meshes.items()
    }
    selected = selected_quality_metrics(options, 3, surface=True)
    all_summaries = {}
    intersection_rows = []
    projection_axes = {"x_mid": (1, 2), "y_mid": (0, 2), "z_mid": (0, 1)}
    for section_name, axes in projection_axes.items():
        target = directory / "boundary_faces" / section_name
        target.mkdir(parents=True, exist_ok=True)
        coordinates = {m: sections[m][section_name] for m in meshes}
        fields = {
            m: surface_quad_quality_fields(s, metrics=selected) if selected else {}
            for m, s in coordinates.items()
        }
        summaries = {m: summarize_fields(f) for m, f in fields.items()}
        all_summaries[section_name] = summaries
        if selected:
            write_json(target / "quality_comparison.json", summaries)
            save_quality_table(target / "quality_summary.csv", summaries)
            save_quality_table(target / "paper_surface_quality_table.csv", summaries)
        write_json(
            target / "metadata.json",
            {
                "method_order": list(meshes),
                "logical_section": section_name,
                "logical_position": 0.5,
                "axes": ["xyz"[a] for a in axes],
                "coordinates": "physical",
                "evaluated_metrics": list(selected),
                "quality_definition": "Intrinsic quadrilateral metrics on physical 3-D middle sections; self_intersection tests opposite edges in a local tangent plane.",
            },
        )
        np.savez_compressed(target / "comparison_mesh_data.npz", **coordinates)
        if len(meshes) == 1:
            np.save(target / "mesh.npy", coordinates[next(iter(meshes))])
        if selected:
            np.savez_compressed(
                target / "comparison_quality_data.npz",
                **{f"{m}__{k}": v for m, f in fields.items() for k, v in f.items()},
            )
            rows = []
            for method, f in fields.items():
                rows.extend(
                    {
                        "method": method,
                        "cell_index": i,
                        **{k: float(v[i]) for k, v in f.items()},
                    }
                    for i in range(len(next(iter(f.values()))))
                )
                for metric in (
                    "self_intersection",
                    "intersecting_cell",
                    "concave_vertex_count",
                    "flipping_cell",
                    "degenerate_cell",
                ):
                    if metric not in summaries[method]:
                        continue
                    summary = summaries[method][metric]
                    intersection_rows.append(
                        {
                            "method": method,
                            "section": section_name,
                            "metric": metric,
                            "affected_cell_count": summary["affected_cell_count"],
                            "cell_count": summary["cell_count"],
                            "affected_cell_fraction": summary["affected_cell_fraction"],
                        }
                    )
            write_csv(target / "quality_data.csv", rows)
            if len(meshes) == 1:
                method = next(iter(meshes))
                np.savez_compressed(target / "quality_data.npz", **fields[method])
                write_json(target / "quality_summary.json", summaries[method])
            else:
                write_json(target / "quality_summary.json", summaries)
        if not plot_enabled(options):
            continue
        names = list(meshes)
        for metric in [None, *next(iter(fields.values()))]:
            combined = (
                np.concatenate([fields[m][metric] for m in names])
                if metric
                else np.array([0.0, 1.0])
            )
            cmap, norm = quality_plot_style(metric, combined)
            points = np.concatenate(
                [s[..., list(axes)].reshape(-1, 2) for s in coordinates.values()]
            )
            low, high = points.min(axis=0), points.max(axis=0)
            padding = max(float((high - low).max()), 1e-8) * 0.04
            fig, panels = plt.subplots(
                1, len(names), figsize=(4.5 * len(names), 4.5), squeeze=False
            )
            for ax, method in zip(panels[0], names):
                s = coordinates[method]
                cells = quad_cells(s)
                verts = s[..., list(axes)].reshape(-1, 2)
                colors = cmap(norm(fields[method][metric])) if metric else "none"
                ax.add_collection(
                    PolyCollection(
                        verts[cells],
                        facecolors=colors,
                        edgecolors="black",
                        linewidths=0.3,
                    )
                )
                ax.set_xlim(low[0] - padding, high[0] + padding)
                ax.set_ylim(low[1] - padding, high[1] + padding)
                ax.set_aspect("equal")
                ax.set(
                    xlabel="xyz"[axes[0]],
                    ylabel="xyz"[axes[1]],
                    title=method_label(method),
                )
            if metric:
                fig.colorbar(
                    ScalarMappable(norm=norm, cmap=cmap),
                    ax=panels[0].tolist(),
                    label=metric,
                    fraction=0.02,
                    pad=0.03,
                )
            fig.suptitle(section_name)
            save_figure(
                fig, target / (f"quality_{metric}" if metric else "mesh"), options
            )
    if selected:
        write_json(directory / "surface_quality_summary.json", all_summaries)
        if intersection_rows:
            write_csv(directory / "self_intersection_table.csv", intersection_rows)
        write_csv(
            directory / "surface_quality_table.csv",
            [
                {"method": m, "section": s, "metric": metric, **values}
                for s, methods in all_summaries.items()
                for m, summary in methods.items()
                for metric, values in summary.items()
            ],
            [
                "method",
                "section",
                "metric",
                "minimum",
                "mean",
                "maximum",
                "affected_cell_count",
                "cell_count",
                "affected_cell_fraction",
            ],
        )


def plot_volumes(directory, meshes, fields, shape, options):
    """Shared-scale physical surface colors and complete wireframe comparisons."""
    names = ordered_methods(meshes)
    all_points = np.concatenate(list(meshes.values()))
    limits = (all_points.min(axis=0), all_points.max(axis=0))
    cells = HexMeshXYZ(*shape, next(iter(meshes.values()))).cells
    for metric in [None, *next(iter(fields.values()))]:
        values = (
            np.concatenate([fields[m][metric] for m in names])
            if metric
            else np.array([0, 1])
        )
        cmap, norm = quality_plot_style(metric, values)
        fig = plt.figure(figsize=(5 * len(names), 5))
        panels = []
        for i, method in enumerate(names, 1):
            ax = fig.add_subplot(1, len(names), i, projection="3d")
            draw_volume(
                ax,
                meshes[method],
                cells,
                fields[method].get(metric),
                limits,
                {**ZOOM_DEFAULTS, "volume_style": "surface"},
                norm,
                cmap,
                metric is not None,
            )
            ax.set_title(method_label(method))
            panels.append(ax)
        if metric:
            fig.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                ax=panels,
                label=metric,
                fraction=0.02,
                pad=0.03,
            )
        save_figure(
            fig,
            Path(directory) / (f"quality_{metric}" if metric else "mesh_comparison"),
            options,
        )


def save_comparison(directory, meshes, qualities, shape, metadata, options):
    """Compare available methods in the fixed TFI/PDE/MeshONet/HMONet order."""
    meshes = {name: meshes[name] for name in ordered_methods(meshes)}
    qualities = {name: qualities[name] for name in meshes}
    metadata = {**metadata, "method_order": list(meshes)}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not selected_quality_metrics(options, len(shape)):
        np.savez_compressed(directory / "comparison_mesh_data.npz", **meshes)
        write_json(directory / "metadata.json", metadata)
        if len(shape) == 3:
            save_sections(directory, meshes, shape, options)
            if plot_enabled(options):
                plot_volumes(directory, meshes, {m: {} for m in meshes}, shape, options)
        elif plot_enabled(options):
            from src.utils.visualize import plot_quad_meshXY

            fig, axes = plt.subplots(
                1, len(meshes), figsize=(5 * len(meshes), 5), squeeze=False
            )
            for ax, method in zip(axes[0], meshes):
                plot_quad_meshXY(qualities[method]["mesh"], ax=ax, show_node_id=False)
                ax.set_title(method_label(method))
            fig.tight_layout()
            save_figure(fig, directory / "mesh_comparison", options)
        return {m: {} for m in meshes}
    if len(shape) == 2:
        summary = save_quality_comparison(
            directory,
            meshes,
            *shape,
            metadata,
            quality_cache=qualities,
            make_plots=options["save_plots"],
            save_png=options["save_png"],
            save_pdf=options["save_pdf"],
            save_distribution_plots=options["save_distribution_plots"],
        )
    else:
        summary = {m: q["summary"] for m, q in qualities.items()}
        np.savez_compressed(directory / "comparison_mesh_data.npz", **meshes)
        np.savez_compressed(
            directory / "comparison_quality_data.npz",
            **{
                f"{m}__{k}": v
                for m, q in qualities.items()
                for k, v in q["fields"].items()
            },
        )
        write_json(directory / "metadata.json", metadata)
        write_json(
            directory / "quality_summary.json",
            {"metadata": metadata, "methods": summary},
        )
        rows = [
            {
                "method": m,
                "cell_index": i,
                **{k: float(v[i]) for k, v in q["fields"].items()},
            }
            for m, q in qualities.items()
            for i in range(len(next(iter(q["fields"].values()))))
        ]
        write_csv(directory / "comparison_quality_data.csv", rows)
        save_sections(directory, meshes, shape, options)
        if plot_enabled(options):
            plot_volumes(
                directory,
                meshes,
                {m: q["fields"] for m, q in qualities.items()},
                shape,
                options,
            )
            from .quality_3d import plot_surface_slice_comparison

            plot_surface_slice_comparison(
                directory,
                {m: q["fields"] for m, q in qualities.items()},
                *shape,
                save_png=options["save_png"],
                save_pdf=options["save_pdf"],
            )
    save_quality_table(directory / "quality_summary.csv", summary)
    save_quality_table(
        directory
        / (
            "paper_volume_quality_table.csv"
            if len(shape) == 3
            else "paper_quality_table.csv"
        ),
        summary,
    )
    # Keep the mesh and its metadata at the group root for standalone zooms;
    # all quality colors and their numeric backing data live together.
    quality_directory = directory / "grid_quality"
    quality_directory.mkdir(exist_ok=True)
    for path in list(directory.iterdir()):
        if path.is_file() and (
            path.name.startswith(
                (
                    "quality_",
                    "distribution_",
                    "comparison_quality_",
                    "pairwise_quality_",
                    "paper_quality_",
                    "paper_volume_quality_",
                )
            )
        ):
            path.replace(quality_directory / path.name)
    return summary


def save_point_errors(directory, meshes, raw, shape, options):
    """Persist coordinate MSE per point and maximum vertex MSE per cell."""
    if "PDE_Winslow" not in meshes:
        return {}
    reference = meshes["PDE_Winslow"]
    compared = {
        m: p
        for m, p in meshes.items()
        if m.startswith(("HMONet", "MeshONet", "PI-MeshONet", "PI-HMONet", "Attention-PI-HMONet"))
    }
    compared.update({f"{m}_raw": p for m, p in raw.items()})
    compared = {name: compared[name] for name in ordered_methods(compared)}
    if not compared:
        return {}
    cells = (
        QuadMeshXY(*shape, reference).faces
        if len(shape) == 2
        else HexMeshXYZ(*shape, reference).cells
    )
    points = {m: np.mean((p - reference) ** 2, axis=-1) for m, p in compared.items()}
    cell_values = {m: v[cells].max(axis=1) for m, v in points.items()}
    directory = Path(directory)
    np.savez_compressed(directory / "point_loss_data.npz", **points)
    np.savez_compressed(directory / "cell_loss_data.npz", **cell_values)
    write_csv(
        directory / "point_loss_data.csv",
        [
            dict(
                point_index=i,
                **{
                    f"winslow_{axis}": float(reference[i, j])
                    for j, axis in enumerate("xyz"[: len(shape)])
                },
                **{m: float(v[i]) for m, v in points.items()},
            )
            for i in range(len(reference))
        ],
    )
    write_csv(
        directory / "cell_loss_data.csv",
        [
            dict(cell_index=i, **{m: float(v[i]) for m, v in cell_values.items()})
            for i in range(len(cells))
        ],
    )
    summary = {
        m: {
            "mse_vs_winslow": float(v.mean()),
            "max_point_mse_vs_winslow": float(v.max()),
            "mean_cell_max_vertex_mse_vs_winslow": float(cell_values[m].mean()),
            "max_cell_max_vertex_mse_vs_winslow": float(cell_values[m].max()),
        }
        for m, v in points.items()
    }
    write_json(
        directory / "point_loss_summary.json",
        {
            "definition": "Mean squared coordinate components against this evaluation's PDE; boundary nodes included, no dimension multiplier.",
            "methods": summary,
        },
    )
    if plot_enabled(options):
        positive = [v[v > 0] for v in points.values() if np.any(v > 0)]
        positive = np.concatenate(positive) if positive else np.array([1e-16])
        color_min = max(float(positive.min()), np.finfo(float).tiny)
        color_max = max(float(positive.max()), color_min * 10)
        norm = LogNorm(color_min, color_max)
        cmap = plt.get_cmap("jet")
        fig = plt.figure(figsize=(5 * len(points), 9))
        for column, (method, values) in enumerate(points.items(), 1):
            ax = fig.add_subplot(
                2, len(points), column, projection="3d" if len(shape) == 3 else None
            )
            colored = ax.scatter(
                *reference.T,
                c=np.maximum(values, color_min),
                cmap=cmap,
                norm=norm,
                s=12,
            )
            ax.set_title(f"{method}: point MSE")
            if len(shape) == 2:
                ax.set_aspect("equal")
            fig.colorbar(colored, ax=ax, label="Point coordinate MSE")
            cell_axis = fig.add_subplot(
                2,
                len(points),
                len(points) + column,
                projection="3d" if len(shape) == 3 else None,
            )
            clipped = np.maximum(cell_values[method], color_min)
            if len(shape) == 2:
                colored_cells = PolyCollection(
                    reference[cells],
                    array=clipped,
                    cmap=cmap,
                    norm=norm,
                    edgecolors="black",
                    linewidths=0.3,
                )
                cell_axis.add_collection(colored_cells)
                cell_axis.autoscale_view()
                cell_axis.set_aspect("equal")
            else:
                draw_volume(
                    cell_axis,
                    reference,
                    cells,
                    clipped,
                    (reference.min(axis=0), reference.max(axis=0)),
                    {**ZOOM_DEFAULTS, "volume_style": "surface"},
                    norm,
                    cmap,
                    True,
                )
            cell_axis.set_title(f"{method}\nMaximum vertex MSE per cell")
            fig.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                ax=cell_axis,
                label="Maximum vertex MSE",
            )
        fig.tight_layout()
        save_figure(fig, directory / "08_point_loss_values", options)
        if options["save_distribution_plots"]:
            fig, ax = plt.subplots(figsize=(9, 6))
            bins = np.geomspace(color_min, color_max, 31)
            for method, values in points.items():
                ax.hist(
                    np.maximum(values, color_min),
                    bins=bins,
                    histtype="step",
                    linewidth=2,
                    label=method,
                )
            ax.set(
                xscale="log",
                yscale="log",
                xlabel="Point coordinate MSE against Winslow",
                ylabel="Point count",
                title="Model prediction loss comparison",
            )
            ax.legend()
            fig.tight_layout()
            save_figure(fig, directory / "09_point_loss_comparison", options)
    return summary
