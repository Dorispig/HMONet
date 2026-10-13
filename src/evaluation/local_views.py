"""Run configured zooms using the same saved data as standalone zoom commands."""

from argparse import Namespace
from pathlib import Path
from src.utils.quality_selection import selected_quality_metrics
from .artifacts import read_json


def prune_disabled_quality_zooms(directory, dimension, options):
    """Remove obsolete quality zoom artifacts using their recorded filenames."""
    zoom_root = (Path(directory) / "zoom_views").resolve()
    for marker in zoom_root.rglob("*zoom_metadata.json"):
        metadata = read_json(marker, {})
        settings = metadata.get("options", metadata)
        metric = settings.get("metric")
        selected = selected_quality_metrics(
            options,
            dimension,
            surface=dimension == 3 and settings.get("mode", "slice") == "slice",
        )
        if metric in selected or settings.get("plot") == "mesh":
            continue
        if "files" in metadata:
            names = [*metadata["files"], marker.name]
        else:
            stem = marker.name.removesuffix("_zoom_metadata.json")
            names = [marker.name] + [
                f"{stem}_{view}.{extension}"
                for view in ("zoom", "overview_and_zoom")
                for extension in ("png", "pdf")
            ]
        for name in names:
            target = (marker.parent / name).resolve()
            if not target.is_relative_to(zoom_root):
                raise ValueError("Invalid path in zoom metadata")
            if target.is_file():
                target.unlink()


def run_local_views(directory, dimension, options):
    prune_disabled_quality_zooms(directory, dimension, options)
    outputs = []
    for job in options.get("zoom_views", []):
        job = dict(job)
        if not job.pop("enabled", True):
            continue
        groups = job.pop("groups", None)
        if groups and Path(directory).name not in groups:
            continue
        metric = job.get(
            "metric",
            (
                "minimum_scaled_jacobian"
                if dimension == 3 and job.get("mode") == "volume"
                else "scaled_jacobian"
            ),
        )
        selected = selected_quality_metrics(
            options,
            dimension,
            surface=dimension == 3 and job.get("mode", "slice") == "slice",
        )
        if metric not in selected:
            if job.get("center_on") == "worst" or job.get("plot") == "quality":
                print(f"  skip quality/worst zoom: {metric} is disabled", flush=True)
                continue
            # Explicit coordinate/index crops remain useful without quality.
            job["plot"] = "mesh"
        if dimension == 3:
            from .zoom_3d import generate_zoom

            job.setdefault("save_plots", options["save_plots"])
            job.setdefault("save_png", options["save_png"])
            job.setdefault("save_pdf", options["save_pdf"])
            job.setdefault("force", options["force_reanalyze"])
            outputs.append(generate_zoom(directory, job))
        else:
            from scripts.evaluate.plot_result_zoom import (
                load_comparison,
                resolve_bounds,
                save_zoom,
            )

            needs_metric = (
                job.get("plot", "both") != "mesh" or job.get("center_on") == "worst"
            )
            metric = metric if needs_metric else None
            args = Namespace(
                xlim=job.get("xlim"),
                ylim=job.get("ylim"),
                i_range=job.get("i_range"),
                j_range=job.get("j_range"),
                cell_index=job.get("cell_index"),
                center_on=job.get("center_on"),
                cell_padding=job.get("cell_padding", 2),
                color_scale=job.get("color_scale", "global"),
                output_dir=None,
                parameter_dir=None,
                save_png=options["save_plots"] and options["save_png"],
                save_pdf=options["save_plots"] and options["save_pdf"],
            )
            meshes, fields, nx, ny = load_comparison(Path(directory), metric)
            xlim, ylim, selection = resolve_bounds(meshes, fields, nx, ny, metric, args)
            plots = (
                ("mesh", "quality")
                if job.get("plot", "both") == "both"
                else (job["plot"],)
            )
            for plot in plots:
                save_zoom(
                    Path(directory),
                    meshes,
                    fields if plot == "quality" else {},
                    nx,
                    ny,
                    plot,
                    metric if plot == "quality" else None,
                    xlim,
                    ylim,
                    selection,
                    args,
                )
            outputs.append(str(Path(directory) / "zoom_views"))
    return outputs
