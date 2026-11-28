"""Generate reusable training, timing, and quality comparison figures.

Six-method plots use archived quality and timing records. The legacy
five-method path can also derive boundary variants from saved raw meshes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from src.engine.training_comparison import ensure_training_comparison
from src.evaluation.inference import postprocess
from src.evaluation.quality_output import compute_quad_quality
from src.metrics import evaluate_hexahedral_grid_quality
from src.utils.quality_selection import HEX_METRICS, QUAD_METRICS


BOUNDARY_PROCESSING_GROUPS = {
    "raw": ("TFI", "PDE", "MeshONet raw", "PI-MeshONet raw", "HMONet"),
    "direct": ("TFI", "PDE", "MeshONet direct", "PI-MeshONet direct", "HMONet"),
    "idw": ("TFI", "PDE", "MeshONet IDW", "PI-MeshONet IDW", "HMONet"),
}
SIX_METHOD_GROUP = "TFI_PDE_M_PI_H_RM"
RAW_COMPARISON = Path("comparisons/CCAP/H_raw__M_raw") / SIX_METHOD_GROUP
LOWER_IS_BETTER = {
    "mse_vs_pde",
    "max_angle",
    "skewness",
    "aspect_ratio",
    "center_condition_number",
    "edge_aspect_ratio",
    "concave_vertex_count",
    "self_intersection",
    "intersecting_cell",
    "flipping_cell",
    "degenerate_cell",
    "flipped_cell",
}
FLAG_METRICS = {
    "concave_vertex_count",
    "self_intersection",
    "intersecting_cell",
    "flipping_cell",
    "degenerate_cell",
    "flipped_cell",
}


def dataset_name(value: str) -> str:
    if value in {"", ".", ".."} or any(sep in value for sep in ("/", "\\", ":")):
        raise argparse.ArgumentTypeError("Use a short dataset directory name")
    return value


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def boundary_indices(shape: tuple[int, ...]) -> np.ndarray:
    """Return structured boundary nodes for x-fastest flattened storage."""
    if len(shape) == 2:
        nx, ny = shape
        indices = np.arange(nx * ny, dtype=int).reshape(ny, nx)
        mask = np.zeros((ny, nx), dtype=bool)
        mask[[0, -1], :] = True
        mask[:, [0, -1]] = True
    elif len(shape) == 3:
        nx, ny, nz = shape
        indices = np.arange(nx * ny * nz, dtype=int).reshape(nz, ny, nx)
        mask = np.zeros((nz, ny, nx), dtype=bool)
        mask[[0, -1], :, :] = True
        mask[:, [0, -1], :] = True
        mask[:, :, [0, -1]] = True
    else:
        raise ValueError("Only 2-D and 3-D structured shapes are supported")
    return indices[mask]


def worst_key(metric: str) -> str:
    return "maximum" if metric in LOWER_IS_BETTER or metric in FLAG_METRICS else "minimum"


def quality_axis_limits(metric: str, values: list[float]) -> tuple[float, float | None] | None:
    """Choose a readable, non-misleading y range for a quality comparison."""
    finite = [float(value) for value in values if math.isfinite(value)]
    if not finite:
        return None
    low, high = min(finite), max(finite)
    if metric in FLAG_METRICS or low <= 0.0:
        return (0.0, None) if low >= 0.0 else None
    span = high - low
    reference = max(abs(low), abs(high), 1.0e-12)
    if span <= 0.20 * reference:
        padding = max(0.25 * span, 0.0025 * reference)
        return max(0.0, low - padding), high + padding
    return 0.0, None


def _mesh_path(root: Path, method: str) -> Path | None:
    if method == "TFI":
        path = root / "TFI" / "mesh.npy"
    elif method == "PDE":
        path = root / "PDE" / "mesh.npy"
    elif method == "MeshONet":
        path = root / "MeshONet" / "raw" / "mesh.npy"
    elif method == "PI-MeshONet":
        path = root / "PI-MeshONet" / "raw" / "mesh.npy"
    elif method == "HMONet":
        candidates = sorted(
            (root / "HMONet").glob("*/raw/mesh.npy"),
            key=lambda item: ("CCAP" not in item.parts, str(item)),
        )
        return candidates[0] if candidates else None
    else:
        raise KeyError(method)
    return path if path.is_file() else None


def _bundle_mesh(root: Path, method: str) -> np.ndarray | None:
    file_name = (
        "generated_raw_model_meshes.npz"
        if method in {"HMONet", "MeshONet", "PI-MeshONet"}
        else "generated_meshes.npz"
    )
    path = root / file_name
    if not path.is_file():
        return None
    keys = {
        "TFI": ("TFI",),
        "PDE": ("PDE_Winslow", "PDE"),
        "HMONet": ("HMONet",),
        "MeshONet": ("MeshONet",),
        "PI-MeshONet": ("PI-MeshONet",),
    }[method]
    with np.load(path) as payload:
        for key in keys:
            if key in payload:
                return np.asarray(payload[key], dtype=float)
    return None


def load_base_meshes(root: Path, shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    expected = math.prod(shape)
    meshes = {}
    for method in ("TFI", "PDE", "HMONet", "MeshONet", "PI-MeshONet"):
        path = _mesh_path(root, method)
        mesh = np.asarray(np.load(path), dtype=float) if path else _bundle_mesh(root, method)
        if mesh is None:
            raise FileNotFoundError(f"Missing reusable {method} mesh below {root}")
        mesh = mesh.reshape(-1, len(shape))
        if mesh.shape != (expected, len(shape)):
            raise ValueError(
                f"{method} mesh has shape {mesh.shape}; expected {(expected, len(shape))}"
            )
        meshes[method] = mesh
    return meshes


def timed_projection(
    raw: np.ndarray,
    exact: np.ndarray,
    indices: np.ndarray,
    mode: str,
    *,
    power: float,
    repeats: int,
    chunk_size: int,
) -> tuple[np.ndarray, float]:
    samples = []
    result = None
    for _ in range(max(repeats, 1)):
        started = time.perf_counter()
        result = postprocess(
            raw,
            exact,
            indices,
            mode,
            power=power,
            chunk_size=chunk_size,
        )
        samples.append(time.perf_counter() - started)
    return result, float(np.median(samples))


def load_timings(root: Path) -> dict:
    path = root / "evaluation_timing.json"
    if path.is_file():
        return load_json(path).get("methods", {})
    path = root / "results.json"
    return load_json(path).get("timings", {}) if path.is_file() else {}


def inference_seconds(record: dict) -> float:
    return float(
        record.get("network_inference_seconds", record.get("total_model_seconds", record.get("total_seconds", 0.0)))
        or 0.0
    )


def quality_statistics(
    meshes: dict[str, np.ndarray], shape: tuple[int, ...], metrics: tuple[str, ...]
) -> dict[str, dict[str, dict[str, float]]]:
    result = {}
    for label, mesh in meshes.items():
        if len(shape) == 2:
            fields = compute_quad_quality(mesh, *shape, metrics=metrics)["fields"]
        else:
            grid = mesh.reshape(*shape[::-1], 3)
            fields = evaluate_hexahedral_grid_quality(grid, metrics=metrics)
        result[label] = {}
        for metric, raw in fields.items():
            values = np.asarray(raw, dtype=float)
            finite = values[np.isfinite(values)]
            key = worst_key(metric)
            result[label][metric] = {
                "mean": float(finite.mean()) if finite.size else math.nan,
                "worst": float(finite.max() if key == "maximum" else finite.min())
                if finite.size
                else math.nan,
                "worst_key": key,
                "cell_count": int(values.size),
            }
    return result


def mse_statistics(
    meshes: dict[str, np.ndarray], reference: np.ndarray
) -> dict[str, dict[str, dict[str, float]]]:
    """Return all-point coordinate MSE and maximum pointwise MSE vs PDE."""
    result = {}
    reference = np.asarray(reference, dtype=float)
    for label, mesh in meshes.items():
        values = np.mean((np.asarray(mesh, dtype=float) - reference) ** 2, axis=-1)
        finite = values[np.isfinite(values)]
        result[label] = {
            "mse_vs_pde": {
                "mean": float(finite.mean()) if finite.size else math.nan,
                "worst": float(finite.max()) if finite.size else math.nan,
                "worst_key": "maximum",
                "cell_count": int(values.size),
            }
        }
    return result


def _number(value: float) -> str:
    if not math.isfinite(value):
        return "NaN"
    if value != 0 and (abs(value) < 1.0e-3 or abs(value) >= 1.0e4):
        return f"{value:.2e}"
    return f"{value:.4g}"


def save_figure(fig, stem: Path, formats: tuple[str, ...]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_training_comparisons(
    training_root: Path,
    results_dataset_root: Path,
    dimension: int,
    dataset: str,
) -> None:
    record = ensure_training_comparison(training_root, dimension, dataset)
    if record["status"] == "waiting_for_models":
        missing = ", ".join(record["missing_models"])
        print(f"[training comparison] skipped; missing {missing}", flush=True)
        return
    source = Path(record["directory"])
    target = results_dataset_root / "dataset_comparisons" / "training"
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "best_mse_bar.png",
        "training_time_bar.png",
        "training_comparison.csv",
        "training_comparison.json",
    ):
        if (source / name).is_file():
            shutil.copy2(source / name, target / name)


def plot_inference_times(
    output: Path,
    timings: dict,
    projection_times: dict[str, dict[str, float]],
    formats: tuple[str, ...],
    *,
    include_resmetamesh: bool = False,
) -> list[dict]:
    methods = ("TFI", "PDE", "MeshONet", "PI-MeshONet", "HMONet")
    if include_resmetamesh:
        methods += ("ResMetaMesh",)
    source_names = {
        "ResMetaMesh": "ResMetaMesh",
        "TFI": "TFI",
        "PDE": "PDE_Winslow",
        "HMONet": "HMONet",
        "MeshONet": "MeshONet",
        "PI-MeshONet": "PI-MeshONet",
    }
    x = np.arange(len(methods), dtype=float)
    rows = []
    fig, axis = plt.subplots(figsize=(9.4, 5.6))
    for index, method in enumerate(methods):
        network = inference_seconds(timings.get(source_names[method], {}))
        variants = ("raw",) if method not in {"MeshONet", "PI-MeshONet"} else ("raw", "direct", "idw")
        totals = {}
        for variant in variants:
            post_seconds = projection_times.get(method, {}).get(variant, 0.0)
            total = network + post_seconds
            totals[variant] = total
            rows.append(
                {
                    "method": method,
                    "boundary_processing": variant,
                    "network_or_solver_seconds": network,
                    "boundary_processing_seconds": post_seconds,
                    "total_seconds": total,
                }
            )
        # The three totals share one method position and one width.  Draw the
        # tallest IDW total first, then direct, and raw last.  Their different
        # heights remain visible as three vertical colour regions while each
        # recorded value still represents a complete end-to-end total.
        layer_order = ("idw", "direct", "raw") if len(variants) == 3 else ("raw",)
        layer_width = 0.58
        colors = {"raw": "#4C78A8", "direct": "#F2CF5B", "idw": "#E45756"}
        for zorder, variant in enumerate(layer_order, start=2):
            axis.bar(
                index,
                max(totals[variant], 1.0e-12),
                layer_width,
                color="#008D85" if method == "ResMetaMesh" else colors[variant],
                zorder=zorder,
            )
        if len(variants) == 3:
            label_offsets = {"raw": -0.13, "direct": 0.13, "idw": 0.0}
            text_offsets = {"raw": (-8, 4), "direct": (8, 13), "idw": (0, 4)}
            for variant in ("raw", "direct", "idw"):
                axis.annotate(
                    _number(totals[variant]),
                    (index + label_offsets[variant], max(totals[variant], 1.0e-12)),
                    xytext=text_offsets[variant],
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
        else:
            axis.annotate(
                _number(totals["raw"]),
                (index, max(totals["raw"], 1.0e-12)),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    axis.set_yscale("log")
    axis.set_xticks(x, methods)
    axis.set_ylabel("Total generation time (s, log scale)")
    title = "Six-method" if include_resmetamesh else "Five-method"
    axis.set_title(f"{title} generation time with boundary-processing totals")
    axis.grid(True, axis="y", which="both", alpha=0.25)
    axis.margins(y=0.16)
    axis.legend(
        handles=[
            Patch(facecolor="#4C78A8", label="raw"),
            Patch(facecolor="#F2CF5B", label="raw + direct"),
            Patch(facecolor="#E45756", label="raw + IDW"),
            *([Patch(facecolor="#008D85", label="ResMetaMesh incl. adaptation")]
              if include_resmetamesh else []),
        ],
        frameon=False,
        ncol=3,
    )
    fig.tight_layout()
    save_figure(fig, output / "inference_time_bar", formats)
    with (output / "inference_time.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(output / "inference_time.json", rows)
    return rows


def plot_quality_bar(
    output: Path,
    labels: tuple[str, ...],
    values: list[float],
    *,
    metric: str,
    statistic: str,
    group_title: str,
    formats: tuple[str, ...],
) -> None:
    colors = ("#4C78A8", "#59A14F", "#F28E2B", "#B279A2", "#E15759", "#008D85")
    fig, axis = plt.subplots(figsize=(9.2, 5.4))
    bars = axis.bar(np.arange(len(labels)), values, color=colors[: len(labels)], width=0.66)
    axis.bar_label(bars, labels=[_number(value) for value in values], padding=3, fontsize=9)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=12, ha="right")
    direction = "↓" if metric in LOWER_IS_BETTER or metric in FLAG_METRICS else "↑"
    metric_label = "Coordinate MSE vs PDE" if metric == "mse_vs_pde" else metric
    statistic_label = "Mean" if statistic == "mean" else f"Worst ({worst_key(metric)})"
    axis.set_ylabel(f"{metric_label}: {statistic_label} {direction}")
    axis.set_title(f"{group_title}\n{metric_label} — {statistic_label}")
    axis.grid(True, axis="y", alpha=0.25)
    limits = quality_axis_limits(metric, values)
    if limits is not None:
        lower, upper = limits
        axis.set_ylim(bottom=lower, top=upper)
    fig.tight_layout()
    save_figure(fig, output, formats)


def plot_quality_groups(
    output: Path,
    statistics: dict[str, dict[str, dict[str, float]]],
    metrics: tuple[str, ...],
    formats: tuple[str, ...],
) -> list[dict]:
    rows = []
    for metric in metrics:
        metric_root = output / "quality" / metric
        metric_root.mkdir(parents=True, exist_ok=True)
        for pattern in ("*.png", "*.pdf"):
            for stale in metric_root.glob(pattern):
                stale.unlink()
        for group_name, labels in BOUNDARY_PROCESSING_GROUPS.items():
            if "ResMetaMesh" in statistics:
                labels += ("ResMetaMesh",)
            title = "Six-method" if len(labels) == 6 else "Five-method"
            for statistic in ("mean", "worst"):
                values = [statistics[label][metric][statistic] for label in labels]
                plot_quality_bar(
                    metric_root / f"{group_name}_{statistic}",
                    labels,
                    values,
                    metric=metric,
                    statistic=statistic,
                    group_title=f"{title} comparison — {group_name.upper()} boundary processing",
                    formats=formats,
                )
            for label in labels:
                item = statistics[label][metric]
                rows.append(
                    {
                        "metric": metric,
                        "group": group_name,
                        "method": label,
                        "mean": item["mean"],
                        "worst": item["worst"],
                        "worst_key": item["worst_key"],
                        "cell_count": item["cell_count"],
                    }
                )
    with (output / "quality_statistics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(output / "quality_statistics.json", rows)
    return rows


def selected_metrics(dimension: int, requested: tuple[str, ...]) -> tuple[str, ...]:
    available = tuple(
        metric for metric in (QUAD_METRICS if dimension == 2 else HEX_METRICS)
        if not (dimension == 2 and metric == "cell_area")
    )
    available += ("mse_vs_pde",)
    if requested == ("all",):
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown {dimension}D quality metrics: {', '.join(unknown)}")
    return tuple(metric for metric in available if metric in requested)


def discover_jobs(
    dataset_root: Path,
    grids: tuple[str, ...] | None,
    values: tuple[str, ...] | None,
) -> list[Path]:
    selected_grids = set(grids or ())
    selected_values = set(values or ())
    jobs = []
    for grid in sorted(path for path in dataset_root.iterdir() if path.is_dir() and "x" in path.name):
        if selected_grids and grid.name not in selected_grids:
            continue
        for value in sorted(path for path in grid.iterdir() if path.is_dir()):
            if selected_values and value.name not in selected_values:
                continue
            if (value / "results.json").is_file() or (value / "evaluation_timing.json").is_file():
                jobs.append(value)
    return jobs


def load_saved_six_method_statistics(root: Path, metrics: tuple[str, ...]):
    """Read matched raw, direct and IDW evaluations without running any solver."""
    raw_directory = root / RAW_COMPARISON
    raw = load_json(raw_directory / "evaluation_result.json")
    shape = tuple(raw["grid_shape"])
    cell_count = math.prod(n - 1 for n in shape)
    statistics = {}
    projection_times = {}
    raw_timings = raw["timings"]
    for group, directory in (
        ("raw", raw_directory),
        ("direct", root / "comparisons/CCAP/H_raw__M_direct__PI_direct/TFI_PDE_M_PI_H"),
        ("idw", root / "comparisons/CCAP/H_raw__M_idw__PI_idw/TFI_PDE_M_PI_H"),
    ):
        record = raw if group == "raw" else load_json(directory / "evaluation_result.json")
        if any(record[key] != raw[key] for key in ("grid_shape", "parameters", "case")):
            raise ValueError(f"Mismatched saved evaluation in {directory}")
        labels = BOUNDARY_PROCESSING_GROUPS[group]
        methods = ("TFI", "PDE_Winslow", "MeshONet", "PI-MeshONet", "HMONet")
        pairs = zip((*labels, "ResMetaMesh"), (*methods, "ResMetaMesh")) if group == "raw" else zip(labels[2:4], methods[2:4])
        saved_mse = (
            load_json(directory / "point_loss_summary.json")["methods"]
            if "mse_vs_pde" in metrics else {}
        )
        for label, method in pairs:
            method_metrics = record["metrics"][method]
            items = {}
            for metric in metrics:
                worst = worst_key(metric)
                if metric == "mse_vs_pde":
                    if method == "TFI":
                        meshes = {m: np.load(_mesh_path(root, m)).reshape(-1, len(shape))
                                  for m in ("TFI", "PDE")}
                        loss = mse_statistics({"TFI": meshes["TFI"]}, meshes["PDE"])["TFI"]["mse_vs_pde"]
                        mean, value = loss["mean"], loss["worst"]
                    elif method == "PDE_Winslow":
                        mean = value = 0.0
                    else:
                        loss = saved_mse[method]
                        mean = loss["mse_vs_winslow"]
                        value = loss["max_point_mse_vs_winslow"]
                    count = math.prod(shape)
                else:
                    saved = method_metrics[metric]
                    mean, value = saved["mean"], saved[worst]
                    count = cell_count
                items[metric] = {"mean": mean, "worst": value,
                                 "worst_key": worst, "cell_count": count}
            statistics[label] = items
            if method in ("MeshONet", "PI-MeshONet"):
                timing = record["timings"][method]
                if timing["checkpoint_sha256"] != raw_timings[method]["checkpoint_sha256"]:
                    raise ValueError(f"Mismatched {method} checkpoint in {directory}")
                projection_times.setdefault(method, {})[group] = timing["boundary_projection_seconds"]
    return raw, statistics, projection_times


def process_saved_six_method_job(args, root: Path, metrics: tuple[str, ...]) -> dict:
    record, stats, projection_times = load_saved_six_method_statistics(root, metrics)
    output = root / "summary_comparisons" / "six_methods"
    output.mkdir(parents=True, exist_ok=True)
    timing_rows = plot_inference_times(
        output, record["timings"], projection_times, args.formats, include_resmetamesh=True,
    )
    quality_rows = plot_quality_groups(output, stats, metrics, args.formats)
    manifest = {
        "dimension": args.dimension,
        "dataset": args.dataset,
        "grid_shape": record["grid_shape"],
        "parameter": record["parameter"],
        "value": record["value"],
        "metrics": list(metrics),
        "method_order": list(record["method_order"]),
        "evaluation_source": "saved_evaluation",
        "quality_figure_count": len(metrics) * 6 * len(args.formats),
        "timing_rows": len(timing_rows),
        "quality_rows": len(quality_rows),
    }
    save_json(output / "comparison_manifest.json", manifest)
    return {"root": str(root), "output": str(output), **manifest}


def process_job(args, root: Path, metrics: tuple[str, ...]) -> dict:
    if args.dimension == 2 and (
        args.include_resmetamesh or (root / RAW_COMPARISON / "evaluation_result.json").is_file()
    ):
        return process_saved_six_method_job(args, root, metrics)
    metadata_path = root / "results.json"
    metadata = load_json(metadata_path) if metadata_path.is_file() else load_json(root / "evaluation_parameters.json")
    shape = tuple(int(value) for value in metadata["grid_shape"])
    meshes = load_base_meshes(root, shape)
    ids = boundary_indices(shape)
    exact = meshes["PDE"][ids]
    projection_meshes = {}
    projection_times: dict[str, dict[str, float]] = {}
    for method in ("MeshONet", "PI-MeshONet"):
        raw = meshes[method]
        projection_meshes[f"{method} raw"] = raw
        projection_times[method] = {"raw": 0.0}
        for mode, label in (("direct", "direct"), ("inverse_distance", "IDW")):
            projected, elapsed = timed_projection(
                raw,
                exact,
                ids,
                mode,
                power=args.idw_power,
                repeats=args.timing_repeats,
                chunk_size=args.idw_chunk_size,
            )
            projection_meshes[f"{method} {label}"] = projected
            projection_times[method]["idw" if label == "IDW" else "direct"] = elapsed
    comparison_meshes = {
        "TFI": meshes["TFI"],
        "PDE": meshes["PDE"],
        "HMONet": meshes["HMONet"],
        **projection_meshes,
    }
    output = root / "summary_comparisons"
    output.mkdir(parents=True, exist_ok=True)
    timing_rows = plot_inference_times(
        output,
        load_timings(root),
        projection_times,
        args.formats,
    )
    stats = quality_statistics(
        comparison_meshes, shape, tuple(metric for metric in metrics if metric != "mse_vs_pde"),
    )
    if "mse_vs_pde" in metrics:
        for label, values in mse_statistics(comparison_meshes, meshes["PDE"]).items():
            stats[label].update(values)
    plotted_metrics = metrics
    quality_rows = plot_quality_groups(output, stats, plotted_metrics, args.formats)
    manifest = {
        "dimension": args.dimension,
        "dataset": args.dataset,
        "grid_shape": list(shape),
        "parameter": metadata.get("parameter"),
        "value": metadata.get("value"),
        "metrics": list(plotted_metrics),
        "quality_figure_count": len(plotted_metrics) * 6,
        "timing_repeats": args.timing_repeats,
        "idw_power": args.idw_power,
        "idw_chunk_size": args.idw_chunk_size,
        "timing_rows": len(timing_rows),
        "quality_rows": len(quality_rows),
    }
    save_json(output / "comparison_manifest.json", manifest)
    return {"root": str(root), "output": str(output), **manifest}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dimension", required=True, type=int, choices=(2, 3))
    result.add_argument("--dataset", required=True, type=dataset_name)
    result.add_argument("--results-root", type=Path, default=Path("outputs/results"))
    result.add_argument("--training-root", type=Path, default=Path("outputs/training"))
    result.add_argument("--grids", nargs="+", help="Optional folder names, e.g. 101x101")
    result.add_argument("--values", nargs="+", help="Optional parameter folder names")
    result.add_argument("--quality-metrics", nargs="+", default=["max_angle"])
    result.add_argument(
        "--include-resmetamesh", action="store_true",
        help="Require saved six-method 2-D results instead of falling back to five methods.",
    )
    result.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=["png"])
    result.add_argument("--timing-repeats", type=int, default=3)
    result.add_argument("--idw-power", type=float, default=2.0)
    result.add_argument("--idw-chunk-size", type=int, default=512)
    result.add_argument("--skip-training-comparison", action="store_true")
    return result


def main(argv=None) -> int:
    command = parser()
    args = command.parse_args(argv)
    if args.include_resmetamesh and args.dimension != 2:
        command.error("--include-resmetamesh currently uses the saved 2-D evaluations")
    args.results_root = (ROOT / args.results_root).resolve() if not args.results_root.is_absolute() else args.results_root
    args.training_root = (ROOT / args.training_root).resolve() if not args.training_root.is_absolute() else args.training_root
    args.formats = tuple(dict.fromkeys(args.formats))
    if args.timing_repeats < 1 or args.idw_power <= 0 or args.idw_chunk_size < 1:
        raise ValueError("timing repeats, IDW power, and chunk size must be positive")
    metrics = selected_metrics(args.dimension, tuple(args.quality_metrics))
    dataset_root = args.results_root / f"{args.dimension}D" / args.dataset
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not args.skip_training_comparison:
        plot_training_comparisons(
            args.training_root,
            dataset_root,
            args.dimension,
            args.dataset,
        )
    jobs = discover_jobs(
        dataset_root,
        tuple(args.grids) if args.grids else None,
        tuple(args.values) if args.values else None,
    )
    if not jobs:
        raise FileNotFoundError("No matching completed evaluation parameter directories")
    records, failures = [], []
    for index, root in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {root}", flush=True)
        try:
            record = process_job(args, root, metrics)
            records.append(record)
            print(f"  saved {record['quality_figure_count']} quality figures -> {record['output']}", flush=True)
        except Exception as error:  # preserve other completed grid/parameter figures
            failures.append({"root": str(root), "error": f"{type(error).__name__}: {error}"})
            print(f"  ERROR: {type(error).__name__}: {error}", flush=True)
    summary_root = dataset_root / "dataset_comparisons"
    save_json(summary_root / "comparison_figure_runs.json", {"completed": records, "failures": failures})
    if failures:
        raise RuntimeError(f"{len(failures)} comparison-figure job(s) failed; see {summary_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
