"""Versioned evaluation artifacts, numeric tables, and raw boundary errors."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def json_value(value):
    """Convert NumPy/path values and represent undefined statistics as null."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_value(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def digest(value):
    return hashlib.sha256(
        json.dumps(json_value(value), sort_keys=True).encode()
    ).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def array_digest(values):
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def write_csv(path, rows, headers=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    headers = headers or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(json_value(rows))


def manifest_valid(directory, signature):
    """Require the requested signature and every recorded artifact unchanged."""
    directory = Path(directory)
    metadata = read_json(directory / "analysis_metadata.json", {})
    if metadata.get("signature") != signature or not metadata.get("files"):
        return False
    for name, expected in metadata["files"].items():
        path = directory / name
        if (
            not path.is_file()
            or path.stat().st_size != expected[0]
            or path.stat().st_mtime_ns != expected[1]
        ):
            return False
    return True


def save_manifest(directory, signature):
    directory = Path(directory)
    owned_prefixes = (
        "mesh",
        "quality_",
        "comparison_",
        "pairwise_quality_",
        "paper_",
        "boundary_",
        "raw_output_",
        "surface_quality_",
        "self_intersection_",
        "point_loss_",
        "cell_loss_",
        "distribution_",
        "invalid_cells",
        "08_point_loss_",
        "09_point_loss_",
        "generation_timings",
    )
    files = {
        p.relative_to(directory).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns]
        for p in directory.rglob("*")
        if p.is_file()
        and p.name != "analysis_metadata.json"
        and "zoom_views" not in p.parts
        and (
            p.name in {"metadata.json", "timings.csv"}
            or p.name.startswith(owned_prefixes)
        )
    }
    write_json(
        directory / "analysis_metadata.json", {"signature": signature, "files": files}
    )


def clear_managed_files(directory):
    """Remove only files recorded by this evaluator, confined to this directory."""
    directory = Path(directory).resolve()
    previous = read_json(directory / "analysis_metadata.json", {})
    for name in previous.get("files", {}):
        target = (directory / name).resolve()
        if not target.is_relative_to(directory):
            raise ValueError("Invalid path in evaluation manifest")
        if target.is_file():
            target.unlink()


def boundary_error(predicted, exact, ids):
    """Compare paired logical boundary nodes using physical Euclidean distance."""
    values = np.asarray(predicted, dtype=float)[ids]
    exact = np.asarray(exact, dtype=float)
    if values.shape != exact.shape:
        raise ValueError("Predicted and exact boundary arrays must match")
    distances = np.linalg.norm(values - exact, axis=-1)
    return {
        "boundary_point_count": len(ids),
        "point_count": len(ids),
        "boundary_max_error": float(distances.max()),
        "boundary_mean_error": float(distances.mean()),
        "boundary_rms_error": float(np.sqrt(np.mean(distances**2))),
        "maximum_euclidean_error": float(distances.max()),
        "mean_euclidean_error": float(distances.mean()),
        "boundary_mse": float(np.mean((values - exact) ** 2)),
        "maximum_error_boundary_index": int(np.argmax(distances)),
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(distances))),
        "definition": "Euclidean distance at corresponding exact logical boundary parameters",
    }, distances


def save_boundary_errors(directory, predicted, exact, ids, method, stage, plots):
    """Persist raw and postprocessed errors independently; never overwrite raw."""
    directory = Path(directory)
    summary, distances = boundary_error(predicted, exact, ids)
    summary.update(method=method, stage=stage)
    write_json(directory / "boundary_error_summary.json", summary)
    rows = []
    for index, (node, error) in enumerate(zip(ids, distances)):
        row = {
            "boundary_index": index,
            "node_index": int(node),
            "euclidean_error": float(error),
        }
        for axis, label in enumerate("xyz"[: exact.shape[1]]):
            row[f"predicted_{label}"] = float(predicted[node, axis])
            row[f"exact_{label}"] = float(exact[index, axis])
        rows.append(row)
    write_csv(directory / "boundary_error_data.csv", rows)
    if plots["save_plots"] and (plots["save_png"] or plots["save_pdf"]):
        import matplotlib.pyplot as plt
        from .quality_output import save_figure_formats

        fig, axis = plt.subplots(figsize=(8, 4))
        axis.plot(np.arange(len(ids)), distances)
        axis.set(
            xlabel="Logical boundary node",
            ylabel="Euclidean error",
            title=f"{method}: {stage}",
        )
        fig.tight_layout()
        save_figure_formats(
            fig,
            directory / "boundary_error",
            save_png=plots["save_png"],
            save_pdf=plots["save_pdf"],
            dpi=180,
        )
        plt.close(fig)
    return summary


def quality_rows(summaries, section=None):
    rows = []
    for method, summary in summaries.items():
        for metric, values in summary.items():
            row = {"method": method, "metric": metric, **values}
            if section is not None:
                row["section"] = section
            rows.append(row)
    return rows


def save_quality_table(path, summaries):
    rows = quality_rows(summaries)
    headers = list(dict.fromkeys(k for row in rows for k in row))
    write_csv(path, rows, headers)


def save_timing_tables(directory, timings):
    """The paper's grid-generation column contains network forward time only."""
    write_json(Path(directory) / "generation_timings.json", timings)
    write_csv(
        Path(directory) / "timings.csv",
        [dict(method=m, **v) for m, v in timings.items()],
        [
            "method",
            "network_inference_seconds",
            "inference_batch_size",
            "batch_inference_seconds",
            "inference_time_basis",
            "curve_kernel_backend",
            "effective_inference_warmup_runs",
            "inference_warmup_runs",
            "inference_timing_repeats",
            "inference_timing_statistic",
            "inference_discarded_samples",
            "inference_used_samples",
            "boundary_projection_seconds",
            "total_model_seconds",
            "total_seconds",
            "cache_hit",
            "checkpoint",
            "training_seconds",
            "epoch",
        ],
    )
    rows = []
    for method, values in timings.items():
        rows.append(
            {
                "Method": method,
                "Training time (s)": values.get("training_seconds"),
                "Iterations": values.get("epoch"),
                "Grid generation time (s)": values.get(
                    "network_inference_seconds", values.get("total_seconds")
                ),
                "Inference batch size": values.get("inference_batch_size", 1),
                "Batch forward time (s)": values.get("batch_inference_seconds"),
                "Timing basis": values.get("inference_time_basis", "single_grid"),
                "Timing statistic": values.get(
                    "inference_timing_statistic", "single_run"
                ),
                "Measured runs": values.get("inference_timing_repeats", 1),
                "Runs after trimming": values.get("inference_used_samples", 1),
                "Boundary post-processing time (s)": values.get(
                    "boundary_projection_seconds", 0.0
                ),
                "Generation reused": values.get("cache_hit", False),
            }
        )
    write_csv(Path(directory) / "paper_timing_table.csv", rows)
