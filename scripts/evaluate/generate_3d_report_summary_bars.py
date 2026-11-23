"""Generate four five-method comparison bars from archived 3-D report CSVs.

Only values already present in each report are used. Missing IDW or timing
records are rendered as N/A rather than estimated from another dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


METHODS = ("TFI", "PDE", "MeshONet", "PI-MeshONet", "HMONet")
COLORS = {
    "TFI": "#4C78A8",
    "PDE": "#59A14F",
    "MeshONet": "#F28E2B",
    "PI-MeshONet": "#B279A2",
    "HMONet": "#E15759",
}


def value(row: dict, *names: str) -> str:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return str(row[name]).strip()
    return ""


def canonical_method(text: str) -> tuple[str, str]:
    normalized = text.strip().replace("（", "(").replace("）", ")").casefold()
    if normalized == "tfi":
        return "TFI", "raw"
    if normalized in {"pde", "pde_winslow", "pde-winslow"}:
        return "PDE", "raw"
    if normalized == "hmonet":
        return "HMONet", "raw"
    base = "PI-MeshONet" if "pi-meshonet" in normalized else "MeshONet"
    mode = "idw" if "idw" in normalized else "direct" if "direct" in normalized else "raw"
    return base, mode


def grid_size(text: str) -> int:
    numbers = [int(item) for item in re.findall(r"\d+", text)]
    return math.prod(numbers) if numbers else 0


def number(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def load_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def select_case(rows: list[dict]) -> tuple[str, float, list[dict]]:
    candidates = []
    for row in rows:
        scope = value(row, "scope", "范围").casefold()
        metric = value(row, "metric", "指标").casefold()
        if "z_mid" not in scope or metric != "scaled_jacobian":
            continue
        parameter = number(value(row, "parameter", "参数"))
        grid = value(row, "grid", "网格规模")
        if math.isfinite(parameter):
            candidates.append((parameter, grid_size(grid), grid))
    if not candidates:
        raise ValueError("No z_mid scaled_jacobian rows found")
    parameter, _, grid = max(candidates, key=lambda item: (item[0], item[1]))
    selected = [
        row
        for row in rows
        if value(row, "grid", "网格规模") == grid
        and number(value(row, "parameter", "参数")) == parameter
    ]
    return grid, parameter, selected


def quality_values(rows: list[dict], mode: str) -> dict[str, float]:
    result = {}
    for row in rows:
        if value(row, "metric", "指标").casefold() != "scaled_jacobian":
            continue
        if "z_mid" not in value(row, "scope", "范围").casefold():
            continue
        method, row_mode = canonical_method(value(row, "method", "方法"))
        wanted_mode = mode if method in {"MeshONet", "PI-MeshONet"} else "raw"
        if row_mode == wanted_mode:
            result[method] = number(value(row, "worst", "最差值"))
    return result


def timing_values(rows: list[dict]) -> dict[tuple[str, str], float]:
    result = {}
    for row in rows:
        if value(row, "metric", "指标").casefold() != "inference_total_seconds":
            continue
        method, mode = canonical_method(value(row, "method", "方法"))
        result[(method, mode)] = number(
            value(row, "total_seconds", "总时间（s）", "mean", "平均值")
        )
    return result


def add_labels(axis, bars, values) -> None:
    for bar, item in zip(bars, values):
        if math.isfinite(item):
            axis.annotate(
                f"{item:.4g}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        else:
            axis.annotate(
                "N/A",
                (bar.get_x() + bar.get_width() / 2, 0),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#666666",
            )


def plot_quality(output: Path, dataset: str, grid: str, parameter: float, mode: str, rows: list[dict]) -> dict:
    mapping = quality_values(rows, mode)
    values = [mapping.get(method, math.nan) for method in METHODS]
    heights = [item if math.isfinite(item) else 0.0 for item in values]
    figure, axis = plt.subplots(figsize=(9.4, 5.6))
    bars = axis.bar(
        np.arange(len(METHODS)), heights,
        color=[COLORS[item] if math.isfinite(value_) else "#D9D9D9" for item, value_ in zip(METHODS, values)],
        width=0.66,
    )
    for bar, item in zip(bars, values):
        if not math.isfinite(item):
            bar.set_hatch("///")
    add_labels(axis, bars, values)
    finite = [item for item in values if math.isfinite(item)]
    if finite:
        low, high = min(finite), max(finite)
        span = high - low
        if low > 0 and span <= 0.2 * max(high, 1e-12):
            padding = max(0.25 * span, 0.0025 * high)
            axis.set_ylim(max(0.0, low - padding), high + padding)
        elif low < 0:
            padding = max(0.08 * (high - low), 0.02)
            axis.set_ylim(low - padding, high + padding)
            axis.axhline(0.0, color="#444444", linewidth=0.9)
        else:
            axis.set_ylim(bottom=0)
    axis.set_xticks(np.arange(len(METHODS)), METHODS, rotation=10, ha="right")
    axis.set_ylabel("Worst scaled Jacobian (minimum) ↑")
    axis.set_title(f"{dataset} | {grid} | parameter={parameter:g} | z_mid\n{mode.upper()} boundary processing")
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    path = output / f"scaled_jacobian_worst_{mode}.png"
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {method: mapping.get(method) for method in METHODS}


def plot_timing(output: Path, dataset: str, grid: str, parameter: float, rows: list[dict]) -> dict:
    timing = timing_values(rows)
    figure, axis = plt.subplots(figsize=(9.4, 5.6))
    available = [item for item in timing.values() if math.isfinite(item) and item > 0]
    for index, method in enumerate(METHODS):
        modes = ("raw", "direct", "idw") if method in {"MeshONet", "PI-MeshONet"} else ("raw",)
        layer_order = tuple(item for item in ("idw", "direct", "raw") if item in modes)
        colors = {"raw": "#4C78A8", "direct": "#F2CF5B", "idw": "#E45756"}
        any_value = False
        for layer, mode in enumerate(layer_order, start=2):
            item = timing.get((method, mode), math.nan)
            if not math.isfinite(item) or item <= 0:
                continue
            any_value = True
            axis.bar(index, item, 0.58, color=colors[mode], zorder=layer)
            x_offset = {"raw": -0.12, "direct": 0.12, "idw": 0.0}[mode]
            y_offset = {"raw": 4, "direct": 13, "idw": 4}[mode]
            axis.annotate(
                f"{item:.4g}", (index + x_offset, item), xytext=(0, y_offset),
                textcoords="offset points", ha="center", va="bottom", fontsize=7.5,
            )
        if not any_value:
            axis.bar(index, 1.0 if not available else min(available), 0.58, color="#D9D9D9", hatch="///")
            axis.annotate("N/A", (index, 1.0 if not available else min(available)), ha="center", va="bottom")
    if available:
        axis.set_yscale("log")
        axis.set_ylabel("Total generation time (s, log scale)")
    else:
        axis.set_ylim(0, 1.2)
        axis.set_ylabel("Total generation time (report data unavailable)")
    axis.set_xticks(np.arange(len(METHODS)), METHODS)
    axis.set_title(f"{dataset} | {grid} | parameter={parameter:g}\nGeneration time with boundary-processing totals")
    axis.grid(True, axis="y", which="both", alpha=0.25)
    axis.legend(
        handles=[
            Patch(facecolor="#4C78A8", label="raw"),
            Patch(facecolor="#F2CF5B", label="raw + direct"),
            Patch(facecolor="#E45756", label="raw + IDW"),
        ], frameon=False, ncol=3,
    )
    figure.tight_layout()
    path = output / "inference_time.png"
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {f"{method}/{mode}": timing.get((method, mode)) for method in METHODS for mode in (("raw", "direct", "idw") if method in {"MeshONet", "PI-MeshONet"} else ("raw",))}


def process_dataset(dataset_root: Path) -> dict:
    report = dataset_root / "five_method_comparison" / "quality_comparison.csv"
    rows = load_rows(report)
    grid, parameter, selected = select_case(rows)
    output = dataset_root / "five_method_comparison" / "summary_bars"
    output.mkdir(parents=True, exist_ok=True)
    quality = {mode: plot_quality(output, dataset_root.name, grid, parameter, mode, selected) for mode in ("raw", "direct", "idw")}
    timing = plot_timing(output, dataset_root.name, grid, parameter, selected)
    manifest = {
        "dataset": dataset_root.name,
        "grid": grid,
        "parameter": parameter,
        "scope": "z_mid",
        "source": str(report),
        "quality": quality,
        "timing": timing,
    }
    (output / "summary_bars.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(output), **manifest}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=Path("reports/3D"))
    parser.add_argument("--datasets", nargs="*")
    args = parser.parse_args(argv)
    roots = sorted(path for path in args.reports_root.iterdir() if path.is_dir())
    if args.datasets:
        wanted = set(args.datasets)
        roots = [path for path in roots if path.name in wanted]
    records = [process_dataset(path) for path in roots]
    print(json.dumps(records, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
