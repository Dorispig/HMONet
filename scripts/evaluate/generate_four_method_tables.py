"""Generate four-method comparison tables from existing evaluation results.

Command-line usage
------------------
Generate all available cases for one dataset:

    python scripts/evaluate/generate_four_method_tables.py \
        --dataset run_int_0_1000_61_21

Generate both interpolation and extrapolation datasets:

    python scripts/evaluate/generate_four_method_tables.py --dataset both

Restrict the generated tables to selected grids and parameters:

    python scripts/evaluate/generate_four_method_tables.py \
        --dataset run_ext_0_1000_61_21 \
        --grid-shapes 61x21,81x41 \
        --parameters 1.52,1.535

Overwrite existing generated CSV files:

    python scripts/evaluate/generate_four_method_tables.py \
        --dataset both --force

Arguments:
    --dataset       Dataset directory name, or ``both``. If omitted, all
                    datasets containing results.json are discovered.
    --grid-shapes   Comma-separated grid shapes such as ``61x21,101x101``.
    --parameters    Comma-separated parameter values to include.
    --output-dir    Optional result root; defaults to outputs/results/2D.
    --force         Replace existing generated CSV files.

The script reads results.json and writes one case CSV for every
dataset/grid/parameter combination, plus a dataset-level block CSV. It does
not rerun model inference or PDE generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


METHODS = ("TFI", "PDE_Winslow", "MeshONet", "HMONet")
# User-editable defaults. Command-line arguments override these values.
DEFAULT_OUTPUT_DIR = "outputs/results/2D"
DEFAULT_DATASETS = (
    "run_int_0_1000_61_21",
    "run_ext_0_1000_61_21",
)
DEFAULT_GRID_SHAPES = None
DEFAULT_PARAMETERS = None
METHOD_LABELS = {
    "TFI": "TFI",
    "PDE_Winslow": "PDE",
    "MeshONet": "MeshONet",
    "HMONet": "HMONet",
}
QUALITY_SPECS = (
    ("最大角/° ↓", "max_angle", "down"),
    ("最小角/° ↑", "min_angle", "up"),
    ("Jacobian ↑", "jacobian", "up"),
    ("Scaled Jacobian ↑", "scaled_jacobian", "up"),
    ("Skewness ↓", "skewness", "down"),
    ("Aspect ratio ↓", "aspect_ratio", "down"),
    ("Orthogonality ↑", "orthogonality", "up"),
    ("边界层正交性 ↑", "boundary_layer_orthogonality", "up"),
    ("Cell area ↑", "cell_area", "up"),
)
COUNT_SPECS = (
    ("单元自身相交", "self_intersection", "affected_cell_fraction"),
    ("单元间相交", "intersecting_cell", "affected_cell_fraction"),
    ("翻转单元", "flipping_cell", "affected_cell_fraction"),
    ("退化单元", "degenerate_cell", "affected_cell_fraction"),
    ("任意非法单元", "invalid_cells", "fraction"),
)


def number(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0.000000"
    if abs(value) < 1e-4 or abs(value) >= 1e4:
        return f"{value:.6e}"
    return f"{value:.6f}"


def percent(value: float) -> str:
    return f"{100.0 * float(value):.4f}%"


def bold(value: str, is_best: bool) -> str:
    return f"**{value}**" if is_best else value


def best_flags(values: list[float], direction: str) -> list[bool]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return [False] * len(values)
    target = max(finite) if direction == "up" else min(finite)
    return [math.isclose(value, target, rel_tol=1e-10, abs_tol=1e-12) for value in values]


def timing_seconds(result: dict, method: str) -> float:
    timing = result["timings"][method]
    return float(
        timing.get(
            "network_inference_seconds",
            timing.get("total_model_seconds", timing.get("total_seconds", 0.0)),
        )
    )


def boundary_errors(result: dict) -> dict[str, tuple[float, float, float]]:
    errors = {method: (0.0, 0.0, 0.0) for method in ("TFI", "PDE_Winslow")}
    for item in result.get("boundary_errors", []):
        if item.get("stage") == "raw":
            errors[item["method"]] = (
                float(item["boundary_mean_error"]),
                float(item["boundary_max_error"]),
                float(item["boundary_rms_error"]),
            )
    return errors


def load_results(root: Path, dataset: str) -> list[dict]:
    paths = sorted((root / dataset).glob("*/*/results.json"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    return sorted(
        results,
        key=lambda item: (
            item["grid_shape"][0] * item["grid_shape"][1],
            tuple(item["grid_shape"]),
            float(item["value"]),
        ),
    )


def parse_grid_filter(value: str | None) -> set[tuple[int, int]] | None:
    if not value:
        return None
    grids = set()
    for item in value.split(","):
        parts = item.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"Invalid grid shape: {item}")
        grids.add((int(parts[0]), int(parts[1])))
    return grids


def parse_parameter_filter(value: str | None) -> set[float] | None:
    if not value:
        return None
    return {float(item) for item in value.split(",")}


def select_results(results: list[dict], grids, parameters) -> list[dict]:
    return [
        result
        for result in results
        if (grids is None or tuple(result["grid_shape"]) in grids)
        and (parameters is None or float(result["value"]) in parameters)
    ]


def case_rows(dataset: str, result: dict) -> list[list[str]]:
    nx, ny = result["grid_shape"]
    parameter = str(result["value"])
    total_cells = max((nx - 1) * (ny - 1), 1)
    metrics = result["metrics"]
    rows = [
        [
            "数据集", dataset, "参数", parameter,
            "网格规模", f"{nx}x{ny}",
            "网格总单元数", str(total_cells),
        ],
        ["指标", *[METHOD_LABELS[m] for m in METHODS]],
        [],
        ["运行时间 ↓", "", "", "", ""],
    ]
    values = [timing_seconds(result, method) for method in METHODS]
    flags = best_flags(values, "down")
    rows.append(["运行时间/ms", *[bold(f"{1000 * value:.3f}", flags[i]) for i, value in enumerate(values)]])
    rows += [[], ["相交与非法单元比例 ↓", "", "", "", ""], ["指标", *[METHOD_LABELS[m] for m in METHODS]]]
    for label, key, field in COUNT_SPECS:
        fractions = [float(metrics[method][key][field]) for method in METHODS]
        flags = best_flags(fractions, "down")
        cells = [
            bold(f"{int(round(fractions[i] * total_cells))}/{total_cells} ({percent(fractions[i])})", flags[i])
            for i in range(len(METHODS))
        ]
        rows.append([label, *cells])
    rows += [[], ["网格质量：平均值 / 最差值", "", "", "", ""], ["指标", *[METHOD_LABELS[m] for m in METHODS]]]
    for label, key, direction in QUALITY_SPECS:
        worst_key = "minimum" if direction == "up" else "maximum"
        means = [float(metrics[method][key]["mean"]) for method in METHODS]
        worsts = [float(metrics[method][key][worst_key]) for method in METHODS]
        mean_flags = best_flags(means, direction)
        worst_flags = best_flags(worsts, direction)
        rows.append(
            [
                label,
                *[
                    f"{bold(number(means[i]), mean_flags[i])} / {bold(number(worsts[i]), worst_flags[i])}"
                    for i in range(len(METHODS))
                ],
            ]
        )
    errors = boundary_errors(result)
    rows += [[], ["原始输出边界误差 ↓", "", "", "", ""], ["指标", *[METHOD_LABELS[m] for m in METHODS]]]
    for label, index in (("平均误差", 0), ("最大误差", 1), ("RMS误差", 2)):
        values = [errors[method][index] for method in METHODS]
        flags = best_flags(values, "down")
        rows.append([label, *[bold(number(values[i]), flags[i]) for i in range(len(METHODS))]])
    rows += [[], ["结束", f"{nx}x{ny}", parameter], []]
    return rows


def write_csv(path: Path, rows: list[list[str]], force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        csv.writer(stream).writerows(rows)


def write_dataset(dataset: str, results: list[dict], output_root: Path, force: bool) -> None:
    destination = output_root / dataset / "four_methods_tables"
    combined: list[list[str]] = []
    for index, result in enumerate(results):
        rows = case_rows(dataset, result)
        nx, ny = result["grid_shape"]
        parameter = str(result["value"])
        write_csv(destination / f"{nx}x{ny}" / f"{parameter}.csv", rows, force)
        if index:
            combined.append([])
            combined.append(["=" * 20])
            combined.append([])
        combined.extend(rows)
    write_csv(destination / "four_methods_comparison.csv", combined, force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=None, help="Dataset name or both")
    parser.add_argument("--grid-shapes", default=DEFAULT_GRID_SHAPES)
    parser.add_argument("--parameters", default=DEFAULT_PARAMETERS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_root = Path(args.output_dir)
    if args.dataset and args.dataset != "both":
        datasets = [args.dataset]
    elif args.dataset == "both":
        datasets = list(DEFAULT_DATASETS)
    else:
        datasets = sorted(path.name for path in output_root.iterdir() if path.is_dir() and list(path.glob("*/*/results.json")))
    grids = parse_grid_filter(args.grid_shapes)
    parameters = parse_parameter_filter(args.parameters)
    for dataset in datasets:
        results = select_results(load_results(output_root, dataset), grids, parameters)
        if not results:
            raise SystemExit(f"No evaluation results matched for {dataset}")
        write_dataset(dataset, results, output_root, args.force)
        print(f"{dataset}: wrote {len(results)} case table(s) under {output_root / dataset / 'four_methods_tables'}")


if __name__ == "__main__":
    main()
