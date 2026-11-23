"""End-to-end manufactured Poisson evaluation on generated meshes."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.artifacts import write_json
from src.evaluation.mesh_provider import (
    boundary_geometry_error,
    generation_seconds,
    load_evaluated_meshes,
    replace_with_exact_boundary,
)
from src.numerics.downstream.boundary_conditions import build_exact_dirichlet_boundary
from src.numerics.downstream.field_comparison import (
    compare_to_exact_solution,
)
from src.numerics.downstream.poisson_2d import solve_poisson_2d


LEARNED_METHODS = {"MeshONet", "PI-MeshONet", "HMONet"}

def _write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _plot_fields(path, meshes, solutions, title):
    methods = list(solutions)
    if not methods:
        return
    figure, axes = plt.subplots(1, len(methods), figsize=(5.0 * len(methods), 4.1), squeeze=False)
    image = None
    for axis, method in zip(axes[0], methods):
        points, shape = meshes[method]
        nx, ny = shape
        grid = points.reshape(ny, nx, 2)
        field = solutions[method].reshape(ny, nx)
        image = axis.pcolormesh(grid[..., 0], grid[..., 1], field, shading="gouraud", cmap="inferno")
        axis.set_title(method)
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="u")
    figure.suptitle(title)
    figure.subplots_adjust(left=0.05, right=0.94, bottom=0.12, top=0.83, wspace=0.28)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_errors(path, meshes, errors, title, zoom_bounds=None):
    methods = list(errors)
    if not methods:
        return
    maximum = max(float(np.max(np.abs(values))) for values in errors.values())
    maximum = max(maximum, np.finfo(float).eps)
    figure, axes = plt.subplots(1, len(methods), figsize=(5.0 * len(methods), 4.1), squeeze=False)
    image = None
    for axis, method in zip(axes[0], methods):
        points, shape = meshes[method]
        nx, ny = shape
        grid = points.reshape(ny, nx, 2)
        field = np.abs(errors[method]).reshape(ny, nx)
        image = axis.pcolormesh(
            grid[..., 0], grid[..., 1], field, shading="gouraud", cmap="magma", vmin=0, vmax=maximum
        )
        axis.set_title(method)
        axis.set_aspect("equal")
        if zoom_bounds:
            axis.set_xlim(*zoom_bounds[:2])
            axis.set_ylim(*zoom_bounds[2:])
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="|u-u_exact|")
    figure.suptitle(title)
    figure.subplots_adjust(left=0.05, right=0.94, bottom=0.12, top=0.83, wspace=0.28)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def solve_downstream_case(
    *,
    dataset,
    value,
    shape,
    geometry,
    methods,
    protocols,
    evaluation_root,
    output_root,
    problem,
    solver_options,
    manufactured_solution,
    save_plots=True,
    zoom_bounds=None,
):
    """Solve all method/protocol pairs for one value and grid shape."""
    meshes, timings, archive = load_evaluated_meshes(
        evaluation_root, dataset, shape, value, methods
    )
    rows = []
    for protocol in protocols:
        directory = (
            Path(output_root)
            / dataset
            / f"{float(value):g}"
            / "x".join(map(str, shape))
            / manufactured_solution.name
            / protocol
        )
        directory.mkdir(parents=True, exist_ok=True)
        solved_meshes, solutions, errors = {}, {}, {}
        for method, raw_points in meshes.items():
            points = raw_points
            if protocol == "exact_boundary_control" and method in LEARNED_METHODS:
                points = replace_with_exact_boundary(raw_points, geometry, shape)
            boundary = build_exact_dirichlet_boundary(
                points, shape, manufactured_solution.value
            )
            base = {
                "dataset": dataset,
                "parameter_value": float(value),
                "grid_shape": "x".join(map(str, shape)),
                "protocol": protocol,
                "method": method,
                "mesh_archive": str(archive),
                "generation_seconds": generation_seconds(timings, method),
                "manufactured_solution": manufactured_solution.name,
                **boundary_geometry_error(points, geometry, shape),
            }
            try:
                result = solve_poisson_2d(
                    points,
                    shape,
                    boundary,
                    conductivity=float(problem["conductivity"]),
                    source=manufactured_solution.source,
                    **solver_options,
                )
                comparison = compare_to_exact_solution(
                    points,
                    shape,
                    result.solution,
                    manufactured_solution.value,
                    manufactured_solution.gradient,
                )
                row = {
                    **base,
                    "status": "success",
                    **result.metrics(),
                    **{key: value for key, value in comparison.items() if np.isscalar(value)},
                }
                solved_meshes[method] = (points, shape)
                solutions[method] = result.solution
                errors[method] = comparison["nodal_error"]
                (directory / "fields").mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    directory / "fields" / f"{method.lower().replace('-', '_')}.npz",
                    points=points,
                    solution=result.solution,
                    exact=comparison["reference_on_candidate_grid"],
                    reference=comparison["reference_on_candidate_grid"],
                    error=comparison["nodal_error"],
                )
            except Exception as error:
                row = {**base, "status": "failed", "failure": f"{type(error).__name__}: {error}"}
            rows.append(row)
        _write_csv(directory / "metrics.csv", [r for r in rows if r["protocol"] == protocol])
        write_json(directory / "summary.json", [r for r in rows if r["protocol"] == protocol])
        if save_plots:
            _plot_fields(
                directory / "figures" / "solution_comparison.png",
                solved_meshes,
                solutions,
                f"{dataset} | {value:g} | {'x'.join(map(str, shape))} | {protocol}",
            )
            _plot_errors(
                directory / "figures" / "error_comparison.png",
                solved_meshes,
                errors,
                "Absolute error against the manufactured exact solution",
            )
            if zoom_bounds:
                _plot_errors(
                    directory / "figures" / "high_curvature_zoom.png",
                    solved_meshes,
                    errors,
                    "High-curvature-region error against the exact solution",
                    zoom_bounds,
                )
    return rows


def summarize_rows(rows, output_root):
    """Write detailed and method-level mean/std tables."""
    import pandas as pd

    output_root = Path(output_root)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["nominal_h"] = frame["grid_shape"].map(
            lambda value: 1.0
            / max(int(part) - 1 for part in str(value).lower().split("x"))
        )
        frame["l2_convergence_order"] = np.nan
        frame["h1_convergence_order"] = np.nan
        group_columns = [
            "dataset",
            "parameter_value",
            "manufactured_solution",
            "protocol",
            "method",
        ]
        for _, indices in frame.groupby(group_columns, dropna=False).groups.items():
            ordered = frame.loc[list(indices)].sort_values("nominal_h", ascending=False)
            previous = None
            for index, row in ordered.iterrows():
                if previous is not None and row["status"] == previous["status"] == "success":
                    denominator = np.log(previous["nominal_h"] / row["nominal_h"])
                    if denominator > 0:
                        for error_name, order_name in (
                            ("relative_l2_error", "l2_convergence_order"),
                            ("relative_h1_seminorm_error", "h1_convergence_order"),
                        ):
                            coarse = float(previous[error_name])
                            fine = float(row[error_name])
                            if coarse > 0 and fine > 0:
                                frame.loc[index, order_name] = np.log(coarse / fine) / denominator
                previous = row
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "metrics_by_case.csv", index=False, encoding="utf-8-sig")
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column != "parameter_value"
    ]
    successful = frame[frame["status"] == "success"]
    if not successful.empty:
        grouped = successful.groupby(
            ["manufactured_solution", "protocol", "method"]
        )[numeric].agg(["mean", "std"])
        grouped.to_csv(output_root / "mean_std_by_method.csv", encoding="utf-8-sig")
        identity = [
            "dataset",
            "parameter_value",
            "grid_shape",
            "manufactured_solution",
            "protocol",
        ]
        paper_metrics = [
            name
            for name in (
                "relative_l2_error",
                "relative_h1_seminorm_error",
                "l2_convergence_order",
                "h1_convergence_order",
                "nodal_maximum_absolute_error",
                "boundary_max_error",
                "minimum_element_jacobian",
                "assembly_seconds",
                "solve_seconds",
                "generation_seconds",
            )
            if name in successful.columns
        ]
        paper = successful.melt(
            id_vars=[*identity, "method"],
            value_vars=paper_metrics,
            var_name="metric",
            value_name="value",
        ).pivot_table(index=[*identity, "metric"], columns="method", values="value")
        paper.to_csv(output_root / "paper_table.csv", encoding="utf-8-sig")
    success = (
        frame.assign(solve_success=frame["status"].eq("success").astype(int))
        .groupby(["manufactured_solution", "protocol", "method"])["solve_success"]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )
    success.to_csv(output_root / "solver_success.csv", index=False, encoding="utf-8-sig")
    return frame
