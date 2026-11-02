"""End-to-end downstream heat-equation evaluation on generated meshes."""

from __future__ import annotations

import csv
import json
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
from src.numerics.downstream.boundary_conditions import build_dirichlet_boundary
from src.numerics.downstream.field_comparison import (
    boundary_heat_flux,
    compare_to_reference,
)
from src.numerics.downstream.poisson_2d import solve_poisson_2d
from src.numerics.downstream.reference_solution import (
    load_reference,
    reference_identity,
    save_reference,
)


LEARNED_METHODS = {"MeshONet", "PI-MeshONet", "HMONet"}


def _edge_values(problem):
    values = {edge: float(problem["cold_value"]) for edge in problem["cold_edges"]}
    values.update({edge: float(problem["hot_value"]) for edge in problem["hot_edges"]})
    return values


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
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="|u-u_ref|")
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
    reference_shape,
    reference_solution,
    save_plots=True,
    zoom_bounds=None,
):
    """Solve all method/protocol pairs for one value and grid shape."""
    meshes, timings, archive = load_evaluated_meshes(
        evaluation_root, dataset, shape, value, methods
    )
    boundary = build_dirichlet_boundary(
        shape, _edge_values(problem), corner_policy=problem.get("corner_policy", "max")
    )
    rows = []
    for protocol in protocols:
        directory = (
            Path(output_root)
            / dataset
            / f"{float(value):g}"
            / "x".join(map(str, shape))
            / protocol
        )
        directory.mkdir(parents=True, exist_ok=True)
        solved_meshes, solutions, errors = {}, {}, {}
        for method, raw_points in meshes.items():
            points = raw_points
            if protocol == "exact_boundary_control" and method in LEARNED_METHODS:
                points = replace_with_exact_boundary(raw_points, geometry, shape)
            base = {
                "dataset": dataset,
                "parameter_value": float(value),
                "grid_shape": "x".join(map(str, shape)),
                "protocol": protocol,
                "method": method,
                "mesh_archive": str(archive),
                "generation_seconds": generation_seconds(timings, method),
                **boundary_geometry_error(points, geometry, shape),
            }
            try:
                result = solve_poisson_2d(
                    points,
                    shape,
                    boundary,
                    conductivity=float(problem["conductivity"]),
                    source=float(problem["source"]),
                    **solver_options,
                )
                comparison = compare_to_reference(
                    points, shape, result.solution, reference_solution, reference_shape
                )
                row = {
                    **base,
                    "status": "success",
                    **result.metrics(),
                    **{key: value for key, value in comparison.items() if np.isscalar(value)},
                    "hot_boundary_heat_flux": boundary_heat_flux(
                        points,
                        shape,
                        result.solution,
                        edge=problem["hot_edges"][0],
                        conductivity=float(problem["conductivity"]),
                    ),
                }
                solved_meshes[method] = (points, shape)
                solutions[method] = result.solution
                errors[method] = comparison["nodal_error"]
                (directory / "fields").mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    directory / "fields" / f"{method.lower().replace('-', '_')}.npz",
                    points=points,
                    solution=result.solution,
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
                "Absolute error against the PDE-mesh reference",
            )
            if zoom_bounds:
                _plot_errors(
                    directory / "figures" / "high_curvature_zoom.png",
                    solved_meshes,
                    errors,
                    "High-curvature-region error",
                    zoom_bounds,
                )
    return rows


def prepare_reference(
    *,
    dataset,
    value,
    geometry,
    reference_shape,
    evaluation_root,
    output_root,
    problem,
    solver_options,
):
    """Load or solve the high-resolution PDE-mesh reference field."""
    meshes, _, archive = load_evaluated_meshes(
        evaluation_root, dataset, reference_shape, value, ["PDE_Winslow"]
    )
    mesh = meshes["PDE_Winslow"]
    identity = reference_identity(mesh, reference_shape, problem)
    path = Path(output_root) / dataset / f"{float(value):g}" / "reference" / "solution.npz"
    cached = load_reference(path, identity)
    if cached is not None:
        return cached
    boundary = build_dirichlet_boundary(
        reference_shape,
        _edge_values(problem),
        corner_policy=problem.get("corner_policy", "max"),
    )
    result = solve_poisson_2d(
        mesh,
        reference_shape,
        boundary,
        conductivity=float(problem["conductivity"]),
        source=float(problem["source"]),
        **solver_options,
    )
    save_reference(
        path,
        identity,
        result.solution,
        {"mesh_archive": str(archive), "solver_metrics": result.metrics()},
    )
    return result.solution


def summarize_rows(rows, output_root):
    """Write detailed and method-level mean/std tables."""
    import pandas as pd

    output_root = Path(output_root)
    frame = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "metrics_by_case.csv", index=False, encoding="utf-8-sig")
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column != "parameter_value"
    ]
    successful = frame[frame["status"] == "success"]
    if not successful.empty:
        grouped = successful.groupby(["protocol", "method"])[numeric].agg(["mean", "std"])
        grouped.to_csv(output_root / "mean_std_by_method.csv", encoding="utf-8-sig")
        identity = ["dataset", "parameter_value", "grid_shape", "protocol"]
        paper_metrics = [
            name
            for name in (
                "relative_l2_error",
                "relative_h1_seminorm_error",
                "nodal_maximum_absolute_error",
                "boundary_max_error",
                "hot_boundary_heat_flux",
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
        .groupby(["protocol", "method"])["solve_success"]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )
    success.to_csv(output_root / "solver_success.csv", index=False, encoding="utf-8-sig")
    return frame
