"""Two-dimensional dataset generators and their shared generation helpers.

Choose entries in SELECTED_DATASETS and run this module to generate them in
sequence. Keep one public generator per family and share sampling/PDE helpers.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack

import argparse
import inspect
import json
import sys
import tempfile
import time
import uuid
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from scripts.pde.vectorized_winslow_2d import (
    solve_winslow_2d_vectorized as solve_winslow_2d,
)
from scripts.tfi.tfi_2d import tfi_from_curves
from src.data.geometry.boundary_generator_2d import (
    BOUNDARY_GENERATORS,
    smooth_constricted_channel_boundary,
    variable_width_s_channel_boundary,
    gear_spline_boundary,
    rounded_u_notch_boundary,
    circle_airfoil_boundary,
    GenerateRectangleWithTopSemiEllipseBoundary,
    EccentricAnnularSectorBoundary,
    VerticallyAsymmetricConvexTriBoundary,
    generate_boundary,
)
from src.data.dataset import (
    BOUNDARY_GEOMETRY_FILENAME,
    NUMERICAL_DATA_FILENAME,
    PAIRED_FORMAT,
    read_dataset_metadata,
)
from src.data.mesh.topology import GenerateQuadMeshBoundaryIndex
from src.data.geometry.parametric_curve_2d import BoundaryCurveSet2D
from src.data.generation.dataset_naming import (
    build_dataset_name as build_canonical_dataset_name,
    default_version_for_split,
    validate_version,
)

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "2D"

# 按边界构造器名称排序，键对应 DATASET_FAMILIES 中的 function。
# 多个数据集族可共用一个边界构造器；返回值为 BoundaryCurveSet2D。
DATASET_GENERATORS = {
    "circle_airfoil": circle_airfoil_boundary,
    "eccentric_semiannulus": EccentricAnnularSectorBoundary,
    "gear_spline": gear_spline_boundary,
    "rect_top_semiellipse": GenerateRectangleWithTopSemiEllipseBoundary,
    "rounded_u_notch": rounded_u_notch_boundary,
    "smooth_constricted_channel": smooth_constricted_channel_boundary,
    "symmetric_convex_tri": VerticallyAsymmetricConvexTriBoundary,
    "variable_width_s_channel": variable_width_s_channel_boundary,
}

# 按数据集族完整名称排序，与 SELECTED_DATASETS 和专用生成函数一致。
# output_name 为输出简称；parameter 为变化参数，fixed_parameters 为固定几何参数。
# interpolation/extrapolation 指定训练参数区间；equal_interval_counts 控制区间样本分配。
DATASET_FAMILIES = {
    "circle_airfoil_t": {
        "output_name": "air_t",
        "function": "circle_airfoil",
        "parameter": "t",
        "description": (
            "外圆与内部 NACA 四位数翼型、两条连接线组成的四边参数边界；"
            "变化参数为翼型相对厚度 t。"
        ),
        "fixed_parameters": {
            "radius": 1.0,
            "m": 0.02,
            "p": 0.4,
            "c": 1.0,
            "center": [0.0, 0.0],
        },
        # 与现有 air_t 训练、评估配置一致。
        "grid_shape": [21, 20],
        # 按区间长度分配样本数；每个区间内部均匀采样。
        "equal_interval_counts": False,
        "interpolation": [[0.055, 0.25], [0.35, 0.60]],
        "extrapolation": [[0.055, 0.60]],
    },
    "eccentric_semiannulus": {
        "output_name": "esa",
        "function": "eccentric_semiannulus",
        "parameter": "inner_center_x",
        "description": "固定内外圆半径的偏心上半环，变化参数为内圆圆心横坐标。",
        "fixed_parameters": {"outer_radius": 0.5, "inner_radius": 0.1},
        "grid_shape": [20, 20],
        "equal_interval_counts": True,
        "interpolation": [[0.3, 0.45], [0.55, 0.7]],
        "extrapolation": [[0.3, 0.7]],
    },
    "gear_spline": {
        "output_name": "gsp",
        "function": "gear_spline",
        "parameter": "amplitude",
        "description": "四条三次 B 样条波纹边界，端部波幅平滑减弱；固定6周期。",
        "fixed_parameters": {
            "width": 2.0,
            "height": 1.6,
            "teeth": 6,
            "controls_per_wave": 16,
        },
        "grid_shape": [81, 81],
        "equal_interval_counts": True,
        "interpolation": [[0.02, 0.1], [0.15, 0.2]],
        "extrapolation": [[0.02, 0.18]],
    },
    "rect_top_semiellipse": {
        "output_name": "rse",
        "function": "rect_top_semiellipse",
        "parameter": "semi_minor_axis",
        "description": "单位矩形的上边界为半椭圆，变化参数为椭圆短半轴。",
        "fixed_parameters": {"length": 1.0, "height": 1.0},
        "grid_shape": [20, 20],
        "equal_interval_counts": True,
        "interpolation": [[0.0, 0.2], [0.3, 0.5]],
        "extrapolation": [[0.0, 0.4]],
    },
    "rounded_u_notch_compact": {
        "output_name": "run",
        "function": "rounded_u_notch",
        "parameter": "depth",
        "description": "小规模深 U 槽；61x21均匀网格，圆角半径0.25，仅变化depth。",
        "fixed_parameters": {
            "width": 3.0,
            "height": 2.0,
            "notch_width": 0.9,
            "radius": 0.25,
        },
        "grid_shape": [61, 21],
        "equal_interval_counts": True,
        "interpolation": [[0.55, 1.0], [1.3, 1.5]],
        "extrapolation": [[0.55, 1.5]],
    },
    "smooth_constricted_channel": {
        "output_name": "scc",
        "function": "smooth_constricted_channel",
        "parameter": "depth",
        "description": "上下光滑相向凹陷的通道；最小间隙 height-2*depth。",
        "fixed_parameters": {"length": 3.0, "height": 1.0},
        "grid_shape": [41, 41],
        "equal_interval_counts": True,
        "interpolation": [[0.05, 0.25], [0.35, 0.42]],
        "extrapolation": [[0.05, 0.4]],
    },
    "symmetric_convex_tri": {
        "output_name": "tri",
        "function": "symmetric_convex_tri",
        "parameter": "protrusion_distance",
        "description": "单位矩形的上下边界为对称三角折线，变化参数为凸起高度。",
        "fixed_parameters": {"width": 1.0, "side_length": 1.0},
        "grid_shape": [21, 20],
        "equal_interval_counts": True,
        "interpolation": [[0.05, 0.2], [0.3, 0.5]],
        "extrapolation": [[0.05, 0.4]],
    },
    "variable_width_s_channel": {
        "output_name": "vsc",
        "function": "variable_width_s_channel",
        "parameter": "constriction",
        "description": "Bézier 变宽 S 通道；仅变化收缩控制量，弯曲控制量固定。",
        "fixed_parameters": {"length": 4.0, "height": 0.8, "bend": 2.0},
        "grid_shape": [41, 41],
        "equal_interval_counts": True,
        "interpolation": [[0, 0.3], [0.6, 0.85]],
        "extrapolation": [[0, 0.8]],
    },
    "variable_width_s_channel_bend": {
        "output_name": "vsb",
        "function": "variable_width_s_channel",
        "parameter": "bend",
        "description": "变宽 S 通道；只变化弯曲控制量 bend，收缩控制量固定为0.65。",
        "fixed_parameters": {"length": 4.0, "height": 0.8, "constriction": 0.65},
        "grid_shape": [41, 41],
        "equal_interval_counts": True,
        "interpolation": [[0.5, 1.5], [2.5, 3.5]],
        "extrapolation": [[0.5, 3.0]],
    },
}

SPLIT_ALIASES = {
    "ext": "extrapolation",
    "extrapolation": "extrapolation",
    "int": "interpolation",
    "interpolation": "interpolation",
}

# 按数据集族的完整名称排序；每类依次列出内插、外推两个选项。
# 每项格式为 (family, split, version)，内插和外推的默认版本均为0。
# 深 U 槽使用 rounded_u_notch_compact，默认网格为 61×21。
SELECTED_DATASETS = [
    ("circle_airfoil_t", "interpolation", 0),
    ("circle_airfoil_t", "extrapolation", 0),
    ("eccentric_semiannulus", "interpolation", 0),
    ("eccentric_semiannulus", "extrapolation", 0),
    ("gear_spline", "interpolation", 0),
    ("gear_spline", "extrapolation", 0),
    ("rect_top_semiellipse", "interpolation", 0),
    ("rect_top_semiellipse", "extrapolation", 0),
    ("rounded_u_notch_compact", "interpolation", 0),
    ("rounded_u_notch_compact", "extrapolation", 0),
    ("smooth_constricted_channel", "interpolation", 0),
    ("smooth_constricted_channel", "extrapolation", 0),
    ("symmetric_convex_tri", "interpolation", 0),
    ("symmetric_convex_tri", "extrapolation", 0),
    ("variable_width_s_channel", "interpolation", 0),
    ("variable_width_s_channel", "extrapolation", 0),
    ("variable_width_s_channel_bend", "interpolation", 0),
    ("variable_width_s_channel_bend", "extrapolation", 0),
]

# Common settings for every selected dataset. CLI values override these settings.
DEFAULT_GENERATION_OPTIONS = {
    # None uses each family's original grid; CLI overrides still take priority.
    "nx": None,
    "ny": None,
    "size": 1000,
    "version": None,
    "winslow_max_iter": 1000,
    "winslow_tolerance": 1e-6,
    "timing_samples": 3,
    "workers": 1,
}

BOUNDARY_DESCRIPTIONS = {
    "circle_airfoil": (
        "外边界为圆，内边界为上下两段组成的 NACA 四位数翼型，"
        "另有两条直线连接翼型尾缘与圆的右端点；共四条参数曲线。"
    ),
    "eccentric_semiannulus": "内外两条上半圆与左右水平连接线构成的半环形边界。",
    "four_convex_semicircle": "四条具有相同弦长和圆心角的向外凸圆弧。",
    "gear_spline": "四条三次 B 样条波纹边界，端部波幅平滑减弱；周期数由 teeth 参数指定。",
    "rect_top_semiellipse": "左、下、右为直线段，上边界为向外凸的半椭圆。",
    "rectangle": "上下左右均为直线段的矩形边界。",
    "rounded_u_notch": "顶部深 U 形凹槽，四处精确圆角；PDE 收敛仍可能存在局部非法单元，须检查诊断。",
    "smooth_constricted_channel": "上下光滑相向凹陷的通道；最小间隙 height-2*depth。",
    "symmetric_convex_tri": "左右为直线段，上下为可向外凸或向内凹的等腰三角折线。",
    "variable_width_s_channel": "Bézier 变宽 S 通道；vsc 仅变化 constriction，vsb 仅变化 bend。",
}


def _boundary_parameter_summary(boundary_type, parameters):
    """Include omitted geometry defaults when describing explicit parameter sets."""
    defaults = {
        key: parameter.default
        for key, parameter in inspect.signature(
            BOUNDARY_GENERATORS[boundary_type]
        ).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    actual = [dict(defaults, **values) for values in parameters]
    fixed, ranges = {}, {}
    for key in actual[0]:
        values = [row[key] for row in actual]
        if all(np.array_equal(value, values[0]) for value in values):
            fixed[key] = np.asarray(values[0]).tolist()
        elif all(np.isscalar(value) for value in values):
            ranges[key] = [[float(min(values)), float(max(values))]]
    return fixed, ranges


def update_dataset_catalog(output_root=DEFAULT_OUTPUT_ROOT):
    """Rebuild the sorted, parameter-detailed catalog from saved records."""
    from src.data.generation.dataset_catalog_2d import write_catalog

    Path(output_root).mkdir(parents=True, exist_ok=True)
    return write_catalog(output_root, DATASET_FAMILIES, BOUNDARY_DESCRIPTIONS)


def sample_parameter_ranges(ranges, count=1000, *, positive=False):
    """Return parameter values, merged intervals and per-interval counts.

    Overlapping or touching intervals are merged. Samples are allocated in
    proportion to interval length (up to integer rounding), with at least two
    samples per interval so both endpoints are present. Each interval uses
    linspace, and the final vector is sorted with no duplicates. Set positive
    for parameters such as thickness; signed geometry parameters are allowed
    otherwise.
    """
    if isinstance(count, bool) or int(count) != count or count < 2:
        raise ValueError("count must be an integer >= 2")
    count = int(count)
    ranges = np.asarray(ranges, dtype=float)
    if ranges.ndim != 2 or ranges.shape[1] != 2 or len(ranges) == 0:
        raise ValueError("ranges must contain one or more (minimum, maximum) pairs")
    if not np.isfinite(ranges).all():
        raise ValueError("Parameter interval endpoints must be finite")
    if positive and np.any(ranges <= 0):
        raise ValueError("Parameter interval endpoints must be positive")
    if np.any(ranges[:, 0] >= ranges[:, 1]):
        raise ValueError("Each parameter interval must have minimum < maximum")
    merged = []
    for lower, upper in ranges[np.argsort(ranges[:, 0])]:
        if merged and lower <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], float(upper))
        else:
            merged.append([float(lower), float(upper)])
    if count < 2 * len(merged):
        raise ValueError("count must allow at least two endpoints per merged interval")
    lengths = np.diff(np.asarray(merged), axis=1).ravel()
    ideal_counts = count * lengths / lengths.sum()
    counts = np.maximum(2, np.floor(ideal_counts).astype(int))
    while counts.sum() < count:
        counts[np.argmax(ideal_counts - counts)] += 1
    while counts.sum() > count:
        surplus = np.where(counts > 2, counts - ideal_counts, -np.inf)
        counts[np.argmax(surplus)] -= 1
    values = np.concatenate(
        [np.linspace(a, b, int(n)) for (a, b), n in zip(merged, counts)]
    )
    return values, merged, counts.tolist()


def sample_thickness_ranges(t_ranges, count=1000):
    """Sample strictly positive airfoil thickness intervals."""
    return sample_parameter_ranges(t_ranges, count, positive=True)


def to_computational_boundary(boundary, *, source_order=None, tolerance=1e-6):
    """Return any four-curve boundary in increasing computational directions.

    ``computational`` stores bottom/top in increasing xi and left/right in
    increasing eta. ``perimeter`` stores bottom, right, top and left in one
    continuous loop, so top and left must be reversed. The default convention
    comes from boundary.parameter_order; custom inputs can override it here.

    Side labels identify logical sides, not physical x/y extrema. No curve is
    fitted, reordered or resampled. The input is not modified, and repeated
    conversion preserves the same parameterization. Endpoint consistency is
    checked; this does not certify that the entire physical contour is simple.
    """
    if not isinstance(boundary, BoundaryCurveSet2D):
        raise TypeError("boundary must be a BoundaryCurveSet2D")
    order = boundary.parameter_order if source_order is None else source_order
    if order not in {"computational", "perimeter"}:
        raise ValueError("source_order must be 'computational' or 'perimeter'")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    metadata = dict(boundary.metadata)
    metadata["parameter_order"] = "computational"
    metadata["parameter_directions"] = "bottom/top: xi; left/right: eta"
    result = BoundaryCurveSet2D(
        boundary.bottom,
        boundary.right,
        boundary.top.reverse() if order == "perimeter" else boundary.top,
        boundary.left.reverse() if order == "perimeter" else boundary.left,
        closed=boundary.closed,
        metadata=metadata,
        validate=False,
    )
    result.validate(tol=tolerance)
    return result


def _solve_pde_grid(boundary, nx, ny, *, max_iter, tolerance, sample_label):
    """Sample any oriented 2-D boundary and verify its converged PDE grid.

    Return boundary nodes, the (ny,nx,2) target grid and solver diagnostics.
    All dataset families can reuse the same ordering and convergence checks.
    """
    counts = dict(bottom=nx, right=ny, top=nx, left=ny)
    boundary_ids = GenerateQuadMeshBoundaryIndex(ny, nx)
    points = boundary.sample_closed(counts)[:-1]
    initial = tfi_from_curves(boundary, nx=nx, ny=ny).reshape(-1, 2)
    if not np.allclose(initial[boundary_ids], points, atol=1e-10, rtol=0):
        raise RuntimeError(
            f"TFI and sampled boundary ordering disagree: {sample_label}"
        )
    started = time.perf_counter()
    mesh, info = solve_winslow_2d(
        boundary,
        nx,
        ny,
        initial_grid=initial,
        max_iter=int(max_iter),
        tolerance=float(tolerance),
        return_info=True,
    )
    pde_seconds = time.perf_counter() - started
    if not info["converged"]:
        raise RuntimeError(
            f"PDE did not converge: {sample_label}, "
            f"iterations={info['iterations']}, "
            f"relative_difference={info['relative_difference']}"
        )
    boundary_error = float(np.max(np.linalg.norm(mesh[boundary_ids] - points, axis=1)))
    if boundary_error > 1e-10:
        raise RuntimeError(
            f"PDE boundary changed: {sample_label}, error={boundary_error}"
        )
    grid = mesh.reshape(ny, nx, 2)
    a, b, c, d = grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]
    determinants = []
    for first, second, third in ((a, b, c), (b, c, d), (c, d, a), (d, a, b)):
        e, f = second - first, third - second
        determinants.append(e[..., 0] * f[..., 1] - e[..., 1] * f[..., 0])
    invalid = np.any(np.asarray(determinants) <= 0, axis=0)
    return (
        points,
        grid,
        {
            **info,
            "pde_seconds": pde_seconds,
            "nonpositive_corner_cell_count": int(invalid.sum()),
            "minimum_corner_determinant": float(np.min(determinants)),
            "boundary_max_error": boundary_error,
        },
    )


def _solve_dataset_sample(payload):
    """Solve one serializable task; ordered map preserves sample correspondence."""
    descriptor, nx, ny, max_iter, tolerance, label = payload
    start = time.perf_counter()
    boundary = BoundaryCurveSet2D.from_dict(descriptor)
    points, mesh, info = _solve_pde_grid(
        boundary, nx, ny, max_iter=max_iter, tolerance=tolerance, sample_label=label
    )
    return descriptor, points, mesh, info, time.perf_counter() - start


def _generate_pde_dataset(
    name,
    parameters,
    boundary_factory,
    nx,
    ny,
    *,
    output_root,
    metadata,
    pde_max_iter,
    pde_tolerance,
    timing_samples,
    workers=1,
):
    """Generate paired, index-aligned curve records and legacy-style mesh records."""
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("name must be a single nonempty directory name")
    for key, value, minimum in (
        ("nx", nx, 3),
        ("ny", ny, 3),
        ("pde_max_iter", pde_max_iter, 1),
        ("timing_samples", timing_samples, 0),
        ("workers", workers, 1),
    ):
        if isinstance(value, bool) or int(value) != value or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if not np.isfinite(pde_tolerance) or pde_tolerance <= 0:
        raise ValueError("pde_tolerance must be finite and positive")
    nx, ny = int(nx), int(ny)
    count = len(parameters)
    if not count:
        raise ValueError("parameters must contain at least one sample")
    parameter_names = sorted(parameters[0])
    if any(sorted(row) != parameter_names for row in parameters):
        raise ValueError("All samples must use the same parameter keys")
    root = Path(output_root).resolve()
    destination = root / name
    if destination.exists():
        raise FileExistsError(f"Dataset already exists: {destination}")
    root.mkdir(parents=True, exist_ok=True)
    identity = uuid.uuid4().hex
    started = time.perf_counter()
    next_percent = 10
    numerical_records, geometry_records, diagnostics = [], [], []

    # Publish both files together only after every PDE solve and write succeeds.
    with ExitStack() as stack:
        temporary = stack.enter_context(
            tempfile.TemporaryDirectory(prefix=f".{name}_", dir=root)
        )
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        payloads = [
            (
                to_computational_boundary(boundary_factory(**values)).to_dict(),
                nx,
                ny,
                pde_max_iter,
                pde_tolerance,
                f"sample={index}, parameters={values}",
            )
            for index, values in enumerate(parameters)
        ]
        if workers > 1:
            pool = stack.enter_context(ProcessPoolExecutor(max_workers=int(workers)))
            results = pool.map(_solve_dataset_sample, payloads, chunksize=4)
        else:
            results = map(_solve_dataset_sample, payloads)
        for index, (values, result) in enumerate(zip(parameters, results)):
            descriptor, points, mesh, info, elapsed = result
            numerical_records.append(
                {
                    "sample_id": index,
                    "dataset_id": identity,
                    "boundary_points": points.astype(np.float32),
                    "mesh_points": mesh.reshape(-1, 2).astype(np.float32),
                }
            )
            geometry_records.append(
                {
                    "sample_id": index,
                    "dataset_id": identity,
                    "parameters": dict(values),
                    "boundary": descriptor,
                }
            )
            diagnostics.append(
                {
                    "sample_id": index,
                    "parameters": dict(values),
                    **info,
                    "generation_seconds": elapsed,
                }
            )
            if index < timing_samples:
                print(
                    f"sample {index + 1}/{count}: parameters={values}, "
                    f"{elapsed:.3f}s, PDE {info['iterations']} iterations",
                    flush=True,
                )
            completed_percent = (index + 1) * 100 // count
            while completed_percent >= next_percent and next_percent <= 100:
                print(
                    f"Progress: {next_percent}% ({index + 1}/{count})",
                    flush=True,
                )
                next_percent += 10

        # Numeric-file metadata contains no curve descriptions or geometry parameters.
        numerical_metadata = {
            "format": PAIRED_FORMAT,
            "dataset_id": identity,
            "dataset": name,
            "dimension": 2,
            "size": count,
            "grid_shape": [nx, ny],
            "data_role": "mesh",
            "coordinate_dtype": "float32",
            "pde_method": "winslow_iterative",
            "pde_implementation": "vectorized_dirichlet_elimination",
            "boundary_sampling": "uniform_parameter_with_endpoints",
            "boundary_point_count": 2 * (nx + ny) - 4,
        }
        sample_times = [row["generation_seconds"] for row in diagnostics]
        geometry_metadata = {
            **numerical_metadata,
            **metadata,
            "data_role": "geometry",
            "parameter_names": parameter_names,
            "pde_initialization": "tfi",
            "pde_max_iter": int(pde_max_iter),
            "pde_tolerance": float(pde_tolerance),
            "winslow_max_iter": int(pde_max_iter),
            "winslow_tolerance": float(pde_tolerance),
            "timing_samples_requested": int(timing_samples),
            "sample_generation_time_seconds": {
                "first": sample_times[0],
                "last": sample_times[-1],
                "minimum": min(sample_times),
                "maximum": max(sample_times),
                "mean": float(np.mean(sample_times)),
                "total": float(np.sum(sample_times)),
            },
            "all_pde_converged": True,
            "workers": int(workers),
            "quality_check_scope": "local_corner_determinants_only_not_global_overlap",
            "samples_with_nonpositive_corner_cells": sum(
                d["nonpositive_corner_cell_count"] > 0 for d in diagnostics
            ),
            "pde_diagnostics": diagnostics,
            "generation_seconds": time.perf_counter() - started,
        }
        numerical_records[0]["metadata"] = numerical_metadata
        geometry_records[0]["metadata"] = geometry_metadata
        np.save(
            stage / NUMERICAL_DATA_FILENAME,
            np.asarray(numerical_records, dtype=object),
            allow_pickle=True,
        )
        np.save(
            stage / BOUNDARY_GEOMETRY_FILENAME,
            np.asarray(geometry_records, dtype=object),
            allow_pickle=True,
        )
        from src.data.generation.dataset_examples import save_dataset_examples

        save_dataset_examples(
            stage, numerical_records, geometry_metadata, geometry_records
        )
        stage.rename(destination)
    update_dataset_catalog(root)
    print(f"Dataset saved: {destination}", flush=True)
    return destination


def generate_explicit_dataset(
    name,
    boundary_type,
    parameters,
    grid_shape=(21, 20),
    output_root=DEFAULT_OUTPUT_ROOT,
    *,
    pde_max_iter=1000,
    pde_tolerance=1e-6,
    timing_samples=3,
):
    """Generate PDE targets for explicit parameter sets of any registered 2-D boundary."""
    parameters = [dict(values) for values in parameters]
    if not parameters:
        raise ValueError("parameters must contain at least one sample")
    fixed, ranges = _boundary_parameter_summary(boundary_type, parameters)
    nx, ny = grid_shape
    return _generate_pde_dataset(
        name,
        parameters,
        lambda **values: generate_boundary(boundary_type, **values),
        nx,
        ny,
        output_root=output_root,
        metadata={
            "boundary_type": boundary_type,
            "boundary_description": BOUNDARY_DESCRIPTIONS.get(
                boundary_type, boundary_type
            ),
            "fixed_parameters": fixed,
            "parameter_ranges": ranges,
            "parameter_sampling": "explicit_parameter_sets",
        },
        pde_max_iter=pde_max_iter,
        pde_tolerance=pde_tolerance,
        timing_samples=timing_samples,
    )


def normalize_split(split):
    """Normalize the original int/ext aliases, including uppercase spellings."""
    normalized = SPLIT_ALIASES.get(str(split).lower())
    if normalized is None:
        raise ValueError(
            "split must be interpolation/int or extrapolation/ext; "
            f"received {split!r}"
        )
    return normalized


def resolve_family(value):
    """Accept either a registry key or its short output name."""
    if value in DATASET_FAMILIES:
        return value
    matches = [
        family
        for family, specification in DATASET_FAMILIES.items()
        if specification["output_name"] == value
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous dataset family: {value!r}")
    return matches[0]


def build_dataset_name(family, split, version, size, nx, ny):
    """Name datasets as boundary_int/ext_version_size_nx_ny."""
    if family not in DATASET_FAMILIES:
        raise ValueError(f"Unknown dataset family: {family!r}")
    for value in (nx, ny):
        if isinstance(value, bool) or int(value) != value or value < 3:
            raise ValueError("Grid dimensions must be integers >= 3")
    prefix = DATASET_FAMILIES[family]["output_name"]
    return build_canonical_dataset_name(prefix, split, version, size, nx, ny)


def generate_dataset(
    family,
    split,
    version=0,
    size=1000,
    nx=None,
    ny=None,
    winslow_max_iter=1000,
    winslow_tolerance=1e-6,
    timing_samples=3,
    output_dir=None,
    workers=1,
):
    """Generate one configured family/split using exact parametric boundaries.

    The public workflow matches the original generator: normalize the split,
    resolve the family specification, sample the parameter, construct boundaries,
    solve Winslow grids and save the automatically named dataset. Geometry
    parameters remain evenly spaced as required for the new dataset format.
    """
    if family not in DATASET_FAMILIES:
        raise ValueError(f"Unknown dataset family: {family!r}")
    specification = DATASET_FAMILIES[family]
    default_grid = specification.get("grid_shape", [21, 21])
    nx = default_grid[0] if nx is None else nx
    ny = default_grid[1] if ny is None else ny
    if family == "symmetric_convex_tri" and nx % 2 == 0:
        raise ValueError("Triangular boundaries require odd nx to sample the apex")
    split = normalize_split(split)
    version = validate_version(version)
    name = build_dataset_name(family, split, version, size, nx, ny)
    specification = DATASET_FAMILIES[family]
    function = specification["function"]
    if function not in DATASET_GENERATORS:
        raise ValueError(f"Unregistered parametric boundary constructor: {function!r}")
    constructor = DATASET_GENERATORS[function]
    parameter = specification["parameter"]
    values, intervals, interval_counts = sample_parameter_ranges(
        specification[split],
        size,
        positive=(function == "circle_airfoil" and parameter == "t"),
    )
    if specification.get("equal_interval_counts"):
        interval_counts = [
            int(size) // len(intervals) + (i < int(size) % len(intervals))
            for i in range(len(intervals))
        ]
        values = np.concatenate(
            [np.linspace(a, b, n) for (a, b), n in zip(intervals, interval_counts)]
        )
    fixed = {
        key: value.default
        for key, value in inspect.signature(constructor).parameters.items()
        if value.default is not inspect.Parameter.empty and key != parameter
    }
    fixed.update(specification.get("fixed_parameters", {}))
    if parameter in fixed:
        raise ValueError(f"Varying parameter {parameter!r} must not be fixed")

    def boundary_factory(**parameters):
        boundary = constructor(**fixed, **parameters)
        if function == "circle_airfoil":
            airfoil = boundary.top.sample(max(257, int(nx)))
            distance = np.linalg.norm(airfoil - np.asarray(fixed["center"]), axis=1)
            if np.any(distance >= fixed["radius"]):
                raise ValueError(f"Airfoil must lie inside the circle: {parameters}")
        return boundary

    print(f"Dataset: {name}", flush=True)
    print(f"Description: {specification['description']}", flush=True)
    print(f"Split: {split}; parameter: {parameter}; ranges: {intervals}", flush=True)
    print(f"Size: {size}; grid: {nx}x{ny}", flush=True)
    return _generate_pde_dataset(
        name,
        [{parameter: float(value)} for value in values],
        boundary_factory,
        nx,
        ny,
        output_root=DEFAULT_OUTPUT_ROOT if output_dir is None else output_dir,
        metadata={
            "family": family,
            "split": split,
            "version": version,
            "function": function,
            "parameter": parameter,
            "description": specification["description"],
            "parameter_intervals": intervals,
            "boundary_type": function,
            "boundary_description": specification["description"],
            "fixed_parameters": fixed,
            "parameter_ranges": {parameter: intervals},
            "interval_sample_counts": interval_counts,
            "parameter_sampling": (
                "equal_interval_linspace_with_endpoints"
                if specification.get("equal_interval_counts")
                else "length_proportional_linspace_with_endpoints"
            ),
        },
        pde_max_iter=winslow_max_iter,
        pde_tolerance=winslow_tolerance,
        timing_samples=timing_samples,
        workers=workers,
    )


# 数据集专用生成入口，按数据集族完整名称排序。
def generate_circle_airfoil_t_dataset(
    name,
    t_ranges=None,
    nx=21,
    ny=20,
    count=1000,
    *,
    radius=1.0,
    m=0.02,
    p=0.4,
    c=1.0,
    center=(0.0, 0.0),
    output_root=DEFAULT_OUTPUT_ROOT,
    pde_max_iter=1000,
    pde_tolerance=1e-6,
    timing_samples=3,
    split=None,
):
    """Save exact boundary descriptions separately from sampled boundaries/PDE grids.

    Geometry options are Python-only arguments. Circle and airfoil each use
    nx uniform parameter nodes; connectors each use ny, including endpoints.
    Odd nx includes the airfoil leading junction at parameter 0.5. Each record
    in dataset.npy has legacy boundary_points/mesh_points float32 arrays.
    boundary_geometry.npy stores the corresponding curve descriptors and
    parameters. No random interior samples or interpolated targets are saved.
    With no explicit ranges, use the selected split from DATASET_FAMILIES
    (interpolation by default). Explicit ranges without a split are custom.
    """
    if split is None:
        split = "interpolation" if t_ranges is None else "custom"
    if split != "custom":
        split = normalize_split(split)
    if t_ranges is None:
        if split == "custom":
            raise ValueError("Custom datasets require explicit t_ranges")
        t_ranges = DATASET_FAMILIES["circle_airfoil_t"][split]
    values, intervals, interval_counts = sample_thickness_ranges(t_ranges, count)
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center must contain two finite coordinates")
    if not np.isfinite([radius, m, p, c]).all():
        raise ValueError("Geometry parameters must be finite")
    if radius <= 0 or c <= 0 or m < 0 or not 0 < p < 1:
        raise ValueError("Require radius,c > 0, m >= 0, and 0 < p < 1")
    fixed = dict(
        radius=float(radius),
        m=float(m),
        p=float(p),
        c=float(c),
        center=center.tolist(),
    )

    def boundary_factory(t):
        boundary = circle_airfoil_boundary(t=t, **fixed)
        airfoil = boundary.top.sample(max(257, int(nx)))
        if np.any(np.linalg.norm(airfoil - center, axis=1) >= radius):
            raise ValueError(f"Airfoil must lie inside the circle: t={t}")
        return boundary

    return _generate_pde_dataset(
        name,
        [{"t": float(value)} for value in values],
        boundary_factory,
        nx,
        ny,
        output_root=output_root,
        metadata={
            "family": "circle_airfoil_t",
            "split": split,
            "boundary_type": "circle_airfoil",
            "boundary_description": BOUNDARY_DESCRIPTIONS["circle_airfoil"],
            "fixed_parameters": fixed,
            "parameter_ranges": {"t": intervals},
            "requested_t_ranges": np.asarray(t_ranges, dtype=float).tolist(),
            "t_ranges": intervals,
            "interval_sample_counts": interval_counts,
            "parameter_sampling": "length_proportional_linspace_with_endpoints",
        },
        pde_max_iter=pde_max_iter,
        pde_tolerance=pde_tolerance,
        timing_samples=timing_samples,
    )


def generate_eccentric_semiannulus_dataset(split, **options):
    """Generate the original ESA family with fixed inner and outer radii."""
    return generate_dataset("eccentric_semiannulus", split, **options)


def generate_gear_spline_dataset(split, **options):
    """Generate index-aligned gear_spline curve and PDE records."""
    return generate_dataset("gear_spline", split, **options)


def generate_rect_top_semiellipse_dataset(split, **options):
    """Generate the original RSE family with parameter curves and PDE targets."""
    return generate_dataset("rect_top_semiellipse", split, **options)


def generate_rounded_u_notch_compact_dataset(split, **options):
    """Generate the U-notch depth family on its configured 61-by-21 grid."""
    return generate_dataset("rounded_u_notch_compact", split, **options)


def generate_smooth_constricted_channel_dataset(split, **options):
    """Generate index-aligned smooth_constricted_channel curve and PDE records."""
    return generate_dataset("smooth_constricted_channel", split, **options)


def generate_symmetric_convex_tri_dataset(split, **options):
    """Generate the original TRI family; an odd nx retains both apex nodes."""
    return generate_dataset("symmetric_convex_tri", split, **options)


def generate_variable_width_s_channel_bend_dataset(split, **options):
    """Generate S channels varying bend only; keep constriction fixed."""
    return generate_dataset("variable_width_s_channel_bend", split, **options)


def generate_variable_width_s_channel_dataset(split, **options):
    """Generate index-aligned variable_width_s_channel curve and PDE records."""
    return generate_dataset("variable_width_s_channel", split, **options)


def generate_selected_datasets(options=None, output_dir=None):
    """Generate each uncommented selection with shared options and separate outputs."""
    selected = list(SELECTED_DATASETS)
    if not selected:
        raise ValueError("SELECTED_DATASETS is empty; enable at least one entry")
    settings = dict(DEFAULT_GENERATION_OPTIONS)
    if options is not None:
        unknown = set(options) - set(settings)
        if unknown:
            raise ValueError(f"Unknown generation options: {sorted(unknown)}")
        settings.update(options)

    # Check the full selection before starting any expensive PDE solve.
    jobs = []
    names = set()
    destination = Path(DEFAULT_OUTPUT_ROOT if output_dir is None else output_dir)
    for item in selected:
        if len(item) == 2:
            family, split = item
            selected_version = default_version_for_split(normalize_split(split))
        elif len(item) == 3:
            family, split, selected_version = item
        else:
            raise ValueError(
                "Each selected dataset must be (family, split) or "
                "(family, split, version)"
            )
        if family not in DATASET_FAMILIES:
            raise ValueError(f"Unknown dataset family: {family!r}")
        split = normalize_split(split)
        grid = DATASET_FAMILIES[family].get("grid_shape", [21, 21])
        nx = grid[0] if settings["nx"] is None else settings["nx"]
        ny = grid[1] if settings["ny"] is None else settings["ny"]
        if family == "symmetric_convex_tri" and nx % 2 == 0:
            raise ValueError("Triangular boundaries require odd nx to sample the apex")
        version = settings.get("version")
        version = selected_version if version is None else validate_version(version)
        name = build_dataset_name(family, split, version, settings["size"], nx, ny)
        if name in names:
            raise ValueError(f"Duplicate selected dataset: {name}")
        if (destination / name).exists():
            raise FileExistsError(f"Dataset already exists: {destination / name}")
        if DATASET_FAMILIES[family]["function"] not in DATASET_GENERATORS:
            raise ValueError(f"Unregistered parametric boundary for {family!r}")
        ranges = DATASET_FAMILIES[family][split]
        specification = DATASET_FAMILIES[family]
        sample_parameter_ranges(
            ranges,
            settings["size"],
            positive=(
                specification["function"] == "circle_airfoil"
                and specification["parameter"] == "t"
            ),
        )
        names.add(name)
        jobs.append((family, split, version, name, ranges))

    outputs = []
    generation_settings = {
        key: value for key, value in settings.items() if key != "version"
    }
    for index, (family, split, version, name, ranges) in enumerate(jobs, start=1):
        print(f"Generating dataset {index}/{len(jobs)}: {name}", flush=True)
        outputs.append(
            generate_dataset(
                family=family,
                split=split,
                version=version,
                output_dir=output_dir,
                **generation_settings,
            )
        )
    return outputs


def parse_grid_shape(text):
    """Parse the 2-D form of the original --grid-shape option, such as 21x21."""
    try:
        dimensions = tuple(int(value) for value in text.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("grid shape must look like 21x21") from error
    if len(dimensions) != 2 or min(dimensions) < 3:
        raise argparse.ArgumentTypeError("grid shape requires two dimensions >= 3")
    return dimensions


def main(argv=None):
    """Run SELECTED_DATASETS; CLI options only override common generation settings."""
    parser = argparse.ArgumentParser(
        description="Generate every enabled entry in SELECTED_DATASETS with automatic names.",
        allow_abbrev=False,
    )
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--family")
    parser.add_argument("--split", choices=tuple(SPLIT_ALIASES))
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--grid-shape", type=parse_grid_shape)
    parser.add_argument("--size", "--count", dest="size", type=int)
    parser.add_argument(
        "--winslow-max-iter", "--pde-max-iter", dest="winslow_max_iter", type=int
    )
    parser.add_argument(
        "--winslow-tolerance", "--pde-tolerance", dest="winslow_tolerance", type=float
    )
    parser.add_argument("--timing-samples", type=int)
    parser.add_argument("--workers", type=int, help="Parallel PDE workers; default 1")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.grid_shape is not None:
        if args.nx is not None or args.ny is not None:
            parser.error("use --grid-shape or --nx/--ny, not both")
        args.nx, args.ny = args.grid_shape
    options = {
        key: value
        for key, value in vars(args).items()
        if value is not None and key not in ("grid_shape", "output_dir", "family", "split")
    }
    if (args.family is None) != (args.split is None):
        parser.error("--family and --split must be supplied together")
    if args.family is not None:
        return generate_dataset(
            family=resolve_family(args.family),
            split=args.split,
            output_dir=args.output_dir,
            **options,
        )
    return generate_selected_datasets(options, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

