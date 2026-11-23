"""Generate parameterized three-dimensional structured-grid datasets."""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys
import json
import argparse
import shutil
import time
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.geometry.boundary_generator_3d import generate_boundary
from src.numerics.traditional.tfi_3d import tfi_from_surfaces
from src.numerics.traditional.winslow_3d import solve_winslow_3d
from src.data.mesh.sampling import boundary_indices, sample_faces
from src.evaluation.quality_3d import surface_quad_quality_fields
from src.metrics import evaluate_hexahedral_grid_quality
from src.data.generation.dataset_naming import (
    build_dataset_name as build_canonical_dataset_name,
    validate_version,
)
from src.data.generation.dataset_validation import (
    endpoint_indices,
    save_quality_audit,
    warn_saved_dataset,
)



ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "raw" / "3D"
from src.data.geometry.pde_surface_3d import CAP_SCHEME, SIDE_SCHEME

DATASET_GENERATORS = {
    "cylindrical_domain_with_airfoil_hole": "cylindrical_domain_with_airfoil_hole",
    "sidewall_deformed_block": "sidewall_deformed_block",
    "rounded_u_notch_extrusion": "rounded_u_notch_extrusion",
    "tapered_elliptic_bore_block": "tapered_elliptic_bore_block",
    "bent_annular_duct": "bent_annular_duct",
    "lobed_annular_duct": "lobed_annular_duct",
    "square_to_round_transition": "square_to_round_transition",
    "variable_area_s_duct": "variable_area_s_duct",
    "twisted_blade_passage": "twisted_blade_passage",
}

# The nine paper/default families are declared explicitly in SELECTED_DATASETS.
# Historical geometry can still be deserialized; generation exposes nine families.
DATASET_FAMILIES = {
    "cylindrical_domain_with_airfoil_hole": {
        "output_name": "cah_t", "function": "cylindrical_domain_with_airfoil_hole",
        "parameter": "thickness",
        "description": "圆柱外域与贯穿轴向的NACA4翼型孔；变化翼型相对厚度。",
        "fixed_parameters": {"outer_radius": 1.5, "chord": 1.0, "camber": 0.02, "camber_position": 0.4, "length": 2.5, "z_min": 0.0},
        "grid_shape": [21, 21, 11],
        "interpolation": [[0.06, 0.15], [0.25, 0.35]], "extrapolation": [[0.08, 0.34]],
        "evaluation_values": {"interpolation": [0.18, 0.20, 0.22], "extrapolation": [0.06, 0.38]},
    },
    "sidewall_deformed_block": {
        "output_name": "swb_a", "function": "sidewall_deformed_block",
        "parameter": "sidewall_amplitude",
        "description": "左右侧壁按正弦剖面对称向中心收缩、沿z方向拉伸的块体；变化无量纲收缩幅值。",
        "fixed_parameters": {"length_x": 3.0, "width_y": 2.0, "length_z": 2.5, "z_min": 0.0},
        "grid_shape": [21, 21, 11],
        "interpolation": [[0.03, 0.10], [0.20, 0.30]], "extrapolation": [[0.04, 0.30]],
        "evaluation_values": {"interpolation": [0.13, 0.15, 0.17], "extrapolation": [0.02, 0.34]},
    },
    "rounded_u_notch_extrusion": {
        "output_name": "run_e", "function": "rounded_u_notch_extrusion",
        "parameter": "depth",
        "description": "二维 run 精确轮廓；201x201二维PDE连续端面，内部由三维PDE求解；仅变化槽深。",
        "fixed_parameters": {"width": 3.0, "height": 2.0, "notch_width": 0.9, "radius": 0.25, "length": 2.5, "z_min": 0.0,
                             "cap_nx": 201, "cap_ny": 201, "cap_max_iter": 1000, "cap_tolerance": 1e-10},
        "grid_shape": [41, 41, 11],
        "interpolation": [[0.55, 1.00], [1.30, 1.50]], "extrapolation": [[0.55, 1.50]],
        "evaluation_values": {"interpolation": [1.10, 1.15, 1.20], "extrapolation": [1.52, 1.55]},
    },
    "tapered_elliptic_bore_block": {
        "output_name": "teb_b", "function": "tapered_elliptic_bore_block",
        "parameter": "minor_axis_out",
        "description": (
            "四条标准直边参数化的矩形块体中带轴向渐变椭圆贯通孔；"
            "变化出口椭圆短半轴，不使用射线与矩形求交。"
        ),
        "fixed_parameters": {"inner_major_axis": 0.32, "inner_minor_axis_in": 0.18, "outer_half_width": 1.5, "outer_half_height": 1.0, "length": 2.5, "z_min": 0.0},
        "grid_shape": [21, 21, 11],
        "interpolation": [[0.12, 0.18], [0.27, 0.32]], "extrapolation": [[0.12, 0.32]],
        "evaluation_values": {"interpolation": [0.21, 0.225, 0.24], "extrapolation": [0.08, 0.36]},
    },
    "bent_annular_duct": {
        "output_name": "bad_a", "function": "bent_annular_duct",
        "parameter": "bend_angle_deg",
        "description": "沿圆弧中心线扫掠的环形弯管；变化弯曲角。",
        "fixed_parameters": {"inner_radius": 0.25, "outer_radius": 0.60, "bend_radius": 3.0, "z_min": 0.0},
        "grid_shape": [21, 21, 11],
        "interpolation": [[20.0, 45.0], [65.0, 90.0]], "extrapolation": [[25.0, 85.0]],
        "evaluation_values": {"interpolation": [50.0, 55.0, 60.0], "extrapolation": [10.0, 105.0]},
    },
    "lobed_annular_duct": {
        "output_name": "lad_a", "function": "lobed_annular_duct",
        "parameter": "amplitude",
        "description": "带轴向相位旋转的星形波瓣环形通道；变化波瓣幅值。",
        "fixed_parameters": {"inner_radius": 0.35, "outer_radius": 1.0, "lobes": 5, "phase_twist_deg": 45.0, "length": 2.5, "z_min": 0.0},
        "grid_shape": [21, 21, 11],
        "interpolation": [[0.02, 0.08], [0.12, 0.18]], "extrapolation": [[0.04, 0.16]],
        "evaluation_values": {"interpolation": [0.09, 0.10, 0.11], "extrapolation": [0.0, 0.20]},
    },
    "square_to_round_transition": {
        "output_name": "srt_r", "function": "square_to_round_transition",
        "parameter": "outlet_radius",
        "description": "矩形入口向圆形出口光滑过渡的变截面管道；变化圆形出口半径。",
        "fixed_parameters": {"inlet_half_width": 1.0, "inlet_half_height": 0.75, "length": 3.0, "z_min": 0.0},
        "grid_shape": [21, 21, 11],
        "interpolation": [[0.58, 0.68], [0.80, 0.90]], "extrapolation": [[0.55, 0.92]],
        "evaluation_values": {"interpolation": [0.72, 0.74, 0.76], "extrapolation": [0.52, 0.96]},
    },
    "variable_area_s_duct": {
        "output_name": "sad_c", "function": "variable_area_s_duct",
        "parameter": "area_ratio_out",
        "description": "沿平面 S 形中心线扫掠的变面积进气道；变化出口/入口面积比。",
        "fixed_parameters": {"inlet_half_width": 0.55, "inlet_half_height": 0.40, "centerline_offset": 0.25, "length": 3.2, "z_min": 0.0},
        "grid_shape": [21, 21, 11],
        "interpolation": [[0.60, 0.82], [1.08, 1.30]], "extrapolation": [[0.55, 1.35]],
        "evaluation_values": {"interpolation": [0.90, 0.95, 1.00], "extrapolation": [0.50, 1.42]},
    },
    "twisted_blade_passage": {
        "output_name": "tbp_a", "function": "twisted_blade_passage",
        "parameter": "twist_angle_deg",
        "description": "单个叶片间流道，两个相邻叶片面随展向改变安装角；变化总扭转角。",
        "fixed_parameters": {"chord": 2.0, "pitch": 1.0, "height": 1.5, "camber_amplitude": 0.22, "blade_thickness": 0.12, "z_min": 0.0},
        "grid_shape": [21, 21, 11],
        "interpolation": [[0.0, 30.0], [60.0, 90.0]], "extrapolation": [[0.0, 90.0]],
        "evaluation_values": {"interpolation": [40.0, 45.0, 50.0], "extrapolation": [0.0, 100.0]},
    },
}

SPLIT_ALIASES = {
    "ext": "extrapolation",
    "extrapolation": "extrapolation",
    "int": "interpolation",
    "interpolation": "interpolation",
}

# Surface accuracy is independent of the volume/network-input resolution.
for _specification in DATASET_FAMILIES.values():
    _specification['fixed_parameters'].update(
        cap_nx=201, cap_ny=201, cap_max_iter=1000, cap_tolerance=1e-10)

DEFAULT_DATASET_VERSION = 2

SELECTED_DATASETS = [
    # The no-argument batch creates the nine current extrapolation sets.
    # Existing directories are reported and skipped; interpolation sets remain
    # explicit because each family uses two disjoint parameter intervals.
    ("cylindrical_domain_with_airfoil_hole", "extrapolation", DEFAULT_DATASET_VERSION, 21, 21, 11),
    ("sidewall_deformed_block", "extrapolation", DEFAULT_DATASET_VERSION, 21, 21, 11),
    ("rounded_u_notch_extrusion", "extrapolation", DEFAULT_DATASET_VERSION, 41, 41, 11),
    ("tapered_elliptic_bore_block", "extrapolation", DEFAULT_DATASET_VERSION, 21, 21, 11),
    ("bent_annular_duct", "extrapolation", DEFAULT_DATASET_VERSION, 21, 21, 11),
    ("lobed_annular_duct", "extrapolation", DEFAULT_DATASET_VERSION, 21, 21, 11),
    ("square_to_round_transition", "extrapolation", DEFAULT_DATASET_VERSION, 21, 21, 11),
    ("variable_area_s_duct", "extrapolation", DEFAULT_DATASET_VERSION, 21, 21, 11),
    ("twisted_blade_passage", "extrapolation", DEFAULT_DATASET_VERSION, 21, 21, 11),
]

DEFAULT_GENERATION_OPTIONS = {
    "size": 500,
    "version": None,
    "nx": None,
    "ny": None,
    "nz": None,
    "winslow_max_iter": 1000,
    "winslow_tolerance": 1e-6,
    "timing_samples": 3,
    "workers": 4,
    "solver_backend": "cpu",
    "gpus": None,
    "linear_tolerance": 1e-12,
    "linear_max_iter": 4000,
}


def normalize_split(split):
    """Return interpolation or extrapolation."""
    normalized = SPLIT_ALIASES.get(str(split).lower())
    if normalized is None:
        raise ValueError(
            "split must be interpolation/int or extrapolation/ext; "
            f"received {split!r}"
        )
    return normalized


def resolve_family(value):
    """Accept either the registry key or a short output name such as swb_a."""
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


def build_dataset_name(family, split, version, size, nx, ny, nz):
    """Build boundary_int/ext_version_size_nx_ny_nz."""
    if family not in DATASET_FAMILIES:
        raise ValueError(f"unsupported dataset family: {family}")
    prefix = DATASET_FAMILIES[family]["output_name"]
    return build_canonical_dataset_name(prefix, split, version, size, nx, ny, nz)


def sample_parameter_ranges(ranges, count):
    """Sample each interval uniformly while retaining all interval endpoints."""
    if isinstance(count, bool) or int(count) != count or count < 2:
        raise ValueError("count must be an integer >= 2")
    intervals = np.asarray(ranges, dtype=float)
    if intervals.ndim != 2 or intervals.shape[1] != 2 or len(intervals) == 0:
        raise ValueError("ranges must contain (minimum, maximum) pairs")
    if not np.isfinite(intervals).all() or np.any(intervals[:, 0] >= intervals[:, 1]):
        raise ValueError("every parameter interval must be finite and increasing")
    if count < 2 * len(intervals):
        raise ValueError("count must retain both endpoints of every interval")

    lengths = intervals[:, 1] - intervals[:, 0]
    raw = lengths / lengths.sum() * count
    counts = np.maximum(np.floor(raw).astype(int), 2)
    while counts.sum() < count:
        counts[int(np.argmax(raw - counts))] += 1
    while counts.sum() > count:
        candidates = np.where(counts > 2)[0]
        if not len(candidates):
            raise ValueError("count is too small for the requested intervals")
        index = int(candidates[np.argmax(counts[candidates] - raw[candidates])])
        counts[index] -= 1

    values = [
        np.linspace(lower, upper, int(number), dtype=float)
        for (lower, upper), number in zip(intervals, counts)
    ]
    merged = np.concatenate(values)
    if len(np.unique(merged)) != count:
        raise ValueError("parameter intervals overlap and produce duplicate samples")
    return np.sort(merged)


def update_dataset_catalog(output_root=DEFAULT_OUTPUT_ROOT):
    """Rebuild CATALOG.md without replacing the family-level README guide."""
    root = Path(output_root)
    records = []
    if root.is_dir():
        for directory in root.iterdir():
            metadata_path = directory / "metadata.json"
            if not directory.is_dir() or not metadata_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if metadata.get("dimension") == 3:
                records.append((directory.name, metadata))
    records.sort(key=lambda item: item[0])
    text = [
        "# 三维数据集目录（自动生成）",
        "",
        "本文件由三维数据集生成器自动更新，只列出当前磁盘上实际存在的数据集。",
        "边界族、命名和复现说明见 [README.md](README.md)。",
        "",
        "## 当前数据集",
        "",
        "| 数据集 | 参数边界 | 变化参数 | 训练范围 | 数据量 | 网格规模 |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    if records:
        for name, metadata in records:
            ranges = metadata.get("parameter_ranges", {})
            text.append(
                "| `{}` | `{}` | `{}` | `{}` | {} | {} |".format(
                    name,
                    metadata.get("boundary_type", "unknown"),
                    metadata.get("parameter", "unknown"),
                    json.dumps(ranges, ensure_ascii=False),
                    metadata.get("size", "unknown"),
                    "×".join(map(str, metadata.get("grid_shape", []))),
                )
            )
    else:
        text.append("| — | — | — | — | — | — |")
    text += [
        "",
        "每个数据集保存六面边界采样、参数曲面描述、Winslow PDE 网格、参数值和生成元数据。",
        "改变参数范围、边界公式或边界定向时，应递增版本号；只改变样本数或网格规模时保持版本号。",
        "",
    ]
    destination = root / "CATALOG.md"
    temporary = root / "CATALOG.tmp"
    temporary.write_text("\n".join(text), encoding="utf-8")
    temporary.replace(destination)
    return destination


def _audit_volume_grid(mesh):
    finite = bool(np.isfinite(mesh).all())
    cell_count = int(np.prod(np.asarray(mesh.shape[:3]) - 1))
    if not finite:
        return {
            "cell_count": cell_count,
            "invalid_cells": None,
            "minimum_scaled_jacobian": None,
            "finite_coordinates": False,
            "all_cells_legal": False,
        }
    fields = evaluate_hexahedral_grid_quality(
        mesh, metrics=["minimum_scaled_jacobian", "flipped_cell", "degenerate_cell"]
    )
    flipped = fields["flipped_cell"] > 0
    degenerate = fields["degenerate_cell"] > 0
    invalid = int(np.count_nonzero(flipped | degenerate))
    minimum = float(fields["minimum_scaled_jacobian"].min())
    return {
        "cell_count": cell_count,
        "invalid_cells": invalid,
        "flipped_cells": int(np.count_nonzero(flipped)),
        "degenerate_cells": int(np.count_nonzero(degenerate)),
        "minimum_scaled_jacobian": minimum if np.isfinite(minimum) else None,
        "finite_coordinates": True,
        "all_cells_legal": invalid == 0,
    }


def _validated_boundary_samples(geometry, shape, sample_index):
    """Reject invalid prescribed face grids before starting the volume solve."""
    try:
        faces, boundary_grid = sample_faces(geometry, shape)
    except ValueError as error:
        raise ValueError(f"Sample {sample_index} has invalid boundary faces: {error}") from error
    bad_faces = []
    for name, face in faces.items():
        fields = surface_quad_quality_fields(
            face, metrics=("self_intersection", "flipping_cell", "degenerate_cell")
        )
        invalid = np.logical_or.reduce([values > 0 for values in fields.values()])
        if invalid.any():
            bad_faces.append(f"{int(invalid.sum())} invalid quadrilaterals on {name}")
    if bad_faces:
        raise ValueError(f"Sample {sample_index} has invalid prescribed boundary faces: "
                         + ", ".join(bad_faces))
    return boundary_grid.reshape(-1, 3)[boundary_indices(shape)]


def _solve_dataset_sample_3d(payload, solver_options=None):
    from src.numerics.traditional.winslow_torch import generation_solver_context
    with generation_solver_context(solver_options):
        return _solve_dataset_sample_3d_impl(payload, solver_options or {})


def _solve_dataset_sample_3d_impl(payload, solver_options):
    """Solve one picklable 3-D sample while preserving its input identity."""
    (
        sample_index,
        boundary_type,
        parameters,
        nx,
        ny,
        nz,
        winslow_max_iter,
        winslow_tolerance,
    ) = payload
    started = time.perf_counter()
    geometry = generate_boundary(boundary_type, **parameters)
    exact = _validated_boundary_samples(geometry, (nx, ny, nz), sample_index)
    initial = tfi_from_surfaces(geometry, xi=nx, eta=ny, zeta=nz)
    # An inverted TFI starter can be repaired by the PDE solve.
    try:
        mesh, info = solve_winslow_3d(
            geometry,
            nx,
            ny,
            nz,
            initial_grid=initial,
            max_iter=winslow_max_iter,
            tolerance=winslow_tolerance,
            return_info=True,
            **solver_options,
        )
    except FloatingPointError as error:
        # A numerical breakdown yields no PDE mesh. Keep the diagnostic
        # starter at this index and mark it explicitly as a fallback.
        mesh = initial
        info = {
            "converged": False,
            "solver_error": f"{type(error).__name__}: {error}",
            "mesh_source": "tfi_fallback",
        }
    info = dict(info)
    info["solver"] = "winslow_3d"
    mesh = mesh.reshape(nz, ny, nx, 3)
    return (
        geometry.to_dict(),
        exact,
        mesh,
        info,
        time.perf_counter() - started,
    )


def validate_existing_dataset(path):
    """Do not silently reuse a pre-PDE-cap dataset with the same public name."""
    path = Path(path)
    required = ('metadata.json', 'geometry.json', 'boundary_samples.npy', 'pde_mesh.npy')
    if not all((path / name).is_file() for name in required):
        raise FileExistsError(f'Incomplete existing dataset: {path}; use --force to regenerate it.')
    metadata = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('cap_parameterization') != CAP_SCHEME:
        raise ValueError(
            f'Existing dataset uses a different end-cap parameterization: {path}. '
            'Use a different version/output root, or explicitly --force to regenerate. '
            'Existing files were preserved.')


def generate_dataset(
    name,
    boundary_type,
    parameters,
    grid_shape=(21, 21, 11),
    output_root="data/raw/3D",
    winslow_max_iter=1000,
    winslow_tolerance=1e-6,
    timing_samples=3,
    workers=1,
    generation_metadata=None,
    overwrite=False,
    solver_backend="cpu",
    gpus=None,
    linear_tolerance=1e-12,
    linear_max_iter=4000,
):
    """Generate matching parameter surfaces, boundary samples, and PDE meshes."""
    if isinstance(workers, bool) or int(workers) != workers or int(workers) < 1:
        raise ValueError("workers must be an integer >= 1")
    workers = int(workers)
    nx, ny, nz = grid_shape
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("name must be a single nonempty directory name")
    root = Path(output_root).resolve()
    out = root / name
    if out.exists() and not overwrite:
        if boundary_type in DATASET_GENERATORS:
            validate_existing_dataset(out)
        print(f"[skip] Dataset already exists: {out}", flush=True)
        return out
    from src.data.generation.gpu_samples import validate_gpu_options, gpu_sample_map
    gpu_ids = validate_gpu_options(solver_backend, gpus)
    if not 0 < linear_tolerance < 1 or linear_max_iter < 1:
        raise ValueError('Require linear_tolerance in (0,1) and linear_max_iter >=1')
    parameters = list(parameters)
    if not parameters:
        raise ValueError("parameters must contain at least one sample")
    geoms = [None] * len(parameters)
    boundary_samples = [None] * len(parameters)
    meshes = [None] * len(parameters)
    diagnostics = [None] * len(parameters)
    if timing_samples < 0:
        raise ValueError("timing_samples must be non-negative")
    # CPU preflight preserves the existing endpoint check. GPU workers build
    # and validate their own faces before any volume solve (no CPU cap rebuild).
    for index in (endpoint_indices(len(parameters)) if solver_backend == 'cpu' else []):
        geometry = generate_boundary(boundary_type, **parameters[index])
        _validated_boundary_samples(geometry, (nx, ny, nz), index)
    payloads = [
        (
            sample_index,
            boundary_type,
            params,
            nx,
            ny,
            nz,
            winslow_max_iter,
            winslow_tolerance,
        )
        for sample_index, params in enumerate(parameters)
    ]
    progress_interval = max(1, len(parameters) // 10)
    def collect(results, indexed=False):
        for completed, item in enumerate(results, start=1):
            sample_index, result = item if indexed else (completed-1,item)
            geometry, exact, mesh, info, elapsed = result
            geoms[sample_index] = geometry
            boundary_samples[sample_index] = exact
            meshes[sample_index] = mesh
            diagnostics[sample_index] = info
            if sample_index < timing_samples:
                print(
                    f"Sample {sample_index + 1}/{len(parameters)}: {elapsed:.6f} s",
                    flush=True,
                )
            if completed % progress_interval == 0 or completed == len(parameters):
                print(
                    f"Progress: {completed}/{len(parameters)} "
                    f"({100.0 * completed / len(parameters):.0f}%)",
                    flush=True,
                )

    if solver_backend == 'cuda':
        from contextlib import closing
        with closing(gpu_sample_map(_solve_dataset_sample_3d,payloads,gpu_ids,
                                    linear_tolerance=linear_tolerance,linear_max_iter=linear_max_iter)) as results:
            collect(results, indexed=True)
    elif workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            collect(pool.map(_solve_dataset_sample_3d, payloads, chunksize=1))
    else:
        collect(map(_solve_dataset_sample_3d, payloads))
    extreme_legality_checks = [
        {
            "sample_id": index,
            "parameters": parameters[index],
            "mesh_source": diagnostics[index].get("mesh_source", "pde"),
            **_audit_volume_grid(meshes[index]),
        }
        for index in endpoint_indices(len(parameters))
    ]
    # Retain any prior output if boundary validation or the solve aborts.
    if out.exists():
        if out.resolve().parent != root or out.is_symlink():
            raise ValueError(f"Cannot overwrite dataset outside output root: {out}")
        if not overwrite:
            raise FileExistsError(f"Dataset appeared during generation: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    np.save(out / "boundary_samples.npy", np.asarray(boundary_samples))
    np.save(out / "pde_mesh.npy", np.asarray(meshes))
    np.save(
        out / "sample_parameters.npy",
        np.asarray([[p[k] for k in sorted(p)] for p in parameters]),
    )
    (out / "geometry.json").write_text(
        json.dumps(
            {
                "type": "dataset_geometry_3d",
                "dimension": 3,
                "samples": [
                    {"sample_id": i, "parameters": p, "boundary": g}
                    for i, (p, g) in enumerate(zip(parameters, geoms))
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    scope = "endpoint_pde_grids_hex_local_validity"
    audit = save_quality_audit(out, name, (nx, ny, nz), extreme_legality_checks, diagnostics, scope)
    metadata = {
        "dataset": name,
        "dimension": 3,
        "size": len(parameters),
        "grid_shape": [nx, ny, nz],
        "boundary_type": boundary_type,
        "pde_method": "winslow_iterative",
        "pde_solver": "winslow_3d",
        "cap_parameterization": CAP_SCHEME,
        "side_parameterization": SIDE_SCHEME,
        "geometry_revision": 3,
        "pde_max_iter": int(winslow_max_iter),
        "pde_tolerance": float(winslow_tolerance),
        "workers": workers,
        "solver_backend": solver_backend,
        "gpus": gpu_ids,
        "gpu_worker_count": len(gpu_ids),
        "linear_tolerance": linear_tolerance if solver_backend == 'cuda' else None,
        "linear_max_iter": linear_max_iter if solver_backend == 'cuda' else None,
        "quality_check_scope": scope,
        "illegal_samples": audit["illegal_samples"],
        "extreme_grid_legality_checks": extreme_legality_checks,
        "all_extreme_grids_legal": audit["illegal_samples"] == 0,
        "all_pde_converged": audit["nonconverged_samples"] == 0,
        "pde_diagnostics": diagnostics,
    }
    if generation_metadata:
        metadata.update(generation_metadata)
    (out / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    from src.data.generation.dataset_examples import save_dataset_examples_3d

    save_dataset_examples_3d(
        out,
        meshes=np.asarray(meshes),
        metadata=metadata,
        parameters=parameters,
    )
    update_dataset_catalog(out.parent)
    print(f"Dataset saved: {out}", flush=True)
    warn_saved_dataset(out, audit)
    return out



def generate_family_dataset(
    family,
    split,
    *,
    size=500,
    version=None,
    nx=None,
    ny=None,
    nz=None,
    output_root=DEFAULT_OUTPUT_ROOT,
    winslow_max_iter=1000,
    winslow_tolerance=1e-6,
    timing_samples=3,
    workers=1,
    overwrite=False,
    parameter_ranges=None,
    solver_backend="cpu",
    gpus=None,
    linear_tolerance=1e-12,
    linear_max_iter=4000,
):
    """Generate one registered dataset family and split."""
    if family not in DATASET_FAMILIES:
        raise ValueError(f"unsupported dataset family: {family}")
    split = normalize_split(split)
    specification = DATASET_FAMILIES[family]
    default_shape = specification["grid_shape"]
    raw_shape = (
        default_shape[0] if nx is None else nx,
        default_shape[1] if ny is None else ny,
        default_shape[2] if nz is None else nz,
    )
    if any(
        isinstance(value, bool) or int(value) != value or int(value) < 3
        for value in raw_shape
    ):
        raise ValueError("nx, ny, and nz must be integers >= 3")
    nx, ny, nz = (int(value) for value in raw_shape)

    requested_ranges = specification[split] if parameter_ranges is None else parameter_ranges
    values = sample_parameter_ranges(requested_ranges, size)
    parameter_name = specification["parameter"]
    parameters = [
        dict(specification["fixed_parameters"], **{parameter_name: float(value)})
        for value in values
    ]
    if version is None:
        version = next((item[2] for item in SELECTED_DATASETS if item[0] == family), DEFAULT_DATASET_VERSION)
    version = validate_version(version)
    name = build_dataset_name(family, split, version, size, nx, ny, nz)
    print(f"Dataset: {name}", flush=True)
    print(f"Description: {specification['description']}", flush=True)
    print(f"Parameter: {parameter_name}", flush=True)
    print(f"Training ranges: {requested_ranges}", flush=True)
    print(f"Size: {size}; grid: {nx}x{ny}x{nz}", flush=True)
    if solver_backend == 'cuda':
        print(f"PDE backend: CUDA float64; GPUs: {gpus if gpus is not None else [0]}; one sample per GPU", flush=True)
    else:
        print(f"PDE backend: CPU; workers: {workers}", flush=True)
    return generate_dataset(
        name,
        DATASET_GENERATORS[specification["function"]],
        parameters,
        grid_shape=(nx, ny, nz),
        output_root=output_root,
        winslow_max_iter=winslow_max_iter,
        winslow_tolerance=winslow_tolerance,
        timing_samples=timing_samples,
        workers=workers,
        generation_metadata={
            "dataset_family": family,
            "split": split,
            "version": version,
            "parameter": parameter_name,
            "parameter_ranges": requested_ranges,
            "parameter_ranges_source": (
                "family_default" if parameter_ranges is None else "command_override"
            ),
            "fixed_parameters": specification["fixed_parameters"],
            "evaluation_values": (
                specification["evaluation_values"][split]
                if parameter_ranges is None
                else [
                    float(np.asarray(requested_ranges, dtype=float)[:, 0].min()),
                    float(np.asarray(requested_ranges, dtype=float)[:, 1].max()),
                ]
            ),
        },
        overwrite=overwrite,
        solver_backend=solver_backend,
        gpus=gpus,
        linear_tolerance=linear_tolerance,
        linear_max_iter=linear_max_iter,
    )


def generate_cylindrical_domain_with_airfoil_hole_dataset(split, **kwargs):
    return generate_family_dataset("cylindrical_domain_with_airfoil_hole", split, **kwargs)


def generate_sidewall_deformed_block_dataset(split, **kwargs):
    return generate_family_dataset("sidewall_deformed_block", split, **kwargs)


def generate_rounded_u_notch_extrusion_dataset(split, **kwargs):
    return generate_family_dataset("rounded_u_notch_extrusion", split, **kwargs)


def generate_tapered_elliptic_bore_block_dataset(split, **kwargs):
    return generate_family_dataset("tapered_elliptic_bore_block", split, **kwargs)


def generate_bent_annular_duct_dataset(split, **kwargs):
    return generate_family_dataset("bent_annular_duct", split, **kwargs)


def generate_lobed_annular_duct_dataset(split, **kwargs):
    return generate_family_dataset("lobed_annular_duct", split, **kwargs)


def generate_square_to_round_transition_dataset(split, **kwargs):
    return generate_family_dataset("square_to_round_transition", split, **kwargs)


def generate_variable_area_s_duct_dataset(split, **kwargs):
    return generate_family_dataset("variable_area_s_duct", split, **kwargs)


def generate_twisted_blade_passage_dataset(split, **kwargs):
    return generate_family_dataset("twisted_blade_passage", split, **kwargs)


def generate_selected_datasets(selected=None, **kwargs):
    """Validate all selected jobs, then generate them sequentially."""
    selected = SELECTED_DATASETS if selected is None else list(selected)
    if not selected:
        raise ValueError("SELECTED_DATASETS is empty")
    override_version = kwargs.pop("version", None)
    jobs = []
    names = set()
    destination = Path(kwargs.get("output_root", DEFAULT_OUTPUT_ROOT))
    for item in selected:
        if len(item) == 2:
            family, split = item
            selected_version = DEFAULT_DATASET_VERSION
            selected_shape = None
        elif len(item) == 3:
            family, split, selected_version = item
            selected_shape = None
        elif len(item) == 6:
            family, split, selected_version, selected_nx, selected_ny, selected_nz = item
            selected_shape = (selected_nx, selected_ny, selected_nz)
        else:
            raise ValueError(
                "Each selected dataset must be (family, split) or "
                "(family, split, version), or "
                "(family, split, version, nx, ny, nz)"
            )
        if family not in DATASET_FAMILIES:
            raise ValueError(f"unsupported dataset family: {family}")
        split = normalize_split(split)
        version = selected_version if override_version is None else override_version
        version = validate_version(version)
        specification = DATASET_FAMILIES[family]
        if specification["function"] not in DATASET_GENERATORS:
            raise ValueError(f"unregistered parametric boundary for {family!r}")
        default_shape = (
            selected_shape if selected_shape is not None else specification["grid_shape"]
        )
        shape = tuple(
            default_shape[index] if kwargs.get(axis) is None else kwargs[axis]
            for index, axis in enumerate(("nx", "ny", "nz"))
        )
        name = build_dataset_name(
            family,
            split,
            version,
            kwargs.get("size", DEFAULT_GENERATION_OPTIONS["size"]),
            *shape,
        )
        if name in names:
            raise ValueError(f"duplicate selected dataset: {name}")
        output_path = destination / name
        if output_path.exists() and not kwargs.get("overwrite", False):
            validate_existing_dataset(output_path)
            print(f"[skip] Dataset already exists: {output_path}", flush=True)
            continue
        sample_parameter_ranges(
            specification[split], kwargs.get("size", DEFAULT_GENERATION_OPTIONS["size"])
        )
        names.add(name)
        jobs.append((family, split, version, *shape))

    outputs = []
    for family, split, version, nx, ny, nz in jobs:
        generation_kwargs = {
            key: value for key, value in kwargs.items() if key not in {"nx", "ny", "nz"}
        }
        outputs.append(
            generate_family_dataset(
                family, split, version=version, nx=nx, ny=ny, nz=nz, **generation_kwargs
            )
        )
    return outputs


def parse_grid_shape(text):
    """Parse NXxNYxNZ."""
    try:
        shape = tuple(int(value) for value in str(text).lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("grid shape must look like 11x11x11") from error
    if len(shape) != 3 or min(shape) < 3:
        raise argparse.ArgumentTypeError("grid shape requires three dimensions >= 3")
    return shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the datasets enabled in SELECTED_DATASETS."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--family")
    parser.add_argument("--split", choices=tuple(SPLIT_ALIASES))
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument('--solver-backend', choices=('cpu','cuda'), default=DEFAULT_GENERATION_OPTIONS['solver_backend'])
    parser.add_argument('--gpus', type=int, nargs='+', default=DEFAULT_GENERATION_OPTIONS['gpus'], help='CUDA-visible indices; requires --solver-backend cuda')
    parser.add_argument('--linear-tolerance', type=float, default=DEFAULT_GENERATION_OPTIONS['linear_tolerance'])
    parser.add_argument('--linear-max-iter', type=int, default=DEFAULT_GENERATION_OPTIONS['linear_max_iter'])
    parser.add_argument("--size", type=int, default=DEFAULT_GENERATION_OPTIONS["size"])
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--nz", type=int)
    parser.add_argument("--grid-shape", type=parse_grid_shape)
    parser.add_argument(
        "--parameter-range",
        "--range",
        dest="parameter_ranges",
        action="append",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help=(
            "Override the varying-parameter interval for one --family/--split. "
            "Repeat this option to define disjoint intervals."
        ),
    )
    parser.add_argument(
        "--winslow-max-iter",
        type=int,
        default=DEFAULT_GENERATION_OPTIONS["winslow_max_iter"],
    )
    parser.add_argument(
        "--winslow-tolerance",
        type=float,
        default=DEFAULT_GENERATION_OPTIONS["winslow_tolerance"],
    )
    parser.add_argument(
        "--timing-samples",
        type=int,
        default=DEFAULT_GENERATION_OPTIONS["timing_samples"],
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_GENERATION_OPTIONS["workers"],
        help="Parallel CPU workers for independent PDE samples; default 4.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing same-name 3-D dataset after regenerating it.",
    )
    arguments = parser.parse_args()
    if arguments.grid_shape is not None:
        if any(value is not None for value in (arguments.nx, arguments.ny, arguments.nz)):
            parser.error("use --grid-shape or --nx/--ny/--nz, not both")
        arguments.nx, arguments.ny, arguments.nz = arguments.grid_shape
    if (arguments.family is None) != (arguments.split is None):
        parser.error("--family and --split must be supplied together")
    if arguments.parameter_ranges is not None and arguments.family is None:
        parser.error("--parameter-range requires --family and --split")
    common = dict(
        size=arguments.size,
        version=arguments.version,
        nx=arguments.nx,
        ny=arguments.ny,
        nz=arguments.nz,
        output_root=arguments.output_root,
        winslow_max_iter=arguments.winslow_max_iter,
        winslow_tolerance=arguments.winslow_tolerance,
        timing_samples=arguments.timing_samples,
        workers=arguments.workers,
        overwrite=arguments.force,
        parameter_ranges=arguments.parameter_ranges,
        solver_backend=arguments.solver_backend,
        gpus=arguments.gpus,
        linear_tolerance=arguments.linear_tolerance,
        linear_max_iter=arguments.linear_max_iter,
    )
    if arguments.family is not None:
        family = resolve_family(arguments.family)
        revisions = {item[0]: item[2] for item in SELECTED_DATASETS}
        common["version"] = revisions.get(family, DEFAULT_DATASET_VERSION) if arguments.version is None else arguments.version
        generate_family_dataset(family, arguments.split, **common)
        raise SystemExit(0)
    generate_selected_datasets(
        **common,
    )
