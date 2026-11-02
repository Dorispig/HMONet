"""Generate parameterized three-dimensional structured-grid datasets."""

from pathlib import Path
import sys
import json
import argparse
import time
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.geometry.boundary_generator_3d import generate_boundary
from src.numerics.traditional.tfi_3d import tfi_from_surfaces
from src.numerics.traditional.winslow_3d import solve_winslow_3d
from src.data.mesh.sampling import sample_boundary
from src.metrics import evaluate_hexahedral_grid_quality
from src.data.generation.dataset_naming import (
    build_dataset_name as build_canonical_dataset_name,
    default_version_for_split,
    validate_version,
)



ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "raw" / "3D"

DATASET_GENERATORS = {
    "cylindrical_domain_with_airfoil_hole": "cylindrical_domain_with_airfoil_hole",
    "stadium_cylinder_with_circular_hole": "stadium_cylinder_with_circular_hole",
    "annular_nozzle_diffuser": "annular_nozzle_diffuser",
    "twisted_elliptic_annulus": "twisted_elliptic_annulus",
    "bent_annular_duct": "bent_annular_duct",
    "lobed_annular_duct": "lobed_annular_duct",
}

# The first six entries are the paper datasets, in the same order as the
# boundary generator. Every family has interpolation and extrapolation sets.
DATASET_FAMILIES = {
    "cylindrical_domain_with_airfoil_hole": {
        "output_name": "cah_t", "function": "cylindrical_domain_with_airfoil_hole",
        "parameter": "thickness",
        "description": "圆柱外域与贯穿轴向的NACA4翼型孔；变化翼型相对厚度。",
        "fixed_parameters": {"outer_radius": 1.5, "chord": 1.0, "camber": 0.02, "camber_position": 0.4, "length": 2.5, "z_min": 0.0},
        "grid_shape": [11, 31, 11],
        "interpolation": [[0.06, 0.15], [0.25, 0.35]], "extrapolation": [[0.08, 0.34]],
        "evaluation_values": {"interpolation": [0.18, 0.20, 0.22], "extrapolation": [0.06, 0.38]},
    },
    "stadium_cylinder_with_circular_hole": {
        "output_name": "sch_r", "function": "stadium_cylinder_with_circular_hole",
        "parameter": "hole_radius",
        "description": "操场形柱状外域与同心圆柱孔；变化圆孔半径。",
        "fixed_parameters": {"straight_half_length": 0.8, "stadium_radius": 0.7, "length": 2.5, "z_min": 0.0},
        "grid_shape": [11, 31, 11],
        "interpolation": [[0.12, 0.24], [0.36, 0.48]], "extrapolation": [[0.16, 0.44]],
        "evaluation_values": {"interpolation": [0.27, 0.30, 0.33], "extrapolation": [0.10, 0.50]},
    },
    "annular_nozzle_diffuser": {
        "output_name": "and_ro", "function": "annular_nozzle_diffuser",
        "parameter": "outer_radius_out",
        "description": "同轴环形喷管/扩压器；变化出口外半径。",
        "fixed_parameters": {"inner_radius_in": 0.30, "inner_radius_out": 0.35, "outer_radius_in": 1.0, "length": 2.5, "z_min": 0.0},
        "grid_shape": [11, 31, 11],
        "interpolation": [[1.10, 1.25], [1.40, 1.55]], "extrapolation": [[1.15, 1.50]],
        "evaluation_values": {"interpolation": [1.30, 1.325, 1.35], "extrapolation": [1.05, 1.60]},
    },
    "twisted_elliptic_annulus": {
        "output_name": "tea_a", "function": "twisted_elliptic_annulus",
        "parameter": "twist_angle_deg",
        "description": "同心椭圆环形通道沿轴向共同扭转；变化累计扭转角。",
        "fixed_parameters": {"inner_a": 0.35, "inner_b": 0.22, "outer_a": 1.10, "outer_b": 0.75, "length": 2.5, "z_min": 0.0},
        "grid_shape": [11, 31, 11],
        "interpolation": [[10.0, 30.0], [50.0, 70.0]], "extrapolation": [[15.0, 65.0]],
        "evaluation_values": {"interpolation": [35.0, 40.0, 45.0], "extrapolation": [0.0, 80.0]},
    },
    "bent_annular_duct": {
        "output_name": "bad_a", "function": "bent_annular_duct",
        "parameter": "bend_angle_deg",
        "description": "沿圆弧中心线扫掠的环形弯管；变化弯曲角。",
        "fixed_parameters": {"inner_radius": 0.25, "outer_radius": 0.60, "bend_radius": 3.0, "z_min": 0.0},
        "grid_shape": [11, 31, 11],
        "interpolation": [[20.0, 45.0], [65.0, 90.0]], "extrapolation": [[25.0, 85.0]],
        "evaluation_values": {"interpolation": [50.0, 55.0, 60.0], "extrapolation": [10.0, 105.0]},
    },
    "lobed_annular_duct": {
        "output_name": "lad_a", "function": "lobed_annular_duct",
        "parameter": "amplitude",
        "description": "带轴向相位旋转的星形波瓣环形通道；变化波瓣幅值。",
        "fixed_parameters": {"inner_radius": 0.35, "outer_radius": 1.0, "lobes": 5, "phase_twist_deg": 45.0, "length": 2.5, "z_min": 0.0},
        "grid_shape": [11, 31, 11],
        "interpolation": [[0.02, 0.08], [0.12, 0.18]], "extrapolation": [[0.04, 0.16]],
        "evaluation_values": {"interpolation": [0.09, 0.10, 0.11], "extrapolation": [0.0, 0.20]},
    },
}

SPLIT_ALIASES = {
    "ext": "extrapolation",
    "extrapolation": "extrapolation",
    "int": "interpolation",
    "interpolation": "interpolation",
}

SELECTED_DATASETS = [
    ("cylindrical_domain_with_airfoil_hole", "interpolation", 0),
    ("cylindrical_domain_with_airfoil_hole", "extrapolation", 0),
    ("stadium_cylinder_with_circular_hole", "interpolation", 0),
    ("stadium_cylinder_with_circular_hole", "extrapolation", 0),
    ("annular_nozzle_diffuser", "interpolation", 0),
    ("annular_nozzle_diffuser", "extrapolation", 0),
    ("twisted_elliptic_annulus", "interpolation", 0),
    ("twisted_elliptic_annulus", "extrapolation", 0),
    ("bent_annular_duct", "interpolation", 0),
    ("bent_annular_duct", "extrapolation", 0),
    ("lobed_annular_duct", "interpolation", 0),
    ("lobed_annular_duct", "extrapolation", 0),
]

DEFAULT_GENERATION_OPTIONS = {
    "size": 1000,
    "version": None,
    "nx": None,
    "ny": None,
    "nz": None,
    "sample_count": 400,
    "seed": 20260905,
    "winslow_max_iter": 1000,
    "winslow_tolerance": 1e-6,
    "timing_samples": 3,
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
    """Accept either the registry key or short output name such as sch_r."""
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


def generate_dataset(
    name,
    boundary_type,
    parameters,
    grid_shape=(21, 31, 11),
    sample_count=400,
    output_root="data/raw/3D",
    seed=20260905,
    winslow_max_iter=1000,
    winslow_tolerance=1e-6,
    timing_samples=3,
    generation_metadata=None,
):
    """Generate matching parameter surfaces, boundary samples, and PDE meshes."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = grid_shape
    out = Path(output_root) / name
    out.mkdir(parents=True, exist_ok=True)
    geoms = []
    boundary = []
    physical = []
    meshes = []
    diagnostics = []
    parameters = list(parameters)
    if not parameters:
        raise ValueError("parameters must contain at least one sample")
    if timing_samples < 0:
        raise ValueError("timing_samples must be non-negative")
    progress_interval = max(1, len(parameters) // 10)
    for sample_index, params in enumerate(parameters):
        started = time.perf_counter()
        g = generate_boundary(boundary_type, **params)
        geoms.append(g.to_dict())
        faces = g.surfaces()
        full = np.zeros((nz, ny, nx, 3), float)
        full[:, :, 0, :] = faces["x_min"].sample(ny, nz).transpose(1, 0, 2)
        full[:, :, -1, :] = faces["x_max"].sample(ny, nz).transpose(1, 0, 2)
        full[:, 0, :, :] = faces["y_min"].sample(nx, nz).transpose(1, 0, 2)
        full[:, -1, :, :] = faces["y_max"].sample(nx, nz).transpose(1, 0, 2)
        full[0, :, :, :] = faces["z_min"].sample(nx, ny).transpose(1, 0, 2)
        full[-1, :, :, :] = faces["z_max"].sample(nx, ny).transpose(1, 0, 2)
        mask = np.zeros((nz, ny, nx), bool)
        mask[[0, -1], :, :] = True
        mask[:, [0, -1], :] = True
        mask[:, :, [0, -1]] = True
        boundary.append(full[mask])
        logical = rng.random((sample_count, 3))
        physical.append(tfi_from_surfaces(g, parameter_grid=logical))
        initial = tfi_from_surfaces(g, xi=nx, eta=ny, zeta=nz)
        initial_quality = evaluate_hexahedral_grid_quality(
            initial,
            metrics=["minimum_scaled_jacobian", "flipped_cell", "degenerate_cell"],
        )
        minimum_tfi_jacobian = float(initial_quality["minimum_scaled_jacobian"].min())
        invalid_tfi_cells = int(
            np.logical_or(
                initial_quality["flipped_cell"] > 0,
                initial_quality["degenerate_cell"] > 0,
            ).sum()
        )
        if invalid_tfi_cells or minimum_tfi_jacobian <= 0.0:
            raise RuntimeError(
                f"3-D TFI is invalid for sample {sample_index}: "
                f"minimum_scaled_jacobian={minimum_tfi_jacobian:.6e}, "
                f"invalid_cells={invalid_tfi_cells}"
            )
        exact = sample_boundary(g, (nx, ny, nz))
        result, info = solve_winslow_3d(
            g,
            nx,
            ny,
            nz,
            initial_grid=initial,
            max_iter=winslow_max_iter,
            tolerance=winslow_tolerance,
            return_info=True,
        )
        if not info["converged"]:
            raise RuntimeError(
                f"3-D PDE did not converge for sample {len(meshes)}: {info}"
            )
        boundary[-1] = exact
        meshes.append(result.reshape(nz, ny, nx, 3))
        info = dict(info)
        info["tfi_minimum_scaled_jacobian"] = minimum_tfi_jacobian
        info["tfi_invalid_cells"] = invalid_tfi_cells
        diagnostics.append(info)
        elapsed = time.perf_counter() - started
        if sample_index < timing_samples:
            print(
                f"Sample {sample_index + 1}/{len(parameters)}: {elapsed:.6f} s",
                flush=True,
            )
        completed = sample_index + 1
        if completed % progress_interval == 0 or completed == len(parameters):
            print(
                f"Progress: {completed}/{len(parameters)} "
                f"({100.0 * completed / len(parameters):.0f}%)",
                flush=True,
            )
    np.save(out / "boundary_samples.npy", np.asarray(boundary))
    np.save(out / "physical_samples.npy", np.asarray(physical))
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
    metadata = {
        "dataset": name,
        "dimension": 3,
        "size": len(parameters),
        "grid_shape": [nx, ny, nz],
        "boundary_type": boundary_type,
        "sample_count": sample_count,
        "pde_method": "winslow_iterative",
        "pde_max_iter": int(winslow_max_iter),
        "pde_tolerance": float(winslow_tolerance),
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
    return out



def generate_family_dataset(
    family,
    split,
    *,
    size=1000,
    version=0,
    nx=None,
    ny=None,
    nz=None,
    sample_count=400,
    output_root=DEFAULT_OUTPUT_ROOT,
    seed=20260905,
    winslow_max_iter=1000,
    winslow_tolerance=1e-6,
    timing_samples=3,
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

    values = sample_parameter_ranges(specification[split], size)
    parameter_name = specification["parameter"]
    parameters = [
        dict(specification["fixed_parameters"], **{parameter_name: float(value)})
        for value in values
    ]
    version = validate_version(version)
    name = build_dataset_name(family, split, version, size, nx, ny, nz)
    print(f"Dataset: {name}", flush=True)
    print(f"Description: {specification['description']}", flush=True)
    print(f"Parameter: {parameter_name}", flush=True)
    print(f"Training ranges: {specification[split]}", flush=True)
    print(f"Size: {size}; grid: {nx}x{ny}x{nz}", flush=True)
    return generate_dataset(
        name,
        DATASET_GENERATORS[specification["function"]],
        parameters,
        grid_shape=(nx, ny, nz),
        sample_count=sample_count,
        output_root=output_root,
        seed=seed,
        winslow_max_iter=winslow_max_iter,
        winslow_tolerance=winslow_tolerance,
        timing_samples=timing_samples,
        generation_metadata={
            "dataset_family": family,
            "split": split,
            "version": version,
            "parameter": parameter_name,
            "parameter_ranges": specification[split],
            "fixed_parameters": specification["fixed_parameters"],
            "evaluation_values": specification["evaluation_values"][split],
            "seed": int(seed),
        },
    )


def generate_cylindrical_domain_with_airfoil_hole_dataset(split, **kwargs):
    return generate_family_dataset("cylindrical_domain_with_airfoil_hole", split, **kwargs)


def generate_stadium_cylinder_with_circular_hole_dataset(split, **kwargs):
    return generate_family_dataset("stadium_cylinder_with_circular_hole", split, **kwargs)


def generate_annular_nozzle_diffuser_dataset(split, **kwargs):
    return generate_family_dataset("annular_nozzle_diffuser", split, **kwargs)


def generate_twisted_elliptic_annulus_dataset(split, **kwargs):
    return generate_family_dataset("twisted_elliptic_annulus", split, **kwargs)


def generate_bent_annular_duct_dataset(split, **kwargs):
    return generate_family_dataset("bent_annular_duct", split, **kwargs)


def generate_lobed_annular_duct_dataset(split, **kwargs):
    return generate_family_dataset("lobed_annular_duct", split, **kwargs)


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
            selected_version = default_version_for_split(normalize_split(split))
        elif len(item) == 3:
            family, split, selected_version = item
        else:
            raise ValueError(
                "Each selected dataset must be (family, split) or "
                "(family, split, version)"
            )
        if family not in DATASET_FAMILIES:
            raise ValueError(f"unsupported dataset family: {family}")
        split = normalize_split(split)
        version = selected_version if override_version is None else override_version
        version = validate_version(version)
        specification = DATASET_FAMILIES[family]
        if specification["function"] not in DATASET_GENERATORS:
            raise ValueError(f"unregistered parametric boundary for {family!r}")
        default_shape = specification["grid_shape"]
        shape = tuple(
            default_shape[index] if kwargs.get(axis) is None else kwargs[axis]
            for index, axis in enumerate(("nx", "ny", "nz"))
        )
        name = build_dataset_name(
            family, split, version, kwargs.get("size", 1000), *shape
        )
        if name in names:
            raise ValueError(f"duplicate selected dataset: {name}")
        output_path = destination / name
        if output_path.exists():
            raise FileExistsError(f"dataset already exists: {output_path}")
        sample_parameter_ranges(specification[split], kwargs.get("size", 1000))
        names.add(name)
        jobs.append((family, split, version))

    outputs = []
    for family, split, version in jobs:
        outputs.append(generate_family_dataset(family, split, version=version, **kwargs))
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
    parser.add_argument("--size", type=int, default=DEFAULT_GENERATION_OPTIONS["size"])
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--nz", type=int)
    parser.add_argument("--grid-shape", type=parse_grid_shape)
    parser.add_argument(
        "--sample-count", type=int, default=DEFAULT_GENERATION_OPTIONS["sample_count"]
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_GENERATION_OPTIONS["seed"])
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
    arguments = parser.parse_args()
    if arguments.grid_shape is not None:
        if any(value is not None for value in (arguments.nx, arguments.ny, arguments.nz)):
            parser.error("use --grid-shape or --nx/--ny/--nz, not both")
        arguments.nx, arguments.ny, arguments.nz = arguments.grid_shape
    if (arguments.family is None) != (arguments.split is None):
        parser.error("--family and --split must be supplied together")
    common = dict(
        size=arguments.size,
        version=arguments.version,
        nx=arguments.nx,
        ny=arguments.ny,
        nz=arguments.nz,
        sample_count=arguments.sample_count,
        output_root=arguments.output_root,
        seed=arguments.seed,
        winslow_max_iter=arguments.winslow_max_iter,
        winslow_tolerance=arguments.winslow_tolerance,
        timing_samples=arguments.timing_samples,
    )
    if arguments.family is not None:
        common["version"] = 0 if arguments.version is None else arguments.version
        generate_family_dataset(resolve_family(arguments.family), arguments.split, **common)
        raise SystemExit(0)
    generate_selected_datasets(
        **common,
    )
