"""YAML cases and command-line overrides shared by 2-D and 3-D evaluation."""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from .method_order import ordered_methods
from src.models.mapping_modes import (
    RATIONAL_AFFINE,
    mapping_mode_choices,
    normalize_mapping_mode,
)
from src.utils.quality_selection import ALL_METRICS, QUAD_METRICS, middle_section_names, quality_settings

ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    "tfi": "TFI",
    "pde": "PDE_Winslow",
    "pde_winslow": "PDE_Winslow",
    "meshonet": "MeshONet",
    "pimeshonet": "PI-MeshONet",
    "pi-meshonet": "PI-MeshONet",
    "hmonet": "HMONet",
    "qghmonet": "QG-HMONet",
    "qg-hmonet": "QG-HMONet",
    "resmetamesh": "ResMetaMesh",
}
def checkpoint_kind(method, selection=None):
    """Choose the saved model state unless the caller overrides the default."""
    if selection is not None:
        return selection
    return "final" if method.lower() == "resmetamesh" else "best"


DEFAULTS = dict(
    methods=["tfi", "pde", "meshonet", "pimeshonet", "hmonet", "qghmonet"],
    include_tfi=True,
    hmonet_mapping_mode=RATIONAL_AFFINE,
    hmonet_boundary_projection="none",
    qghmonet_boundary_projection="none",
    curve_kernel_backend="auto",
    inference_batch_size=1,
    meshonet_boundary_projection="none",
    pimeshonet_boundary_projection="none",
    inverse_distance_power=2.0,
    device="auto",
    query_batch_size=50000,
    inference_warmup_runs=3,
    inference_timing_repeats=10,
    compute_grid_quality=True,
    quality_metrics="both",
    quality_switches={},
    # Numeric quality summaries are always retained; expensive metric figures
    # are opt-in via --quality-plots.
    quality_plot_metrics=[],
    winslow_max_iter=1000,
    winslow_tolerance=1e-8,
    hmonet_pde_iterations=0,
    output_root="outputs/results",
    # ``case_root`` is an internal queue layout: output_root already names
    # one <dimension>/<dataset>/<model>/<seed>/<tag> scope.
    output_layout="standard",
    reuse_generated_results=True,
    reuse_analysis_results=True,
    save_plots=True,
    save_png=True,
    save_pdf=False,
    save_distribution_plots=False,
    # Retain the ordinary mesh overview and additionally draw the continuous
    # exact parameter boundary on a second comparison figure.  This is a plot-
    # only option: meshes, metrics and boundary-error tables are unchanged.
    exact_boundary_overlay=True,
    compact_outputs=False,
    skip_missing_checkpoints=True,
    force_regenerate=False,
    force_reanalyze=False,
    analysis_only=False,
    generation_only=False,
    require_pde_convergence=True,
    zoom_views=[],
)


def resolve_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def csv_values(value, converter=float):
    """Accept legacy comma values and the shared space-separated CLI form."""
    values = value if isinstance(value, (list, tuple)) else [value]
    return [
        converter(part.strip())
        for item in values
        for part in str(item).split(",")
        if part.strip()
    ]


def compact_grid_shapes(values, dimension):
    """Parse ``N`` or explicit ``Nx...xN`` shapes for either dimension.

    ``--grids 101 201 401`` is the common compact spelling: each scalar is
    expanded to a square 2-D or cubic 3-D output grid.  Explicit rectangular
    or anisotropic forms such as ``11x31x11`` remain available directly via
    ``--grids``; the legacy ``--grid-shapes`` spelling is also supported.
    """
    raw_values = values if isinstance(values, (list, tuple)) else [values]
    result = []
    for raw in raw_values:
        for item in str(raw).split(","):
            text = item.strip().lower()
            if not text:
                continue
            if "x" not in text:
                try:
                    size = int(text)
                except ValueError as error:
                    raise ValueError(f"Invalid grid size {item!r}") from error
                shape = (size,) * int(dimension)
            else:
                shape = tuple(csv_values(text.replace("x", ","), int))
            result.append(shape)
    if any(len(shape) != dimension or min(shape) < 2 for shape in result):
        raise ValueError(f"Expected {dimension}-D shapes with counts >= 2")
    return result


def grid_shapes(value, dimension):
    """Parse legacy comma-separated explicit grid shapes."""
    return compact_grid_shapes(value, dimension)


def selected_methods(values):
    if isinstance(values, str):
        values = values.split(",")
    result = []
    for value in values:
        key = str(value).strip().lower()
        if key not in METHODS:
            raise ValueError(
                f"Unknown method {value!r}; choose tfi,pde,meshonet,pimeshonet,hmonet,qghmonet,resmetamesh"
            )
        if METHODS[key] not in result:
            result.append(METHODS[key])
    if not result:
        raise ValueError("At least one evaluation method is required")
    return ordered_methods(result)


def build_parser(dimension):
    parser = argparse.ArgumentParser(
        description=f"Evaluate exact parametric {dimension}-D boundaries and selected methods."
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / f"configs/{dimension}D/evaluate.yaml"
    )
    parser.add_argument(
        "--train-config", type=Path, default=ROOT / f"configs/{dimension}D/train.yaml"
    )
    parser.add_argument(
        "--case", action="append", help="Repeat to select several YAML cases."
    )
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument(
        "--dataset", help="Alternatively evaluate one stored geometry sample."
    )
    parser.add_argument("--parameter-index", type=int, default=0)
    parser.add_argument(
        "--values",
        nargs="+",
        help="Physical geometry parameter values; use spaces or legacy commas.",
    )
    grid_group = parser.add_mutually_exclusive_group()
    grid_group.add_argument(
        "--grids",
        nargs="+",
        metavar="N_OR_SHAPE",
        help=(
            "Output sizes or explicit shapes, for example 101 201 401 or "
            "11x31x11 21x61x21; scalar N expands to NxN or NxNxN."
        ),
    )
    grid_group.add_argument(
        "--grid-shapes", help="Legacy explicit shapes, for example 21x20,51x50 or 11x11x11."
    )
    parser.add_argument(
        "--methods", help="Comma-separated methods; only these methods are compared."
    )
    parser.add_argument(
        "--model",
        choices=["hmonet", "qghmonet", "meshonet", "pimeshonet", "resmetamesh", "both", "pi", "all"],
    )
    parser.set_defaults(dimension=dimension)
    parser.add_argument(
        "--quality-metrics",
        help="both: all metrics; none: no metrics; otherwise comma-separated metric names.",
    )
    parser.add_argument(
        "--quality-plots",
        nargs="+",
        metavar="METRIC",
        help=(
            "Optional quality figures: 'all' for every enabled metric, or one or "
            "more metric names (for example scaled_jacobian max_angle). "
            "Default: no quality figures; mesh visualizations and summaries remain enabled."
        ),
    )
    for metric in (QUAD_METRICS if dimension == 2 else ALL_METRICS):
        parser.add_argument(
            "--quality-" + metric.replace("_", "-"),
            action=argparse.BooleanOptionalAction,
            default=None,
            help=f"Enable/disable {metric} within --quality-metrics (default: both).",
        )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Explicit checkpoint when exactly one network is selected.",
    )
    parser.add_argument("--hmonet-checkpoint", type=Path)
    parser.add_argument("--qghmonet-checkpoint", type=Path)
    parser.add_argument("--meshonet-checkpoint", type=Path)
    parser.add_argument("--pimeshonet-checkpoint", type=Path)
    parser.add_argument("--resmetamesh-checkpoint", type=Path)
    parser.add_argument("--curve-kernel-backend", choices=["auto", "torch"])
    parser.add_argument(
        "--hmonet-mapping-mode", choices=mapping_mode_choices(allow_both=True)
    )
    for model in ("hmonet", "qghmonet", "meshonet", "pimeshonet"):
        parser.add_argument(
            f"--{model}-boundary-projection",
            choices=["none", "direct", "inverse_distance", "idw"],
            help=(
                "Boundary postprocessing mode. 'none' is the default and "
                "uses the unmodified network output stored in the raw folder."
            ),
        )
    parser.add_argument("--output-root")
    parser.add_argument(
        "--output-layout",
        choices=["standard", "case_root"],
        help=argparse.SUPPRESS,
    )
    for flag in ("device",):
        parser.add_argument(f"--{flag}")
    for flag in (
        "query-batch-size",
        "inference-batch-size",
        "inference-warmup-runs",
        "inference-timing-repeats",
        "winslow-max-iter",
        "hmonet-pde-iterations",
    ):
        parser.add_argument(f"--{flag}", type=int)
    for flag in ("winslow-tolerance", "inverse-distance-power"):
        parser.add_argument(f"--{flag}", type=float)
    for flag in (
        "include-tfi",
        "compute-grid-quality",
        "save-plots",
        "save-png",
        "save-pdf",
        "save-distribution-plots",
        "exact-boundary-overlay",
        "compact-outputs",
        "skip-missing-checkpoints",
        "require-pde-convergence",
        "reuse-generated-results",
        "reuse-analysis-results",
    ):
        parser.add_argument(
            f"--{flag}", action=argparse.BooleanOptionalAction, default=None
        )
    parser.add_argument(
        "--no-plots", dest="save_plots", action="store_false", default=None
    )
    for flag in (
        "force",
        "force-regenerate",
        "force-reanalyze",
        "analysis-only",
        "generation-only",
    ):
        parser.add_argument(f"--{flag}", action="store_true", default=None)
    parser.add_argument(
        "--local", nargs=4, type=float, metavar=("X0", "X1", "Y0", "Y1")
    )
    parser.add_argument("--xlim", help="Physical MIN,MAX")
    parser.add_argument("--ylim", help="Physical MIN,MAX")
    parser.add_argument("--zlim", help="Physical MIN,MAX (3-D volume zoom).")
    parser.add_argument("--metric", help="Metric for a quality zoom.")
    parser.add_argument("--center-on", choices=["worst"])
    parser.add_argument("--cell-padding", type=int, default=2)
    parser.add_argument("--zoom-mode", choices=["slice", "volume"], default="slice")
    parser.add_argument("--slice", choices=["x_mid", "y_mid", "z_mid"], default="z_mid")
    if dimension == 3:
        parser.add_argument(
            "--volume-evaluation", action=argparse.BooleanOptionalAction, default=None,
            help="Compute volume quality and volume figures (default: enabled).",
        )
        parser.add_argument(
            "--middle-sections",
            nargs="+",
            metavar="SECTION",
            help="Evaluate x_mid, y_mid, z_mid (default), all, or none.",
        )
    return parser


def load_configuration(dimension, argv=None):
    args = build_parser(dimension).parse_args(argv)
    path = resolve_path(args.config)
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    return args, config


def evaluation_options(config, case, args):
    result = {
        **deepcopy(DEFAULTS),
        **deepcopy(config.get("evaluation", {})),
        **deepcopy(case.get("evaluation", {})),
    }
    for old, new in (
        ("reuse_generation_cache", "reuse_generated_results"),
        ("reuse_analysis_cache", "reuse_analysis_results"),
    ):
        if old in result:
            result[new] = result[old]
    for key in (
        *DEFAULTS,
        "model",
        "hmonet_checkpoint",
        "qghmonet_checkpoint",
        "meshonet_checkpoint",
        "pimeshonet_checkpoint",
        "resmetamesh_checkpoint",
        "checkpoint",
    ):
        value = getattr(args, key, None)
        if value is not None:
            result[key] = value
    # The CLI spelling intentionally uses ``--quality-plots`` while the
    # configuration spelling is ``quality_plot_metrics``.  This cannot be
    # handled by the generic loop above, because argparse stores the former as
    # ``args.quality_plots``.  Keep an omitted flag distinct from an explicitly
    # empty configuration value: the default remains no metric figures.
    if args.quality_plots is not None:
        result["quality_plot_metrics"] = list(args.quality_plots)
    if args.dimension == 3:
        volume = getattr(args, "volume_evaluation", None)
        if volume is not None:
            result["volume_evaluation"] = volume
        sections = getattr(args, "middle_sections", None)
        if sections is None:
            sections = result.get("middle_sections", "z_mid")
        result["middle_sections"] = list(middle_section_names(sections))
    result["quality_switches"] = {
        **config.get("evaluation", {}).get("quality_switches", {}),
        **case.get("evaluation", {}).get("quality_switches", {}),
    }
    for metric in ALL_METRICS:
        value = getattr(args, "quality_" + metric, None)
        if value is not None:
            result["quality_switches"][metric] = value
    quality_settings(result, args.dimension)
    result["methods"] = selected_methods(result["methods"])
    if args.model is not None and args.methods is None:
        result["methods"] = [
            m
            for m in result["methods"]
            if m not in ("HMONet", "QG-HMONet", "MeshONet", "PI-MeshONet", "ResMetaMesh")
        ]
        result["methods"] += (
            ["MeshONet", "PI-MeshONet", "HMONet", "QG-HMONet"]
            if args.model == "all"
            else (
                ["MeshONet", "HMONet"]
                if args.model == "both"
                else (
                    ["PI-MeshONet"]
                    if args.model == "pi"
                    else [METHODS[args.model]]
                )
            )
        )
    if not result["include_tfi"] and args.methods is None:
        result["methods"] = [m for m in result["methods"] if m != "TFI"]
    if not result["methods"]:
        raise ValueError("At least one evaluation method must remain selected")
    removed = {"hmonet_geometry_mode", "inference_cuda_graph"}.intersection(result)
    if removed:
        raise ValueError(
            "Evaluation setup options were removed: " + ", ".join(sorted(removed))
        )
    if int(result["inference_batch_size"]) < 1:
        raise ValueError("inference_batch_size must be positive")
    if result["curve_kernel_backend"] not in ("auto", "torch"):
        raise ValueError("curve_kernel_backend must be auto or torch")
    if (
        args.checkpoint
        and sum(
            m in ("HMONet", "QG-HMONet", "MeshONet", "PI-MeshONet", "ResMetaMesh")
            for m in result["methods"]
        )
        != 1
    ):
        raise ValueError(
            "--checkpoint requires one selected network; otherwise use --hmonet-checkpoint/--meshonet-checkpoint"
        )
    for key in ("hmonet_mapping_mode",):
        result[key] = normalize_mapping_mode(result[key], allow_both=True)
    for key in (
        "hmonet_boundary_projection",
        "qghmonet_boundary_projection",
        "meshonet_boundary_projection",
        "pimeshonet_boundary_projection",
    ):
        result[key] = {
            "idw": "inverse_distance",
            "inverse_distance_weight": "inverse_distance",
        }.get(result[key], result[key])
        if result[key] not in ("none", "direct", "inverse_distance"):
            raise ValueError(f"Invalid {key}: {result[key]}")
    if args.force:
        result.update(force_regenerate=True, force_reanalyze=True)
    for key in ("query_batch_size", "inference_timing_repeats", "winslow_max_iter"):
        if int(result[key]) < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("inference_warmup_runs", "hmonet_pde_iterations"):
        if int(result[key]) < 0:
            raise ValueError(f"{key} must be nonnegative")
    if result["inverse_distance_power"] <= 0 or result["winslow_tolerance"] <= 0:
        raise ValueError("IDW power and PDE tolerance must be positive")
    if result["analysis_only"] and (
        result["generation_only"] or result["force_regenerate"]
    ):
        raise ValueError(
            "analysis-only conflicts with generation-only/force-regenerate"
        )
    zoom = {}
    if args.local:
        zoom.update(xlim=args.local[:2], ylim=args.local[2:])
    for key in ("xlim", "ylim", "zlim"):
        if getattr(args, key):
            zoom[key] = csv_values(getattr(args, key))
    if args.center_on:
        zoom["center_on"] = args.center_on
    if zoom:
        zoom.update(
            metric=args.metric
            or (
                "minimum_scaled_jacobian"
                if args.zoom_mode == "volume"
                else "scaled_jacobian"
            ),
            cell_padding=args.cell_padding,
            mode=args.zoom_mode,
            slice=args.slice,
        )
        result["zoom_views"] = [*result.get("zoom_views", []), zoom]
    return result


def create_geometry(dimension, case, value):
    parameters = dict(case.get("fixed_parameters", {}))
    parameter = case.get("parameter")
    if parameter:
        parameters[parameter] = value
    family = case.get("boundary_type", case.get("boundary"))
    if dimension == 2:
        from src.data.geometry.boundary_generator_2d import generate_boundary
        from src.data.generation.make_dataset_parametric_2d import to_computational_boundary

        family = {
            "circle_airfoil_t": "circle_airfoil",
            "circle_airfoil_m": "circle_airfoil",
        }.get(family, family)
        geometry = to_computational_boundary(generate_boundary(family, **parameters))
    else:
        from src.data.geometry.boundary_generator_3d import generate_boundary

        geometry = generate_boundary(family, **parameters)
    return geometry, parameters


def checked_dataset_case(name, case, dimension):
    """Use saved generation parameters, and reject incompatible 3-D boundaries.

    No volume PDE mesh is loaded or solved here. A numerical run_e end-cap
    may be constructed. This catches geometry implementation drift before a
    learned mesh is compared with a different traditional domain.
    """
    root = ROOT / f"data/raw/{dimension}D" / name
    if dimension != 3 or not (root / "metadata.json").is_file():
        return case
    import json
    import numpy as np
    from src.data.mesh.sampling import sample_faces, boundary_indices
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    fixed = meta.get("fixed_parameters", {})
    supplied = case.get("fixed_parameters", {})
    if any(key in fixed and fixed[key] != value for key, value in supplied.items()):
        raise ValueError(f"Evaluation fixed_parameters differ from dataset metadata: {name}")
    case = {**case, "fixed_parameters": {**fixed, **supplied}}
    records = json.loads((root / "geometry.json").read_text(encoding="utf-8"))["samples"]
    points = np.load(root / "boundary_samples.npy", mmap_mode="r")
    indices = boundary_indices(meta["grid_shape"])
    for index in sorted({0, len(records) - 1}):
        value = records[index]["parameters"].get(case.get("parameter"))
        geometry, _ = create_geometry(3, case, value)
        _, sampled = sample_faces(geometry, tuple(meta["grid_shape"]))
        expected = sampled.reshape(-1, 3)[indices]
        error = float(np.max(np.abs(expected - points[index])))
        if error > 2e-6 * max(1.0, float(np.abs(expected).max())):
            raise ValueError(
                f"Evaluation boundary mismatch for {name}, sample {index}: max error={error:.8g}. "
                "Dataset and exact evaluation surfaces differ. Regenerate with the current "
                "geometry revision and retrain. The dataset-name version may remain 1; "
                "do not reuse old labels or checkpoints by renaming them."
            )
    return case


def jobs(dimension, args, config):
    """Yield independently evaluated physical objects, without reading training meshes."""
    if args.dataset:
        from src.data.parametric_dataset import (
            read_dataset_metadata,
            read_sample_parameters,
            load_paired_records,
        )
        from src.data.geometry.parametric_curve_2d import BoundaryCurveSet2D
        from src.data.geometry.parametric_geometry_3d import BoundarySurfaceSet3D
        from .artifacts import read_json

        root = resolve_path(args.dataset)
        if not root.exists():
            root = ROOT / f"data/raw/{dimension}D" / args.dataset
        if root.is_file():
            root = root.parent
        meta = read_dataset_metadata(root)
        index = args.parameter_index
        if not 0 <= index < meta["size"]:
            raise IndexError("parameter-index is outside the dataset")
        if dimension == 2:
            records, _ = load_paired_records(root / "boundary_geometry.npy", "geometry")
            geometry = BoundaryCurveSet2D.from_dict(records[index]["boundary"])
        else:
            geometry = BoundarySurfaceSet3D.from_dict(
                read_json(root / "geometry.json")["samples"][index]["boundary"]
            )
            import numpy as np
            from src.data.mesh.sampling import sample_faces, boundary_indices
            _, sampled = sample_faces(geometry, tuple(meta["grid_shape"]))
            expected = sampled.reshape(-1, 3)[boundary_indices(meta["grid_shape"])]
            stored = np.load(root / "boundary_samples.npy", mmap_mode="r")[index]
            if not np.allclose(expected, stored, rtol=2e-6, atol=2e-6):
                raise ValueError(
                    f"Dataset boundary mismatch: {root.name}, sample {index}. "
                    "Regenerate with the current geometry revision before evaluation."
                )
        parameters = read_sample_parameters(root, index)
        case = {
            "train_grid": meta["grid_shape"],
            "split": meta.get("split", "interpolation"),
        }
        shapes = (
            compact_grid_shapes(args.grids, dimension)
            if args.grids
            else grid_shapes(args.grid_shapes, dimension)
            if args.grid_shapes
            else [tuple(meta["grid_shape"])]
        )
        tag = (
            f"{next(iter(parameters.values())):g}"
            if len(parameters) == 1
            and isinstance(next(iter(parameters.values())), (int, float))
            else f"sample_{index:04d}"
        )
        for shape in shapes:
            yield root.name, case, tag, shape, geometry, parameters
        return
    cases = config.get("cases", {})
    names = (
        list(cases)
        if args.all_cases
        else args.case or config.get("evaluation", {}).get("default_cases", [])
    )
    if not names:
        raise ValueError("Select --case or --dataset, or set evaluation.default_cases")
    for name in names:
        case = checked_dataset_case(name, cases[name], dimension)
        if not case.get("enabled", True):
            if args.case:
                raise ValueError(
                    f"Case {name} is disabled: {case.get('disabled_reason', 'not enabled')}"
                )
            continue
        values = csv_values(args.values) if args.values else case.get("values", [None])
        shapes = (
            compact_grid_shapes(args.grids, dimension)
            if args.grids
            else grid_shapes(args.grid_shapes, dimension)
            if args.grid_shapes
            else case.get(
                "grid_shapes", [case.get("grid_shape", case.get("train_grid"))]
            )
        )
        for shape in shapes:
            shape = tuple(map(int, shape))
            if len(shape) != dimension or min(shape) < 2:
                raise ValueError(f"Invalid {dimension}-D shape in case {name}: {shape}")
            for value in values:
                geometry, parameters = create_geometry(dimension, case, value)
                tag = "default" if value is None else f"{value:g}"
                yield name, case, tag, shape, geometry, parameters
