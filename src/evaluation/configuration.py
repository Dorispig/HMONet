"""YAML cases and command-line overrides shared by 2-D and 3-D evaluation."""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from .method_order import ordered_methods
from src.utils.quality_selection import ALL_METRICS, QUAD_METRICS, quality_settings

ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    "tfi": "TFI",
    "pde": "PDE_Winslow",
    "pde_winslow": "PDE_Winslow",
    "meshonet": "MeshONet",
    "pimeshonet": "PI-MeshONet",
    "pi-meshonet": "PI-MeshONet",
    "hmonet": "HMONet",
    "pihmonet": "PI-HMONet",
    "pi-hmonet": "PI-HMONet",
    "attentionpihmonet": "Attention-PI-HMONet",
    "attention-pi-hmonet": "Attention-PI-HMONet",
    "attention_pihmonet": "Attention-PI-HMONet",
}
DEFAULTS = dict(
    methods=["tfi", "pde", "meshonet", "hmonet"],
    include_tfi=True,
    hmonet_mapping_mode="current",
    pihmonet_mapping_mode="current",
    attentionpihmonet_mapping_mode="current",
    pihmonet_boundary_projection="none",
    attentionpihmonet_boundary_projection="none",
    hmonet_boundary_projection="none",
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
    winslow_max_iter=1000,
    winslow_tolerance=1e-8,
    hmonet_pde_iterations=0,
    output_root="outputs/results",
    reuse_generated_results=True,
    reuse_analysis_results=True,
    save_plots=True,
    save_png=True,
    save_pdf=False,
    save_distribution_plots=False,
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
    return [converter(part.strip()) for part in value.split(",") if part.strip()]


def grid_shapes(value, dimension):
    result = [
        tuple(csv_values(item.lower().replace("x", ","), int))
        for item in value.split(",")
    ]
    if any(len(shape) != dimension or min(shape) < 2 for shape in result):
        raise ValueError(f"Expected {dimension}-D shapes with counts >= 2")
    return result


def selected_methods(values):
    if isinstance(values, str):
        values = values.split(",")
    result = []
    for value in values:
        key = str(value).strip().lower()
        if key not in METHODS:
            raise ValueError(
                f"Unknown method {value!r}; choose tfi,pde,meshonet,pimeshonet,hmonet,pihmonet,attentionpihmonet"
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
        "--values", help="Comma-separated physical geometry parameter values."
    )
    parser.add_argument(
        "--grid-shapes", help="For example 21x20,51x50 or 11x11x11,21x21x21."
    )
    parser.add_argument(
        "--methods", help="Comma-separated methods; only these methods are compared."
    )
    parser.add_argument(
        "--model",
        choices=["hmonet", "meshonet", "pimeshonet", "pihmonet", "attentionpihmonet", "both", "pi", "all"],
    )
    parser.set_defaults(dimension=dimension)
    parser.add_argument(
        "--quality-metrics",
        help="both: all metrics; none: no metrics; otherwise comma-separated metric names.",
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
    parser.add_argument("--meshonet-checkpoint", type=Path)
    parser.add_argument("--pimeshonet-checkpoint", type=Path)
    parser.add_argument("--pihmonet-checkpoint", type=Path)
    parser.add_argument("--attentionpihmonet-checkpoint", type=Path)
    parser.add_argument(
        "--pihmonet-mapping-mode",
        choices=["current", "coons", "coons_residual", "both"],
    )
    parser.add_argument(
        "--attentionpihmonet-mapping-mode",
        choices=["current", "coons", "coons_residual", "both"],
    )
    parser.add_argument("--curve-kernel-backend", choices=["auto", "torch"])
    parser.add_argument(
        "--hmonet-mapping-mode", choices=["current", "coons", "coons_residual", "both"]
    )
    for model in ("hmonet", "meshonet", "pimeshonet", "pihmonet", "attentionpihmonet"):
        parser.add_argument(
            f"--{model}-boundary-projection",
            choices=["none", "direct", "inverse_distance", "idw"],
        )
    for flag in ("output-root", "device"):
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
        "meshonet_checkpoint",
        "pimeshonet_checkpoint",
        "pihmonet_checkpoint",
        "attentionpihmonet_checkpoint",
        "checkpoint",
    ):
        value = getattr(args, key, None)
        if value is not None:
            result[key] = value
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
            if m not in ("HMONet", "MeshONet", "PI-MeshONet", "PI-HMONet", "Attention-PI-HMONet")
        ]
        result["methods"] += (
            ["MeshONet", "PI-MeshONet", "HMONet", "PI-HMONet", "Attention-PI-HMONet"]
            if args.model == "all"
            else (
                ["MeshONet", "HMONet"]
                if args.model == "both"
                else (
                    ["PI-MeshONet", "PI-HMONet", "Attention-PI-HMONet"]
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
            m in ("HMONet", "MeshONet", "PI-MeshONet", "PI-HMONet", "Attention-PI-HMONet")
            for m in result["methods"]
        )
        != 1
    ):
        raise ValueError(
            "--checkpoint requires one selected network; otherwise use --hmonet-checkpoint/--meshonet-checkpoint"
        )
    for key in (
        "hmonet_mapping_mode",
        "pihmonet_mapping_mode",
        "attentionpihmonet_mapping_mode",
    ):
        if result[key] == "coons":
            result[key] = "coons_residual"
        if result[key] not in ("current", "coons_residual", "both"):
            raise ValueError(f"Invalid {key}: {result[key]}")
    for key in (
        "hmonet_boundary_projection",
        "meshonet_boundary_projection",
        "pimeshonet_boundary_projection",
        "pihmonet_boundary_projection",
        "attentionpihmonet_boundary_projection",
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


def jobs(dimension, args, config):
    """Yield independently evaluated physical objects, without reading training meshes."""
    if args.dataset:
        from src.data.dataset import (
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
        parameters = read_sample_parameters(root, index)
        case = {
            "train_grid": meta["grid_shape"],
            "split": meta.get("split", "interpolation"),
        }
        shapes = (
            grid_shapes(args.grid_shapes, dimension)
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
        case = cases[name]
        if not case.get("enabled", True):
            if args.case:
                raise ValueError(
                    f"Case {name} is disabled: {case.get('disabled_reason', 'not enabled')}"
                )
            continue
        values = csv_values(args.values) if args.values else case.get("values", [None])
        shapes = (
            grid_shapes(args.grid_shapes, dimension)
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
