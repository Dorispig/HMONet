"""Run the Q1 steady-heat downstream benchmark on evaluated 2-D meshes."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.artifacts import write_json
from src.evaluation.configuration import create_geometry
from src.evaluation.downstream import (
    prepare_reference,
    solve_downstream_case,
    summarize_rows,
)
from src.evaluation.mesh_provider import canonical_method
from src.utils.paths import resolve_project_path


def _grid(value):
    parts = str(value).lower().split("x")
    if len(parts) == 1:
        parts *= 2
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Use N or NxM for a 2-D grid")
    shape = tuple(map(int, parts))
    if min(shape) < 2:
        raise argparse.ArgumentTypeError("Grid counts must be at least 2")
    return shape


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/downstream/poisson_2d.yaml")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--values", nargs="+", type=float)
    parser.add_argument("--grids", nargs="+", type=_grid)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument(
        "--protocols", nargs="+", choices=["raw", "exact_boundary_control"]
    )
    parser.add_argument("--evaluation-root")
    parser.add_argument("--output-root")
    parser.add_argument("--solver", choices=["direct", "cg"])
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=None)
    return parser


def _load_yaml(path):
    path = resolve_project_path(path)
    return yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}, path


def main(argv=None):
    args = build_parser().parse_args(argv)
    config, config_path = _load_yaml(args.config)
    evaluation_config, evaluation_config_path = _load_yaml(
        config.get("evaluation_config", "configs/2D/evaluate.yaml")
    )
    names = args.dataset or config.get("datasets", [])
    if not names:
        raise ValueError("Select at least one dataset in YAML or with --dataset")
    evaluation_root = args.evaluation_root or config.get(
        "evaluation_root", "outputs/results"
    )
    output_root = resolve_project_path(
        args.output_root or config.get("output_root", "outputs/downstream/poisson_2d")
    )
    methods = [
        canonical_method(value)
        for value in (args.methods or config.get("methods", []))
    ]
    protocols = args.protocols or config.get("protocols", ["raw"])
    solver_options = deepcopy(config.get("solver", {}))
    if args.solver:
        solver_options["solver"] = args.solver
    save_plots = config.get("save_plots", True) if args.plots is None else args.plots
    reference_shape = tuple(map(int, config.get("reference", {}).get("grid_shape", [401, 401])))

    all_rows = []
    manifest = {
        "config": str(config_path),
        "configuration": config,
        "evaluation_config": str(evaluation_config_path),
        "evaluation_root": str(resolve_project_path(evaluation_root)),
        "datasets": names,
        "methods": methods,
        "protocols": protocols,
        "reference_shape": reference_shape,
    }
    for name in names:
        if name not in evaluation_config.get("cases", {}):
            raise KeyError(
                f"Dataset/case {name!r} is absent from {evaluation_config_path}"
            )
        case = evaluation_config["cases"][name]
        override = config.get("case_overrides", {}).get(name, {})
        values = args.values or override.get("values") or case.get("values", [])
        shapes = args.grids or [tuple(map(int, value)) for value in override.get("grid_shapes", case.get("grid_shapes", []))]
        if not values or not shapes:
            raise ValueError(f"No values/grid_shapes configured for {name}")
        problem = {**deepcopy(config["problem"]), **deepcopy(override.get("problem", {}))}
        zoom_bounds = override.get("zoom_bounds")
        for value in values:
            geometry, _ = create_geometry(2, case, value)
            reference = prepare_reference(
                dataset=name,
                value=value,
                geometry=geometry,
                reference_shape=reference_shape,
                evaluation_root=evaluation_root,
                output_root=output_root,
                problem=problem,
                solver_options=solver_options,
            )
            for shape in shapes:
                print(
                    f"downstream dataset={name} value={value:g} grid={'x'.join(map(str, shape))}",
                    flush=True,
                )
                all_rows.extend(
                    solve_downstream_case(
                        dataset=name,
                        value=value,
                        shape=shape,
                        geometry=geometry,
                        methods=methods,
                        protocols=protocols,
                        evaluation_root=evaluation_root,
                        output_root=output_root,
                        problem=problem,
                        solver_options=solver_options,
                        reference_shape=reference_shape,
                        reference_solution=reference,
                        save_plots=save_plots,
                        zoom_bounds=zoom_bounds,
                    )
                )
    summarize_rows(all_rows, output_root / "aggregate")
    write_json(output_root / "aggregate" / "run_manifest.json", manifest)
    print(f"downstream_results={output_root}", flush=True)
    return all_rows


if __name__ == "__main__":
    main()
