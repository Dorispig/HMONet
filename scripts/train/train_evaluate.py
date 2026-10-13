"""Train selected datasets, then evaluate the exact checkpoints from this run."""

import argparse
import importlib
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def normalize_evaluate_methods(value):
    """Normalize an optional subset while preserving the evaluator's method order."""
    if value is None:
        return None
    from src.evaluation.configuration import selected_methods

    return [
        "pde" if name == "PDE_Winslow" else name.lower().replace("-", "")
        for name in selected_methods(value)
    ]


def extract_evaluation_model(evaluate_extra, evaluate_methods=None):
    """Consume a model selector while forwarding remaining evaluation overrides."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--model",
        choices=["hmonet", "meshonet", "pimeshonet", "pihmonet", "attentionpihmonet", "both", "pi", "all"],
    )
    selected, remaining = parser.parse_known_args(evaluate_extra)
    if selected.model is None:
        return normalize_evaluate_methods(evaluate_methods), remaining
    if evaluate_methods is not None:
        raise ValueError(
            "Use either --evaluate-methods or --evaluate-args '--model ...', not both"
        )
    models = (
        ["hmonet", "meshonet", "pimeshonet", "pihmonet", "attentionpihmonet"]
        if selected.model == "all"
        else (
            ["hmonet", "meshonet"]
            if selected.model == "both"
            else ["pimeshonet", "pihmonet", "attentionpihmonet"]
            if selected.model == "pi"
            else [selected.model]
        )
    )
    return normalize_evaluate_methods(["tfi", "pde"] + models), remaining


def evaluation_commands(
    summaries, config_path, cases, checkpoint, evaluate_methods=None
):
    """Build per-dataset evaluations and pin checkpoints trained in this run.

    Explicitly requested models that were not trained in this invocation are
    left without a checkpoint argument, so the evaluator resolves their latest
    compatible existing checkpoint.
    """
    selected = normalize_evaluate_methods(evaluate_methods)
    grouped = {}
    for summary in summaries:
        name = summary["dataset_name"]
        if name not in cases:
            raise ValueError(f"Missing evaluation case: {name}")
        grouped.setdefault(name, {})[summary["model"].lower()] = summary
    commands = []
    for name, models in grouped.items():
        methods = selected if selected is not None else ["tfi", "pde"] + list(models)
        command = [
            "--config",
            str(config_path),
            "--case",
            name,
            "--methods",
            ",".join(methods),
        ]
        for model, summary in models.items():
            if model not in methods:
                continue
            path = Path(summary[f"{checkpoint}_checkpoint"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            command += [f"--{model}-checkpoint", str(path)]
            if model in ("hmonet", "pihmonet", "attentionpihmonet"):
                command += [f"--{model}-mapping-mode", summary["hmonet_mapping_mode"]]
        commands.append(command)
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=[2, 3], default=2)
    parser.add_argument("--train-config")
    parser.add_argument("--evaluate-config")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate datasets and cases without training",
    )
    parser.add_argument(
        "--evaluate-methods",
        help="Comma-separated evaluation subset, e.g. tfi,pde,meshonet,hmonet; defaults to all models trained in this run plus TFI/PDE",
    )
    parser.add_argument("--checkpoint", choices=["best", "final"], default="best")
    parser.add_argument(
        "--train-args", default="", help="Quoted training CLI overrides"
    )
    parser.add_argument(
        "--evaluate-args",
        default="",
        help="Quoted evaluation CLI overrides, including --model; cases and checkpoints remain automatic",
    )
    args = parser.parse_args(argv)
    import shlex

    train_extra = shlex.split(args.train_args, posix=True)
    evaluate_extra = shlex.split(args.evaluate_args, posix=True)
    try:
        selected, evaluate_extra = extract_evaluation_model(
            evaluate_extra, args.evaluate_methods
        )
    except ValueError as error:
        parser.error(str(error))
    if any(token.split("=", 1)[0] == "--config" for token in train_extra):
        parser.error("Use --train-config instead of --config inside --train-args")
    reserved = {
        "--config",
        "--case",
        "--all-cases",
        "--dataset",
        "--methods",
        "--model",
        "--checkpoint",
        "--hmonet-checkpoint",
        "--meshonet-checkpoint",
        "--pimeshonet-checkpoint",
        "--pihmonet-checkpoint",
        "--attentionpihmonet-checkpoint",
        "--pihmonet-mapping-mode",
        "--attentionpihmonet-mapping-mode",
        "--hmonet-mapping-mode",
    }
    if any(token.split("=", 1)[0] in reserved for token in evaluate_extra):
        parser.error(
            "Evaluation cases, methods and checkpoints are supplied by this pipeline"
        )
    os.chdir(ROOT)
    train_config = Path(args.train_config or f"configs/{args.dimension}D/train.yaml")
    evaluate_config = Path(
        args.evaluate_config or f"configs/{args.dimension}D/evaluate.yaml"
    )
    config = yaml.safe_load(train_config.read_text(encoding="utf-8"))
    cases = yaml.safe_load(evaluate_config.read_text(encoding="utf-8"))["cases"]
    # Parse dataset/grid overrides before expensive training begins.
    preflight = argparse.ArgumentParser(add_help=False)
    preflight.add_argument(
        "--model",
        choices=["hmonet", "meshonet", "pimeshonet", "pihmonet", "attentionpihmonet", "both", "pi", "all"],
    )
    preflight.add_argument("--dataset")
    preflight.add_argument("--dataset-mode")
    for axis in ("nx", "ny", "nz"):
        preflight.add_argument(f"--{axis}", type=int)
    overrides, _ = preflight.parse_known_args(train_extra)
    choice = overrides.model or config.get("model", {}).get("name", "hmonet")
    trained = (
        {"hmonet", "meshonet", "pimeshonet", "pihmonet", "attentionpihmonet"}
        if choice == "all"
        else (
            {"hmonet", "meshonet"}
            if choice == "both"
            else {"pimeshonet", "pihmonet", "attentionpihmonet"}
            if choice == "pi"
            else {choice}
        )
    )
    # Explicit evaluation methods may include an already trained model. Models
    # trained in this invocation are pinned to their new checkpoints; other
    # requested models are resolved by the evaluator from existing checkpoints.
    from scripts.train.training_common import resolve_dataset_paths
    from src.data.dataset import ParametricDataset2D, ParametricDataset3D

    mode = overrides.dataset_mode or config.get("training", {}).get(
        "dataset", "interpolation"
    )
    paths = (
        [Path(overrides.dataset)]
        if overrides.dataset
        else resolve_dataset_paths(config, mode, ROOT)
    )
    if not paths:
        raise ValueError("No training datasets selected")
    dataset_class = ParametricDataset2D if args.dimension == 2 else ParametricDataset3D
    for path in paths:
        data = dataset_class(path, load_boundary=False)
        for axis, actual in zip(
            ("nx", "ny", "nz")[: args.dimension], data.grid_shape
        ):
            requested = getattr(overrides, axis)
            if requested is not None and int(requested) != int(actual):
                raise ValueError(
                    f"Dataset {path}: explicit --{axis}={requested} differs from "
                    f"saved grid {data.grid_shape}"
                )
        name = Path(path).parent.name if Path(path).is_file() else Path(path).name
        if name not in cases:
            raise ValueError(
                f"Add an evaluation case for {name} before running the pipeline"
            )
        print(
            f"Validated: {name}, {len(data)} samples, grid={data.grid_shape}",
            flush=True,
        )
        del data
    if args.check_only:
        return []
    train = importlib.import_module(f"scripts.train.train_{args.dimension}d")
    evaluate = importlib.import_module(f"scripts.evaluate.evaluate_{args.dimension}d")
    summaries = train.main(["--config", str(train_config)] + train_extra)
    for command in evaluation_commands(
        summaries, evaluate_config, cases, args.checkpoint, selected
    ):
        evaluate.main(command + evaluate_extra)
    return summaries


if __name__ == "__main__":
    main()
