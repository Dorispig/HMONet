"""Train selected datasets, then evaluate the exact checkpoints from this run."""

import argparse
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.model_selection import parse_model_selection
from src.utils.gpu_queue import GpuTask, parse_gpu_ids, run_gpu_queue


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
        help="Model alias or comma-separated model list",
    )
    selected, remaining = parser.parse_known_args(evaluate_extra)
    if selected.model is None:
        return normalize_evaluate_methods(evaluate_methods), remaining
    if evaluate_methods is not None:
        raise ValueError(
            "Use either --evaluate-methods or --evaluate-args '--model ...', not both"
        )
    models = parse_model_selection(selected.model)
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


def _without_value_options(tokens, names):
    """Remove selected one-value options from a forwarded CLI token list."""
    names = set(names)
    result = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        key = token.split("=", 1)[0]
        if key in names:
            if "=" not in token:
                if index + 1 >= len(tokens):
                    raise ValueError(f"Missing value for {token}")
                index += 2
            else:
                index += 1
            continue
        result.append(token)
        index += 1
    return result


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _queued_formal_run(
    args,
    train_extra,
    evaluate_extra,
    models,
    paths,
    evaluate_config,
    cases,
    selected_methods,
    resolved_dataset_mode,
):
    """Dynamically train model/dataset pairs, then evaluate each dataset once."""
    import shlex

    gpu_ids = parse_gpu_ids(args.gpus)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    queue_root = Path(args.queue_log_root or ROOT / "outputs/gpu_queue/formal" / run_id)
    result_root = queue_root / "results"
    if not args.dry_run:
        result_root.mkdir(parents=True, exist_ok=True)
    stripped = _without_value_options(
        train_extra, {"--model", "--dataset", "--dataset-mode", "--device"}
    )
    requested_mode = next(
        (
            train_extra[index + 1]
            for index, token in enumerate(train_extra[:-1])
            if token == "--dataset-mode"
        ),
        resolved_dataset_mode,
    )
    tasks = []
    result_files = []
    script = ROOT / "scripts/train/train_evaluate.py"
    for model in models:
        for path in paths:
            dataset_name = path.parent.name if path.is_file() else path.name
            task_mode = requested_mode
            if task_mode in (None, "both"):
                task_mode = (
                    "interpolation"
                    if "_int_" in dataset_name
                    else "extrapolation" if "_ext_" in dataset_name else None
                )
            if task_mode not in ("interpolation", "extrapolation"):
                raise ValueError(
                    f"Cannot infer interpolation/extrapolation mode for {dataset_name!r}; "
                    "pass --train-args '--dataset-mode interpolation' (or extrapolation)."
                )
            worker_extra = [
                *stripped,
                "--model",
                model,
                "--dataset",
                str(path),
                "--dataset-mode",
                task_mode,
                "--device",
                "cuda:0",
            ]
            result_file = result_root / f"{model}__{dataset_name}.json"
            result_files.append(result_file)
            command = [
                sys.executable,
                str(script),
                "--dimension",
                str(args.dimension),
                "--train-config",
                str(args.train_config or f"configs/{args.dimension}D/train.yaml"),
                "--evaluate-config",
                str(args.evaluate_config or f"configs/{args.dimension}D/evaluate.yaml"),
                "--checkpoint",
                args.checkpoint,
                "--train-args",
                shlex.join(worker_extra),
                "--queue-worker",
                "--queue-result",
                str(result_file),
            ]
            tasks.append(
                GpuTask(
                    name=f"train__{model}__{dataset_name}",
                    command=command,
                )
            )
    training_records = run_gpu_queue(
        tasks,
        gpu_ids,
        cwd=ROOT,
        log_root=queue_root / "training",
        continue_on_error=args.continue_on_error,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return []
    summaries = []
    for result_file in result_files:
        if result_file.is_file():
            summaries.extend(json.loads(result_file.read_text(encoding="utf-8")))
    failed = [record for record in training_records if record["status"] == "failed"]
    if failed and not summaries:
        return []

    evaluation_extra = _without_value_options(evaluate_extra, {"--device"})
    evaluation_tasks = []
    evaluate_script = ROOT / f"scripts/evaluate/evaluate_{args.dimension}d.py"
    for command in evaluation_commands(
        summaries,
        evaluate_config,
        cases,
        args.checkpoint,
        selected_methods,
    ):
        case = command[command.index("--case") + 1]
        evaluation_tasks.append(
            GpuTask(
                name=f"evaluate__{case}",
                command=[
                    sys.executable,
                    str(evaluate_script),
                    *command,
                    *evaluation_extra,
                    "--device",
                    "cuda:0",
                ],
            )
        )
    run_gpu_queue(
        evaluation_tasks,
        gpu_ids,
        cwd=ROOT,
        log_root=queue_root / "evaluation",
        continue_on_error=args.continue_on_error,
        dry_run=False,
    )
    return summaries


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
    parser.add_argument(
        "--gpus",
        nargs="+",
        help="Dynamic GPU queue, e.g. --gpus 0 1 2 3 or --gpus 0,1,2,3.",
    )
    parser.add_argument("--queue-log-root")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--queue-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--queue-result", help=argparse.SUPPRESS)
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
        help="Model alias or comma-separated model list",
    )
    preflight.add_argument("--dataset")
    preflight.add_argument("--dataset-mode")
    for axis in ("nx", "ny", "nz"):
        preflight.add_argument(f"--{axis}", type=int)
    overrides, _ = preflight.parse_known_args(train_extra)
    choice = overrides.model or config.get("model", {}).get("name", "hmonet")
    trained = set(parse_model_selection(choice))
    # Explicit evaluation methods may include an already trained model. Models
    # trained in this invocation are pinned to their new checkpoints; other
    # requested models are resolved by the evaluator from existing checkpoints.
    from src.engine.training import resolve_dataset_paths
    from src.data.parametric_dataset import ParametricDataset2D, ParametricDataset3D

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
    if args.gpus and args.queue_worker:
        parser.error("--gpus cannot be combined with the internal queue worker")
    if args.gpus:
        return _queued_formal_run(
            args,
            train_extra,
            evaluate_extra,
            sorted(trained),
            paths,
            evaluate_config,
            cases,
            selected,
            mode,
        )
    train = importlib.import_module(f"scripts.train.train_{args.dimension}d")
    evaluate = importlib.import_module(f"scripts.evaluate.evaluate_{args.dimension}d")
    summaries = train.main(["--config", str(train_config)] + train_extra)
    if args.queue_worker:
        if not args.queue_result:
            parser.error("Internal queue workers require --queue-result")
        result_path = Path(args.queue_result)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return summaries
    for command in evaluation_commands(
        summaries, evaluate_config, cases, args.checkpoint, selected
    ):
        evaluate.main(command + evaluate_extra)
    return summaries


if __name__ == "__main__":
    main()
