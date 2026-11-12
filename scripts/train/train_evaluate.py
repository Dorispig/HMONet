"""Train selected datasets, then evaluate the exact checkpoints from this run."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.model_selection import parse_model_selection
from src.utils.gpu_queue import GpuTask, parse_gpu_ids, run_gpu_queue


# Keep formal queue-log paths within the same portable budget as ablations.
FORMAL_RUN_TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
FORMAL_MAX_RUN_TAG_LENGTH = 12
FORMAL_PATH_LIMIT = 234


def validate_formal_run_tag(value: str) -> str:
    """Validate the user-visible tag used in formal multi-GPU output paths."""
    value = str(value).strip()
    if not FORMAL_RUN_TAG_PATTERN.fullmatch(value):
        raise ValueError(
            "--run-tag must use letters, digits, '.', '_' or '-', and begin with a letter or digit"
        )
    if len(value) > FORMAL_MAX_RUN_TAG_LENGTH:
        raise ValueError(
            f"--run-tag must contain at most {FORMAL_MAX_RUN_TAG_LENGTH} characters "
            "to keep Windows and server output paths portable"
        )
    return value


def _formal_shape_names(value, fallback):
    """Parse optional comma-separated formal evaluation grids for dry runs."""
    if value is None:
        return ["x".join(map(str, shape)) for shape in fallback]
    names = []
    for item in str(value).split(","):
        pieces = item.strip().lower().split("x")
        if len(pieces) < 2 or not all(piece.isdigit() for piece in pieces):
            return ["x".join(map(str, shape)) for shape in fallback]
        names.append("x".join(pieces))
    return names or ["x".join(map(str, shape)) for shape in fallback]


def _formal_values(value, fallback):
    if value is None:
        return list(fallback)
    try:
        return [float(item.strip()) for item in str(value).split(",") if item.strip()]
    except ValueError:
        return list(fallback)


def report_formal_dry_run_paths(args, train_extra, evaluate_extra, paths, cases, models):
    """Preview formal output paths without writing, warning only on overflow."""
    evaluation_root = Path(
        _command_option(evaluate_extra, "--output-root", "outputs/results")
    )
    if not evaluation_root.is_absolute():
        evaluation_root = ROOT / evaluation_root
    if evaluation_root.name != f"{args.dimension}D":
        evaluation_root /= f"{args.dimension}D"
    train_root = Path(
        _command_option(train_extra, "--output-root", f"outputs/training/{args.dimension}D")
    )
    if not train_root.is_absolute():
        train_root = ROOT / train_root
    configured = yaml.safe_load(
        Path(args.train_config or f"configs/{args.dimension}D/train.yaml").read_text(
            encoding="utf-8"
        )
    ) or {}
    seed = int(_command_option(train_extra, "--seed", configured.get("training", {}).get("seed", 0)))
    mappings = configured.get("model", {}).get("hmonet", {}).get(
        "hmonet_mapping_mode", "corner_corrected_affine"
    )
    model_names = {
        "hmonet": "HMONet", "qghmonet": "QG-HMONet",
        "meshonet": "MeshONet", "pimeshonet": "PI-MeshONet",
    }
    candidates = []
    grid_override = _command_option(evaluate_extra, "--grid-shapes")
    value_override = _command_option(evaluate_extra, "--values")
    for path in paths:
        dataset_name = path.parent.name if path.is_file() else path.name
        case = cases[dataset_name]
        for grid in _formal_shape_names(grid_override, case.get("grid_shapes", [])):
            for value in _formal_values(value_override, case.get("values", [])):
                candidates.append((
                    "formal evaluation",
                    evaluation_root / dataset_name / grid / f"{value:g}"
                    / "p/c/t_p_pi_m_h/q/q_scaled_jacobian_distribution.png",
                ))
        for model in models:
            display = model_names.get(model, model)
            mapping = mappings if model in {"hmonet", "qghmonet"} else None
            training = train_root / dataset_name / display
            if mapping:
                training /= str(mapping)
            candidates.append((
                "formal training",
                training / "20261112_0000" / "tensorboard"
                / "events.out.tfevents.9999999999.HMONet-worker.999999.0",
            ))
        if args.gpus:
            candidates.append((
                "formal queue log",
                ROOT / "outputs/gpu_queue/formal" / dataset_name / "_".join(models)
                / f"seed_{seed}" / args.run_tag / "train_evaluate.log",
            ))
    violations = [
        (category, path, len(str(path.resolve())))
        for category, path in candidates
        if len(str(path.resolve())) > FORMAL_PATH_LIMIT
        or any(len(part) > 255 for part in path.parts)
    ]
    print(
        f"[dry-run paths] checked={len(candidates)} | Windows-safe={FORMAL_PATH_LIMIT} | "
        "Linux-total=4096 | Linux-component=255",
        flush=True,
    )
    if not violations:
        print("[dry-run paths] all selected formal output paths are within both limits.", flush=True)
    else:
        for category, path, length in violations:
            print(
                f"[dry-run paths] WARNING | {category} | length={length}, "
                f"limit={FORMAL_PATH_LIMIT} | {path}",
                flush=True,
            )
        print("[dry-run paths] warnings do not stop this dry run.", flush=True)
    return violations


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


def _formal_queue_fingerprint(
    *,
    dimension,
    dataset,
    models,
    train_config,
    evaluate_config,
    train_arguments,
    evaluate_arguments,
    evaluate_methods,
    checkpoint,
):
    """Create a stable identity for one formal train-and-evaluate queue task."""
    dataset = Path(dataset).resolve()

    def config_digest(value):
        path = Path(value)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        return hashlib.sha256(path.read_bytes()).hexdigest()

    if dataset.is_dir():
        data_files = sorted(dataset.glob("*.npy"))
    else:
        data_files = [dataset]
    dataset_signature = [
        {
            "name": item.name,
            "size": item.stat().st_size,
            "modified_ns": item.stat().st_mtime_ns,
        }
        for item in data_files
    ]
    payload = {
        "dimension": int(dimension),
        "dataset": str(dataset),
        "dataset_signature": dataset_signature,
        "models": list(models),
        "train_config_sha256": config_digest(train_config),
        "evaluate_config_sha256": config_digest(evaluate_config),
        "train_arguments": list(train_arguments),
        "evaluate_arguments": list(evaluate_arguments),
        "evaluate_methods": list(evaluate_methods or []),
        "checkpoint": str(checkpoint),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _command_option(tokens, name, default=None):
    """Read one ``--name value`` or ``--name=value`` option."""
    for index, token in enumerate(tokens):
        if token == name:
            if index + 1 >= len(tokens):
                raise ValueError(f"Missing value for {name}")
            return tokens[index + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return default


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
    """Queue atomic per-dataset train-and-evaluate tasks on selected GPUs."""
    import shlex

    gpu_ids = parse_gpu_ids(args.gpus)
    queue_root = Path(args.queue_log_root or ROOT / "outputs/gpu_queue/formal")
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
    evaluation_extra = _without_value_options(evaluate_extra, {"--device"})
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
            ",".join(models),
            "--dataset",
            str(path),
            "--dataset-mode",
            task_mode,
            "--device",
            "cuda:0",
        ]
        train_config = args.train_config or f"configs/{args.dimension}D/train.yaml"
        evaluation_config = (
            args.evaluate_config or f"configs/{args.dimension}D/evaluate.yaml"
        )
        fingerprint = _formal_queue_fingerprint(
            dimension=args.dimension,
            dataset=path,
            models=models,
            train_config=train_config,
            evaluate_config=evaluation_config,
            train_arguments=worker_extra,
            evaluate_arguments=[*evaluation_extra, "--device", "cuda:0"],
            evaluate_methods=selected_methods,
            checkpoint=args.checkpoint,
        )
        configured = yaml.safe_load(
            (ROOT / train_config).read_text(encoding="utf-8-sig")
            if not Path(train_config).is_absolute()
            else Path(train_config).read_text(encoding="utf-8-sig")
        ) or {}
        seed = int(
            _command_option(
                worker_extra,
                "--seed",
                configured.get("training", {}).get("seed", 0),
            )
        )
        experiment_name = "_".join(models)
        task_directory = (
            Path(dataset_name)
            / experiment_name
            / f"seed_{seed}"
            / args.run_tag
        )
        result_file = queue_root / task_directory / "queue_result.json"
        if not args.dry_run:
            result_file.parent.mkdir(parents=True, exist_ok=True)
        result_files.append(result_file)
        command = [
            sys.executable,
            str(script),
            "--dimension",
            str(args.dimension),
            "--train-config",
            str(train_config),
            "--evaluate-config",
            str(evaluation_config),
            "--checkpoint",
            args.checkpoint,
            "--train-args",
            shlex.join(worker_extra),
            "--evaluate-args",
            shlex.join([*evaluation_extra, "--device", "cuda:0"]),
            "--queue-worker",
            "--queue-result",
            str(result_file),
        ]
        if selected_methods is not None:
            command += ["--evaluate-methods", ",".join(selected_methods)]
        tasks.append(
            GpuTask(
                name=f"train_evaluate__{dataset_name}",
                command=command,
                log_relative_path=task_directory / "train_evaluate.log",
                clear_log_directory=True,
                metadata={
                    "dataset": dataset_name,
                    "experiment": experiment_name,
                    "seed": seed,
                    "run_tag": args.run_tag,
                    "task_fingerprint": fingerprint,
                },
            )
        )
    task_records = run_gpu_queue(
        tasks,
        gpu_ids,
        cwd=ROOT,
        log_root=queue_root,
        continue_on_error=args.continue_on_error,
        dry_run=args.dry_run,
        terminal_interval=args.queue_terminal_interval,
    )
    if args.dry_run:
        return []
    summaries = []
    for result_file in result_files:
        if result_file.is_file():
            summaries.extend(json.loads(result_file.read_text(encoding="utf-8")))
    failed = [record for record in task_records if record["status"] == "failed"]
    if failed and not summaries:
        return []
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
        "--run-tag",
        "--tag",
        dest="run_tag",
        default="default",
        help="Readable queue-log run tag (default: default).",
    )
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
    parser.add_argument(
        "--queue-terminal-interval",
        type=int,
        default=500,
        help="Print one complete queued epoch summary every N epochs.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--queue-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--queue-result", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.run_tag = validate_formal_run_tag(args.run_tag)
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
    if args.dry_run:
        report_formal_dry_run_paths(
            args, train_extra, evaluate_extra, paths, cases, sorted(trained)
        )
        return []
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
    for command in evaluation_commands(
        summaries, evaluate_config, cases, args.checkpoint, selected
    ):
        evaluate.main(command + evaluate_extra)
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


if __name__ == "__main__":
    main()
