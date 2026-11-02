"""Run multiple or all 2-D ablation experiments with shared options."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import run_ablation_2d
from src.utils.gpu_queue import GpuTask, parse_gpu_ids, run_gpu_queue


ALL_ABLATIONS = run_ablation_2d.ABLATIONS


def resolve_ablations(
    values: list[str], available: tuple[str, ...] | list[str] | None = None
) -> list[str]:
    """Resolve experiments while allowing one explicit canonical baseline run."""
    experiments = list(available if available is not None else ALL_ABLATIONS)
    if not values:
        return experiments
    if len(set(values)) != len(values):
        raise ValueError("Ablation names must not contain duplicates")
    include_baseline = run_ablation_2d.BASELINE_ABLATION in values
    if "all" in values:
        allowed = {"all", run_ablation_2d.BASELINE_ABLATION}
        invalid = [value for value in values if value not in allowed]
        if invalid:
            raise ValueError(
                "Use 'all' alone, 'baseline all', or list individual experiments"
            )
        return (
            [run_ablation_2d.BASELINE_ABLATION] if include_baseline else []
        ) + experiments
    unknown = [
        value
        for value in values
        if value != run_ablation_2d.BASELINE_ABLATION and value not in experiments
    ]
    if unknown:
        raise ValueError(f"Unknown ablations: {', '.join(unknown)}")
    selected_experiments = [
        value for value in values if value != run_ablation_2d.BASELINE_ABLATION
    ]
    return (
        [run_ablation_2d.BASELINE_ABLATION] if include_baseline else []
    ) + selected_experiments


def _shared_declared_config(
    experiment_root: Path, ablations: list[str], key: str
) -> Path | None:
    """Resolve one shared config declared consistently by selected experiments."""
    declared = []
    for ablation in ablations:
        definition = experiment_root / ablation / "experiment.yaml"
        value = run_ablation_2d.declared_config_path(definition, key)
        if value is not None and value not in declared:
            declared.append(value)
    if len(declared) > 1:
        rendered = ", ".join(str(path) for path in declared)
        raise ValueError(
            f"Selected ablations declare different {key} files: {rendered}"
        )
    return declared[0] if declared else None


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=["all"],
        help=(
            "Experiments to run, 'all', or an explicit 'baseline' followed by "
            "one or more experiments/'all'. Available experiments: "
            + ", ".join(ALL_ABLATIONS)
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset name/path; repeat for multiple datasets. Config defaults apply when omitted.",
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--evaluation-grid-shapes",
        "--grid-shapes",
        dest="evaluation_grid_shapes",
        nargs="+",
        metavar="NXxNY",
    )
    parser.add_argument("--evaluation-values", nargs="+", type=float)
    parser.add_argument(
        "--evaluation-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate evaluation figures (default: enabled).",
    )
    parser.add_argument("--experiment-set", default=run_ablation_2d.DEFAULT_EXPERIMENT_SET)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--evaluate-config",
        type=Path,
        default=None,
    )
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--run-tag", default="default")
    parser.add_argument("--device")
    parser.add_argument(
        "--gpus",
        nargs="+",
        help="Dynamic GPU queue, for example --gpus 0 1 2 3 or --gpus 0,1,2,3.",
    )
    parser.add_argument(
        "--queue-log-root",
        type=Path,
        help="Directory for one log per dynamically scheduled task.",
    )
    parser.add_argument(
        "--queue-terminal-interval",
        type=int,
        default=500,
        help="Print one complete queued epoch summary every N epochs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def _forward_common_arguments(args, ablation: str) -> list[str]:
    baseline_config = getattr(
        args,
        "baseline_config",
        getattr(args, "train_config", ROOT / "experiments" / run_ablation_2d.DEFAULT_EXPERIMENT_SET / "baseline/experiment.yaml"),
    )
    command = [
        "--ablation",
        ablation,
        "--config",
        str(args.config),
        "--baseline-config",
        str(baseline_config),
        "--evaluate-config",
        str(args.evaluate_config),
        "--experiment-root",
        str(args.experiment_root),
        "--run-tag",
        args.run_tag,
    ]
    for dataset in args.dataset or []:
        command += ["--dataset", dataset]
    if args.seeds:
        command += ["--seeds", *map(str, args.seeds)]
    if getattr(args, "evaluation_grid_shapes", None):
        command += [
            "--evaluation-grid-shapes",
            *map(str, args.evaluation_grid_shapes),
        ]
    if getattr(args, "evaluation_values", None):
        command += ["--evaluation-values", *map(str, args.evaluation_values)]
    if not getattr(args, "evaluation_plots", True):
        command.append("--no-evaluation-plots")
    if args.device:
        command += ["--device", args.device]
    for flag in (
        "dry_run",
        "train_only",
        "evaluate_only",
        "force",
        "continue_on_error",
    ):
        if getattr(args, flag):
            command.append("--" + flag.replace("_", "-"))
    return command


def _write_batch_summary(args, summary):
    if args.dry_run:
        return
    output = args.experiment_root.resolve() / "batch_runs"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{args.run_tag}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _configured_variants(args, ablation: str):
    if ablation == run_ablation_2d.BASELINE_ABLATION:
        return [run_ablation_2d.Variant("baseline", "baseline", ())]
    definition_path = args.experiment_root / ablation / "experiment.yaml"
    definition = run_ablation_2d._load_yaml_mapping(definition_path)
    variants = run_ablation_2d.variants_from_definition(definition)
    if variants:
        return variants
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    return run_ablation_2d.VARIANT_BUILDERS[ablation](config)


def _run_dynamic_gpu_queue(args, selected):
    """Expand every experiment/dataset/seed/variant into one GPU task."""
    if args.device:
        raise ValueError("Use either --device for serial execution or --gpus for the queue")
    gpu_ids = parse_gpu_ids(args.gpus)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    datasets = args.dataset or config.get("datasets", [])
    seeds = args.seeds or config.get("seeds", [])
    if not datasets or not seeds:
        raise ValueError("The GPU queue requires at least one dataset and one seed")
    tasks = []
    for ablation in selected:
        variants = _configured_variants(args, ablation)
        for dataset in datasets:
            for variant in variants:
                for seed in seeds:
                    worker_args = copy.copy(args)
                    worker_args.dataset = [str(dataset)]
                    worker_args.seeds = [int(seed)]
                    worker_args.device = "cuda:0"
                    worker_args.gpus = None
                    command = [
                        sys.executable,
                        str(ROOT / "scripts/experiments/run_ablation_2d.py"),
                        *_forward_common_arguments(worker_args, ablation),
                        "--variants",
                        variant.name,
                        "--defer-finalization",
                    ]
                    tasks.append(
                        GpuTask(
                            name=(
                                f"{ablation}__{Path(str(dataset)).name}__"
                                f"{variant.name}__seed_{seed}"
                            ),
                            command=command,
                        )
                    )
    log_root = args.queue_log_root or (
        args.experiment_root / "queue_logs" / args.run_tag
    )
    records = run_gpu_queue(
        tasks,
        gpu_ids,
        cwd=ROOT,
        log_root=log_root,
        continue_on_error=args.continue_on_error,
        dry_run=args.dry_run,
        terminal_interval=args.queue_terminal_interval,
    )
    if args.dry_run:
        return {
            "run_tag": args.run_tag,
            "selected_ablations": selected,
            "status": "dry_run",
            "tasks": records,
        }

    failed = [record for record in records if record["status"] == "failed"]
    # A deferred worker deliberately avoids shared aggregation and baseline
    # validation. Re-enter each experiment once after the queue drains; every
    # completed training/evaluation is fingerprint-reused, then summaries are
    # written without concurrent file races.
    finalize_records = []
    if not failed:
        for ablation in selected:
            finalize_args = copy.copy(args)
            finalize_args.force = False
            finalize_args.dry_run = False
            finalize_args.device = "cuda:0"
            finalize_args.gpus = None
            command = _forward_common_arguments(finalize_args, ablation)
            print(f"\n=== Finalizing ablation: {ablation} ===", flush=True)
            completed = run_ablation_2d.main(command)
            finalize_records.append(
                {"ablation": ablation, "run_count": len(completed)}
            )
    summary = {
        "run_tag": args.run_tag,
        "selected_ablations": selected,
        "status": "completed_with_failures" if failed else "completed",
        "gpu_ids": gpu_ids,
        "tasks": records,
        "finalization": finalize_records,
    }
    _write_batch_summary(args, summary)
    return summary


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_ablation_2d.validate_experiment_set_name(args.experiment_set)
    suite_root = (ROOT / "experiments" / args.experiment_set).resolve()
    args.experiment_root = (args.experiment_root or suite_root).resolve()
    args.config = (args.config or suite_root / "configs/ablation_2d.yaml").resolve()
    if args.train_only and args.evaluate_only:
        raise ValueError("--train-only and --evaluate-only are mutually exclusive")
    available = tuple(
        path.name
        for path in sorted(args.experiment_root.iterdir())
        if (
            path.is_dir()
            and path.name != run_ablation_2d.BASELINE_ABLATION
            and (path / "experiment.yaml").is_file()
        )
    )
    selected = resolve_ablations(args.ablations, available)
    if (
        run_ablation_2d.BASELINE_ABLATION in selected
        and not (suite_root / "baseline/experiment.yaml").is_file()
    ):
        raise FileNotFoundError(
            f"Baseline configuration not found: {suite_root / 'baseline/experiment.yaml'}"
        )
    args.baseline_config = (
        args.baseline_config
        or _shared_declared_config(
            args.experiment_root, selected, "baseline_config"
        )
        or suite_root / "baseline/experiment.yaml"
    ).resolve()
    args.evaluate_config = (
        args.evaluate_config
        or _shared_declared_config(
            args.experiment_root, selected, "evaluation_config"
        )
        or suite_root / "baseline/evaluate.yaml"
    ).resolve()
    if args.gpus:
        return _run_dynamic_gpu_queue(args, selected)
    started = datetime.now()
    records = []
    failures = []
    baseline_compatibility_failures = []
    for ablation in selected:
        command = _forward_common_arguments(args, ablation)
        print(f"\n=== Running ablation: {ablation} ===", flush=True)
        begin = time.perf_counter()
        try:
            completed = run_ablation_2d.main(command)
            records.append(
                {
                    "ablation": ablation,
                    "status": "completed",
                    "run_count": len(completed),
                    "elapsed_seconds": time.perf_counter() - begin,
                    "arguments": command,
                }
            )
        except Exception as error:
            failure = {
                "ablation": ablation,
                "status": "failed",
                "error": repr(error),
                "elapsed_seconds": time.perf_counter() - begin,
                "arguments": command,
            }
            records.append(failure)
            failures.append(failure)
            if isinstance(error, run_ablation_2d.BaselineCompatibilityError):
                # The individual experiment has already completed every variant's
                # training and per-model evaluation. Defer this scientific-validity
                # error so the remaining selected ablations can also finish.
                baseline_compatibility_failures.append(failure)
                continue
            if not args.continue_on_error:
                summary = {
                    "run_tag": args.run_tag,
                    "selected_ablations": selected,
                    "started_at": started.isoformat(timespec="seconds"),
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "status": "failed",
                    "experiments": records,
                }
                _write_batch_summary(args, summary)
                raise
    summary = {
        "run_tag": args.run_tag,
        "selected_ablations": selected,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed_with_failures" if failures else "completed",
        "experiments": records,
    }
    _write_batch_summary(args, summary)
    if baseline_compatibility_failures:
        names = ", ".join(
            item["ablation"] for item in baseline_compatibility_failures
        )
        raise run_ablation_2d.BaselineCompatibilityError(
            "Baseline configuration mismatch affected these ablations: "
            f"{names}. All selected ablations finished their runnable training "
            "and per-model evaluation; baseline comparison and aggregate "
            "evaluation were skipped. See each experiment's "
            "baseline_compatibility_errors.json."
        )
    return summary


if __name__ == "__main__":
    main()
