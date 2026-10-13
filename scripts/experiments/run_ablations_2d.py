"""Run multiple or all 2-D ablation experiments with shared options."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import run_ablation_2d


ALL_ABLATIONS = run_ablation_2d.ABLATIONS


def resolve_ablations(values: list[str]) -> list[str]:
    """Expand ``all`` and reject unknown or duplicate experiment names."""
    if not values:
        return list(ALL_ABLATIONS)
    if "all" in values:
        if len(values) != 1:
            raise ValueError("Use 'all' alone or list individual ablations")
        return list(ALL_ABLATIONS)
    unknown = [value for value in values if value not in ALL_ABLATIONS]
    if unknown:
        raise ValueError(f"Unknown ablations: {', '.join(unknown)}")
    if len(set(values)) != len(values):
        raise ValueError("Ablation names must not contain duplicates")
    return values


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=["all"],
        help=(
            "Experiments to run, or 'all'. Available: "
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


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not __import__("re").fullmatch(r"ex[1-9][0-9]*", args.experiment_set):
        raise ValueError("--experiment-set must use the form ex1, ex2, ...")
    suite_root = (ROOT / "experiments" / args.experiment_set).resolve()
    args.experiment_root = (args.experiment_root or suite_root).resolve()
    args.config = (args.config or suite_root / "configs/ablation_2d.yaml").resolve()
    args.baseline_config = (args.baseline_config or suite_root / "baseline/experiment.yaml").resolve()
    args.evaluate_config = (args.evaluate_config or suite_root / "baseline/evaluate.yaml").resolve()
    if args.train_only and args.evaluate_only:
        raise ValueError("--train-only and --evaluate-only are mutually exclusive")
    available = tuple(
        path.name
        for path in sorted(args.experiment_root.iterdir())
        if path.is_dir() and (path / "experiment.yaml").is_file()
    )
    if args.ablations == ["all"]:
        selected = list(available)
    else:
        unknown = [name for name in args.ablations if name not in available]
        if unknown:
            raise ValueError(
                f"Ablations not defined in {args.experiment_set}: {', '.join(unknown)}"
            )
        selected = list(args.ablations)
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
