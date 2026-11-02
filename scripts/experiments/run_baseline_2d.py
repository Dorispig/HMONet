"""Optionally train the unchanged 2-D HMONet reference configuration.

Example:
  python scripts/experiments/run_baseline_2d.py --dataset air_t_int_0_1000_21_20 --dataset air_t_ext_0_1000_21_20 --seeds 20260802 20260917 20261031

Training summaries, checkpoints, and baseline evaluation results are written
below experiments/<experiment-set>/baseline/.
Ablation scripts do not require this command first. All parameter variants search
the complete experiments tree by training and evaluation fingerprints.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import run_ablation_2d, run_ablations_2d


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", help="Dataset name/path; repeat for multiple datasets.")
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
        help="Dynamic GPU queue, for example --gpus 0 1 2 3.",
    )
    parser.add_argument("--queue-log-root", type=Path)
    parser.add_argument(
        "--queue-terminal-interval",
        type=int,
        default=500,
        help="Print one complete queued epoch summary every N epochs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Also evaluate the baseline after training.")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Reuse an existing baseline checkpoint and run evaluation only.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    run_ablation_2d.validate_experiment_set_name(args.experiment_set)
    suite_root = (ROOT / "experiments" / args.experiment_set).resolve()
    args.experiment_root = (args.experiment_root or suite_root).resolve()
    args.config = (args.config or suite_root / "configs/ablation_2d.yaml").resolve()
    args.baseline_config = (args.baseline_config or suite_root / "baseline/experiment.yaml").resolve()
    args.evaluate_config = (args.evaluate_config or suite_root / "baseline/evaluate.yaml").resolve()
    command = [
        "--ablation", "baseline",
        "--config", str(args.config),
        "--baseline-config", str(args.baseline_config),
        "--evaluate-config", str(args.evaluate_config),
        "--experiment-root", str(args.experiment_root),
        "--run-tag", args.run_tag,
    ]
    if args.evaluate_only:
        command.append("--evaluate-only")
    elif not args.evaluate:
        command.append("--train-only")
    for dataset in args.dataset or []:
        command += ["--dataset", dataset]
    if args.seeds:
        command += ["--seeds", *map(str, args.seeds)]
    if args.evaluation_grid_shapes:
        command += [
            "--evaluation-grid-shapes",
            *map(str, args.evaluation_grid_shapes),
        ]
    if args.evaluation_values:
        command += ["--evaluation-values", *map(str, args.evaluation_values)]
    if not args.evaluation_plots:
        command.append("--no-evaluation-plots")
    if args.device:
        command += ["--device", args.device]
    if args.dry_run:
        command.append("--dry-run")
    if args.force:
        command.append("--force")
    if args.continue_on_error:
        command.append("--continue-on-error")
    if args.gpus:
        if args.device:
            parser.error("Use either --device or --gpus, not both")
        batch_command = [
            "--experiment-set", args.experiment_set,
            "--ablations", "baseline",
            "--config", str(args.config),
            "--baseline-config", str(args.baseline_config),
            "--evaluate-config", str(args.evaluate_config),
            "--experiment-root", str(args.experiment_root),
            "--run-tag", args.run_tag,
            "--gpus", *args.gpus,
            "--queue-terminal-interval", str(args.queue_terminal_interval),
        ]
        for dataset in args.dataset or []:
            batch_command += ["--dataset", dataset]
        if args.seeds:
            batch_command += ["--seeds", *map(str, args.seeds)]
        if args.evaluation_grid_shapes:
            batch_command += [
                "--evaluation-grid-shapes",
                *map(str, args.evaluation_grid_shapes),
            ]
        if args.evaluation_values:
            batch_command += [
                "--evaluation-values", *map(str, args.evaluation_values)
            ]
        if not args.evaluation_plots:
            batch_command.append("--no-evaluation-plots")
        if args.queue_log_root:
            batch_command += ["--queue-log-root", str(args.queue_log_root)]
        if args.dry_run:
            batch_command.append("--dry-run")
        if args.force:
            batch_command.append("--force")
        if args.continue_on_error:
            batch_command.append("--continue-on-error")
        if args.evaluate_only:
            batch_command.append("--evaluate-only")
        elif not args.evaluate:
            batch_command.append("--train-only")
        return run_ablations_2d.main(batch_command)
    return run_ablation_2d.main(command)


if __name__ == "__main__":
    main()
