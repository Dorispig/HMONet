"""Optionally train the unchanged 2-D HMONet reference configuration.

Example:
  python scripts/experiments/run_baseline_2d.py --dataset air_t_int_0_1000_21_20 --dataset air_t_ext_0_1000_21_20 --seeds 20260802 20260917 20261031

The training summaries and checkpoints are written below experiments/<experiment-set>/baseline/.
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

from scripts.experiments import run_ablation_2d


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
    args = build_parser().parse_args(argv)
    if not __import__("re").fullmatch(r"ex[1-9][0-9]*", args.experiment_set):
        raise ValueError("--experiment-set must use the form ex1, ex2, ...")
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
        (
            "--evaluate-only"
            if args.evaluate_only
            else ("--continue-on-error" if args.evaluate else "--train-only")
        ),
    ]
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
    if args.device:
        command += ["--device", args.device]
    if args.dry_run:
        command.append("--dry-run")
    if args.force:
        command.append("--force")
    if args.continue_on_error:
        command.append("--continue-on-error")
    return run_ablation_2d.main(command)


if __name__ == "__main__":
    main()
