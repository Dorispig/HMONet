"""Compact command-line entry point for 2-D experiment suites.

Examples::

    python scripts/experiments/run.py ex9 e9_1
    python scripts/experiments/run.py ex9 baseline e9_1 e9_2 e9_3 e9_4 --grids 101 201 401
    python scripts/experiments/run.py ex9 all --dataset run_ext_0_1000_61_21

Dataset, seeds and evaluation grids fall back to the selected suite configs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.experiments import run_ablations_2d


def _grid_name(value: str) -> str:
    text = str(value).lower().strip()
    if "x" in text:
        parts = text.split("x")
        if len(parts) != 2 or parts[0] != parts[1]:
            raise argparse.ArgumentTypeError("Use N or NxN for a square 2-D grid")
        text = parts[0]
    try:
        size = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Grid size must be an integer") from error
    if size < 2:
        raise argparse.ArgumentTypeError("Grid size must be at least 2")
    return f"{size}x{size}"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", help="Experiment suite, for example ex3 or ex9")
    parser.add_argument(
        "experiments",
        nargs="*",
        default=["all"],
        help="Experiment names, baseline, or all",
    )
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--seed", "--seeds", dest="seeds", nargs="+", type=int)
    parser.add_argument("--grids", nargs="+", type=_grid_name)
    parser.add_argument("--values", nargs="+", type=float)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--device")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--plots", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def command_from_args(args):
    command = [
        "--experiment-set", args.suite,
        "--ablations", *(args.experiments or ["all"]),
        "--run-tag", args.tag,
    ]
    for dataset in args.dataset or ():
        command += ["--dataset", dataset]
    if args.seeds:
        command += ["--seeds", *map(str, args.seeds)]
    if args.grids:
        command += ["--evaluation-grid-shapes", *args.grids]
    if args.values:
        command += ["--evaluation-values", *map(str, args.values)]
    if args.device:
        command += ["--device", args.device]
    if not args.plots:
        command.append("--no-evaluation-plots")
    for name in ("train_only", "evaluate_only", "force", "dry_run", "continue_on_error"):
        if getattr(args, name):
            command.append("--" + name.replace("_", "-"))
    return command


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run_ablations_2d.main(command_from_args(args))


if __name__ == "__main__":
    main()
