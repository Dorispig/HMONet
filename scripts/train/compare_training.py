"""Build formal training comparison figures from completed training histories."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.engine.training_comparison import ensure_training_comparison


def dataset_name(value):
    if value in ("", ".", "..") or any(separator in value for separator in ("/", "\\", ":")):
        raise argparse.ArgumentTypeError("Use a short dataset name, without a directory.")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", required=True, type=int, choices=(2, 3))
    parser.add_argument("--dataset", required=True, nargs="+", type=dataset_name)
    parser.add_argument(
        "--training-root",
        default=Path("outputs/training"),
        type=Path,
        help="Parent directory of 2D/ and 3D/ (default: outputs/training).",
    )
    args = parser.parse_args(argv)
    training_root = ROOT / args.training_root
    failed = False
    for name in dict.fromkeys(args.dataset):
        comparison = ensure_training_comparison(training_root, args.dimension, name)
        status = comparison["status"]
        if status == "waiting_for_models":
            missing = ", ".join(comparison["missing_models"])
            status = f"missing completed MSE history for {missing}"
            failed = True
        elif status == "matplotlib_unavailable":
            status = "matplotlib is required to generate training figures"
            failed = True
        print(f"{name} | {status} | {comparison['directory']}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
