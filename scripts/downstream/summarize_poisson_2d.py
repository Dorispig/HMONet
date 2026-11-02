"""Rebuild aggregate downstream tables from retained per-case metrics files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.downstream import summarize_rows
from src.utils.paths import resolve_project_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/downstream/poisson_2d")
    args = parser.parse_args(argv)
    root = resolve_project_path(args.root)
    paths = [path for path in root.rglob("metrics.csv") if "aggregate" not in path.parts]
    if not paths:
        raise FileNotFoundError(f"No downstream metrics.csv files below {root}")
    rows = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    summarize_rows(rows.to_dict("records"), root / "aggregate")
    print(f"summarized={len(rows)} rows from {len(paths)} files", flush=True)


if __name__ == "__main__":
    main()
