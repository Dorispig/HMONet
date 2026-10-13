"""Generate first/middle/last example images for existing 3-D datasets.

Examples
--------
Generate examples for every stored three-dimensional dataset::

    python scripts/data/generate_dataset_examples_3d.py --all

Generate one named dataset::

    python scripts/data/generate_dataset_examples_3d.py \
        --dataset sch_r_ext_0_1000_11_31_11
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.generation.dataset_examples import save_dataset_examples_3d


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate first/middle/last PDE grid images for 3-D datasets."
    )
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/3D")
    args = parser.parse_args(argv)
    if args.all:
        directories = sorted(
            path for path in args.data_root.iterdir() if path.is_dir()
        )
    else:
        if not args.dataset:
            parser.error("use --all or at least one --dataset")
        directories = [args.data_root / name for name in args.dataset]

    for directory in directories:
        if not (directory / "pde_mesh.npy").is_file():
            raise FileNotFoundError(directory / "pde_mesh.npy")
        output = save_dataset_examples_3d(directory)
        print(f"Generated examples: {output}", flush=True)


if __name__ == "__main__":
    main()
