"""Audit every stored sample and optionally refresh three grid example images."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from src.data.dataset import read_dataset_metadata
from src.data.generation.dataset_quality_2d import audit_grid
from src.data.generation.dataset_examples import save_dataset_examples


def audit_dataset(directory):
    directory = Path(directory)
    metadata = read_dataset_metadata(directory)
    nx, ny = metadata["grid_shape"]
    records = np.load(directory / "dataset.npy", allow_pickle=True)
    details = [
        {"sample_id": i, **audit_grid(row["mesh_points"], nx, ny)}
        for i, row in enumerate(records)
    ]
    result = {
        "dataset": directory.name,
        "size": len(records),
        "grid_shape": [nx, ny],
        "illegal_samples": sum(not r["all_cells_legal"] for r in details),
        "self_intersecting_samples": sum(
            r["self_intersecting_cells"] > 0 for r in details
        ),
        "overlapping_convex_samples": sum(
            r["overlapping_convex_cells"] > 0 for r in details
        ),
        "samples": details,
    }
    (directory / "quality_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        f"{directory.name}: {result['illegal_samples']}/{len(records)} illegal",
        flush=True,
    )
    return {k: v for k, v in result.items() if k != "samples"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--examples", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2] / "data/raw/2D"
    directories = sorted(p.parent for p in root.glob("*/dataset.npy"))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(audit_dataset, directories))
    out = Path(__file__).resolve().parents[2] / "outputs/dataset_quality_audit.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.examples:
        for directory in directories:
            save_dataset_examples(directory)
            print(f"Examples: {directory.name}", flush=True)


if __name__ == "__main__":
    main()
