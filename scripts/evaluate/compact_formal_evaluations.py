"""Compact formal evaluation outputs while preserving summaries and images.

Usage:
    python scripts/evaluate/compact_formal_evaluations.py --dry-run
    python scripts/evaluate/compact_formal_evaluations.py
    python scripts/evaluate/compact_formal_evaluations.py --dimensions 2D
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate.evaluation_common import evaluation_identity_from_record
from src.evaluation.artifacts import digest, read_json, write_json
from src.evaluation.retention import (
    IMAGE_SUFFIXES,
    compact_evaluation_directory,
    reorganize_evaluation_images,
)


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def compact_result(result_file: Path, dry_run: bool, images_only: bool = False) -> dict:
    root = result_file.parent
    record = read_json(result_file, {})
    parameters = read_json(root / "evaluation_parameters.json", {})
    options = parameters.get("evaluation", {})
    needs_fingerprint = not record.get("evaluation_fingerprint") and bool(options)
    removable = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path != result_file
        and path.suffix.lower() not in IMAGE_SUFFIXES
    ]
    if not dry_run:
        if needs_fingerprint:
            identity = evaluation_identity_from_record(record, options)
            record["evaluation_identity"] = identity
            record["evaluation_fingerprint"] = digest(identity)
            write_json(result_file, record)
        if images_only:
            reorganize_evaluation_images(root)
        else:
            compact_evaluation_directory(root)
    return {
        "result": str(result_file),
        "removed_files": 0 if images_only else len(removable),
        "removed_bytes": 0 if images_only else sum(file_size(path) for path in removable),
        "preserved_detail_files": len(removable) if images_only else 0,
        "preserved_detail_bytes": (
            sum(file_size(path) for path in removable) if images_only else 0
        ),
        "retained_images": sum(
            1
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        "fingerprint_added": needs_fingerprint,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dimensions",
        nargs="*",
        choices=("2D", "3D"),
        default=("2D", "3D"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Reorganize images without removing numeric caches or detail tables.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_root = (ROOT / "outputs" / "results").resolve()
    if output_root != (ROOT / "outputs" / "results").resolve():
        raise RuntimeError(f"Unsafe output root: {output_root}")
    results = []
    for dimension in args.dimensions:
        root = output_root / dimension
        if root.is_dir():
            results.extend(
                compact_result(path, args.dry_run, args.images_only)
                for path in sorted(root.rglob("results.json"))
            )
    total = {
        "cases": len(results),
        "removed_files": sum(item["removed_files"] for item in results),
        "removed_bytes": sum(item["removed_bytes"] for item in results),
        "retained_images": sum(item["retained_images"] for item in results),
        "fingerprints_added": sum(item["fingerprint_added"] for item in results),
        "preserved_detail_files": sum(
            item["preserved_detail_files"] for item in results
        ),
        "preserved_detail_bytes": sum(
            item["preserved_detail_bytes"] for item in results
        ),
    }
    print(
        json.dumps(
            {"dry_run": args.dry_run, "images_only": args.images_only, "total": total},
            indent=2,
        )
    )
    return total


if __name__ == "__main__":
    main()
