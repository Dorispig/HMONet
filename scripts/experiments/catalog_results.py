"""Build a non-destructive catalog of all historical experiment runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def build_catalog(experiments_root: Path) -> dict:
    runs = []
    for manifest_path in sorted(experiments_root.rglob("run_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runs.append(
            {
                "manifest": manifest_path.relative_to(ROOT).as_posix(),
                "suite": manifest_path.relative_to(experiments_root).parts[0],
                "ablation": manifest.get("ablation"),
                "variant": manifest.get("variant"),
                "dataset": manifest.get("dataset"),
                "seed": manifest.get("seed"),
                "run_tag": manifest.get("run_tag"),
                "status": manifest.get("status"),
                "training_fingerprint": manifest.get("training_fingerprint"),
                "evaluation_count": len(manifest.get("evaluations") or {}),
            }
        )
    return {
        "schema_version": 1,
        "policy": (
            "Historical result directories remain in place so paths embedded in "
            "manifests stay valid; the runner discovers them recursively."
        ),
        "run_count": len(runs),
        "runs": runs,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "experiments")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = (args.output or root / "results_catalog.json").resolve()
    output.write_text(
        json.dumps(build_catalog(root), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
