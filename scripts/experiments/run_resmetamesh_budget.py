"""Compare ResMetaMesh pre-training budgets on the same Adam trajectory."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate.evaluate_2d import main as evaluate
from scripts.train.train_resmetamesh import train_resmetamesh
from src.data.parametric_dataset import ParametricDataset2D
from src.evaluation.inference import resolve_device

CASE = "air_t_ext_1_500_41_41"
DEFAULT_BUDGETS = (5000, 10000, 15000, 20000)


def save_snapshot(root, state, epoch):
    steps = state.optimizer_updates
    directory = root / "weights" / f"u{steps}"
    directory.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu() for key, value in state.model.state_dict().items()
            },
            "config": state.saved_config,
            "grid": state.config["grid"],
            "model_name": "resmetamesh",
            "epoch": epoch,
            "optimizer_updates": steps,
        },
        directory / "final.pt",
    )
    with (directory / "training_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {"training_time_seconds": perf_counter() - state.started,
             "epoch": epoch, "optimizer_updates": steps},
            stream, indent=2,
        )
    print(f"Saved ResMetaMesh after {steps} Adam updates", flush=True)


def case_directory(root, n, value):
    return root / "2D" / CASE / f"{n}x{n}" / f"{value:g}"


def reuse_references(source_root, target_root, grids, values):
    if source_root == target_root:
        return
    for n in grids:
        for value in values:
            source = case_directory(source_root, n, value) / ".cache"
            target = case_directory(target_root, n, value) / ".cache"
            for method in ("TFI", "PDE"):
                if (source / method / "generation_metadata.json").is_file() and not (target / method).exists():
                    shutil.copytree(source / method, target / method)


def read_case(directory, n, value, budget, training_seconds):
    # This method combination uses one comparison directory, independent of HMONet settings.
    candidates = list((directory / "comparisons").glob("**/comparison_mesh_data.npz"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one complete comparison in {directory}")
    comparison = candidates[0].parent
    with (comparison / "grid_quality/paper_quality_table.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        quality = {(row["method"], row["metric"]): row for row in csv.DictReader(stream)}
    timing = json.loads((comparison / "generation_timings.json").read_text(encoding="utf-8-sig"))
    rows = []
    with np.load(candidates[0]) as meshes:
        reference = meshes["PDE_Winslow"].reshape(n, n, 2)[1:-1, 1:-1]
        for method in ("TFI", "PDE_Winslow", "ResMetaMesh"):
            mesh = meshes[method].reshape(n, n, 2)[1:-1, 1:-1]
            method_timing = timing[method]
            row = {
                "grid": f"{n}x{n}",
                "t": value,
                "method": method,
                "updates": budget if method == "ResMetaMesh" else "",
                "pretraining_s": training_seconds if method == "ResMetaMesh" else "",
                "generation_s": method_timing.get("total_model_seconds", method_timing.get("total_seconds")),
                "interior_mse_vs_pde": float(np.square(mesh - reference).mean()),
                "invalid_cells": int(quality[method, "invalid_cells"]["count"]),
            }
            for metric, worst in (
                ("scaled_jacobian", "minimum"), ("min_angle", "minimum"), ("max_angle", "maximum"),
            ):
                values = quality[method, metric]
                row[f"{metric}_mean"] = float(values["mean"])
                row[f"{metric}_worst"] = float(values[worst])
            rows.append(row)
    return rows


def write_summary(root, rows, grids, values):
    with (root / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# air_t ResMetaMesh 预训练预算对照", "",
        "各检查点来自同一次预训练。指标来自逐几何适配后的输出，生成耗时包含适配。",
        "成对质量值为均值 / 最差值，分别加粗最优值。内部点 MSE 的最优比较排除 PDE 自身。",
    ]
    metrics = (
        ("Pretraining time (s)", ("pretraining_s",), (min,)),
        ("Generation time (s)", ("generation_s",), (min,)),
        ("Interior MSE vs. PDE", ("interior_mse_vs_pde",), (min,)),
        ("Invalid cells", ("invalid_cells",), (min,)),
        ("Scaled Jacobian", ("scaled_jacobian_mean", "scaled_jacobian_worst"), (max, max)),
        ("Minimum angle (deg)", ("min_angle_mean", "min_angle_worst"), (max, max)),
        ("Maximum angle (deg)", ("max_angle_mean", "max_angle_worst"), (min, min)),
    )
    for n in grids:
        grid = f"{n}x{n}"
        for value in values:
            selected = [row for row in rows if row["grid"] == grid and row["t"] == value]
            labels = [f"{r['updates']} updates" if r["updates"] else r["method"] for r in selected]
            lines += ["", f"## {grid}, t = {value:g}", "", "| Metric | " + " | ".join(labels) + " |",
                      "| --- |" + " ---: |" * len(labels)]
            for label, fields, reductions in metrics:
                columns = []
                for field, reduce in zip(fields, reductions):
                    eligible = [r[field] for r in selected if r[field] != "" and (
                        field != "interior_mse_vs_pde" or r["method"] != "PDE_Winslow"
                    )]
                    best = reduce(eligible)
                    columns.append([
                        "—" if r[field] == "" else f"**{r[field]:.6g}**"
                        if r[field] == best and (field != "interior_mse_vs_pde" or r["method"] != "PDE_Winslow")
                        else f"{r[field]:.6g}"
                        for r in selected
                    ])
                lines.append("| " + " | ".join([label, *map(" / ".join, zip(*columns))]) + " |")
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--grids", nargs="+", type=int, default=(101, 201, 401))
    parser.add_argument("--values", nargs="+", type=float, default=(0.62, 0.65))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/rm_budget")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--run-tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--train-config", type=Path, default=ROOT / "configs/2D/train.yaml")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate snapshots from an existing run tag.")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_tag):
        parser.error("--run-tag must contain only letters, digits, hyphens and underscores")
    budgets = sorted(set(args.updates))
    if min(budgets) < 1 or min(args.grids) < 3:
        parser.error("Update counts must be positive and grids need at least three nodes per axis")
    device = resolve_device(args.device)
    root = args.output_root.resolve() / args.run_tag
    if not args.evaluate_only:
        cfg = yaml.safe_load(args.train_config.read_text(encoding="utf-8-sig"))
        cfg["training"]["seed"] = args.seed
        cfg["resmetamesh_training"]["pretrain_updates"] = budgets[-1]
        dataset = ParametricDataset2D(ROOT / "data/raw/2D" / CASE)
        root.mkdir(parents=True, exist_ok=False)

        def capture(state, epoch):
            if state.optimizer_updates in budgets:
                save_snapshot(root, state, epoch)

        train_resmetamesh(
            dataset, cfg, "extrapolation", SimpleNamespace(flat_run_root=root / "run"),
            device, on_update=capture,
        )

    snapshots = {step: root / "weights" / f"u{step}" / "final.pt" for step in budgets}
    missing = [str(p) for p in snapshots.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing budget checkpoints: " + ", ".join(missing))
    rows = []
    reference_root = ROOT / "outputs/results"
    for budget, checkpoint in snapshots.items():
        output = root / "eval" / f"u{budget}"
        reuse_references(reference_root, output, args.grids, args.values)
        evaluate([
            "--case", CASE, "--methods", "tfi,pde,resmetamesh",
            "--resmetamesh-checkpoint", str(checkpoint), "--output-root", str(output),
            "--device", str(device), "--no-compact-outputs", "--no-skip-missing-checkpoints",
            "--grids", *map(str, args.grids), "--values", *map(str, args.values),
        ])
        reference_root = output
        metadata = json.loads((checkpoint.parent / "training_summary.json").read_text(encoding="utf-8"))
        for n in args.grids:
            for value in args.values:
                case_rows = read_case(
                    case_directory(output, n, value), n, value, budget, metadata["training_time_seconds"]
                )
                rows.extend(case_rows if budget == budgets[0] else case_rows[-1:])
    write_summary(root, rows, args.grids, args.values)
    print(f"Comparison: {root / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
