"""Compare jointly scaled ResMetaMesh decoders and LoRA adapters on a 2-D dataset."""

import argparse
import csv
from copy import deepcopy
from datetime import datetime
import json
import re
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_resmetamesh_adaptation_steps import adaptation_snapshots, compare_grid
from scripts.experiments.run_resmetamesh_budget import CASE
from scripts.train.train_resmetamesh import train_resmetamesh
from src.data.parametric_dataset import ParametricDataset2D
from src.evaluation.configuration import create_geometry
from src.evaluation.inference import resolve_device
from src.models.resmetamesh import ResMetaMesh, decode_mesh

WIDTHS = (50, 96, 160, 276)


def scaled_architecture(architecture, options, width):
    reference_width = architecture["width"]
    rank = max(1, round(options["lora_rank"] * width / reference_width))
    return {**architecture, "width": width}, rank


def grid_references(run, dataset_name, grids, values):
    reference_root = run / "eval/u5000" if (run / "eval/u5000").is_dir() else run
    references = {}
    for n in grids:
        for value in values:
            comparisons = reference_root / "2D" / dataset_name / f"{n}x{n}" / f"{value:g}" / "comparisons"
            formal = comparisons / "CCAP/H_raw__M_raw/TFI_PDE_M_PI_H/comparison_mesh_data.npz"
            archives = [formal] if formal.is_file() else list(comparisons.rglob("comparison_mesh_data.npz"))
            if len(archives) != 1:
                raise FileNotFoundError(f"Expected one unambiguous PDE reference archive in {comparisons}")
            with np.load(archives[0]) as meshes:
                references[n, value] = meshes["PDE_Winslow"].reshape(-1, 2)
    return references


def write_reports(root, dataset_name, rows):
    with (root / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {dataset_name} ResMetaMesh 主干与 LoRA 容量对照", "",
        "各档独立预训练。预训练耗时截止训练结束，适配耗时包含 TFI 基础项准备。",
        "第 0 步为测试几何适配前，第 0 步和适配后的网格均用相同 PDE 参考解。",
    ]
    groups = dict.fromkeys((row["grid"], row["t"], row["adaptation_steps"]) for row in rows)
    fields = (
        "width", "rank", "backbone_parameters", "pretrain_parameters", "adaptation_parameters",
        "training_s", "adaptation_s", "interior_mse_vs_pde", "invalid_cells", "intersecting_cells",
        "flipping_cells", "worst_scaled_jacobian", "status",
    )
    labels = (
        "Width", "LoRA rank", "Backbone params", "Pretrain params", "Adapt params",
        "Training (s)", "Adaptation (s)", "Interior MSE vs. PDE", "Invalid cells", "Intersecting cells",
        "Flipped cells", "Worst Scaled Jacobian", "状态",
    )
    for grid, value, steps in groups:
        lines += ["", f"## {grid}, t = {value:g}, 适配 {steps} 步", "",
                  "| " + " | ".join(labels) + " |", "|" + " --- |" * len(fields)]
        for row in rows:
            if (row["grid"], row["t"], row["adaptation_steps"]) != (grid, value, steps):
                continue
            cells = [f"{row[key]:.6g}" if isinstance(row[key], (int, float))
                     else (str(row[key]) or "—").replace("|", "/") for key in fields]
            lines.append("| " + " | ".join(cells) + " |")
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_run", type=Path, help="Formal evaluation root or existing budget experiment with saved PDE references")
    parser.add_argument("--dataset", default=CASE)
    parser.add_argument("--widths", type=int, nargs="+", default=WIDTHS)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--adaptation-steps", type=int, default=150)
    parser.add_argument("--grids", type=int, nargs="+", default=(101,))
    parser.add_argument("--values", type=float, nargs="+")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-config", type=Path, default=ROOT / "configs/2D/train.yaml")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/rm_capacity")
    args = parser.parse_args(argv)
    if min(*args.widths, args.updates, args.adaptation_steps) < 1 or min(args.grids) < 3:
        parser.error("Widths and step counts must be positive and grid sizes must be at least 3")
    widths = sorted(set(args.widths))
    grids = sorted(set(args.grids))
    device = resolve_device(args.device)
    eval_cfg = yaml.safe_load((ROOT / "configs/2D/evaluate.yaml").read_text(encoding="utf-8-sig"))
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.dataset) or args.dataset not in eval_cfg["cases"]:
        parser.error(f"Unknown 2-D evaluation case: {args.dataset}")
    case = eval_cfg["cases"][args.dataset]
    values = sorted(set(case["values"] if args.values is None else args.values))
    references = grid_references(args.reference_run, args.dataset, grids, values)
    geometries = {value: create_geometry(2, case, value)[0] for value in values}
    base_cfg = yaml.safe_load(args.train_config.read_text(encoding="utf-8-sig"))
    base_arch = base_cfg["model"]["resmetamesh"]
    base_opts = base_cfg["resmetamesh_training"]
    dataset = ParametricDataset2D(ROOT / "data/raw/2D" / args.dataset)
    root = args.output_root.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    print(f"Output: {root}", flush=True)
    rows = []
    for width in widths:
        config = deepcopy(base_cfg)
        architecture, rank = scaled_architecture(base_arch, base_opts, width)
        config["model"]["resmetamesh"] = architecture
        config["training"]["seed"] = args.seed
        # Limit the activation footprint at large widths, while using the same
        # points and accumulating their gradients before the same Adam update.
        point_chunk = min(base_opts["derivative_batch_size"],
                          max(1, base_opts["collocation_count"] * base_arch["width"] // width))
        config["resmetamesh_training"].update(
            pretrain_updates=args.updates, adaptation_steps=args.adaptation_steps,
            lora_rank=rank, derivative_batch_size=point_chunk,
        )
        variant = root / f"w{width}"
        print(f"Training width={width}, rank={rank}, point chunk={point_chunk}", flush=True)
        train_resmetamesh(
            dataset, config, "extrapolation", SimpleNamespace(flat_run_root=variant),
            device,
        )
        summary = json.loads((variant / "training/training_summary.json").read_text(encoding="utf-8-sig"))
        if summary.get("completed_optimizer_updates") != args.updates:
            raise RuntimeError(f"Width {width} did not reach its Adam update budget")
        saved = torch.load(variant / "checkpoints/final.pt", map_location="cpu", weights_only=True)
        model = ResMetaMesh(2, len(dataset), **architecture)
        model.load_state_dict(saved["model_state_dict"])
        model.to(device).eval().requires_grad_(False)
        backbone_count = sum(p.numel() for p in model.decoder.parameters())
        output = model.decoder[-1]
        adaptation_options = {
            **saved["config"]["resmetamesh_training"],
            "derivative_batch_size": base_opts["derivative_batch_size"],
        }
        for value, geometry in geometries.items():
            print(f"Adapting width={width}, rank={rank}, t={value:g}", flush=True)
            states, failure = adaptation_snapshots(
                model, geometry, adaptation_options, args.seed,
                device, [0, args.adaptation_steps],
            )
            for n in grids:
                folder = variant / "grids" / f"{value:g}" / f"{n}x{n}"
                folder.mkdir(parents=True, exist_ok=True)
                for step in (0, args.adaptation_steps):
                    row = {
                        "width": width, "rank": rank, "backbone_parameters": backbone_count,
                        "pretrain_parameters": backbone_count + model.latents.numel(),
                        "adaptation_parameters": model.latent_dim + rank * (output.in_features + output.out_features),
                        "updates": args.updates, "training_s": summary["training_time_seconds"],
                        "point_chunk": point_chunk, "t": value, "grid": f"{n}x{n}", "adaptation_steps": step,
                        "adaptation_s": "", "interior_mse_vs_pde": "", "invalid_cells": "",
                        "intersecting_cells": "", "flipping_cells": "", "worst_scaled_jacobian": "",
                        "status": failure or "已完成",
                    }
                    if step in states:
                        (latent, left, right), elapsed = states[step]
                        points = decode_mesh(model, geometry, (n, n), latent, (left, right))
                        if np.isfinite(points).all():
                            np.save(folder / f"step_{step}.npy", points)
                            row.update(compare_grid(points, references[n, value], n),
                                       adaptation_s=elapsed if step else 0, status="已完成")
                        else:
                            row["status"] = "网格坐标非有限"
                    rows.append(row)
            write_reports(root, args.dataset, rows)
    print(f"Comparison: {root / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
