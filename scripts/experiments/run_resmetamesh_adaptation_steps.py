"""Compare ResMetaMesh adaptation budgets along a shared 2-D L-BFGS trajectory."""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_resmetamesh_budget import CASE, case_directory
from src.evaluation.configuration import create_geometry
from src.evaluation.inference import resolve_device
from src.evaluation.quality_output import compute_quad_quality
from src.models.resmetamesh import _synchronize, adapt_shape, decode_mesh, ResMetaMesh

QUALITY_METRICS = (
    "scaled_jacobian", "concave_vertex_count", "intersecting_cell",
    "flipping_cell", "degenerate_cell", "self_intersection",
)


def adaptation_snapshots(model, geometry, options, seed, device, steps):
    """Observe the real optimizer without changing adaptation or public defaults."""
    snapshots = {}
    completed = 0
    _synchronize(device)
    started = perf_counter()

    def capture(optimizer, step):
        if step in steps:
            parameters = optimizer.param_groups[0]["params"]
            _synchronize(device)
            snapshots[step] = (tuple(p.detach().clone() for p in parameters), perf_counter() - started)

    def before(optimizer, args, kwargs):
        if completed == 0 and isinstance(optimizer, torch.optim.LBFGS):
            capture(optimizer, 0)
        return None

    def after(optimizer, args, kwargs):
        nonlocal completed
        if isinstance(optimizer, torch.optim.LBFGS):
            completed += 1
            capture(optimizer, completed)

    failure = None
    pre_hook = register_optimizer_step_pre_hook(before)
    post_hook = register_optimizer_step_post_hook(after)
    try:
        adapt_shape(model, geometry, {**options, "adaptation_steps": max(steps)}, seed, device)
    except FloatingPointError as error:
        failure = f"第 {min(completed + 1, max(steps))} 步检测到失败 ({error})"
    finally:
        pre_hook.remove()
        post_hook.remove()
    if failure is None and set(snapshots) != set(steps):
        raise RuntimeError(f"Saved adaptation steps {sorted(snapshots)}, expected {steps}")
    return snapshots, failure


def compare_grid(points, reference, n):
    quality = compute_quad_quality(points, n, n, metrics=QUALITY_METRICS)
    fields = quality["fields"]
    interior = (points - reference).reshape(n, n, 2)[1:-1, 1:-1]
    return {
        "invalid_cells": quality["summary"]["invalid_cells"]["count"],
        "intersecting_cells": int(np.count_nonzero(fields["intersecting_cell"] > 0)),
        "flipping_cells": int(np.count_nonzero(fields["flipping_cell"] > 0)),
        "worst_scaled_jacobian": float(fields["scaled_jacobian"].min()),
        "interior_mse_vs_pde": float(np.square(interior).mean()),
    }


def write_report(root, rows):
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# air_t ResMetaMesh 测试时适配步数对照", ""]
    groups = dict.fromkeys((r["pretrain_updates"], r["t"], r["grid"]) for r in rows)
    for budget, value, grid in groups:
        lines.extend([
            f"## 预训练 {budget} 次，t = {value:g}，{grid}", "",
            "| 适配步数 | 非法 | 相交 | 翻转 | 最差 Scaled Jacobian | 内部 MSE vs. PDE | 累计耗时 (s) | 状态 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        fields = ("adaptation_steps", "invalid_cells", "intersecting_cells", "flipping_cells",
                  "worst_scaled_jacobian", "interior_mse_vs_pde", "elapsed_s")
        for row in rows:
            if (row["pretrain_updates"], row["t"], row["grid"]) == (budget, value, grid):
                formatted = ["—" if row[k] == "" else f"{row[k]:.6g}" for k in fields]
                lines.append("| " + " | ".join([*formatted, row["status"]]) + " |")
        lines.append("")
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("budget_run", type=Path, help="Existing ResMetaMesh pre-training budget experiment directory")
    parser.add_argument("--pretrain-updates", nargs="+", type=int, default=(5000, 20000))
    parser.add_argument("--steps", nargs="+", type=int, default=(100, 150, 200, 300, 500))
    parser.add_argument("--values", nargs="+", type=float, default=(0.62, 0.65))
    parser.add_argument("--grids", nargs="+", type=int, default=(101,))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/rm_adapt")
    args = parser.parse_args(argv)
    if min(args.steps) < 1 or min(args.grids) < 3 or min(args.pretrain_updates) < 1:
        parser.error("Positive step counts and grids of at least three nodes are required")
    steps = sorted({0, *args.steps})
    budgets = sorted(set(args.pretrain_updates))
    grids = sorted(set(args.grids))
    values = sorted(set(args.values))
    device = resolve_device(args.device)
    evaluation = yaml.safe_load((ROOT / "configs/2D/evaluate.yaml").read_text(encoding="utf-8-sig"))
    case = evaluation["cases"][CASE]
    snapshots = {steps: args.budget_run / "weights" / f"u{steps}" / "final.pt" for steps in budgets}
    references = {}
    for value in values:
        for n in grids:
            case_root = case_directory(args.budget_run / "eval" / f"u{budgets[0]}", n, value)
            files = list((case_root / "comparisons").rglob("comparison_mesh_data.npz"))
            if len(files) != 1:
                raise FileNotFoundError(f"Expected one comparison mesh archive in {case_root}")
            with np.load(files[0]) as meshes:
                references[n, value] = meshes["PDE_Winslow"].reshape(-1, 2)
    for path in snapshots.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    root = args.output_root.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    print(f"Output: {root}", flush=True)
    rows = []
    for budget, path in snapshots.items():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        weights = saved["model_state_dict"]
        config = saved["config"]
        model = ResMetaMesh(2, len(weights["latents"]), **config["model"]["resmetamesh"])
        model.load_state_dict(weights)
        model.to(device).eval().requires_grad_(False)
        for value in values:
            geometry, *_ = create_geometry(2, case, value)
            print(f"Adapting u{budget}, t={value:g}, through step {max(steps)}", flush=True)
            states, failure = adaptation_snapshots(
                model, geometry, config["resmetamesh_training"], config["training"]["seed"], device, steps,
            )
            if failure is not None:
                print(f"u{budget} t={value:g}: {failure}", flush=True)
            for n in grids:
                directory = root / f"u{budget}" / f"{value:g}" / f"{n}x{n}"
                directory.mkdir(parents=True)
                for step in steps:
                    row = {
                        "pretrain_updates": budget, "t": value, "grid": f"{n}x{n}", "adaptation_steps": step,
                        "elapsed_s": "", "invalid_cells": "", "intersecting_cells": "", "flipping_cells": "",
                        "worst_scaled_jacobian": "", "interior_mse_vs_pde": "", "status": failure or "已完成",
                    }
                    if step in states:
                        (latent, left, right), elapsed = states[step]
                        points = decode_mesh(model, geometry, (n, n), latent, (left, right))
                        if np.isfinite(points).all():
                            np.save(directory / f"step_{step}.npy", points)
                            row.update(elapsed_s=elapsed, **compare_grid(points, references[n, value], n), status="已完成")
                        else:
                            row["status"] = "网格坐标非有限"
                    rows.append(row)
            write_report(root, rows)
    print(f"Comparison: {root / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
