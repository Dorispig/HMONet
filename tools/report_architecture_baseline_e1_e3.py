"""Build the current Ex11 baseline/e1/e2/e3 quality and mesh comparison report."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET = "air_t_ext_1_500_21_21"
SEED = "20261030"
TAG = "physics_v1"
PARAMETERS = ("0.62", "0.65")
GRIDS = ("101x101", "201x201", "401x401")
METHODS = (
    ("PDE", "PDE_Winslow", None),
    ("Baseline", "HMONet", "baseline"),
    ("e1 ModifiedMLP", "HMONet", "e1_modified_mlp"),
    ("e2 Cross-attention", "HMONet", "e2_cross_attention"),
    ("e3 Boundary decoder", "HMONet", "e3_boundary_decoder"),
)
METRICS = (
    ("最大角 (°)", "max_angle", "mean", "maximum", "low"),
    ("最小角 (°)", "min_angle", "mean", "minimum", "high"),
    ("Jacobian", "jacobian", "mean", "minimum", "high"),
    ("Scaled Jacobian", "scaled_jacobian", "mean", "minimum", "high"),
    ("Skewness", "skewness", "mean", "maximum", "low"),
    ("长宽比", "aspect_ratio", "mean", "maximum", "low"),
    ("正交性", "orthogonality", "mean", "minimum", "high"),
    ("边界层正交性", "boundary_layer_orthogonality", "mean", "minimum", "high"),
)
COUNT_METRICS = (
    ("凹单元", "concave_vertex_count"),
    ("自相交单元", "self_intersection"),
    ("相交单元", "intersecting_cell"),
    ("翻转单元", "flipping_cell"),
    ("退化单元", "degenerate_cell"),
    ("非法单元", "invalid_cells"),
)


def result_path(variant: str | None, grid: str, parameter: str) -> Path:
    if variant in (None, "baseline"):
        return (
            ROOT / "experiments" / "ex3" / "baseline" / DATASET
            / "evaluation" / SEED / TAG / grid / parameter / "results.json"
        )
    return (
        ROOT / "experiments" / "ex3" / "eval" / DATASET / f"seed_{SEED}"
        / TAG / variant / grid / parameter / "results.json"
    )


def method_directory(variant: str | None, grid: str, parameter: str) -> Path:
    return result_path(variant, grid, parameter).parent


def load_metrics(grid: str, parameter: str):
    loaded = {}
    for display, key, variant in METHODS:
        source_variant = "baseline" if variant is None else variant
        payload = json.loads(result_path(source_variant, grid, parameter).read_text(encoding="utf-8"))
        loaded[display] = payload["metrics"][key]
    return loaded


def compact(value: float) -> str:
    value = float(value)
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude < 1e-3 or magnitude >= 1e4:
        return f"{value:.4e}"
    if magnitude >= 10:
        return f"{value:.4f}"
    return f"{value:.6f}"


def best_learning_methods(values, direction):
    learning = {name: pair for name, pair in values.items() if name != "PDE"}
    # Worst value is the primary discriminator; mean is the tie breaker.
    key = (lambda pair: (pair[1], pair[0])) if direction == "low" else (
        lambda pair: (-pair[1], -pair[0])
    )
    optimum = min(key(pair) for pair in learning.values())
    return {name for name, pair in learning.items() if key(pair) == optimum}


def build_tables():
    report = [
        "# Ex11 baseline/e1/e2/e3 与 PDE 网格质量对比",
        "",
        f"数据集：`{DATASET}`；随机种子：`{SEED}`；run-tag：`{TAG}`。",
        "连续质量指标为 `平均值 / 最差值`；离散合法性指标为 `数量 / 占比`。",
        "粗体表示四种学习模型中的最佳结果（先比较最差值，再比较平均值）。",
        "",
    ]
    win_counts = {name: 0 for name, _, _ in METHODS if name != "PDE"}
    for grid in GRIDS:
        for parameter in PARAMETERS:
            metrics = load_metrics(grid, parameter)
            report += [f"## {grid}，参数 {parameter}", ""]
            headers = ["质量指标"] + [name for name, _, _ in METHODS]
            report.append("| " + " | ".join(headers) + " |")
            report.append("|" + "|".join(["---"] * len(headers)) + "|")
            for label, key, mean_key, worst_key, direction in METRICS:
                pairs = {
                    name: (values[key][mean_key], values[key][worst_key])
                    for name, values in metrics.items()
                }
                best = best_learning_methods(pairs, direction)
                for name in best:
                    win_counts[name] += 1
                cells = []
                for name in headers[1:]:
                    text = f"{compact(pairs[name][0])} / {compact(pairs[name][1])}"
                    cells.append(f"**{text}**" if name in best else text)
                report.append("| " + " | ".join([label] + cells) + " |")
            for label, key in COUNT_METRICS:
                pairs = {}
                for name, values in metrics.items():
                    item = values[key]
                    if key == "invalid_cells":
                        count, fraction = item["count"], item["fraction"]
                    else:
                        count = item.get("affected_cell_count", item.get("total", 0))
                        fraction = item.get("affected_cell_fraction", 0)
                    pairs[name] = (float(count), float(fraction))
                learning_min = min((pair[1], pair[0]) for name, pair in pairs.items() if name != "PDE")
                best = {
                    name for name, pair in pairs.items()
                    if name != "PDE" and (pair[1], pair[0]) == learning_min
                }
                cells = []
                for name in headers[1:]:
                    count, fraction = pairs[name]
                    text = f"{int(count)} / {compact(fraction)}"
                    cells.append(f"**{text}**" if name in best else text)
                report.append("| " + " | ".join([label] + cells) + " |")
            report.append("")
    return "\n".join(report), win_counts


def font(size: int, bold: bool = False):
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf") if bold else Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/msyhbd.ttc") if bold else Path("C:/Windows/Fonts/msyh.ttc"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def mesh_path(display: str, variant: str | None, parameter: str) -> Path:
    directory = method_directory("baseline" if variant is None else variant, "101x101", parameter)
    method = "PDE" if display == "PDE" else "HMONet/raw"
    return directory / "plots" / method / "mesh.png"


def labelled_panel(image: Image.Image, label: str, local: bool) -> Image.Image:
    width, height = image.size
    if local:
        # Leading-edge/high-curvature neighbourhood where the minimum scaled
        # Jacobian and worst angles are concentrated for this geometry family.
        crop = image.crop((int(0.285 * width), int(0.315 * height), int(0.535 * width), int(0.69 * height)))
        crop.thumbnail((430, 560), Image.Resampling.LANCZOS)
    else:
        crop = image.crop((int(0.07 * width), int(0.04 * height), int(0.93 * width), int(0.97 * height)))
        crop.thumbnail((430, 410), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (450, crop.height + 58), "white")
    panel.paste(crop, ((panel.width - crop.width) // 2, 52))
    draw = ImageDraw.Draw(panel)
    box = draw.textbbox((0, 0), label, font=font(25, bold=True))
    draw.text(((panel.width - (box[2] - box[0])) / 2, 13), label, fill="#17243A", font=font(25, bold=True))
    return panel


def build_comparison(parameter: str, local: bool, output: Path):
    panels = []
    for display, _, variant in METHODS:
        with Image.open(mesh_path(display, variant, parameter)) as source:
            panels.append(labelled_panel(source.convert("RGB"), display, local))
    title_height = 70
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels) + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    title = (
        f"{DATASET} | parameter={parameter} | 101x101 | "
        + ("worst-quality leading-edge detail" if local else "mesh comparison")
    )
    box = draw.textbbox((0, 0), title, font=font(29, bold=True))
    draw.text(((canvas.width - (box[2] - box[0])) / 2, 18), title, fill="#17243A", font=font(29, bold=True))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, title_height))
        x += panel.width
    canvas.save(output, dpi=(180, 180))


def main():
    output = ROOT / "reports" / "architecture_suite" / "baseline_e1_e3_air_t"
    output.mkdir(parents=True, exist_ok=True)
    report, wins = build_tables()
    (output / "quality_comparison.md").write_text(report, encoding="utf-8")
    (output / "learning_model_wins.json").write_text(
        json.dumps(wins, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for parameter in PARAMETERS:
        build_comparison(parameter, False, output / f"mesh_comparison_{parameter}.png")
        build_comparison(parameter, True, output / f"worst_quality_detail_{parameter}.png")
    print(output)
    print(json.dumps(wins, ensure_ascii=False))


if __name__ == "__main__":
    main()
