"""Generate a complete Markdown audit of finished 2-D ablation experiments."""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

EXPERIMENTS = {
    "initial_learning_rate": ("初始学习率", "lr_2e-4"),
    "jacobian_enabled": ("Jacobian损失开关", "jac_on"),
    "jacobian_weight": ("Jacobian损失权重", "jac_w_1e-2"),
    "mapping_mode": ("模型内映射方式", "current"),
    "sampling_mode": ("内部点采样方式", "uniform"),
    "target_interpolation": ("训练真实值插值方式", "curved_boundary_cell"),
    "training_sample_count": ("每批随机采样点数量", "samples_400"),
}

ANALYSIS = {
    "initial_learning_rate": (
        "2e-4在最差scaled Jacobian、最差偏斜和最差正交性之间取得最均衡结果；"
        "4e-4的固定网格MSE略低，但总训练时间更长。1e-3和6e-4出现非零相交率，稳定性不足。",
        "建议保留为超参数选择证据，但不必占正文大篇幅；正文给出最终学习率，完整表放附录。",
    ),
    "jacobian_enabled": (
        "权重0.01的Jacobian损失没有改善总体质量，最差scaled Jacobian、偏斜、长宽比和正交性均略差；"
        "平均单epoch成本约由0.373 s增至0.751 s，总训练成本超过两倍。两组均未出现相交。",
        "若论文不把Jacobian损失作为主要创新，正文无需强调其正向作用；应在附录或讨论中说明大权重无收益。",
    ),
    "jacobian_weight": (
        "1e-4在最差scaled Jacobian、最差偏斜、最差长宽比和最差正交性上最好，且所有权重均无相交；"
        "但平均质量差异很小，1e-4训练时间约为无Jacobian的2.2倍。0.1和0.01没有呈现与成本相称的收益。",
        "如果最终模型使用Jacobian损失，必须保留该实验，并选择1e-4附近的小权重；否则可作为负结果放附录。",
    ),
    "mapping_mode": (
        "current模式的最差scaled Jacobian、偏斜和正交性明显优于Coons残差，并保持单元间相交率为0；"
        "Coons残差虽然训练更快、最差长宽比较低，但平均单元间相交比例达到约3%，不适合作为最终映射。",
        "必须保留。这是边界硬约束形式的直接消融，能够支撑论文选择current映射的核心论证。",
    ),
    "sampling_mode": (
        "自适应采样改善了最差scaled Jacobian、偏斜、长宽比和正交性，且两种采样均无相交；"
        "代价是单epoch时间约增加27%，总训练时间约增加11%，固定网格MSE略差。",
        "若最终采用自适应采样，应作为正文实验；若最终采用均匀采样，可放附录说明质量与成本取舍。",
    ),
    "target_interpolation": (
        "曲边修正将最差scaled Jacobian由约0.165提高到0.254，最差偏斜由约0.894降至0.836，"
        "最差正交性由约0.187提高到0.284，同时固定网格MSE降低；训练总成本增加约21%。",
        "必须保留。这直接证明曲边真实值修正改善边界附近网格，是HMONet训练策略的重要组成部分。",
    ),
    "training_sample_count": (
        "采样点数量与质量并非单调关系。400点总体均衡；100点训练成本最低但部分最差指标下降；"
        "1200点和800点显著增加训练成本，却没有稳定提升质量。50点因需要更多epoch，总时长也未降低。",
        "建议作为附录中的超参数敏感性实验。正文只需说明选择400点是质量、稳定性和成本的折中。",
    ),
}


def number(value, digits=6):
    if value is None or not np.isfinite(value):
        return "—"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.3e}"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def cell_status(row, prefix):
    count = row.get(f"{prefix}__affected_cell_count__mean_across_seeds", np.nan)
    total = row.get(f"{prefix}__cell_count__mean_across_seeds", np.nan)
    fraction = row.get(f"{prefix}__affected_cell_fraction__mean_across_seeds", np.nan)
    if not np.isfinite(count):
        return "—"
    return f"{number(count, 2)}/{number(total, 0)} ({number(100 * fraction, 4)}%)"


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def metric(row, name, statistic):
    return number(row.get(f"{name}__{statistic}__mean_across_seeds", np.nan))


def parameter_label(value):
    return number(float(value), 6)


def variant_label(name, baseline):
    return f"**{name}（冻结基线）**" if name == baseline else str(name)


def quality_tables(case, baseline):
    case = case.sort_values("variant")
    angle_rows, shape_rows, legality_rows = [], [], []
    for _, row in case.iterrows():
        variant = variant_label(row["variant"], baseline)
        angle_rows.append(
            [
                variant,
                metric(row, "max_angle", "mean"),
                metric(row, "max_angle", "maximum"),
                metric(row, "min_angle", "mean"),
                metric(row, "min_angle", "minimum"),
                metric(row, "jacobian", "mean"),
                metric(row, "jacobian", "minimum"),
                metric(row, "scaled_jacobian", "mean"),
                metric(row, "scaled_jacobian", "minimum"),
            ]
        )
        shape_rows.append(
            [
                variant,
                metric(row, "skewness", "mean"),
                metric(row, "skewness", "maximum"),
                metric(row, "aspect_ratio", "mean"),
                metric(row, "aspect_ratio", "maximum"),
                metric(row, "orthogonality", "mean"),
                metric(row, "orthogonality", "minimum"),
                metric(row, "cell_area", "mean"),
                metric(row, "cell_area", "minimum"),
                metric(row, "boundary_layer_orthogonality", "mean"),
                metric(row, "boundary_layer_orthogonality", "minimum"),
            ]
        )
        invalid_count = row.get("invalid_cells__count__mean_across_seeds", np.nan)
        invalid_fraction = row.get("invalid_cells__fraction__mean_across_seeds", np.nan)
        legality_rows.append(
            [
                variant,
                f"{number(invalid_count, 2)} ({number(100 * invalid_fraction, 4)}%)",
                cell_status(row, "self_intersection"),
                cell_status(row, "intersecting_cell"),
                cell_status(row, "flipping_cell"),
                cell_status(row, "degenerate_cell"),
                cell_status(row, "concave_vertex_count"),
            ]
        )
    return (
        markdown_table(
            [
                "变体",
                "最大角均值↓",
                "最大角最差↓",
                "最小角均值↑",
                "最小角最差↑",
                "Jacobian均值↑",
                "Jacobian最差↑",
                "Scaled J均值↑",
                "Scaled J最差↑",
            ],
            angle_rows,
        ),
        markdown_table(
            [
                "变体",
                "偏斜均值↓",
                "偏斜最差↓",
                "长宽比均值↓",
                "长宽比最差↓",
                "正交性均值↑",
                "正交性最差↑",
                "面积均值",
                "面积最差↑",
                "边界层正交均值↑",
                "边界层正交最差↑",
            ],
            shape_rows,
        ),
        markdown_table(
            ["变体", "非法单元", "自相交", "单元间相交", "翻转", "退化", "凹顶点影响"],
            legality_rows,
        ),
    )


def cost_table(data, baseline):
    rows = []
    for variant, group in data.groupby("variant", sort=True):
        first = group.iloc[0]
        rows.append(
            [
                variant_label(variant, baseline),
                number(first.get("completed_epochs__mean_across_seeds"), 1),
                number(first.get("best_epoch__mean_across_seeds"), 1),
                number(first.get("training_seconds__mean_across_seeds"), 2),
                number(first.get("average_epoch_seconds__mean_across_seeds"), 5),
                number(group["network_inference_seconds__mean_across_seeds"].mean(), 7),
                number(group["boundary_projection_seconds__mean_across_seeds"].mean(), 7),
                number(group["total_model_seconds__mean_across_seeds"].mean(), 7),
                number(first.get("fixed_grid_best_mse__mean_across_seeds")),
            ]
        )
    return markdown_table(
        [
            "变体",
            "完成epoch",
            "最佳epoch",
            "训练总时长/s",
            "单epoch/s",
            "网络推理/s",
            "边界投影/s",
            "网格生成/s",
            "最佳固定网格MSE↓",
        ],
        rows,
    )


def averaged_table(data, baseline):
    rows = []
    for variant, group in data.groupby("variant", sort=True):
        rows.append(
            [
                variant_label(variant, baseline),
                number(group["scaled_jacobian__mean__mean_across_seeds"].mean()),
                number(group["scaled_jacobian__minimum__mean_across_seeds"].mean()),
                number(group["skewness__mean__mean_across_seeds"].mean()),
                number(group["skewness__maximum__mean_across_seeds"].mean()),
                number(group["aspect_ratio__mean__mean_across_seeds"].mean()),
                number(group["aspect_ratio__maximum__mean_across_seeds"].mean()),
                number(group["orthogonality__mean__mean_across_seeds"].mean()),
                number(group["orthogonality__minimum__mean_across_seeds"].mean()),
                number(100 * group["self_intersection__affected_cell_fraction__mean_across_seeds"].mean(), 4),
                number(100 * group["intersecting_cell__affected_cell_fraction__mean_across_seeds"].mean(), 4),
                number(group["total_model_seconds__mean_across_seeds"].mean(), 7),
            ]
        )
    return markdown_table(
        [
            "变体",
            "Scaled J均值↑",
            "Scaled J最差↑",
            "偏斜均值↓",
            "偏斜最差↓",
            "长宽比均值↓",
            "长宽比最差↓",
            "正交性均值↑",
            "正交性最差↑",
            "自相交/%↓",
            "单元间相交/%↓",
            "生成时间/s↓",
        ],
        rows,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="paper_v1")
    parser.add_argument("--output", default="docs/completed_ablation_report.md")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output

    lines = [
        "# 已完成二维消融实验完整对比",
        "",
        "统计口径：表中数值为三个随机种子的均值；最差值根据指标方向取最大值或最小值。"
        "冻结基线使用学习率2e-4、Jacobian权重0.01、current映射、均匀采样、曲边IDW真实值和400个随机点。",
        "",
        "已完成并纳入统计：初始学习率、Jacobian开关、Jacobian权重、映射方式、采样方式、"
        "真实值插值方式和训练采样点数量。`normal_loss`与`point_weighting`尚无完整训练评估结果，未纳入结论。",
        "",
    ]
    for experiment, (title, baseline) in EXPERIMENTS.items():
        root = ROOT / "experiments" / experiment
        files = sorted(root.glob(f"*/aggregate/{args.run_tag}/*/mean_variance_by_case.csv"))
        if not files:
            continue
        lines += [f"## {title}（`{experiment}`）", "", f"冻结基线变体：`{baseline}`。", ""]
        all_data = []
        for path in files:
            data = pd.read_csv(path)
            dataset = str(data.iloc[0]["dataset"])
            grid = str(data.iloc[0]["grid_shape"])
            all_data.append(data)
            lines += [f"### 数据集 `{dataset}`，评估网格 `{grid}`", "", "#### 运行成本与网格生成时间", "", cost_table(data, baseline), ""]
            for value, case in data.groupby("value", sort=True):
                lines += [f"#### 参数 `{case.iloc[0]['parameter']}={parameter_label(value)}`", ""]
                angle, shape, legality = quality_tables(case, baseline)
                lines += ["**角度与Jacobian质量**", "", angle, "", "**畸变、面积和边界层质量**", "", shape, "", "**非法与相交情况：平均影响单元数/总单元数（比例）**", "", legality, ""]
        combined = pd.concat(all_data, ignore_index=True)
        finding, paper = ANALYSIS[experiment]
        lines += [
            "### 跨参数和网格规模平均概览",
            "",
            averaged_table(combined, baseline),
            "",
            "### 结果分析",
            "",
            finding,
            "",
            "**论文取舍：** " + paper,
            "",
            "---",
            "",
        ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
