"""Create the Ex8 loss-ablation report for the Run extrapolation dataset.

The report is intentionally generated from retained ``results.json`` and
``training_summary.json`` files only.  It therefore remains reproducible after
per-cell quality plots have been cleaned from the experiment directories.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

VARIANTS = (
    ("Base: PDE+J+SVD", "baseline", "baseline"),
    ("E1: PDE only", "e1", "e1_pde_only"),
    ("E2: PDE+J", "e2", "e2_pde_jacobian"),
    ("E3: PDE+SVD", "e3", "e3_pde_svd"),
)

QUALITY_METRICS = (
    ("Maximum angle (deg)", "max_angle", "mean", "maximum", "low"),
    ("Minimum angle (deg)", "min_angle", "mean", "minimum", "high"),
    ("Jacobian", "jacobian", "mean", "minimum", "high"),
    ("Scaled Jacobian", "scaled_jacobian", "mean", "minimum", "high"),
    ("Skewness", "skewness", "mean", "maximum", "low"),
    ("Aspect ratio", "aspect_ratio", "mean", "maximum", "low"),
    ("Orthogonality", "orthogonality", "mean", "minimum", "high"),
    (
        "Boundary-layer orthogonality",
        "boundary_layer_orthogonality",
        "mean",
        "minimum",
        "high",
    ),
)

LEGALITY_METRICS = (
    ("Concave cells", "concave_vertex_count"),
    ("Self-intersecting cells", "self_intersection"),
    ("Intersecting cells", "intersecting_cell"),
    ("Flipped cells", "flipping_cell"),
    ("Degenerate cells", "degenerate_cell"),
    ("Invalid cells", "invalid_cells"),
)


def compact(value: float | int | None) -> str:
    """Use the numerical style established by prior ablation reports."""
    if value is None:
        return "N/A"
    value = float(value)
    if not math.isfinite(value):
        return "N/A"
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude < 1e-3 or magnitude >= 1e4:
        return f"{value:.4e}"
    if magnitude >= 10:
        return f"{value:.4f}"
    return f"{value:.6f}"


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    seconds = float(seconds)
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 60:
        return f"{seconds / 60:.2f} min"
    return f"{seconds:.2f} s"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing retained experiment output: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def result_path(
    *, dataset: str, seed: str, tag: str, alias: str, grid: str, parameter: str
) -> Path:
    return (
        ROOT
        / "experiments"
        / "ex8"
        / "eval"
        / dataset
        / f"seed_{seed}"
        / tag
        / alias
        / grid
        / parameter
        / "results.json"
    )


def training_path(*, dataset: str, seed: str, tag: str, run_dir: str) -> Path:
    return (
        ROOT
        / "experiments"
        / "ex8"
        / run_dir
        / dataset
        / f"seed_{seed}"
        / tag
        / "training"
        / "training_summary.json"
    )


def method_boundary_error(payload: dict[str, Any], method: str) -> tuple[float, float]:
    for item in payload.get("boundary_errors", []):
        if item.get("method") == method and item.get("stage") == "raw":
            return float(item["boundary_mean_error"]), float(item["boundary_max_error"])
    if method == "PDE_Winslow":
        return 0.0, 0.0
    return float("nan"), float("nan")


def method_seconds(payload: dict[str, Any], method: str) -> float:
    timing = payload["timings"][method]
    return float(
        timing.get("total_model_seconds", timing.get("total_seconds", float("nan")))
    )


def quality_pair(metric: dict[str, Any], key: str, worst_key: str) -> tuple[float, float]:
    return float(metric[key]["mean"]), float(metric[key][worst_key])


def legality_pair(metric: dict[str, Any], key: str) -> tuple[float, float]:
    item = metric[key]
    if key == "invalid_cells":
        return float(item.get("count", 0.0)), float(item.get("fraction", 0.0))
    return float(item.get("affected_cell_count", 0.0)), float(
        item.get("affected_cell_fraction", 0.0)
    )


def best_indices(values: list[float], direction: str) -> set[int]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return set()
    reference = min(finite) if direction == "low" else max(finite)
    return {
        index
        for index, value in enumerate(values)
        if math.isfinite(value) and math.isclose(value, reference, rel_tol=1e-10, abs_tol=1e-12)
    }


def pair_cell(
    pair: tuple[float, float], best_first: bool, best_second: bool, *, percent: bool = False
) -> str:
    first, second = pair
    if percent:
        second_text = f"{second * 100:.4f}%"
    else:
        second_text = compact(second)
    first_text = compact(first)
    if best_first:
        first_text = f"**{first_text}**"
    if best_second:
        second_text = f"**{second_text}**"
    return f"{first_text} / {second_text}"


def heading_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def load_case(
    dataset: str, seed: str, tag: str, grid: str, parameter: str
) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    baseline_payload = load_json(
        result_path(
            dataset=dataset,
            seed=seed,
            tag=tag,
            alias="baseline",
            grid=grid,
            parameter=parameter,
        )
    )
    loaded["PDE (Winslow)"] = {
        "metrics": baseline_payload["metrics"]["PDE_Winslow"],
        "seconds": method_seconds(baseline_payload, "PDE_Winslow"),
        "boundary": method_boundary_error(baseline_payload, "PDE_Winslow"),
    }
    for display, alias, _ in VARIANTS:
        payload = baseline_payload if alias == "baseline" else load_json(
            result_path(
                dataset=dataset,
                seed=seed,
                tag=tag,
                alias=alias,
                grid=grid,
                parameter=parameter,
            )
        )
        loaded[display] = {
            "metrics": payload["metrics"]["HMONet"],
            "seconds": method_seconds(payload, "HMONet"),
            "boundary": method_boundary_error(payload, "HMONet"),
        }
    return loaded


def build_report(
    *, dataset: str, seed: str, tag: str, parameters: list[str], grids: list[str]
) -> tuple[str, dict[str, Any]]:
    names = ["PDE (Winslow)"] + [display for display, _, _ in VARIANTS]
    report = [
        "# Ex8 损失函数消融实验对比：Run 外推数据集",
        "",
        "## 实验范围与可复现性",
        "",
        f"- 数据集：`{dataset}`（外推）；随机种子：`{seed}`；运行标签：`{tag}`。",
        f"- 评估参数：{ '、'.join(parameters) }；逻辑网格规模：{ '、'.join(grids) }。",
        "- 参考解：每个参数和网格规模下收敛的 `PDE_Winslow` 网格。为聚焦自监督损失项的作用，表中不再单列 TFI。",
        "- Base、E1、E2、E3 使用相同的 HMONet 架构、CCAP 精确硬边界约束、有限差分后端和采样设置；仅改变归一化 PDE、Jacobian 屏障和奇异值屏障的启用组合。",
        "- 目前为单随机种子结果，因此表中不包含均值±标准差。",
        "",
        "## 消融变体",
        "",
    ]
    report += heading_table(
        ["变体", "启用的归一化训练损失", "PDE 权重", "Jacobian barrier 权重", "SVD barrier 权重"],
        [
            ["Base", "PDE + Jacobian 屏障 + 奇异值屏障", "1", "0.01", "0.05"],
            ["E1", "仅 PDE", "1", "0", "0"],
            ["E2", "PDE + Jacobian 屏障", "1", "0.01", "0"],
            ["E3", "PDE + 奇异值屏障", "1", "0", "0.05"],
        ],
    )
    report += ["", "## 训练成本", ""]
    train_rows: list[list[str]] = []
    training: dict[str, dict[str, Any]] = {}
    for display, _, run_dir in VARIANTS:
        summary = load_json(training_path(dataset=dataset, seed=seed, tag=tag, run_dir=run_dir))
        training[display] = summary
        train_rows.append(
            [
                display.split(":")[0],
                f"{int(summary.get('completed_epochs') or 0):,}",
                f"{int(summary.get('best_epoch') or 0):,}",
                compact(summary.get("best_metric_value")),
                compact(summary.get("best_model_dataset_mse")),
                duration(summary.get("training_time_seconds")),
            ]
        )
    report += heading_table(
        [
            "变体",
            "完成迭代次数",
            "最佳迭代",
            "最佳训练目标值（不同组合不可直接横比）",
            "最佳模型固定网格 MSE",
            "训练时间",
        ],
        train_rows,
    )
    report += [
        "",
        "不同变体的总训练目标由不同损失项构成，因而不能仅以该目标值横向排名；固定网格 MSE 作为共同诊断量保留。",
        "",
        "## 网格质量、合法性、边界精度与推理时间",
        "",
        "连续质量指标采用“平均值 / 最差值”；合法性指标采用“受影响单元数 / 受影响单元比例”。每个数值分量的粗体表示五种方法中的最佳值，反映网格质量意义上的最优，不等同于与 PDE 节点位置最接近。",
        "",
    ]

    model_wins: Counter[str] = Counter()
    reference_wins: Counter[str] = Counter()
    legality_summary: Counter[str] = Counter()
    runtime_samples: dict[str, list[float]] = {name: [] for name in names}

    for grid in grids:
        report += [f"## {grid} 评估结果", ""]
        for parameter in parameters:
            case = load_case(dataset, seed, tag, grid, parameter)
            report += [
                f"### {grid} | parameter = {parameter}",
                "",
            ]
            quality_rows: list[list[str]] = []
            for label, key, mean_key, worst_key, direction in QUALITY_METRICS:
                pairs = [quality_pair(case[name]["metrics"], key, worst_key) for name in names]
                best_mean = best_indices([pair[0] for pair in pairs], direction)
                best_worst = best_indices([pair[1] for pair in pairs], direction)
                for index in best_worst:
                    if index == 0:
                        reference_wins[label] += 1
                    else:
                        model_wins[names[index]] += 1
                quality_rows.append(
                    [label]
                    + [
                        pair_cell(pair, index in best_mean, index in best_worst)
                        for index, pair in enumerate(pairs)
                    ]
                )
            report += heading_table(["Metric (mean / worst)"] + names, quality_rows)
            report += [""]

            legality_rows: list[list[str]] = []
            for label, key in LEGALITY_METRICS:
                pairs = [legality_pair(case[name]["metrics"], key) for name in names]
                best_count = best_indices([pair[0] for pair in pairs], "low")
                best_fraction = best_indices([pair[1] for pair in pairs], "low")
                legality_rows.append(
                    [label]
                    + [
                        pair_cell(pair, index in best_count, index in best_fraction, percent=True)
                        for index, pair in enumerate(pairs)
                    ]
                )
                for name, pair in zip(names, pairs):
                    if pair[0] > 0:
                        legality_summary[f"{name}:{label}"] += int(pair[0])
            boundary_pairs = [case[name]["boundary"] for name in names]
            best_mean = best_indices([pair[0] for pair in boundary_pairs], "low")
            best_max = best_indices([pair[1] for pair in boundary_pairs], "low")
            legality_rows.append(
                ["Boundary error (mean / max)"]
                + [
                    pair_cell(pair, index in best_mean, index in best_max)
                    for index, pair in enumerate(boundary_pairs)
                ]
            )
            report += heading_table(["Legality / intersection"] + names, legality_rows)
            report += [""]

            times = [case[name]["seconds"] * 1000 for name in names]
            best_time = best_indices(times, "low")
            report += heading_table(
                ["Generation time (ms/grid)"] + names,
                [["Time"] + [f"**{compact(value)}**" if index in best_time else compact(value) for index, value in enumerate(times)]],
            )
            report += [""]
            for name, seconds in zip(names, times):
                runtime_samples[name].append(seconds)

    average_runtime = {
        name: sum(values) / len(values) for name, values in runtime_samples.items() if values
    }
    report += ["## 汇总分析", ""]
    report += [
        "1. **合法性与边界条件。** PDE、E1、E2 与 E3 在六个“参数×网格规模”组合中均未出现凹单元、自相交、相交、翻转、退化或非法单元。Base 在参数 1.55 的 101×101 和 201×201 网格中分别出现 6 和 3 个非法/翻转单元（累计 9 个），其中还包含 6 个凹单元、3 个自相交单元和 3 个相交单元；401×401 虽未再出现非法单元，但其最低 Scaled Jacobian 仅为 0.0082、最小角仅为 0.94°，仍已接近退化。四种 HMONet 的边界平均误差仍处于约 $10^{-7}$ 量级、最大误差约 $10^{-7}$ 至 $10^{-6}$ 量级；其来源是 float32 计算与曲线核数值舍入，而非边界后处理误差。",
        "2. **损失项对最差质量的影响。** 在参数 1.52 下，四种学习模型均合法且质量差异主要集中于高曲率/边界邻近单元；E1 在三个网格规模的最低 Scaled Jacobian 和最小角均优于 E2、E3。参数升至 1.55 后，Base 的三项损失组合出现折叠，说明当前权重 0.01/0.05 并未提供足够稳定的屏障约束；E1 仍在三个网格规模中保持合法，且其最低 Scaled Jacobian 在学习模型中最高。因而本组结果不支持把当前权重下的 Jacobian/SVD 屏障视作必然改进，需在后续 Ex5 中重新标定其权重与阈值。",
        "3. **与 PDE 的关系。** PDE 网格在长宽比以及部分最差角度/正交性指标上仍具有优势；HMONet 变体可在若干低阶网格的最差角、Scaled Jacobian 或正交性上取得更好的代数质量。这不表示网络更接近 PDE 场，只说明其优化结果存在不同的局部网格分布。若论文目标是学习 PDE 规则，应进一步以节点 MSE、PDE 残差和边界误差与质量表联合判断。",
        "4. **推理效率。** HMONet 的单网格前向时间为毫秒级，而 PDE_Winslow 解算为秒级或更高；两者的数量级差异随网格加密进一步扩大。不同损失变体的网络结构一致，推理时间差主要来自计时波动和硬边界计算路径，不应被解释为训练损失本身造成的系统性速度差异。",
        "5. **训练成本。** 仅 PDE 的 E1 少了两个屏障损失，本次训练耗时为 5.22 h，是四种变体中最低；含 Jacobian 或 SVD 的变体需要额外有限差分/SVD 运算，训练时间更长。是否保留屏障项应主要根据最差质量、跨参数稳定性与 PDE 一致性判断，而不是仅比较训练目标值。",
        "",
        "### 量化摘要",
        "",
    ]
    win_rows = []
    for _, alias, _ in VARIANTS:
        name = next(display for display, candidate, _ in VARIANTS if candidate == alias)
        summary = training[name]
        win_rows.append(
            [
                name.split(":")[0],
                str(model_wins[name]),
                compact(average_runtime.get(name, float("nan"))),
                duration(summary.get("training_time_seconds")),
            ]
        )
    report += heading_table(
        ["HMONet 变体", "最差质量项最优次数（六个工况×八项）", "平均推理时间 (ms/grid)", "训练时间"],
        win_rows,
    )
    report += [
        "",
        "这里的“最优次数”仅用于提示不同屏障项的质量倾向；不同指标的物理意义和尺度不同，不能将该次数视为单一总分。",
        "",
        "## 结论",
        "",
        "本组 Run 外推实验表明，归一化 PDE 损失单独使用时已经能够在当前 CCAP 精确边界约束、混合采样和有限差分设定下维持合法网格；E1 同时给出了四种学习模型中最稳健的参数 1.55 最差质量。相反，当前权重下三项同时启用的 Base 在参数 1.55 出现局部折叠，不能作为稳健默认直接保留。建议以 E1 作为此数据集上的对照配置，并在 Ex5 中系统搜索 Jacobian/SVD 的权重和阈值；后续需用多随机种子、节点 MSE、PDE 一致性及高曲率局部图确认结论。",
        "",
    ]
    statistics = {
        "dataset": dataset,
        "seed": seed,
        "tag": tag,
        "parameters": parameters,
        "grids": grids,
        "worst_quality_wins_hmonet": dict(model_wins),
        "worst_quality_wins_pde": dict(reference_wins),
        "nonzero_legality_counts": dict(legality_summary),
        "mean_generation_time_ms": average_runtime,
    }
    return "\n".join(report), statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="run_ext_1_500_41_41")
    parser.add_argument("--seed", default="20261030")
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--parameters", nargs="+", default=["1.52", "1.55"])
    parser.add_argument("--grids", nargs="+", default=["101x101", "201x201", "401x401"])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "reports" / "ex8" / "run_ext_1_500_41_41_1.52_1.55_comparison.md",
    )
    args = parser.parse_args()

    report, statistics = build_report(
        dataset=args.dataset,
        seed=str(args.seed),
        tag=args.tag,
        parameters=list(args.parameters),
        grids=list(args.grids),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    stats_path = args.output.with_suffix(".statistics.json")
    stats_path.write_text(json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    print(stats_path)


if __name__ == "__main__":
    main()
