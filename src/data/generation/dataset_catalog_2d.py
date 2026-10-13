"""Generate an auditable Chinese catalog from saved dataset metadata."""

import inspect
import json
from pathlib import Path
from src.data.dataset import (
    read_dataset_metadata,
    NUMERICAL_DATA_FILENAME,
    BOUNDARY_GEOMETRY_FILENAME,
)
from src.data.geometry.boundary_generator_2d import BOUNDARY_GENERATORS

PARAMETERS = {
    "length": ("通道长度；rse 中为底边长度", "正数"),
    "height": ("基础高度或入口出口竖向宽度", "正数；须满足深度约束"),
    "width": ("基础矩形宽度", "正数"),
    "depth": (
        "每侧凹陷深度（scc）或顶部槽深度（run）",
        "scc: 0<=depth<height/2；run: 0<depth<height",
    ),
    "bend": (
        "S 中心线 Bézier 控制点偏移，不等于最大实际位移",
        "有限实数；vsc固定2，vsb变化bend",
    ),
    "constriction": (
        "S 通道宽度控制点收缩量；最小竖向间隙=height*(1-5*constriction/8)",
        "0<=constriction<1",
    ),
    "teeth": ("每条边波纹控制波形周期数", "正整数"),
    "amplitude": (
        "波纹控制点偏移；实际样条幅值略有衰减",
        "0<=amplitude<0.2*min(width,height)",
    ),
    "controls_per_wave": ("每周期控制点分辨率，非几何变化参数", "整数>=12"),
    "notch_width": ("U 槽两竖直侧壁的间距", "0<notch_width<width"),
    "radius": (
        "外圆半径（airfoil）或U槽圆角半径",
        "外圆须包含翼型；U槽: 0<r<min((width-notch_width)/2,notch_width/2,depth/2)",
    ),
    "m": ("NACA 最大相对弯度", "0<=m；本批固定0.02"),
    "p": ("NACA 最大弯度位置/弦长", "0<p<1"),
    "t": ("NACA 相对厚度", "正数且翼型在外圆内"),
    "c": ("NACA 弦长", "正数且翼型在外圆内"),
    "center": ("外圆中心二维坐标", "两个有限实数"),
    "semi_minor_axis": ("顶部半椭圆的竖向半轴", ">=0"),
    "outer_radius": ("外半圆半径", ">inner_radius>0"),
    "inner_radius": ("内半圆半径", "0<inner_radius<outer_radius"),
    "inner_center_x": (
        "内半圆圆心横坐标",
        "abs(inner_center_x-outer_radius)+inner_radius<outer_radius",
    ),
    "side_length": ("三角边界两侧直线长度", "正数"),
    "protrusion_distance": (
        "上下三角形凸出高度；负值向内凹",
        "side_length+2*min(0,h)>0",
    ),
}


def write_catalog(root, families, descriptions):
    root = Path(root)
    records = []
    for directory in root.iterdir():
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        try:
            m = read_dataset_metadata(directory)
        except (OSError, ValueError, KeyError):
            continue
        audit_path = directory / "quality_audit.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            m["audit_illegal_samples"] = audit["illegal_samples"]
        if m.get("dimension") == 2:
            records.append(
                (m.get("boundary_type", m.get("family", "unknown")), directory, m)
            )
    records.sort(key=lambda x: (x[0], x[1].name))
    text = [
        "# 二维数据集说明",
        "",
        "按边界完整名称排序；只列出当前磁盘上实际存在的数据集。",
        "",
        "## 保存格式与对应关系",
        "",
        f"每个数据集含 `{NUMERICAL_DATA_FILENAME}`（边界均匀采样点、PDE网格）和 `{BOUNDARY_GEOMETRY_FILENAME}`（参数边界描述和参数）。同索引 sample_id、dataset_id 必须对应。",
        "上下边界参数方向为左到右，左右为下到上；闭合遍历时上边界和左边界反向。每条边均匀参数采样，包含端点，闭合拼接处去重。",
        "训练几何参数等间隔采样并包含区间端点；新数据的两个内插训练区间各500条，外推训练区间1000条。网格真值是 Winslow PDE 迭代结果（TFI仅用于初始化），mesh_points 形状为 (ny*nx, 2)。",
        "PDE收敛仅代表迭代变化达到容差，并不保证网格合法。新数据的每样本 pde_diagnostics 记录非正角点行列式单元数；该检查不包括全局单元间相交。另存的 quality_audit.json 是逐条完整合法性验收：凸单元内部重叠通过分离轴检测，或在所有角点行列式为正且外边界简单时使用全局单射性判定。",
        "",
        "## 数据集清单",
        "",
        "| 边界 | 数据集 | 数量 | 网格 | 变化参数及范围 |",
        "|---|---|---:|---|---|",
    ]
    for family, directory, m in records:
        text.append(
            f"| {family} | [{directory.name}](#{directory.name}) | {m.get('size')} | {m.get('grid_shape')} | {m.get('parameter_ranges', m.get('t_ranges'))} |"
        )
    text[4:4] = (
        ["## 目录", ""]
        + [f"- [{directory.name}](#{directory.name})" for _, directory, _ in records]
        + [""]
    )
    flagged = [
        (
            directory.name,
            m.get(
                "audit_illegal_samples",
                m.get("samples_with_nonpositive_corner_cells", 0),
            ),
        )
        for _, directory, m in records
        if m.get(
            "audit_illegal_samples", m.get("samples_with_nonpositive_corner_cells", 0)
        )
        > 0
    ]
    if flagged:
        text[4:4] = [
            "**重要：以下数据集虽 PDE 收敛，但存在局部非法网格，不建议直接用作监督训练真值：** "
            + "；".join(f"{name}：{count}条" for name, count in flagged)
            + "。",
            "",
        ]
    previous = None
    for family, directory, m in records:
        if family != previous:
            text += [
                "",
                f"## {family}",
                "",
                descriptions.get(family, m.get("description", "")),
                "",
            ]
            if family == "rounded_u_notch":
                text += [
                    "版本说明：61×21 小规模数据的圆角半径为0.25；当前可用数据以本目录实际列出的数据集为准，训练与评估应使用对应的深度范围和固定几何参数。",
                    "",
                ]
            constructor = BOUNDARY_GENERATORS.get(family)
            if constructor:
                text += [
                    f"边界函数：`src/data/geometry/boundary_generator_2d.py::{constructor.__name__}`。",
                    "",
                    "| 参数 | 定义 | 函数默认值 | 合法范围/约束 |",
                    "|---|---|---|---|",
                ]
                for key, value in inspect.signature(constructor).parameters.items():
                    meaning, limits = PARAMETERS.get(
                        key, ("见边界函数定义", "见边界构造校验")
                    )
                    text.append(f"| {key} | {meaning} | `{value.default}` | {limits} |")
            matches = [(k, v) for k, v in families.items() if v["function"] == family]
            for k, v in matches:
                text += [
                    "",
                    f"生成函数：`generate_dataset('{k}', split, size=1000)`；简称 `{v['output_name']}`。",
                ]
            previous = family
        text += [
            "",
            f'<a id="{directory.name}"></a>',
            f"### {directory.name}",
            "",
            f"- 类型：{m.get('split')}；数量：{m.get('size')}；网格：{m.get('grid_shape')}。",
            f"- 固定几何参数：`{json.dumps(m.get('fixed_parameters',{}),ensure_ascii=False)}`。",
            f"- 变化参数与训练范围：`{json.dumps(m.get('parameter_ranges',m.get('t_ranges',{})),ensure_ascii=False)}`。",
            f"- PDE：最大迭代 {m.get('pde_max_iter',m.get('winslow_max_iter'))}，容差 {m.get('pde_tolerance',m.get('winslow_tolerance'))}，全部收敛 {m.get('all_pde_converged','未记录')}。",
            f"- 参数采样：`{m.get('parameter_sampling','见metadata')}`。",
        ]
        if "audit_illegal_samples" in m:
            text.append(
                f"- 全量合法性检查：**{m['audit_illegal_samples']}/{m['size']} 条含非法单元**；[逐条诊断]({directory.name}/quality_audit.json)。凸单元采用分离轴检测内部重叠；非凸单元已作为非法单元记录，不将接缝接触记为重叠。"
            )
        if "samples_with_nonpositive_corner_cells" in m:
            text.append(
                f"- **含非正角点行列式单元的样本：{m['samples_with_nonpositive_corner_cells']}/{m['size']}**。出现此项时不可将收敛等同于有效训练真值。"
            )
        text += [
            f"- [首样本网格]({directory.name}/example/first.png)；[中间样本网格]({directory.name}/example/middle.png)；[末样本网格]({directory.name}/example/last.png)。",
            f"- [网格与采样点]({directory.name}/{NUMERICAL_DATA_FILENAME})；[参数边界]({directory.name}/{BOUNDARY_GEOMETRY_FILENAME})。",
        ]
    destination = root / "README.md"
    temporary = root / "README.catalog.tmp"
    temporary.write_text("\n".join(text) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
