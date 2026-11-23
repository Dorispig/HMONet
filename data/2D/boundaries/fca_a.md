# fca_a：four_convex_arcs

四条具有相同弦长和圆心角的外凸圆弧；变化参数为每条圆弧的圆心角。

生成函数：`src/data/geometry/boundary_generator_2d.py::four_convex_semicircle_boundary`。

可变参数：`arc_angle_deg`；函数默认值：`90`。默认样本数500，数据集节点规模：`41×41`。

内插训练区间：`[[30.0, 50.0], [70.0, 90.0]]`；外推训练区间：`[[30.0, 120.0]]`；外推评估参数：`[125.0, 130.0]`。

固定参数：

```json
{
  "chord": 1.0
}
```

## 边界公式与特点

下列参数均在单位区间，角度标为 deg 的输入在求值前转换为弧度。

令 $\alpha=\pi a/180$，$R=c/[2\sin(\alpha/2)]$，$d=R\cos(\alpha/2)$。
四个圆心为 $(c/2,d)$、$(c-d,c/2)$、$(c/2,c-d)$、$(d,c/2)$。
各圆弧连接正方形对应两端点；底、右边角度增加 $\alpha$，顶、左边角度减少 $\alpha$。
$C(q)=O+R(\cos(\theta_0\pm q\alpha),\sin(\theta_0\pm q\alpha))$，
$\theta_0=\operatorname{atan2}(P_{0y}-O_y,P_{0x}-O_x)$。
大于90度时连接点附近存在凹角。外推训练区间为30°–120°；120°的41×41节点PDE网格检测到4个非法凹单元。生成器保存全部样本，并在生成结束后汇总首尾参数的合法性警告。

## 可视化

![可变参数与默认值](../../../outputs/boundaries/2D/fca_a/parameter_comparison.png)

![默认参数](../../../outputs/boundaries/2D/fca_a/default.png)

重现：在项目根目录运行 `python scripts/visualize/render_registered_boundaries.py --dimension 2 --families fca_a`。
