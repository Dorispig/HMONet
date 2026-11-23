# rse：rect_top_semiellipse

单位矩形的上边界为半椭圆，变化参数为椭圆短半轴。

生成函数：`src/data/geometry/boundary_generator_2d.py::GenerateRectangleWithTopSemiEllipseBoundary`。

可变参数：`semi_minor_axis`；函数默认值：`0.5`。默认样本数500，数据集节点规模：`41×41`。

内插训练区间：`[[0.0, 0.2], [0.3, 0.5]]`；外推训练区间：`[[0.0, 0.4]]`；外推评估参数：`[0.42, 0.46, 0.5, 0.55, 0.6]`。

固定参数：

```json
{
  "length": 1.0,
  "height": 1.0
}
```

## 边界公式与特点

下列参数均在单位区间，角度标为 deg 的输入在求值前转换为弧度。

底边 $(Lq,0)$，左/右边 $(0,Hq)$、$(L,Hq)$，顶边为
$(L/2+(L/2)\cos[\pi(1-q)],H+b\sin[\pi(1-q)])$。
$b$ 控制半椭圆凸起高度。

## 可视化

![可变参数与默认值](../../../outputs/boundaries/2D/rse/parameter_comparison.png)

![默认参数](../../../outputs/boundaries/2D/rse/default.png)

重现：在项目根目录运行 `python scripts/visualize/render_registered_boundaries.py --dimension 2 --families rse`。
