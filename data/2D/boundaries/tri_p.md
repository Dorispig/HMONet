# tri_p：symmetric_convex_tri

单位矩形的上下边界为对称三角折线，变化参数为凸起高度。

生成函数：`src/data/geometry/boundary_generator_2d.py::VerticallyAsymmetricConvexTriBoundary`。

可变参数：`protrusion_distance`；函数默认值：`0.5`。默认样本数500，数据集节点规模：`41×41`。

内插训练区间：`[[0.05, 0.2], [0.3, 0.5]]`；外推训练区间：`[[0.05, 0.4]]`；外推评估参数：`[0.42, 0.48]`。

固定参数：

```json
{
  "width": 1.0,
  "side_length": 1.0
}
```

## 边界公式与特点

下列参数均在单位区间，角度标为 deg 的输入在求值前转换为弧度。

令 $f(q)=1-|2q-1|$。底/顶边分别为 $(Wq,-pf(q))$、$(Wq,H+pf(q))$；
左右边为 $(0,Hq)$、$(W,Hq)$。$q=1/2$ 为两个尖点，须用奇数节点规模采样到尖点。

## 可视化

![可变参数与默认值](../../../outputs/boundaries/2D/tri_p/parameter_comparison.png)

![默认参数](../../../outputs/boundaries/2D/tri_p/default.png)

重现：在项目根目录运行 `python scripts/visualize/render_registered_boundaries.py --dimension 2 --families tri_p`。
