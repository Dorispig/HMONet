# esa：eccentric_semiannulus

固定内外圆半径的偏心上半环，变化参数为内圆圆心横坐标。

生成函数：`src/data/geometry/boundary_generator_2d.py::EccentricAnnularSectorBoundary`。

可变参数：`inner_center_x`；函数默认值：`0.5`。默认样本数500，数据集节点规模：`41×41`。

内插训练区间：`[[0.3, 0.45], [0.55, 0.7]]`；外推训练区间：`[[0.3, 0.7]]`；外推评估参数：`[0.72, 0.75]`。

固定参数：

```json
{
  "outer_radius": 0.5,
  "inner_radius": 0.1
}
```

## 边界公式与特点

下列参数均在单位区间，角度标为 deg 的输入在求值前转换为弧度。

令 $\theta(q)=\pi(1-q)$。底/顶边分别为
$(c+r\cos\theta,r\sin\theta)$、$(R+R\cos\theta,R\sin\theta)$。
左右边线性连接对应弧端点。要求 $|c-R|+r<R$；偏心位置控制两侧间隙。

## 可视化

![可变参数与默认值](../../../outputs/boundaries/2D/esa/parameter_comparison.png)

![默认参数](../../../outputs/boundaries/2D/esa/default.png)

重现：在项目根目录运行 `python scripts/visualize/render_registered_boundaries.py --dimension 2 --families esa`。
