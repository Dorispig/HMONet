# air_t：circle_airfoil_t

外圆与内部 NACA 四位数翼型、两条连接线组成的四边参数边界；变化参数为翼型相对厚度 t。

生成函数：`src/data/geometry/boundary_generator_2d.py::circle_airfoil_boundary`。

可变参数：`t`；函数默认值：`0.12`。默认样本数500，数据集节点规模：`41×41`。

内插训练区间：`[[0.055, 0.25], [0.35, 0.6]]`；外推训练区间：`[[0.055, 0.6]]`；外推评估参数：`[0.62, 0.65]`。

固定参数：

```json
{
  "radius": 1.0,
  "m": 0.02,
  "p": 0.4,
  "c": 1.0,
  "center": [
    0.0,
    0.0
  ]
}
```

## 边界公式与特点

下列参数均在单位区间，角度标为 deg 的输入在求值前转换为弧度。

令 $s\in[0,1]$，半厚度为
$$y_t=5tc(0.2969\sqrt{s}-0.1260s-0.3516s^2+0.2843s^3-0.1036s^4).$$
代码中的弯度为 $y_c=m(2ps-s^2)/p^2$（$s<p$），否则为
$m(1-2p+2ps-s^2)/(1-p)^2$；$\phi=\arctan[2m(p-s)/d]$，其中 $d=p^2$ 或 $(1-p)^2$。
上下表面为 $(cs\mp y_t\sin\phi+x_0,\ y_c\pm y_t\cos\phi+y_0)$。
外圆为 $(x_c+R\cos\theta,y_c^{\rm center}+R\sin\theta)$。连接线统一采用
$L(A,B;q)=(1-q)A+qB$。翼型曲线沿上下支分段取值，末项 -0.1036 对应闭合尾缘。


## 可视化

![可变参数与默认值](../../../outputs/boundaries/2D/air_t/parameter_comparison.png)

![默认参数](../../../outputs/boundaries/2D/air_t/default.png)

重现：在项目根目录运行 `python scripts/visualize/render_registered_boundaries.py --dimension 2 --families air_t`。
