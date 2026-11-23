# gsp_a：gear_spline

四条三次 B 样条波纹边界，端部波幅平滑减弱；固定6周期。

生成函数：`src/data/geometry/boundary_generator_2d.py::gear_spline_boundary`。

可变参数：`amplitude`；函数默认值：`0.16`。默认样本数500，数据集节点规模：`41×41`。

内插训练区间：`[[0.02, 0.1], [0.15, 0.2]]`；外推训练区间：`[[0.02, 0.18]]`；外推评估参数：`[0.19, 0.2]`。

固定参数：

```json
{
  "width": 2.0,
  "height": 1.6,
  "teeth": 6,
  "controls_per_wave": 16
}
```

## 边界公式与特点

下列参数均在单位区间，角度标为 deg 的输入在求值前转换为弧度。

令 $n=k m+1$（齿数 $k$，每波控制点数 $m$），$u_i=i/(n-1)$，
$o_i=a\sin(2\pi k u_i)\sin^2(\pi u_i)$。
四壁的三次 B 样条控制点依次为 $(Wu_i,-o_i)$、$(W+o_i,Hu_i)$、
$(Wu_i,H+o_i)$、$(-o_i,Hu_i)$。曲线 $C(q)=\sum_i N_{i,3}(q)P_i$，
节点为端点重复四次的均匀夹持节点。$a$ 是控制点偏移幅度，实际曲线幅度略小。

## 可视化

![可变参数与默认值](../../../outputs/boundaries/2D/gsp_a/parameter_comparison.png)

![默认参数](../../../outputs/boundaries/2D/gsp_a/default.png)

重现：在项目根目录运行 `python scripts/visualize/render_registered_boundaries.py --dimension 2 --families gsp_a`。
