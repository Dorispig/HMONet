# run：rounded_u_notch_compact

深 U 槽；默认使用41x41均匀网格，圆角半径0.25，仅变化depth。

生成函数：`src/data/geometry/boundary_generator_2d.py::rounded_u_notch_boundary`。

可变参数：`depth`；函数默认值：`1.65`。默认样本数500，数据集节点规模：`41×41`。

内插训练区间：`[[0.55, 1.0], [1.3, 1.5]]`；外推训练区间：`[[0.55, 1.5]]`；外推评估参数：`[1.52, 1.55]`。

固定参数：

```json
{
  "width": 3.0,
  "height": 2.0,
  "notch_width": 0.9,
  "radius": 0.25
}
```

## 边界公式与特点

下列参数均在单位区间，角度标为 deg 的输入在求值前转换为弧度。

令 $a=(W-w)/2,b=(W+w)/2,\ell=H-d$。底边 $(Wq,0)$，左右边 $(0,Hq)$、$(W,Hq)$。
顶边按弧长分配参数，依次连接下表九段。直线端点之间线性插值；圆弧为
$O+r(\cos\theta,\sin\theta)$，角度沿表中起止值线性变化。

| 段 | 端点或圆心 | 角度（圆弧） |
|---|---|---|
| 1 | $(0,H)\to(a-r,H)$ | — |
| 2 | $(a-r,H-r)$ | $\pi/2\to0$ |
| 3 | $(a,H-r)\to(a,\ell+r)$ | — |
| 4 | $(a+r,\ell+r)$ | $\pi\to3\pi/2$ |
| 5 | $(a+r,\ell)\to(b-r,\ell)$ | — |
| 6 | $(b-r,\ell+r)$ | $-\pi/2\to0$ |
| 7 | $(b,\ell+r)\to(b,H-r)$ | — |
| 8 | $(b+r,H-r)$ | $\pi\to\pi/2$ |
| 9 | $(b+r,H)\to(W,H)$ | — |

圆弧与直线相切，但连接处曲率有跳变；深凹区对粗网格尤其敏感。


## 可视化

![可变参数与默认值](../../../outputs/boundaries/2D/run/parameter_comparison.png)

![默认参数](../../../outputs/boundaries/2D/run/default.png)

重现：在项目根目录运行 `python scripts/visualize/render_registered_boundaries.py --dimension 2 --families run`。
