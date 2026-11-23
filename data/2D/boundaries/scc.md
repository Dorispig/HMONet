# scc：smooth_constricted_channel

上下光滑相向凹陷的通道；最小间隙 height-2*depth。

生成函数：`src/data/geometry/boundary_generator_2d.py::smooth_constricted_channel_boundary`。

可变参数：`depth`；函数默认值：`0.4`。默认样本数500，数据集节点规模：`41×41`。

内插训练区间：`[[0.05, 0.25], [0.35, 0.42]]`；外推训练区间：`[[0.05, 0.4]]`；外推评估参数：`[0.42, 0.45, 0.47]`。

固定参数：

```json
{
  "length": 3.0,
  "height": 1.0
}
```

## 边界公式与特点

下列参数均在单位区间，角度标为 deg 的输入在求值前转换为弧度。

令 $f(q)=64q^3(1-q)^3$。底/顶边分别为
$(Lq,df(q))$、$(Lq,H-df(q))$，左右边为竖直线。
最窄间隙 $H-2d>0$。端点一、二阶导数与直线光滑衔接。

## 可视化

![可变参数与默认值](../../../outputs/boundaries/2D/scc/parameter_comparison.png)

![默认参数](../../../outputs/boundaries/2D/scc/default.png)

重现：在项目根目录运行 `python scripts/visualize/render_registered_boundaries.py --dimension 2 --families scc`。
