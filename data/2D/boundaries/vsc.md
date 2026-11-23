# vsc：variable_width_s_channel

Bézier 变宽 S 通道；仅变化收缩控制量，弯曲控制量固定。

生成函数：`src/data/geometry/boundary_generator_2d.py::variable_width_s_channel_boundary`。

可变参数：`constriction`；函数默认值：`0.65`。默认样本数500，数据集节点规模：`41×41`。

内插训练区间：`[[0, 0.3], [0.6, 0.85]]`；外推训练区间：`[[0, 0.8]]`；外推评估参数：`[0.85, 0.9, 0.95]`。

固定参数：

```json
{
  "length": 4.0,
  "height": 0.8,
  "bend": 2.0
}
```

## 边界公式与特点

下列参数均在单位区间，角度标为 deg 的输入在求值前转换为弧度。

令 $B_i^5(q)=\binom5i q^i(1-q)^{5-i}$，
$x_i=Li/5$，$c_i=(0,0,b,-b,0,0)_i$，
$g_i=H(1,1,1-a,1-a,1,1)_i$。
上下壁分别为 $\sum_{i=0}^5B_i^5(q)(x_i,c_i+(H\pm g_i)/2)$；左右壁线性连接端点。
中心线反向弯曲，$a<1$ 保持正间隙。

## 可视化

![可变参数与默认值](../../../outputs/boundaries/2D/vsc/parameter_comparison.png)

![默认参数](../../../outputs/boundaries/2D/vsc/default.png)

重现：在项目根目录运行 `python scripts/visualize/render_registered_boundaries.py --dimension 2 --families vsc`。
