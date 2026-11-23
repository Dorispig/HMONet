# swb_a：sidewall_deformed_block

左右侧壁按正弦剖面对称向中心收缩、沿z方向拉伸的块体；变化无量纲收缩幅值。

生成函数：`src/data/geometry/boundary_generator_3d.py::sidewall_deformed_block_boundary`。

可变参数：`sidewall_amplitude`；函数默认值：`0.15`。默认样本数500，数据集节点规模：`21×21×11`。

内插训练区间：`[[0.03, 0.1], [0.2, 0.3]]`；外推训练区间：`[[0.04, 0.3]]`；外推评估参数：`[0.02, 0.34]`。

固定参数：

```json
{
  "length_x": 3.0,
  "width_y": 2.0,
  "length_z": 2.5,
  "z_min": 0.0,
  "cap_nx": 201,
  "cap_ny": 201,
  "cap_max_iter": 1000,
  "cap_tolerance": 1e-6
}
```

## 边界公式与特点

下列参数均在单位区间，角度标为 deg 的输入在求值前转换为弧度。

原始参考映射为
$$F(u,v,w)=\left(L_x[\tfrac12+(u-\tfrac12)(1-a\sin\pi v)],\ L_yv,\ z_0+L_zw\right).$$
左右两侧面沿整个 z 方向内凹，只有 v 两端不收缩；不是局部凹坑。

## 当前正式参数面（九类统一）

上述 $F$ 定义原几何；四侧面保持 $S(q,r)=F|_{u=0,1\ \mathrm{or}\ v=0,1}$。
在原侧面的单位矩形参数域对四边做 TFI 得到恒等映射 $(q,r)$，所以不改变解析曲面的内部形状。
**不是用四条空间棱边重新构造一个不同的 Coons 侧面。**

两个端面分别在各自平面坐标内以 **201×201节点** 求二维 Winslow 方程：
$$\alpha X_{uu}-2\beta X_{uv}+\gamma X_{vv}=0,\quad
\alpha=\|X_v\|^2,\ \beta=X_u\cdot X_v,\ \gamma=\|X_u\|^2.$$
以原精确四边 $B,T,L,R$ 构造 Coons 基准面
$$C=(1-v)B(u)+vT(u)+(1-u)L(v)+uR(v)
-\sum_{i,j\in\{0,1\}}\ell_i(u)\ell_j(v)P_{ij},\quad \ell_0(t)=1-t,\ \ell_1(t)=t.$$
将稠密 PDE 节点的位移 $D_{ij}=X^{\rm PDE}_{ij}-C(u_i,v_j)$ 插值为三次张量积样条，连续端面为
$$S(u,v)=C(u,v)+\sum_{i,j}N_{i,3}(u)M_{j,3}(v)A_{ij}.$$
外圈位移控制系数为零，四条精确边不受拟合偏移。上下端面各自求解，不复用上端面的求值结果；不进行参数重分布。
每个样本保存两端面的样条控制系数、节点向量和原精确边界定义，不另存重复的201×201求解网格。
网络输入仍是数据集指定规模的六面采样点（共享棱边去重），HMONet调用同一连续参数面；体内标签由三维Winslow求解器生成。
密集检查和当前采样规模检查均通过并不等于数学上证明任意分辨率均无翻转。

## 可视化

![可变参数与默认值](../../../outputs/boundaries/3D/swb_a/parameter_comparison.png)

![默认参数](../../../outputs/boundaries/3D/swb_a/default.png)

![逻辑中截面](../../../outputs/boundaries/3D/swb_a/z_mid_comparison.png)

z_mid 表示逻辑坐标 zeta=0.5，不保证是全局 z 常数平面；图中用截面内的正交坐标 s、t 展示轮廓。截面图仅展示边界轮廓，不冒充三维 PDE 内部网格。

重现：在项目根目录运行 `python scripts/visualize/render_registered_boundaries.py --dimension 3 --families swb_a`。
