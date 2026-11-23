# cah_t：cylindrical_domain_with_airfoil_hole

圆柱外域与贯穿轴向的NACA4翼型孔；变化翼型相对厚度。

生成函数：`src/data/geometry/boundary_generator_3d.py::cylindrical_domain_with_airfoil_hole`。

可变参数：`thickness`；函数默认值：`0.12`。默认样本数500，数据集节点规模：`21×21×11`。

内插训练区间：`[[0.06, 0.15], [0.25, 0.35]]`；外推训练区间：`[[0.08, 0.34]]`；外推评估参数：`[0.06, 0.38]`。

固定参数：

```json
{
  "outer_radius": 1.5,
  "chord": 1.0,
  "camber": 0.02,
  "camber_position": 0.4,
  "length": 2.5,
  "z_min": 0.0,
  "cap_nx": 201,
  "cap_ny": 201,
  "cap_max_iter": 1000,
  "cap_tolerance": 1e-6
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

设 $I(v)$ 为尾缘→上表面→前缘→下表面→尾缘的翼型轮廓，$O(v)=R(\cos2\pi v,\sin2\pi v)$。
原始参考映射 $F(u,v,w)=((1-u)I(v)+uO(v),z_0+Lw)$。
四个侧面是该映射在 $u=0,1$ 或 $v=0,1$ 上的限制，两个参数接缝几何重合。

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

![可变参数与默认值](../../../outputs/boundaries/3D/cah_t/parameter_comparison.png)

![默认参数](../../../outputs/boundaries/3D/cah_t/default.png)

![逻辑中截面](../../../outputs/boundaries/3D/cah_t/z_mid_comparison.png)

z_mid 表示逻辑坐标 zeta=0.5，不保证是全局 z 常数平面；图中用截面内的正交坐标 s、t 展示轮廓。截面图仅展示边界轮廓，不冒充三维 PDE 内部网格。

重现：在项目根目录运行 `python scripts/visualize/render_registered_boundaries.py --dimension 3 --families cah_t`。
