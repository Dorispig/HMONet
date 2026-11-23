# run_e：rounded_u_notch_extrusion

局部质量诊断：[depth=1.52、41×41×11 的凹六面体局部图](../../../outputs/boundaries/3D/run_e/depth1p52_41x41x11_concave_hex_detail.png)。该网格有16个角点Jacobian非正的单元，均有凹面；不能将该统计直接等同于16个相交单元。逐单元诊断见 [depth1p52_hex_diagnostics.json](../../../outputs/boundaries/3D/run_e/depth1p52_hex_diagnostics.json)。

加密到81×81×11后，depth=1.52和1.55的端面检查通过，PDE体网格翻转单元数均为0，最小scaled Jacobian分别约0.1093和0.1004。该结果仅针对这两个参数和此规模。

二维 run 精确轮廓；201x201二维PDE连续端面，内部由三维PDE求解；仅变化槽深。

生成函数：`src/data/geometry/boundary_generator_3d.py::rounded_u_notch_extrusion_boundary`。

可变参数：`depth`；函数默认值：`1`。默认样本数500，数据集节点规模：`41×41×11`。

内插训练区间：`[[0.55, 1.0], [1.3, 1.5]]`；外推训练区间：`[[0.55, 1.5]]`；外推评估参数：`[1.52, 1.55]`。

固定参数：

```json
{
  "width": 3.0,
  "height": 2.0,
  "notch_width": 0.9,
  "radius": 0.25,
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

记上面顶边为 $T(u)$，底边 $B(u)=(Wu,0)$。原始参考映射
$F(u,v,w)=((1-v)B(u)+vT(u),z_0+Lw)$ 仅用于定义精确侧面和端面的四条边。
最终端面由下面的二维 PDE + 样条式确定，体内由三维 PDE 求解；不将二维 PDE 网格直接拉伸成标签。

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

![可变参数与默认值](../../../outputs/boundaries/3D/run_e/parameter_comparison.png)

![默认参数](../../../outputs/boundaries/3D/run_e/default.png)

![逻辑中截面](../../../outputs/boundaries/3D/run_e/z_mid_comparison.png)

z_mid 表示逻辑坐标 zeta=0.5，不保证是全局 z 常数平面；图中用截面内的正交坐标 s、t 展示轮廓。截面图仅展示边界轮廓，不冒充三维 PDE 内部网格。

重现：在项目根目录运行 `python scripts/visualize/render_registered_boundaries.py --dimension 3 --families run_e`。

run_e默认规模按要求改为41×41×11。实测槽深0.55、1.50的端面及体网格通过；1.52存在16个翻转体单元，1.55的两个端面各有2个内凹四边形。此设置不表示所有评估参数均合法，既有端面合法性拦截仍生效；61×21×11的通过记录保留在核验报告中。
