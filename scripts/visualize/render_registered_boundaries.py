"""Render the nine registered families per dimension and their reproducible notes."""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.data.generation.make_dataset_parametric_2d import DATASET_FAMILIES as TWO, DATASET_GENERATORS
from src.data.generation.make_dataset_parametric_3d import DATASET_FAMILIES as THREE
from src.data.geometry.boundary_generator_3d import BOUNDARY_GENERATORS
from scripts.visualize.render_boundary_family_catalog import PAPER_2D

TWO = {name: dict(TWO[name]) for name in PAPER_2D}
_cases = yaml.safe_load((ROOT/'configs/2D/evaluate.yaml').read_text(encoding='utf-8'))['cases']
for _spec in TWO.values():
    _case = _cases[_spec['output_name']+'_ext_1_500_41_41']
    _spec['evaluation_values'] = {'extrapolation': _case['values']}

NACA = r'''令 $s\in[0,1]$，半厚度为
$$y_t=5tc(0.2969\sqrt{s}-0.1260s-0.3516s^2+0.2843s^3-0.1036s^4).$$
代码中的弯度为 $y_c=m(2ps-s^2)/p^2$（$s<p$），否则为
$m(1-2p+2ps-s^2)/(1-p)^2$；$\phi=\arctan[2m(p-s)/d]$，其中 $d=p^2$ 或 $(1-p)^2$。
上下表面为 $(cs\mp y_t\sin\phi+x_0,\ y_c\pm y_t\cos\phi+y_0)$。
外圆为 $(x_c+R\cos\theta,y_c^{\rm center}+R\sin\theta)$。连接线统一采用
$L(A,B;q)=(1-q)A+qB$。翼型曲线沿上下支分段取值，末项 -0.1036 对应闭合尾缘。
'''
RUN = r'''令 $a=(W-w)/2,b=(W+w)/2,\ell=H-d$。底边 $(Wq,0)$，左右边 $(0,Hq)$、$(W,Hq)$。
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
'''
FORMULAS = {
    'air_t': NACA,
    'esa': r'''令 $\theta(q)=\pi(1-q)$。底/顶边分别为
$(c+r\cos\theta,r\sin\theta)$、$(R+R\cos\theta,R\sin\theta)$。
左右边线性连接对应弧端点。要求 $|c-R|+r<R$；偏心位置控制两侧间隙。''',
    'rse': r'''底边 $(Lq,0)$，左/右边 $(0,Hq)$、$(L,Hq)$，顶边为
$(L/2+(L/2)\cos[\pi(1-q)],H+b\sin[\pi(1-q)])$。
$b$ 控制半椭圆凸起高度。''',
    'run': RUN,
    'scc': r'''令 $f(q)=64q^3(1-q)^3$。底/顶边分别为
$(Lq,df(q))$、$(Lq,H-df(q))$，左右边为竖直线。
最窄间隙 $H-2d>0$。端点一、二阶导数与直线光滑衔接。''',
    'vsc': r'''令 $B_i^5(q)=\binom5i q^i(1-q)^{5-i}$，
$x_i=Li/5$，$c_i=(0,0,b,-b,0,0)_i$，
$g_i=H(1,1,1-a,1-a,1,1)_i$。
上下壁分别为 $\sum_{i=0}^5B_i^5(q)(x_i,c_i+(H\pm g_i)/2)$；左右壁线性连接端点。
中心线反向弯曲，$a<1$ 保持正间隙。''',
    'tri_p': r'''令 $f(q)=1-|2q-1|$。底/顶边分别为 $(Wq,-pf(q))$、$(Wq,H+pf(q))$；
左右边为 $(0,Hq)$、$(W,Hq)$。$q=1/2$ 为两个尖点，须用奇数节点规模采样到尖点。''',
    'gsp_a': r'''令 $n=k m+1$（齿数 $k$，每波控制点数 $m$），$u_i=i/(n-1)$，
$o_i=a\sin(2\pi k u_i)\sin^2(\pi u_i)$。
四壁的三次 B 样条控制点依次为 $(Wu_i,-o_i)$、$(W+o_i,Hu_i)$、
$(Wu_i,H+o_i)$、$(-o_i,Hu_i)$。曲线 $C(q)=\sum_i N_{i,3}(q)P_i$，
节点为端点重复四次的均匀夹持节点。$a$ 是控制点偏移幅度，实际曲线幅度略小。''',
    'fca_a': r'''令 $\alpha=\pi a/180$，$R=c/[2\sin(\alpha/2)]$，$d=R\cos(\alpha/2)$。
四个圆心为 $(c/2,d)$、$(c-d,c/2)$、$(c/2,c-d)$、$(d,c/2)$。
各圆弧连接正方形对应两端点；底、右边角度增加 $\alpha$，顶、左边角度减少 $\alpha$。
$C(q)=O+R(\cos(\theta_0\pm q\alpha),\sin(\theta_0\pm q\alpha))$，
$\theta_0=\operatorname{atan2}(P_{0y}-O_y,P_{0x}-O_x)$。
大于90度时连接点附近存在凹角。外推训练区间为30°–120°；120°的41×41节点PDE网格检测到4个非法凹单元。生成器保存全部样本，并在生成结束后汇总首尾参数的合法性警告。''',
    'cah_t': NACA + r'''
设 $I(v)$ 为尾缘→上表面→前缘→下表面→尾缘的翼型轮廓，$O(v)=R(\cos2\pi v,\sin2\pi v)$。
原始参考映射 $F(u,v,w)=((1-u)I(v)+uO(v),z_0+Lw)$。
四个侧面是该映射在 $u=0,1$ 或 $v=0,1$ 上的限制，两个参数接缝几何重合。''',
    'swb_a': r'''原始参考映射为
$$F(u,v,w)=\left(L_x[\tfrac12+(u-\tfrac12)(1-a\sin\pi v)],\ L_yv,\ z_0+L_zw\right).$$
左右两侧面沿整个 z 方向内凹，只有 v 两端不收缩；不是局部凹坑。''',
    'run_e': RUN + r'''
记上面顶边为 $T(u)$，底边 $B(u)=(Wu,0)$。原始参考映射
$F(u,v,w)=((1-v)B(u)+vT(u),z_0+Lw)$ 仅用于定义精确侧面和端面的四条边。
最终端面由下面的二维 PDE + 样条式确定，体内由三维 PDE 求解；不将二维 PDE 网格直接拉伸成标签。''',
    'teb_b': r'''令 $O(v)$ 从 $(A,-B)$ 出发，依次沿
$(A,B),(-A,B),(-A,-B),(A,-B)$ 线性行进，每条边分配四分之一参数区间（不是按物理弧长分配）。
$d(v)=(O_x/A,O_y/B)/\|(O_x/A,O_y/B)\|$；$b(w)=(1-w)b_0+wb_1$；
$I(v,w)=(a d_x,b(w)d_y)$。
$$F(u,v,w)=((1-u)I(v,w)+uO(v),z_0+Lw).$$
矩形外边界直接用直线段参数化，不用射线求交；上述归一化方向仅用于椭圆孔上对应点。''',
    'bad_a': r'''令 $r=(1-u)r_i+ur_o$，$\theta=2\pi v$，$\beta=(\pi a/180)w$。
$$F(u,v,w)=((R-r\cos\theta)\sin\beta,\ -r\sin\theta,\ z_0+R(1-\cos\beta)+r\cos\theta\cos\beta).$$
两端面为不同朝向的平面，分别建立各自局部二维坐标求解 PDE。''',
    'lad_a': r'''令 $\theta=2\pi v$，$\phi=(\pi\omega/180)w$，
$r_o(v,w)=R_o[1+a\cos(k\theta+\phi)]$，$r=(1-u)R_i+ur_o(v,w)$。
$$F(u,v,w)=(r\cos\theta,r\sin\theta,z_0+Lw).$$
外壁保留波瓣和轴向相位旋转，不用空间 Coons 平面替代。''',
    'srt_r': r'''令 $s=2u-1,t=2v-1$，$D=(s\sqrt{1-t^2/2},t\sqrt{1-s^2/2})$，
$h(w)=w^2(3-2w)$。
$$F(u,v,w)=((1-h)(As,Bt)+h rD,\ z_0+Lw).$$
入口矩形，出口圆形；侧面保留光滑过渡原几何。''',
    'sad_c': r'''令 $s=2u-1,t=2v-1$，$D=(s\sqrt{1-t^2/2},t\sqrt{1-s^2/2})$，
$C(w)=(a\sin2\pi w,0,z_0+Lw)$，$N(w)=(L,0,-2\pi a\cos2\pi w)/\sqrt{L^2+(2\pi a\cos2\pi w)^2}$，
$k(w)=\sqrt{1-w+\rho w}$。
$$F(u,v,w)=C(w)+A k(w)D_xN(w)+B k(w)D_y(0,1,0).$$
沿 S 形中心线扫掠椭圆截面，$\rho$ 是出口/入口面积比。''',
    'tbp_a': r'''令 $x=cu$，$\gamma=(\pi a/180)(w-1/2)$，
$m=b\sin\pi u+(x-c/2)\tan\gamma$，$d=t\sin\pi u$。
上下叶片面在截面中的位置为 $y_-=m-p/2+d$，$y_+=m+p/2-d$。
$$F(u,v,w)=(x,(1-v)y_-+vy_+,z_0+Hw).$$
这是单个叶片间流道；随展向改变安装角，未计算或输入曲率、法向特征。''',
}

CAP_NOTE = r'''
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
'''


def draw_2d(ax, boundary, color):
    for curve in boundary.curves().values():
        p = curve.sample(801)
        ax.plot(p[:, 0], p[:, 1], color=color, lw=1.25)
    ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.grid(alpha=.15)


def draw_3d(ax, boundary, color, section=False):
    arrays=[]
    if section:
        q=np.linspace(0,1,401)
        for name in ('x_min','x_max','y_min','y_max'):
            arrays.append(getattr(boundary,name).evaluate(q,np.full_like(q,.5)))
        points=np.concatenate(arrays)
        center=points.mean(axis=0)
        _,_,vh=np.linalg.svd(points-center,full_matrices=False)
        normal=vh[-1]
        normal*=1 if normal[np.argmax(np.abs(normal))]>=0 else -1
        basis=np.eye(3)[np.argmin(np.abs(normal))]
        horizontal=basis-normal*np.dot(basis,normal)
        horizontal/=np.linalg.norm(horizontal)
        vertical=np.cross(normal,horizontal)
        for points in arrays:
            local=(points-center)@np.stack([horizontal,vertical],axis=1)
            ax.plot(local[:,0],local[:,1],color=color,lw=1.4)
        ax.set_aspect('equal');ax.set_xlabel('Local s');ax.set_ylabel('Local t')
        ax.grid(alpha=.15)
        return
    else:
        for surface in boundary.surfaces().values():
            p=surface.sample(41,41);arrays.append(p.reshape(-1,3))
            ax.plot_wireframe(*np.moveaxis(p,-1,0),rstride=4,cstride=4,color=color,lw=.42,alpha=.8)
    extent=np.ptp(np.concatenate(arrays),axis=0)
    ax.set_box_aspect(np.maximum(extent,.1*extent.max()))
    ax.view_init(elev=24,azim=-58)
    ax.set_axis_off()


def relative_link(target, directory):
    """Link documentation to figures without moving the generated images."""
    try:
        return Path(os.path.relpath(target, directory)).as_posix()
    except ValueError:  # Custom figure/document roots may be on different drives.
        return Path(target).resolve().as_posix()


def build(dimension, family, spec, output, dpi, documentation_root=None):
    constructor=(DATASET_GENERATORS if dimension==2 else BOUNDARY_GENERATORS)[spec['function']]
    variable=spec['parameter']
    default=float(inspect.signature(constructor).parameters[variable].default)
    low,high=spec['extrapolation'][0][0],spec['extrapolation'][-1][-1]
    values=list(dict.fromkeys([low,default,high,*spec['evaluation_values']['extrapolation']]))
    folder=output/f'{dimension}D'/spec['output_name'];folder.mkdir(parents=True,exist_ok=True)
    documentation_root = ROOT/'data' if documentation_root is None else Path(documentation_root)
    document = documentation_root/f'{dimension}D'/'boundaries'/f'{spec["output_name"]}.md'
    document.parent.mkdir(parents=True,exist_ok=True)
    objects=[]
    for value in values:
        objects.append(constructor(**dict(spec['fixed_parameters'],**{variable:value})))
    colors=['#245b91','#b45124','#278078','#765390','#4e6879']
    def render(filename, indices, section=False):
        fig=plt.figure(figsize=(4.4*len(indices),4.2),layout='constrained')
        for j,i in enumerate(indices,1):
            ax=fig.add_subplot(1,len(indices),j,projection='3d' if dimension==3 and not section else None)
            if dimension==2:draw_2d(ax,objects[i],colors[i%len(colors)])
            else:draw_3d(ax,objects[i],colors[i%len(colors)],section)
            tag=' (default)' if values[i]==default else ''
            ax.set_title(f'{spec["output_name"]}: {variable}={values[i]:g}{tag}',fontsize=10)
        fig.savefig(folder/filename,dpi=dpi);plt.close(fig)
    render('parameter_comparison.png',list(range(len(values))))
    render('default.png',[values.index(default)])
    if dimension==3:render('z_mid_comparison.png',list(range(len(values))),True)
    for i,value in enumerate(values):render('parameter_'+format(value,'.8g').replace('.','p')+'.png',[i])
    source=Path(inspect.getsourcefile(constructor)).relative_to(ROOT).as_posix()
    note=f'# {spec["output_name"]}：{family}\n\n{spec["description"]}\n\n'
    note+=f'生成函数：`{source}::{constructor.__name__}`。\n\n'
    note+=f'可变参数：`{variable}`；函数默认值：`{default:g}`。默认样本数500，数据集节点规模：`'+ '×'.join(map(str,spec['grid_shape']))+'`。\n\n'
    note+=f'内插训练区间：`{spec["interpolation"]}`；外推训练区间：`{spec["extrapolation"]}`；外推评估参数：`{spec["evaluation_values"]["extrapolation"]}`。\n\n'
    note+='固定参数：\n\n```json\n'+json.dumps(spec['fixed_parameters'],ensure_ascii=False,indent=2)+'\n```\n\n'
    note+='## 边界公式与特点\n\n下列参数均在单位区间，角度标为 deg 的输入在求值前转换为弧度。\n\n'+FORMULAS[spec['output_name']]+'\n'
    if dimension==3:note+=CAP_NOTE
    note+='\n## 可视化\n\n![可变参数与默认值]('+relative_link(folder/'parameter_comparison.png',document.parent)+')\n\n![默认参数]('+relative_link(folder/'default.png',document.parent)+')\n'
    if dimension==3:note+='\n![逻辑中截面]('+relative_link(folder/'z_mid_comparison.png',document.parent)+')\n\nz_mid 表示逻辑坐标 zeta=0.5，不保证是全局 z 常数平面；图中用截面内的正交坐标 s、t 展示轮廓。截面图仅展示边界轮廓，不冒充三维 PDE 内部网格。\n'
    note+='\n重现：在项目根目录运行 `python scripts/visualize/render_registered_boundaries.py --dimension '+str(dimension)+' --families '+spec['output_name']+'`。\n'
    document.write_text(note,encoding='utf-8')
    return objects[values.index(default)]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dimension',choices=['2','3','both'],default='both')
    p.add_argument('--families',nargs='+',default=['all'])
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs/boundaries')
    p.add_argument('--documentation-root',type=Path,default=ROOT/'data',help='Formula notes: <root>/<dimension>D/boundaries/<family>.md')
    p.add_argument('--dpi',type=int,default=220)
    args=p.parse_args(argv)
    for dim,registry in ((2,TWO),(3,THREE)):
        if args.dimension not in ('both',str(dim)):continue
        selected=[(name,spec) for name,spec in registry.items() if args.families==['all'] or name in args.families or spec['output_name'] in args.families]
        defaults=[]
        for name,spec in selected:
            print(f'Rendering {dim}D {spec["output_name"]}',flush=True)
            defaults.append((spec,build(dim,name,spec,args.output_root,args.dpi,args.documentation_root)))
        if len(defaults)==9:
            for section in ([False,True] if dim==3 else [False]):
                fig=plt.figure(figsize=(13.5,12),layout='constrained')
                for i,(spec,boundary) in enumerate(defaults,1):
                    ax=fig.add_subplot(3,3,i,projection='3d' if dim==3 and not section else None)
                    if dim==2:draw_2d(ax,boundary,'#245b91')
                    else:draw_3d(ax,boundary,'#245b91',section)
                    ax.set_title(spec['output_name'])
                filename=f'overview_{dim}D'+('_z_mid' if section else '')+'.png'
                fig.savefig(args.output_root/filename,dpi=args.dpi);plt.close(fig)
    index='# 当前十八类参数边界\n\n此目录保存边界图片；公式说明保存在data对应维度目录，链接见下表。三维另含逻辑 z_mid 截面。\n\n'
    index+='九宫格总览：[二维](overview_2D.png)、[三维立体](overview_3D.png)、[三维中截面](overview_3D_z_mid.png)。\n\n'
    for dim,registry in ((2,TWO),(3,THREE)):
        index+=f'## {dim}D\n\n'
        index+='\n'.join(f'- [{s["output_name"]}]('+relative_link(args.documentation_root/f'{dim}D'/'boundaries'/f'{s["output_name"]}.md',args.output_root)+f')：{s["description"]}' for s in registry.values())+'\n\n'
    args.output_root.mkdir(parents=True,exist_ok=True)
    (args.output_root/'README.md').write_text(index,encoding='utf-8')


if __name__=='__main__':main()
