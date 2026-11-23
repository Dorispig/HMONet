"""Process-local projection optimization. Does not edit any project source.

FDLayout is explicit producer metadata: never infer stencil order from length.
Only center-difference 19-point 3D templates are supported. Query-autograd uses
the unreduced path. Boundary representations and spline coefficients unchanged.
"""
import sys
sys.path.insert(0,'E:/projects/HMONet')
from contextlib import contextmanager
from functools import lru_cache
import torch
from src.data.backend import parametric_torch as backend
from src.models import model as model_module
from src.data.geometry.pde_cap_3d import RunPdeFace3D
from src.numerics.differential import stencil_offsets

BASE_FACES=backend.evaluate_boundary_surfaces_torch
BASE_EDGES=backend.evaluate_selected_boundary_surfaces_torch
MODE='base';LAYOUT=None

@lru_cache(None)
def representatives(axes,device):
    offsets=stencil_offsets(3,device='cpu',dtype=torch.float64).tolist()
    unique=[];first=[];inverse=[]
    for i,o in enumerate(offsets):
        key=tuple(o[a] for a in axes)
        if key not in unique:unique.append(key);first.append(i)
        inverse.append(unique.index(key))
    return torch.tensor(first,device=device),torch.tensor(inverse,device=device)

class FDLayout:
    def __init__(self,queries):
        if queries.ndim!=4 or queries.shape[-2:]!=(19,3):raise ValueError('Require producer 19-point layout')
        self.batch,self.centers=queries.shape[:2]
        self.storage=queries.untyped_storage().data_ptr()
    def matches(self,queries,edge=False):
        n=self.centers*19*(12 if edge else 1)
        # Edge queries are deliberately constructed by CCAP from these centers;
        # they have their own storage, but exact fixed 12-edge ordering.
        return all(x.shape==(self.batch,n) and not x.requires_grad for x in queries) and (edge or all(x.untyped_storage().data_ptr()==self.storage for x in queries))

def can_share_caps(boundaries):
    if not isinstance(boundaries,(tuple,list)):return False
    return all(type(b.z_min) is RunPdeFace3D and type(b.z_max) is RunPdeFace3D
        and b.z_min.fixed_axis==b.z_max.fixed_axis==2
        and b.z_min.fixed_value==0 and b.z_max.fixed_value==1
        and b.z_min.cap is b.z_max.cap for b in boundaries)

def faces(boundaries,xi,eta,zeta):
    if MODE=='base' or not isinstance(boundaries,(tuple,list)):
        return BASE_FACES(boundaries,xi,eta,zeta)
    share=can_share_caps(boundaries)
    reduce=MODE=='both' and LAYOUT is not None and LAYOUT.matches((xi,eta,zeta))
    if not share and not reduce:return BASE_FACES(boundaries,xi,eta,zeta)
    names=('x_min','x_max','y_min','y_max','z_min','z_max')[:5 if share else 6]
    axes=((1,2),(1,2),(0,2),(0,2),(0,1),(0,1))[:len(names)]
    q=(xi,eta,zeta);us=[];vs=[]
    for a in axes:
        u,v=q[a[0]],q[a[1]]
        if reduce:
            first,_=representatives(a,xi.device)
            u=u.reshape(xi.shape[0],LAYOUT.centers,19).index_select(2,first).flatten(1)
            v=v.reshape(xi.shape[0],LAYOUT.centers,19).index_select(2,first).flatten(1)
        us.append(u);vs.append(v)
    ss=[getattr(b,n) for n in names for b in boundaries]
    result=backend.evaluate_surfaces_torch(ss,torch.cat(us),torch.cat(vs))
    parts=list(result.reshape(len(names),xi.shape[0],-1,3).unbind(0))
    if reduce:
        parts=[p.reshape(xi.shape[0],LAYOUT.centers,9,3).index_select(2,representatives(a,xi.device)[1]).flatten(1,2) for p,a in zip(parts,axes)]
    if share:
        # Match float32 reference: cast operands before their addition.
        z0=xi.new_tensor([b.z_max.z_min for b in boundaries])[:,None,None]
        length=xi.new_tensor([b.z_max.length for b in boundaries])[:,None,None]
        top=(z0+length).expand(-1,xi.shape[1],1)
        parts.append(torch.cat((parts[-1][...,:2],top),-1))
    return torch.stack(parts,2)

def edges(boundaries,xi,eta,zeta,face_ids):
    if MODE!='both' or LAYOUT is None or not LAYOUT.matches((xi,eta,zeta),edge=True):
        return BASE_EDGES(boundaries,xi,eta,zeta,face_ids)
    b,c=LAYOUT.batch,LAYOUT.centers
    qs=[x.reshape(b,12,c,19) for x in (xi,eta,zeta)]
    firsts=[representatives((axis,),xi.device)[0] for axis in (2,1,0)]
    index=torch.stack([firsts[i//4] for i in range(12)])[None,:,None,:].expand(b,12,c,3)
    small=[x.gather(3,index).flatten(1) for x in qs]
    ids=face_ids.reshape(12,c,19).gather(2,index[0]).flatten()
    values=BASE_EDGES(boundaries,*small,ids).reshape(b,12,c,3,3)
    inv=torch.stack([representatives((axis,),xi.device)[1] for axis in (2,)*4+(1,)*4+(0,)*4])
    return values.gather(3,inv[None,:,None,:,None].expand(b,12,c,19,3)).flatten(1,3)

@contextmanager
def optimized(mode,layout=None):
    global MODE,LAYOUT
    previous=(MODE,LAYOUT);MODE,LAYOUT=mode,layout
    oldfaces=model_module.evaluate_boundary_surfaces_torch;oldedges=model_module.evaluate_selected_boundary_surfaces_torch
    model_module.evaluate_boundary_surfaces_torch=faces
    model_module.evaluate_selected_boundary_surfaces_torch=edges
    try:yield
    finally:
        MODE,LAYOUT=previous
        model_module.evaluate_boundary_surfaces_torch=oldfaces
        model_module.evaluate_selected_boundary_surfaces_torch=oldedges
