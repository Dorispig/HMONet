"""Offline-only geometry optimizations; no repository source is edited."""
import ctypes as ct
import sys
from pathlib import Path
from contextlib import contextmanager
import torch
import src.models.model as mm
import src.models.stencil_projection as sp
import src.data.geometry.pde_cap_3d as pc
from src.data.backend.parametric_torch import evaluate_surfaces_torch

OLD_EDGES = mm.central_stencil_edges
OLD_SPLINE = pc._same_knots_residual
SHARED = FUSED = False
LIBS = []
KERNELS = {}

def direct_edges(boundaries, xi, eta, zeta):
    batch, count = xi.shape
    if not isinstance(boundaries,(list,tuple)):
        boundaries=[boundaries]*batch
    coords=[q.reshape(batch,-1,19) for q in (xi,eta,zeta)]
    x,y,z=[coords[axis].index_select(2,sp._projection_map((axis,),xi.device)[0]).flatten(1)
           for axis in range(3)]
    zero,one=torch.zeros_like(x),torch.ones_like(x)
    # Same incident faces and same query order as the production mask route.
    uv=[(zero,z),(zero,z),(one,z),(one,z),
        (y,zero),(y,zero),(y,one),(y,one),
        (x,zero),(x,zero),(x,one),(x,one)]
    result=[None]*12
    for face,indices in ((0,(0,2,4,6)),(1,(1,3,5,7)),(2,(8,10)),(3,(9,11))):
        surfaces=[getattr(g,sp._NAMES[face]) for g in boundaries]
        u=torch.cat([uv[i][0] for i in indices],1)
        v=torch.cat([uv[i][1] for i in indices],1)
        values=evaluate_surfaces_torch(surfaces,u,v).reshape(batch,len(indices),-1,3)
        for j,index in enumerate(indices):result[index]=values[:,j]
    return torch.stack([
        result[i].reshape(batch,-1,3,3).index_select(2,sp._projection_map((axis,),xi.device)[1]).reshape(batch,count,3)
        for i,axis in enumerate(sp._EDGE_AXES)],2)

def compile_kernel(dtype):
    if dtype in KERNELS:return KERNELS[dtype]
    torch.empty(0, device='cuda')  # Establish the Torch CUDA context before driver loading.
    folder=Path(sys.prefix)/'bin'
    LIBS.append(ct.WinDLL(str(folder/'nvrtc-builtins64_121.dll')))
    rtc=ct.WinDLL(str(folder/'nvrtc64_120_0.dll'));driver=ct.WinDLL('nvcuda.dll');LIBS.extend((rtc,driver))
    scalar='float' if dtype==torch.float32 else 'double'
    source=f'''extern "C" __global__ void gather_weight(const {scalar}* c,
       const long long* iu,const long long* iv,const {scalar}* bu,const {scalar}* bv,
       {scalar}* out,long long total,int n,int nu,int nv,int channels,int shared) {{
       long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;
       if(t>=total) return;
       int ch=t%channels; long long p=t/channels;
       int j=p%4; p/=4; int i=p%4; p/=4;
       int q=p%n; int b=p/n; long long k=((long long)(shared?0:b)*n+q)*4;
       long long ci=(((long long)b*nu+iu[k+i])*nv+iv[k+j])*channels+ch;
       {scalar} a=c[ci]*bu[k+i]; out[t]=a*bv[k+j];
    }}'''
    prog=ct.c_void_p()
    def ok(code):
        if code:raise RuntimeError(f'CUDA/NVRTC status {code}')
    ok(rtc.nvrtcCreateProgram(ct.byref(prog),source.encode(),b'offline_gather.cu',0,None,None))
    opts=[b'--gpu-architecture=compute_89',b'--std=c++11',b'--fmad=false']
    code=rtc.nvrtcCompileProgram(prog,len(opts),(ct.c_char_p*len(opts))(*opts))
    if code:
        size=ct.c_size_t();rtc.nvrtcGetProgramLogSize(prog,ct.byref(size));buf=ct.create_string_buffer(size.value)
        rtc.nvrtcGetProgramLog(prog,buf);raise RuntimeError(buf.value.decode())
    size=ct.c_size_t();ok(rtc.nvrtcGetPTXSize(prog,ct.byref(size)));ptx=ct.create_string_buffer(size.value)
    ok(rtc.nvrtcGetPTX(prog,ptx));rtc.nvrtcDestroyProgram(ct.byref(prog))
    module=ct.c_void_p();fn=ct.c_void_p()
    ok(driver.cuModuleLoadData(ct.byref(module),ptx));ok(driver.cuModuleGetFunction(ct.byref(fn),module,b'gather_weight'))
    KERNELS[dtype]=(driver,module,fn)
    return KERNELS[dtype]

def fused_gather(c,iu,iv,bu,bv,batch,n):
    driver,module,fn=compile_kernel(c.dtype)
    # Preserve the original Torch reduction tree; fuse only gather + two multiplies.
    out=c.new_empty(batch,n,4,4,c.shape[-1])
    tensors=(c,iu,iv,bu,bv,out)
    args=[ct.c_void_p(t.data_ptr()) for t in tensors]
    args += [ct.c_longlong(out.numel()),ct.c_int(n),ct.c_int(c.shape[1]),ct.c_int(c.shape[2]),ct.c_int(c.shape[-1]),ct.c_int(iu.shape[0]==1)]
    pointers=(ct.c_void_p*len(args))(*[ct.cast(ct.byref(a),ct.c_void_p) for a in args])
    status=driver.cuLaunchKernel(fn,ct.c_uint((out.numel()+255)//256),1,1,256,1,1,0,
                               ct.c_void_p(torch.cuda.current_stream(c.device).cuda_stream),pointers,None)
    if status:raise RuntimeError(f'cuLaunchKernel {status}')
    return out.sum((-3,-2))

def spline(caps,u,v):
    if u.requires_grad or v.requires_grad:return OLD_SPLINE(caps,u,v)
    tensors=[cap.tensors(u) for cap in caps]
    tu,tv=tensors[0][:2]
    c=torch.stack([item[2] for item in tensors])
    # SHARED is explicit metadata supplied only for shared-uniform query batches.
    uq,vq=(u[:1],v[:1]) if SHARED else (u,v)
    iu,bu=pc._cubic_basis(uq,tu);iv,bv=pc._cubic_basis(vq,tv)
    if FUSED and u.is_cuda:
        return fused_gather(c,iu,iv,bu,bv,u.shape[0],u.shape[1])
    batch=torch.arange(len(caps),device=u.device)[:,None,None,None]
    local=c[batch,iu[..., :,None],iv[...,None,:],:]
    return (local*bu[..., :,None,None]*bv[...,None,:,None]).sum((-3,-2))

@contextmanager
def mode(name,shared_queries=True):
    global SHARED,FUSED
    assert name in ('baseline','A','B','C','AC','ABC')
    SHARED=shared_queries and name in ('B','ABC')
    FUSED=name in ('C','AC','ABC')
    mm.central_stencil_edges=direct_edges if name in ('A','AC','ABC') else OLD_EDGES
    pc._same_knots_residual=spline if name in ('B','C','AC','ABC') else OLD_SPLINE
    try:yield
    finally:
        mm.central_stencil_edges=OLD_EDGES;pc._same_knots_residual=OLD_SPLINE
        SHARED=FUSED=False
