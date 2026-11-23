"""Offline NVRTC experiment: one thread computes cubic bases and the full cap sum."""
import ctypes as ct
import sys
from pathlib import Path
import torch
import src.data.geometry.pde_cap_3d as pc

ORIGINAL = pc._same_knots_residual
_libs = []
_kernels = {}


def _compile(dtype):
    if dtype in _kernels:
        return _kernels[dtype]
    torch.empty(0,device='cuda')
    folder=Path(sys.prefix)/'bin'
    _libs.append(ct.WinDLL(str(folder/'nvrtc-builtins64_121.dll')))
    rtc=ct.WinDLL(str(folder/'nvrtc64_120_0.dll'))
    driver=ct.WinDLL('nvcuda.dll')
    _libs.extend((rtc,driver))
    scalar='float' if dtype==torch.float32 else 'double'
    source='''
    template<typename T> __device__ void cubic(T q,const T* knots,int nk,int* indices,T* values) {
        int lo=0,hi=nk;
        while(lo<hi){int mid=(lo+hi)/2;if(knots[mid]<=q)lo=mid+1;else hi=mid;}
        int span=lo-1;
        if(span<3)span=3;
        if(span>nk-5)span=nk-5;
        T old[4],next[4];old[0]=(T)1;
        for(int d=1;d<4;d++){
            T saved=(T)0;
            for(int i=0;i<d;i++){
                T left=q-knots[span+1-d+i];
                T right=knots[span+i+1]-q;
                T term=old[i]/(left+right);
                next[i]=saved+right*term;
                saved=left*term;
            }
            next[d]=saved;
            for(int i=0;i<=d;i++)old[i]=next[i];
        }
        for(int i=0;i<4;i++){indices[i]=span-3+i;values[i]=old[i];}
    }
    extern "C" __global__ void fused(const TYPE* c,const TYPE* ku,const TYPE* kv,
       const TYPE* u,const TYPE* v,TYPE* out,int batch,int n,int nu,int nv,int nku,int nkv,int channels){
       int t=blockIdx.x*blockDim.x+threadIdx.x;
       if(t>=batch*n)return;
       int b=t/n;
       int iu[4],iv[4];TYPE bu[4],bv[4];
       cubic<TYPE>(u[t],ku,nku,iu,bu);
       cubic<TYPE>(v[t],kv,nkv,iv,bv);
       for(int ch=0;ch<channels;ch++){
           TYPE sum=(TYPE)0;
           for(int i=0;i<4;i++)for(int j=0;j<4;j++){
               long long offset=(((long long)b*nu+iu[i])*nv+iv[j])*channels+ch;
               TYPE x=(c[offset]*bu[i])*bv[j];
               sum+=x;
           }
           out[(long long)t*channels+ch]=sum;
       }
    }
    '''.replace('TYPE',scalar)
    program=ct.c_void_p()
    def ok(code):
        if code:raise RuntimeError(f'CUDA/NVRTC status {code}')
    ok(rtc.nvrtcCreateProgram(ct.byref(program),source.encode(),b'fused_cap.cu',0,None,None))
    options=[b'--gpu-architecture=compute_89',b'--std=c++11',b'--fmad=false']
    status=rtc.nvrtcCompileProgram(program,len(options),(ct.c_char_p*len(options))(*options))
    if status:
        n=ct.c_size_t();rtc.nvrtcGetProgramLogSize(program,ct.byref(n));buf=ct.create_string_buffer(n.value)
        rtc.nvrtcGetProgramLog(program,buf);raise RuntimeError(buf.value.decode())
    n=ct.c_size_t();ok(rtc.nvrtcGetPTXSize(program,ct.byref(n)));ptx=ct.create_string_buffer(n.value)
    ok(rtc.nvrtcGetPTX(program,ptx));rtc.nvrtcDestroyProgram(ct.byref(program))
    module=ct.c_void_p();function=ct.c_void_p()
    ok(driver.cuModuleLoadData(ct.byref(module),ptx));ok(driver.cuModuleGetFunction(ct.byref(function),module,b'fused'))
    _kernels[dtype]=(driver,module,function)
    return _kernels[dtype]


def fused_residual(caps,u,v):
    # Preserve the differentiable Torch path for coordinate derivatives.
    if u.requires_grad or v.requires_grad or not u.is_cuda:
        return ORIGINAL(caps,u,v)
    tensors=[cap.tensors(u) for cap in caps]
    ku,kv=tensors[0][:2]
    c=torch.stack([item[2] for item in tensors]).contiguous()
    u,v=u.contiguous(),v.contiguous()
    out=u.new_empty((*u.shape,c.shape[-1]))
    driver,module,fn=_compile(u.dtype)
    objects=[ct.c_void_p(t.data_ptr()) for t in (c,ku,kv,u,v,out)]
    objects += [ct.c_int(x) for x in (u.shape[0],u.shape[1],c.shape[1],c.shape[2],ku.numel(),kv.numel(),c.shape[-1])]
    pointers=(ct.c_void_p*len(objects))(*[ct.cast(ct.byref(obj),ct.c_void_p) for obj in objects])
    status=driver.cuLaunchKernel(fn,(out.numel()//c.shape[-1]+255)//256,1,1,256,1,1,0,
                                ct.c_void_p(torch.cuda.current_stream(c.device).cuda_stream),pointers,None)
    if status:raise RuntimeError(f'cuLaunchKernel {status}')
    return out
