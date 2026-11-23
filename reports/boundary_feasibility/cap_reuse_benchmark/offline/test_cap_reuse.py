import sys,os,time,json,pickle,statistics,hashlib
from pathlib import Path
sys.path.insert(0,'E:/projects/HMONet')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import torch,numpy as np,yaml
from cap_reuse_offline import optimized,FDLayout,faces,BASE_FACES
from scripts.visualize.render_boundary_family_catalog import boundary_3d
from src.models.model import HMONet3D
from src.data.mesh.sampling import sample_faces,boundary_indices
from src.numerics.differential import stencil_queries
from src.losses.physics import elliptic_grid_scale_invariant_pde_loss

OUT=Path('C:/Users/24846/tmp/cap_reuse_results');OUT.mkdir(exist_ok=True)
torch.set_num_threads(2);torch.manual_seed(20261030)
cfg=yaml.safe_load(Path('E:/projects/HMONet/configs/3D/train.yaml').read_text(encoding='utf-8'))
cfg=dict(cfg['model']['common'],**cfg['model']['hmonet']);SHAPE=(11,21,11)
paths=['src/data/geometry/pde_cap_3d.py','src/data/backend/parametric_torch.py','src/models/model.py','configs/3D/train.yaml']
def hashes():return {p:hashlib.sha256(Path('E:/projects/HMONet',p).read_bytes()).hexdigest() for p in paths}
R=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,source_hash_before=hashes(),shape=SHAPE,checks=[],training=[],inference=[],projection=[],build=[],seed=20261030)
def save(): (OUT/'results.json').write_text(json.dumps(R,indent=2))
geometries=[]
for value in (.55,1.50,1.52,1.55):
    path=OUT/f'run_{value:g}.pkl';start=time.perf_counter()
    if path.exists():
        with path.open('rb') as f:b=pickle.load(f)
    else:
        b=boundary_3d('rounded_u_notch_extrusion',value)
        with path.open('wb') as f:pickle.dump(b,f)
    geometries.append(b);R['build'].append(dict(value=value,seconds=time.perf_counter()-start,cap_shape=b.z_min.cap.solver_info['grid_shape']))
    print('GEOMETRY',value,R['build'][-1],flush=True);save()

points=[]
for b in geometries:
    _,g=sample_faces(b,SHAPE);points.append(g.reshape(-1,3)[boundary_indices(SHAPE)])
points=np.stack(points)
def inputs(batch,dtype):
    ii=np.arange(batch)%len(geometries)
    return torch.tensor(points[ii],device='cuda',dtype=dtype).unbind(-1),[geometries[i] for i in ii]
def make_model(dtype):return HMONet3D(*SHAPE,**cfg).cuda().to(dtype)
def pred(model,coords,q,geos,mode,layout=None):
    with optimized(mode,layout):return torch.cat(model(*coords,*q.unbind(-1),boundary_geometry=geos),-1)
def delta(a,b):
    diff=(a-b).abs();return dict(max_abs=float(diff.max()),relative_l2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a).clamp_min(1e-30)),equal=bool(torch.equal(a,b)))

# Same full PDE template, same weights, all parameters: output/loss/gradient check.
for dtype,batch,count in ((torch.float64,4,24),(torch.float32,64,400)):
    coords,geos=inputs(batch,dtype);model=make_model(dtype)
    centers=.01+.98*torch.rand(batch,count,3,device='cuda',dtype=dtype)
    full=stencil_queries(centers,[.01]*3);layout=FDLayout(full);q=full.flatten(1,2)
    versions={}
    for mode in ('base','caps','both'):
        model.zero_grad(set_to_none=True)
        output=pred(model,coords,q,geos,mode,layout)
        loss=elliptic_grid_scale_invariant_pde_loss(output.reshape(batch,count,19,3),[.01]*3)
        loss.backward();grad=torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        versions[mode]=(output.detach(),loss.detach(),grad.detach())
    for mode in ('caps','both'):
        rec=dict(test='full_model_pde_gradients',dtype=str(dtype),batch=batch,centers=count,mode=mode,
                 output=delta(versions['base'][0],versions[mode][0]),loss=delta(versions['base'][1],versions[mode][1]),gradient=delta(versions['base'][2],versions[mode][2]),reference_loss=float(versions['base'][1]))
        R['checks'].append(rec);print('CHECK',rec,flush=True)
        assert rec['output']['max_abs'] < (1e-6 if dtype==torch.float32 else 1e-12)
        assert rec['gradient']['relative_l2'] < (1e-4 if dtype==torch.float32 else 1e-9)
    save();del model,versions,output,loss,grad,full,q,layout
    torch.cuda.empty_cache()

# Continuous coordinate derivatives: FD reduction deliberately falls back.
coords,geos=inputs(4,torch.float64);model=make_model(torch.float64)
query=.02+.96*torch.rand(4,20,3,device='cuda',dtype=torch.float64);versions={}
for mode in ('base','caps','both'):
    q=query.detach().clone().requires_grad_(True)
    output=pred(model,coords,q,geos,mode)
    g=torch.autograd.grad(output.square().sum(),q,create_graph=True)[0]
    gg=torch.autograd.grad(g.sum(),q)[0]
    versions[mode]=(output.detach(),g.detach(),gg.detach())
for mode in ('caps','both'):
    rec=dict(test='query_derivatives',mode=mode,output=delta(versions['base'][0],versions[mode][0]),first=delta(versions['base'][1],versions[mode][1]),second=delta(versions['base'][2],versions[mode][2]))
    R['checks'].append(rec);print('DERIVATIVE',rec,flush=True);assert rec['second']['max_abs']<1e-9
save();del model,versions,q,g,gg,output

def stats(v):return dict(mean_ms=statistics.mean(v),median_ms=statistics.median(v),std_ms=statistics.stdev(v),samples_ms=v)
def timing(operations,warm=5,repeats=30):
    modes=list(operations);times={m:[] for m in modes};mem={m:0 for m in modes}
    for rep in range(-warm,repeats):
        order=modes[rep%len(modes):]+modes[:rep%len(modes)]
        for m in order:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            value=operations[m]()
            torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-start)
            if rep>=0:times[m].append(elapsed);mem[m]=max(mem[m],torch.cuda.max_memory_allocated())
            del value
    return {m:dict(**stats(times[m]),peak_allocated_bytes=mem[m]) for m in modes}

for batch in (8,64):
    coords,geos=inputs(batch,torch.float32);models={m:make_model(torch.float32) for m in ('base','caps','both')}
    for m in ('caps','both'):models[m].load_state_dict(models['base'].state_dict())
    opts={m:torch.optim.Adam(models[m].parameters(),lr=.0004) for m in models}
    centers=.01+.98*torch.rand(batch,400,3,device='cuda');full=stencil_queries(centers,[.01]*3);layout=FDLayout(full);q=full.flatten(1,2)
    R['parameters']=sum(p.numel() for p in models['base'].parameters())
    def train(m):
        opts[m].zero_grad(set_to_none=True)
        output=pred(models[m],coords,q,geos,m,layout)
        loss=elliptic_grid_scale_invariant_pde_loss(output.reshape(batch,400,19,3),[.01]*3)
        loss.backward();torch.nn.utils.clip_grad_norm_(models[m].parameters(),1.);opts[m].step()
        return loss
    rec=dict(batch=batch,centers=400,queries=7600,times=timing({m:lambda m=m:train(m) for m in models}))
    R['training'].append(rec);print('TRAIN',batch,{m:round(v['mean_ms'],3) for m,v in rec['times'].items()},flush=True);save()
    # Isolate boundary face+CCAP edge contribution in the same real stencil.
    def projection(m):
        with optimized(m,layout):return faces(geos,*q.unbind(-1))
    with torch.no_grad():times=timing({m:lambda m=m:projection(m) for m in models})
    R['projection'].append(dict(batch=batch,centers=400,times=times));save()
    del models,opts,full,q,layout;torch.cuda.empty_cache()

model=make_model(torch.float32).eval()
for batch,grid,count in ((1,(21,41,21),None),(1,(41,81,41),None),(64,None,400),(64,None,7600)):
    coords,geos=inputs(batch,torch.float32)
    if grid:
        # Highest-depth geometry for single-grid timing.
        geos=[geometries[-1]];coords=torch.tensor(points[-1:],device='cuda',dtype=torch.float32).unbind(-1)
        x,y,z=[torch.linspace(0,1,n,device='cuda') for n in grid];zz,yy,xx=torch.meshgrid(z,y,x,indexing='ij');q=torch.stack((xx,yy,zz),-1).reshape(1,-1,3)
    else:q=torch.rand(batch,count,3,device='cuda')
    with torch.inference_mode():
        # Both has no FD layout for arbitrary/full-grid inference, so is caps.
        times=timing({m:lambda m=m:pred(model,coords,q,geos,m) for m in ('base','caps')})
        a=pred(model,coords,q,geos,'base');b=pred(model,coords,q,geos,'caps');err=delta(a,b)
        assert err['max_abs']<1e-6
    rec=dict(batch=batch,grid=grid,count=q.shape[1],times=times,output_error=err)
    R['inference'].append(rec);print('INFER',batch,grid,count,{m:round(v['mean_ms'],3) for m,v in times.items()},flush=True);save()
    del a,b,q
R['source_hash_after']=hashes();assert R['source_hash_before']==R['source_hash_after'];R['complete']=True;save()
print('COMPLETE',flush=True)
