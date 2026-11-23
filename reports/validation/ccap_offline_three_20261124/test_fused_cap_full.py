import sys,json,time,statistics,subprocess
from pathlib import Path
sys.path[:0]=['E:/projects/HMONet','C:/Users/24846/tmp']
import numpy as np,torch,yaml
import src.data.geometry.pde_cap_3d as pc
from offline_fused_cap import fused_residual,ORIGINAL
from src.data.parametric_dataset import ParametricDataset3D
from src.data.mesh.sampling import sample_boundary
from src.models import HMONet3D
from src.engine.training import predict_coordinates,sample_shared_random_points
from src.losses.physics import stencil_queries,elliptic_grid_scale_invariant_pde_loss
from scripts.benchmark.benchmark_ccap_stencil import options

torch.set_num_threads(1);torch.cuda.set_device(0)
ROOT=Path('E:/projects/HMONet')
CFG=yaml.safe_load((ROOT/'configs/3D/train.yaml').read_text(encoding='utf-8-sig'))
SOURCE=Path('C:/Users/24846/tmp/formal3d_preflight_20261123/audit.json')
OUT=Path('C:/Users/24846/tmp/test_fused_cap_full.json')
def gpu():
 return subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
report={'scope':'Offline-only NVRTC fused cubic basis, coefficient gather, 16-term sum. Formal repository code unchanged.',
        'hardware_before':gpu(),'validation':[],'benchmarks':{}}
def save():OUT.write_text(json.dumps(report,indent=2),encoding='utf-8')
def evaluate(model,boundary,gs,queries,steps,fused):
 pc._same_knots_residual=fused_residual if fused else ORIGINAL
 try:
  model.zero_grad(set_to_none=True)
  pred=predict_coordinates(model,boundary,queries.flatten(-3,-2),gs,central_stencil=True).reshape(queries.shape)
  loss=elliptic_grid_scale_invariant_pde_loss(pred,steps)
  loss.backward()
  grad=torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
  return pred.detach().clone(),loss.detach().clone(),grad.detach().clone()
 finally:pc._same_knots_residual=ORIGINAL
def diff(a,b):
 return {'output_max_abs':float((a[0]-b[0]).abs().max()),'loss_abs':float((a[1]-b[1]).abs()),
         'gradient_max_abs':float((a[2]-b[2]).abs().max()),
         'gradient_relative_l2':float((a[2]-b[2]).norm()/a[2].norm().clamp_min(1e-30)),
         'all_finite':all(bool(torch.isfinite(x).all()) for x in b)}
sources=json.loads(SOURCE.read_text())
for source in sources:
 ds=ParametricDataset3D(source['dataset']);gs=[ds.boundary[0],ds.boundary[-1]]
 shape=(41,41,11) if source['short']=='run_e' else (21,21,11)
 entry={'family':source['short'],'shape':shape,'tests':[]};report['validation'].append(entry)
 for dtype in (torch.float32,torch.float64):
  boundary=torch.tensor(np.stack([sample_boundary(g,shape) for g in gs]),device='cuda',dtype=dtype)
  torch.manual_seed(20261030)
  model=HMONet3D(*shape,**options(CFG,'hmonet')).to(device='cuda',dtype=dtype).train()
  steps=boundary.new_tensor([.01]*3)
  for seed in (31,47):
   torch.manual_seed(seed)
   centers=sample_shared_random_points(2,64,3,'cuda',margin=steps).to(dtype)
   queries=stencil_queries(centers,steps)
   a=evaluate(model,boundary,gs,queries,steps,False)
   b=evaluate(model,boundary,gs,queries,steps,True)
   entry['tests'].append(dict(dtype=str(dtype),seed=seed,**diff(a,b)))
  # Coordinate-gradient queries must take the unmodified generic path.
  cap=gs[0].z_min
  u=torch.rand(1,17,device='cuda',dtype=dtype,requires_grad=True)
  v=torch.rand(1,17,device='cuda',dtype=dtype,requires_grad=True)
  ordinary=ORIGINAL([cap],u,v)
  alternate=fused_residual([cap],u,v)
  entry['tests'].append(dict(dtype=str(dtype),coordinate_autograd_fallback_equal=torch.equal(ordinary,alternate),
                            output_requires_grad=alternate.requires_grad))
  del model,boundary
  torch.cuda.empty_cache()
 save();print('VALIDATED',source['short'],flush=True)

# Match the formal batch-size-64, 400-center, 19-stencil workload.
data=Path('C:/Users/24846/tmp/teb_scheme_comparison_20261124/proposed_0.12/data/teb_b_timing_one_sample')
from src.data.geometry.parametric_geometry_3d import BoundarySurfaceSet3D
descriptor=json.loads((data/'geometry.json').read_text())['samples'][0]['boundary']
gs=[BoundarySurfaceSet3D.from_dict(descriptor) for _ in range(64)]
boundary=torch.tensor(np.load(data/'boundary_samples.npy'),device='cuda',dtype=torch.float32).repeat(64,1,1)
torch.manual_seed(20261030)
model=HMONet3D(21,21,11,**options(CFG,'hmonet')).cuda().train()
optimizer=torch.optim.Adam(model.parameters(),lr=CFG['hmonet_training']['learning_rate'])
steps=boundary.new_tensor([.01]*3)
torch.manual_seed(31)
centers=sample_shared_random_points(64,400,3,'cuda',margin=steps)
queries=stencil_queries(centers,steps)
caps=[g.z_min for g in gs]
u,v=torch.rand(64,3600,device='cuda'),torch.rand(64,3600,device='cuda')
def event_time(fn):
 start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
 start.record();fn();end.record();torch.cuda.synchronize()
 return start.elapsed_time(end)
for _ in range(3):ORIGINAL(caps,u,v);fused_residual(caps,u,v)
times={'original':[],'fused':[]}
for k in range(10):
 for name,fn in ((('original',ORIGINAL),('fused',fused_residual)) if k%2==0 else (('fused',fused_residual),('original',ORIGINAL))):
  times[name].append(event_time(lambda:fn(caps,u,v)))
report['benchmarks']['one_cap_64x3600']={name:dict(samples_ms=values,mean_ms=statistics.mean(values),median_ms=statistics.median(values)) for name,values in times.items()}
def iteration(fused):
 pc._same_knots_residual=fused_residual if fused else ORIGINAL
 try:
  torch.cuda.synchronize();begin=time.perf_counter()
  optimizer.zero_grad(set_to_none=True)
  pred=predict_coordinates(model,boundary,queries.flatten(-3,-2),gs,central_stencil=True).reshape(queries.shape)
  torch.cuda.synchronize();mid=time.perf_counter()
  loss=elliptic_grid_scale_invariant_pde_loss(pred,steps)
  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(),CFG['training']['gradient_clip_norm'])
  optimizer.step()
  torch.cuda.synchronize();end=time.perf_counter()
  return dict(forward_ms=1000*(mid-begin),full_update_ms=1000*(end-begin),loss=float(loss.detach()))
 finally:pc._same_knots_residual=ORIGINAL
for _ in range(2):iteration(False);iteration(True)
records={'original':[],'fused':[]}
for k in range(10):
 for name,fused in ((('original',False),('fused',True)) if k%2==0 else (('fused',True),('original',False))):
  records[name].append(iteration(fused))
 print('TIMED',k+1,flush=True)
report['benchmarks']['full_training_batch64_400']={name:dict(rows=values,mean_forward_ms=statistics.mean(x['forward_ms'] for x in values),
    mean_full_update_ms=statistics.mean(x['full_update_ms'] for x in values)) for name,values in records.items()}
report['hardware_after']=gpu();save()
print(json.dumps({key:{name:{k:v for k,v in block.items() if k.startswith('mean_')} for name,block in value.items()} for key,value in report['benchmarks'].items()},indent=2))
print(OUT)
