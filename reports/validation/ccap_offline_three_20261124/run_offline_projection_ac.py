import sys,json,gc,time,hashlib,subprocess
from pathlib import Path
sys.path[:0]=['E:/projects/HMONet','C:/Users/24846/tmp']
import numpy as np
import torch,yaml
from offline_projection_three import mode,compile_kernel
from src.models import HMONet3D
from src.data.parametric_dataset import ParametricDataset3D,read_sample_parameters
from src.data.mesh.sampling import sample_boundary
from src.engine.training import predict_coordinates,sample_shared_random_points
from src.losses.physics import stencil_queries,elliptic_grid_scale_invariant_pde_loss
from scripts.benchmark.benchmark_ccap_stencil import options

torch.set_num_threads(1);torch.cuda.set_device(0)
ROOT=Path('E:/projects/HMONet');OUTPUT=Path('C:/Users/24846/tmp/offline_projection_ac_results.json')
cfg=yaml.safe_load((ROOT/'configs/3D/train.yaml').read_text(encoding='utf-8-sig'))
names=['baseline','AC']
files=['src/models/model.py','src/models/stencil_projection.py','src/data/geometry/pde_cap_3d.py',
       'src/data/geometry/pde_surface_3d.py','src/data/backend/parametric_torch.py','scripts/train/train_3d.py','configs/3D/train.yaml']
def hashes():return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
def gpu():return subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,power.draw,clocks.sm','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
report={'before_hashes':hashes(),'gpu_before':gpu(),'validation':[], 'benchmark':{},
 'definitions':{'baseline':'Production central-stencil reuse already enabled',
 'A':'Fixed edge-to-face batches without dynamic masks/nonzero/device-to-host checks',
 'B':'Common-query/common-knot basis reuse; independent cap coefficients retained',
 'C':'CUDA fused coefficient gather and two multiplies; original Torch sum retained, fmad=false',
 'ABC':'All three combined'},
 'scope':'Offline monkey patches only; surface maps, interpolation and CCAP arithmetic unchanged.'}
def save():OUTPUT.write_text(json.dumps(report,indent=2),encoding='utf-8')
def evaluate(model,boundary,geometries,queries,steps,name,shared=True):
 model.zero_grad(set_to_none=True)
 with mode(name,shared):
  pred=predict_coordinates(model,boundary,queries.flatten(-3,-2),geometries,central_stencil=True).reshape(queries.shape)
  loss=elliptic_grid_scale_invariant_pde_loss(pred,steps)
  loss.backward()
  grad=torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
  return [pred.detach().clone(),loss.detach().clone(),grad.detach().clone()]
def compare(reference,actual):
 dif=[float((a-b).abs().max()) for a,b in zip(reference,actual)]
 return dict(output_max_abs=dif[0],loss_abs=dif[1],gradient_max_abs=dif[2],
             gradient_relative_l2=float((actual[2]-reference[2]).norm()/reference[2].norm().clamp_min(1e-30)),
             finite=all(bool(torch.isfinite(v).all()) for v in actual),bitwise_equal=all(torch.equal(a,b) for a,b in zip(reference,actual)))

for dtype in (torch.float32,torch.float64):compile_kernel(dtype)
sources=json.loads(Path('C:/Users/24846/tmp/formal3d_preflight_20261123/audit.json').read_text())
for source in sources:
 ds=ParametricDataset3D(source['dataset']);gs=[ds.boundary[0],ds.boundary[-1]]
 shape=(41,41,11) if source['short']=='run_e' else (21,21,11)
 r={'family':source['short'],'source':source['dataset'],'shape':shape,
    'parameters':[read_sample_parameters(source['dataset'],i) for i in (0,len(ds)-1)],'checks':[]}
 report['validation'].append(r)
 for dtype in (torch.float32,torch.float64):
  torch.manual_seed(20261030);model=HMONet3D(*shape,**options(cfg,'hmonet')).to(device='cuda',dtype=dtype)
  boundary=torch.tensor(np.stack([sample_boundary(g,shape) for g in gs]),device='cuda',dtype=dtype)
  steps=boundary.new_tensor([.01]*3)
  for seed in (31,47,83):
   torch.manual_seed(seed)
   centers=sample_shared_random_points(2,400,3,'cuda',margin=steps).to(dtype)
   centers=centers.clone()
   centers[:, :8] = centers.new_tensor([[x,y,z] for x in (.01,.99) for y in (.01,.99) for z in (.01,.99)])
   queries=stencil_queries(centers,steps)
   ref=evaluate(model,boundary,gs,queries,steps,'baseline')
   for name in names[1:]:
    actual=evaluate(model,boundary,gs,queries,steps,name)
    r['checks'].append(dict(dtype=str(dtype),seed=seed,variant=name,**compare(ref,actual)))
  # A different query set per sample must bypass the shared-basis shortcut.
  if dtype==torch.float64:
   centers=.01+.98*torch.rand(2,31,3,device='cuda',dtype=dtype)
   centers=centers.clone()
   centers[:, :8] = centers.new_tensor([[x,y,z] for x in (.01,.99) for y in (.01,.99) for z in (.01,.99)])
   queries=stencil_queries(centers,steps)
   ref=evaluate(model,boundary,gs,queries,steps,'baseline',False)
   actual=evaluate(model,boundary,gs,queries,steps,'ABC',False)
   r['nonshared_query_check']=compare(ref,actual)
  del model,boundary,ref,actual;gc.collect();torch.cuda.empty_cache()
 save();print('VALIDATED',source['short'],all(c['bitwise_equal'] for c in r['checks']),flush=True)
 del ds,gs;gc.collect();torch.cuda.empty_cache()

# Reuse the exact full-step timing protocol from the preceding benchmark.
ns={};exec(Path('C:/Users/24846/tmp/benchmark_training_half_width_3d.py').read_text().split('\nfor name,cls in ')[0],ns)
original_predict=ns['predict_coordinates']
def optimized_predict(model,*a,**kw):
 kw['central_stencil']=True
 return original_predict(model,*a,**kw)
ns['predict_coordinates']=optimized_predict
iteration=ns['iteration'];shape=ns['shape']
models={};opts={}
for name in names:
 torch.manual_seed(20261030);models[name]=HMONet3D(*shape,**options(cfg,'hmonet')).cuda().train()
 opts[name]=torch.optim.Adam(models[name].parameters(),lr=cfg['hmonet_training']['learning_rate'])
 report['benchmark'][name]={'iterations':[]}
steps=ns['steps'];boundary=ns['boundary'];gs=ns['geometries']
torch.manual_seed(456);centers=sample_shared_random_points(64,400,3,'cuda',margin=steps)
queries=stencil_queries(centers,steps)
ref=evaluate(models['baseline'],boundary,gs,queries,steps,'baseline')
report['batch64_equality']={}
for name in names[1:]:
 actual=evaluate(models['baseline'],boundary,gs,queries,steps,name)
 report['batch64_equality'][name]=compare(ref,actual)
del ref,actual;gc.collect();torch.cuda.empty_cache()
for i in range(3):
 for name in names:
  with mode(name):iteration(models[name],opts[name],'hmonet',100+i)
for i in range(15):
 order=names[i%len(names):]+names[:i%len(names)]
 if i%2:order=order[::-1]
 for name in order:
  with mode(name):row=iteration(models[name],opts[name],'hmonet',20261030+i)
  row['reserved_MiB']=torch.cuda.memory_reserved()/2**20
  report['benchmark'][name]['iterations'].append(row)
 print('TIMING',i+1,{n:round(report['benchmark'][n]['iterations'][-1]['total_ms'],2) for n in names},flush=True)
 save()
for name in names:
 rows=report['benchmark'][name]['iterations']
 for stat,fn in [('mean',np.mean),('median',np.median),('min',np.min),('max',np.max),('std',np.std)]:
  report['benchmark'][name][stat]={k:float(fn([row[k] for row in rows])) for k in rows[0]}
report['benchmark_setup']={'family':'teb_b','parameter':.12,'shape':shape,'batch':64,'centers':400,
 'dtype':'float32','repeats':15,'warmups':3,'order':'Rotating and reversing five variants in each round',
 'data':'Same geometry repeated as 64 independent surface objects; identical initial models and random centers for each paired round',
 'timing':'Includes sampling, forward, normalized PDE loss, backward, clipping, Adam; excludes IO, diagnostic MSE and compilation'}
report['after_hashes']=hashes();report['production_unchanged']=report['before_hashes']==report['after_hashes']
report['gpu_after']=gpu();save();print('DONE',OUTPUT,flush=True)

