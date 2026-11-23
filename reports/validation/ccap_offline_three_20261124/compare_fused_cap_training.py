"""Paired offline convergence comparison; repository implementation remains untouched."""
import sys,json,time,copy,statistics,subprocess,os
from pathlib import Path
sys.path[:0]=['E:/projects/HMONet','C:/Users/24846/tmp']
import numpy as np,torch,yaml
import src.data.geometry.pde_cap_3d as pc
from offline_fused_cap import fused_residual,ORIGINAL
from src.data.parametric_dataset import ParametricDataset3D
from src.models import HMONet3D
from src.engine.training import predict_coordinates,sample_shared_random_points,interpolate_pde_mesh
from src.losses.physics import stencil_queries,elliptic_grid_scale_invariant_pde_loss
from scripts.benchmark.benchmark_ccap_stencil import options

torch.set_num_threads(1);torch.cuda.set_device(0)
root=Path('E:/projects/HMONet')
cfg=yaml.safe_load((root/'configs/3D/train.yaml').read_text(encoding='utf-8-sig'))
sources=json.loads(Path('C:/Users/24846/tmp/formal3d_preflight_20261123/audit.json').read_text())
control=os.environ.get('FUSED_CAP_CONTROL')=='1'
out=Path('C:/Users/24846/tmp/fused_cap_training_control.json' if control else 'C:/Users/24846/tmp/fused_cap_training_comparison.json')
steps=int(sys.argv[1]) if len(sys.argv)>1 else 1000
def gpu():
 return subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
report={'scope':'Offline paired Adam training. Original versus original control.' if control else 'Offline paired Adam training. Current exact repository vs offline fused cap only.',
 'iterations':steps,'samples_per_family':2,'centers_per_sample':400,'initial_seed':20261030,
 'validation_points_per_sample':400,'families':[],'hardware_start':gpu(),
 'notes':'Same initial model and same random points per step; one model per variant, separate Adam. Sample points remain inside 0.01 margin; no checkpoint or full formal early stopping.'}
def save():out.write_text(json.dumps(report,indent=2),encoding='utf-8')
def call(mode,model,boundary,geometries,query):
 pc._same_knots_residual=fused_residual if mode=='fused' and not control else ORIGINAL
 try:return predict_coordinates(model,boundary,query,geometries,central_stencil=True)
 finally:pc._same_knots_residual=ORIGINAL
def relative_parameter_difference(a,b):
 with torch.no_grad():
  x=torch.cat([p.detach().flatten() for p in a.parameters()])
  y=torch.cat([p.detach().flatten() for p in b.parameters()])
  return float((x-y).norm()/x.norm().clamp_min(1e-30)),float((x-y).abs().max())

for short in (('teb_b',) if control else ('run_e','teb_b')):
 source=next(s for s in sources if s['short']==short)
 ds=ParametricDataset3D(source['dataset'])
 geometry=[ds.boundary[0],ds.boundary[-1]]
 shape=ds.grid_shape
 index=[0,len(ds)-1]
 boundary=torch.tensor(ds.boundary_points[index],device='cuda',dtype=torch.float32)
 mesh=torch.tensor(ds.mesh_points[index],device='cuda',dtype=torch.float32)
 torch.manual_seed(20261030)
 base=HMONet3D(*shape,**options(cfg,'hmonet')).cuda().train()
 models={'original':base,'fused':copy.deepcopy(base)}
 opts={name:torch.optim.Adam(model.parameters(),lr=cfg['hmonet_training']['learning_rate']) for name,model in models.items()}
 step=boundary.new_tensor([.01]*3)
 torch.manual_seed(98765)
 val_points=sample_shared_random_points(2,400,3,'cuda',margin=step)
 val_queries=stencil_queries(val_points,step)
 val_target=interpolate_pde_mesh(mesh,val_points,shape)
 family={'family':short,'shape':shape,'training':[],'validation':[],'initial_difference':relative_parameter_difference(*models.values())}
 report['families'].append(family)
 def validate(index):
  row={'iteration':index}
  outputs={}
  for name,model in models.items():
   model.eval()
   with torch.no_grad():
    pred=call(name,model,boundary,geometry,val_queries.flatten(-3,-2)).reshape(val_queries.shape)
    pde=float(elliptic_grid_scale_invariant_pde_loss(pred,step))
    mesh_mse=float((pred[:,:,0]-val_target).square().mean())
    outputs[name]=pred[:,:,0].detach().clone()
   model.train()
   row[name]={'normalized_pde_loss':pde,'mesh_mse':mesh_mse}
  row['pde_loss_ratio_fused_to_original']=row['fused']['normalized_pde_loss']/max(row['original']['normalized_pde_loss'],1e-30)
  row['mesh_mse_ratio_fused_to_original']=row['fused']['mesh_mse']/max(row['original']['mesh_mse'],1e-30)
  row['output_max_abs']=float((outputs['original']-outputs['fused']).abs().max())
  row['parameter_relative_l2'],row['parameter_max_abs']=relative_parameter_difference(*models.values())
  family['validation'].append(row)
 validate(0)
 for index in range(1,steps+1):
  torch.manual_seed(20261030+index)
  centers=sample_shared_random_points(2,400,3,'cuda',margin=step)
  queries=stencil_queries(centers,step)
  row={'iteration':index}
  for name in (('original','fused') if index%2 else ('fused','original')):
   model=models[name]
   opts[name].zero_grad(set_to_none=True)
   pred=call(name,model,boundary,geometry,queries.flatten(-3,-2)).reshape(queries.shape)
   loss=elliptic_grid_scale_invariant_pde_loss(pred,step)
   if not torch.isfinite(loss):raise FloatingPointError((short,index,name,'loss'))
   loss.backward()
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['gradient_clip_norm'])
   if not torch.isfinite(norm):raise FloatingPointError((short,index,name,'grad'))
   opts[name].step()
   row[name]=float(loss.detach())
  family['training'].append(row)
  if index%50==0 or index==1:
   validate(index)
   print(short,index,'loss',row['original'],row['fused'],
         'validation_ratio',family['validation'][-1]['pde_loss_ratio_fused_to_original'],flush=True)
   save()
 family['summary']={
  'initial_pde_loss':family['validation'][0]['original']['normalized_pde_loss'],
  'final_original_pde_loss':family['validation'][-1]['original']['normalized_pde_loss'],
  'final_fused_pde_loss':family['validation'][-1]['fused']['normalized_pde_loss'],
  'final_original_mesh_mse':family['validation'][-1]['original']['mesh_mse'],
  'final_fused_mesh_mse':family['validation'][-1]['fused']['mesh_mse'],
  'final_output_max_abs':family['validation'][-1]['output_max_abs'],
  'final_parameter_relative_l2':family['validation'][-1]['parameter_relative_l2'],
  'max_validation_loss_ratio_deviation':max(abs(x['pde_loss_ratio_fused_to_original']-1) for x in family['validation'][1:]),
  'max_validation_mse_ratio_deviation':max(abs(x['mesh_mse_ratio_fused_to_original']-1) for x in family['validation'][1:]),
 }
 save();print('FINISHED',short,family['summary'],flush=True)
 del models,opts,base,boundary,mesh,ds
 torch.cuda.empty_cache()
report['hardware_end']=gpu();save()
print(out)
