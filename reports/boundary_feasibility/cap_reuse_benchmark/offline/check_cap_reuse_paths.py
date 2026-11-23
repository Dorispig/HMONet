import json,pickle
from pathlib import Path
import torch
from cap_reuse_offline import optimized,FDLayout,faces,edges,BASE_FACES,BASE_EDGES
from src.data.backend import parametric_torch as backend
from src.numerics.differential import stencil_queries

root=Path('C:/Users/24846/tmp/cap_reuse_results')
with (root/'run_1.55.pkl').open('rb') as f:b=pickle.load(f)
torch.set_num_threads(2)
q0=.01+.98*torch.rand(2,4,3,device='cuda');full=stencil_queries(q0,[.01]*3);layout=FDLayout(full);q=full.flatten(1,2)
geos=[b,b];records=[]
original=backend.evaluate_surfaces_torch
for mode in ('base','caps','both'):
    calls=[]
    def count(ss,u,v):
        calls.append([len(ss),u.shape[1]])
        return original(ss,u,v)
    backend.evaluate_surfaces_torch=count
    with optimized(mode,layout):out=faces(geos,*q.unbind(-1))
    backend.evaluate_surfaces_torch=original
    if mode=='base':reference=out
    assert torch.equal(reference,out)
    records.append(dict(mode=mode,surface_evaluation_batches=calls,query_pairs=sum(x*y for x,y in calls),equal=True))
# Dense end-cap evaluation includes between-knot samples and all four edges.
u,v=torch.meshgrid(torch.linspace(0,1,401,device='cuda',dtype=torch.float64),torch.linspace(0,1,401,device='cuda',dtype=torch.float64),indexing='ij')
q=torch.stack((u,v,torch.full_like(u,.37)),-1).reshape(1,-1,3)
with torch.no_grad():
    a=BASE_FACES([b],*q.unbind(-1))
    with optimized('both'):c=faces([b],*q.unbind(-1))
error=float((a-c).abs().max());assert error==0
result=dict(calls=records,dense_face_grid=[401,401],dense_max_abs=error,dense_equal=True)
(root/'path_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
