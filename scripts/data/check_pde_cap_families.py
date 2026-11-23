"""Check current 3-D PDE caps and volume meshes at training/evaluation extremes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.data.generation.make_dataset_parametric_3d import (
    DATASET_FAMILIES, resolve_family, _validated_boundary_samples, _audit_volume_grid,
)
from src.data.geometry.boundary_generator_3d import generate_boundary
from src.data.geometry.parametric_geometry_3d import BoundarySurfaceSet3D
from src.data.backend.parametric_torch import evaluate_boundary_surfaces_torch
from src.numerics.traditional.tfi_3d import tfi_from_surfaces
from src.numerics.traditional.winslow_3d import solve_winslow_3d


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families',nargs='+',default=['all'])
    parser.add_argument('--split',choices=['ext','int'],default='ext')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/validation/pde_caps_3d/endpoint_checks.json')
    args=parser.parse_args(argv)
    families=list(DATASET_FAMILIES) if args.families==['all'] else [resolve_family(n) for n in args.families]
    split={'ext':'extrapolation','int':'interpolation'}[args.split]
    records=[];args.output.parent.mkdir(parents=True,exist_ok=True)
    for family in families:
        spec=DATASET_FAMILIES[family]
        values=sorted(set([v for interval in spec[split] for v in interval]+spec['evaluation_values'][split]))
        for value in values:
            start=time.perf_counter()
            record=dict(family=family,short=spec['output_name'],value=value,shape=spec['grid_shape'],split=split)
            try:
                geometry=generate_boundary(family,**dict(spec['fixed_parameters'],**{spec['parameter']:value}))
                restored=BoundarySurfaceSet3D.from_dict(geometry.to_dict())
                q=np.random.default_rng(3).random((90,3))
                actual=evaluate_boundary_surfaces_torch([restored],*torch.tensor(q[None],dtype=torch.float64).unbind(-1)).numpy()[0]
                record['numpy_torch_error']=float(np.max(np.abs(actual-np.stack(restored.evaluate_faces(*q.T),1))))
                shape=spec['grid_shape'];_validated_boundary_samples(restored,shape,0)
                mesh,info=solve_winslow_3d(restored,*shape,initial_grid=tfi_from_surfaces(restored,*shape),max_iter=1000,tolerance=1e-6,return_info=True)
                record.update(info=info,audit=_audit_volume_grid(mesh.reshape(*shape[::-1],3)))
                record['status']='ok' if info['converged'] and record['audit']['all_cells_legal'] and record['numpy_torch_error']<1e-12 else 'failed_check'
            except Exception as error:
                record.update(status='error',error=str(error))
            record['seconds']=time.perf_counter()-start;records.append(record)
            args.output.write_text(json.dumps(records,indent=2),encoding='utf-8')
            print(json.dumps(record),flush=True)
    return records


if __name__=='__main__':main()
