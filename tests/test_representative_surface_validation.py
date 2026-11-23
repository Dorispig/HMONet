"""Representative auditing must survive caching, CPU workers and GPU dispatch."""
import json
import numpy as np
import pytest

from src.data.generation import make_dataset_parametric_3d as gen
from src.data.geometry import pde_surface_3d as caps
from src.data.geometry.boundary_generator_3d import generate_boundary


def test_midpoint_selection_with_disjoint_and_unsorted_intervals():
    values=[12., 0., 14., 1., 10., 4., 3., 11., 2., 13.]
    ids=gen.representative_sample_indices([{'p':v} for v in values],
           {'parameter':'p','parameter_ranges':[[0,4],[10,14]]})
    assert sorted(values[i] for i in ids)==[0,2,4,10,12,14]
    assert gen.representative_sample_indices([{'p':0.}])==[0]
    ids=gen.representative_sample_indices([{'p':v} for v in (0.,.2,.4,.6,.8,1.)])
    assert ids==[0,2,5]


def test_cap_cache_cannot_bypass_requested_audit(monkeypatch):
    from src.data.generation import dataset_quality_2d
    caps._build_cached.cache_clear()
    calls=[]
    original=dataset_quality_2d.audit_grid
    def audit(mesh,nx,ny):
        calls.append((nx,ny))
        return original(mesh,nx,ny)
    monkeypatch.setattr(dataset_quality_2d,'audit_grid',audit)
    params=dict(sidewall_amplitude=.1,cap_nx=11,cap_ny=11)
    with caps.cap_validation_context(False):
        unchecked=generate_boundary('sidewall_deformed_block',**params)
    assert calls==[]
    checked=generate_boundary('sidewall_deformed_block',**params)
    assert calls==[(11,11)]*4  # PDE plus interpolated mesh, separately for both caps
    assert unchecked.z_min.solver_info['interpolation_checks']=={}
    assert checked.z_min.solver_info['legality_checked']
    assert set(checked.z_min.solver_info['interpolation_checks'])=={'11x11'}
    np.testing.assert_array_equal(unchecked.z_min.coefficients,checked.z_min.coefficients)
    assert checked.z_min.solver_info['tolerance']==1e-6


def test_unchecked_faces_still_reject_nonfinite_values(monkeypatch):
    monkeypatch.setattr(gen,'sample_faces',lambda *args: ({},np.full((3,3,3,3),np.nan)))
    with pytest.raises(ValueError,match='nonfinite'):
        gen._validated_boundary_samples(None,(3,3,3),0,validate=False)


@pytest.mark.parametrize('workers',[1,2])
def test_serial_and_spawned_dataset_workers_use_same_audit_selection(tmp_path,monkeypatch,workers):
    from src.data.generation import dataset_examples
    monkeypatch.setattr(dataset_examples,'save_dataset_examples_3d',lambda *args,**kwargs:None)
    monkeypatch.setattr(gen,'update_dataset_catalog',lambda *args:None)
    parameters=[dict(sidewall_amplitude=float(p),cap_nx=11,cap_ny=11)
                for p in np.linspace(.02,.10,5)]
    out=gen.generate_dataset('small','sidewall_deformed_block',parameters,
        grid_shape=(5,5,5),output_root=tmp_path,workers=workers,timing_samples=0)
    meta=json.loads((out/'metadata.json').read_text())
    assert meta['legality_check_sample_ids']==[0,2,4]
    assert [d['boundary_legality_checked'] for d in meta['pde_diagnostics']]==[True,False,True,False,True]
    saved=json.loads((out/'geometry.json').read_text())
    for i,sample in enumerate(saved['samples']):
        from src.data.geometry.parametric_geometry_3d import BoundarySurfaceSet3D
        boundary=BoundarySurfaceSet3D.from_dict(sample['boundary'])
        assert boundary.z_min.solver_info['legality_checked']==(i in (0,2,4))


def test_gpu_payload_applies_same_policy(monkeypatch):
    from src.numerics.traditional.generation_backend import current_solver_options
    calls=[]
    def fake(payload,options,*,validate):
        calls.append((caps._CHECK_CAPS.get(),current_solver_options(),validate))
    monkeypatch.setattr(gen,'_solve_dataset_sample_3d_impl',fake)
    payload=(0,'box',{},3,3,3,10,1e-6)
    options={'solver_backend':'cuda','device':'cuda:0'}
    for flag in (False,True):gen._solve_dataset_sample_3d(payload+(flag,),options)
    assert calls==[(False,options,False),(True,options,True)]
    assert caps._CHECK_CAPS.get() is True
    assert current_solver_options()=={}


def test_all_nine_family_defaults():
    assert len(gen.DATASET_FAMILIES)==9
    for spec in gen.DATASET_FAMILIES.values():
        assert spec['fixed_parameters']['cap_tolerance']==1e-6
        assert spec['fixed_parameters']['cap_nx']==201
        assert spec['fixed_parameters']['cap_ny']==201
