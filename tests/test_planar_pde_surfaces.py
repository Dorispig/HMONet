"""One serializable cap contract shared by all nine formal 3-D families."""
import json
from unittest.mock import patch
import numpy as np
import pytest
import torch

from src.data.generation.make_dataset_parametric_3d import DATASET_FAMILIES
from src.data.geometry.boundary_generator_3d import BOUNDARY_GENERATORS
from src.data.geometry.parametric_geometry_3d import BoundarySurfaceSet3D
from src.data.geometry.pde_surface_3d import PdeCapSurface3D, CAP_SCHEME, SIDE_SCHEME
from src.data.backend.parametric_torch import evaluate_boundary_surfaces_torch


@pytest.mark.parametrize('family', list(DATASET_FAMILIES))
def test_independent_caps_exact_edges_unchanged_sides_and_roundtrip(family):
    spec=DATASET_FAMILIES[family]
    params=dict(spec['fixed_parameters'], cap_nx=31, cap_ny=31)
    params[spec['parameter']]=sum(spec['extrapolation'][0])/2
    factory=BOUNDARY_GENERATORS[family]
    boundary=factory(**params)
    original=factory.__wrapped__(**{k:v for k,v in params.items() if not k.startswith('cap_')})
    assert isinstance(boundary.z_min,PdeCapSurface3D)
    assert isinstance(boundary.z_max,PdeCapSurface3D)
    assert boundary.z_min is not boundary.z_max
    assert boundary.metadata['cap_parameterization']==CAP_SCHEME
    assert boundary.metadata['side_parameterization']==SIDE_SCHEME
    q=np.linspace(0,1,67);o=np.ones_like(q);z=np.zeros_like(q)
    for name in ('x_min','x_max','y_min','y_max'):
        np.testing.assert_array_equal(getattr(boundary,name).sample(17,19),getattr(original,name).sample(17,19))
    for name in ('z_min','z_max'):
        for u,v in ((q,z),(q,o),(z,q),(o,q)):
            np.testing.assert_allclose(getattr(boundary,name).evaluate(u,v),getattr(original,name).evaluate(u,v),atol=3e-14,rtol=0)
    with patch('src.data.geometry.pde_surface_3d._build_cached', side_effect=AssertionError('must not solve on load')):
        restored=BoundarySurfaceSet3D.from_dict(json.loads(json.dumps(boundary.to_dict())))
    query=torch.tensor(np.random.default_rng(7).uniform(.03,.97,(2,19,3)),dtype=torch.float64,requires_grad=True)
    value=evaluate_boundary_surfaces_torch([restored,restored],*query.unbind(-1))
    expected=np.stack(restored.evaluate_faces(*np.moveaxis(query.detach().numpy(),-1,0)),-2)
    np.testing.assert_allclose(value.detach().numpy(),expected,atol=3e-13,rtol=0)
    first=torch.autograd.grad(value.square().sum(),query,create_graph=True)[0]
    second=torch.autograd.grad(first.sum(),query)[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
