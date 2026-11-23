"""Projection reuse must preserve the PDE objective and parameter gradients."""
import pytest
import torch
import numpy as np

from src.models import HMONet3D
from src.models.stencil_projection import can_reuse_central_projections
from src.data.geometry.boundary_generator_3d import box_boundary
from src.data.mesh.sampling import sample_boundary
from src.engine.training import predict_coordinates
from src.losses.physics import stencil_queries, elliptic_grid_scale_invariant_pde_loss


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_central_reuse_output_loss_and_parameter_gradients(device, dtype):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(123)
    shape = (5, 7, 5)
    geometries = [box_boundary(), box_boundary(xmax=1.7, ymax=1.3)]
    boundary = torch.tensor(np.stack([sample_boundary(g, shape) for g in geometries]),
                            device=device, dtype=dtype)
    model = HMONet3D(*shape, hmonet_mapping_mode='CCAP').to(device=device, dtype=dtype)
    centers = .05 + .9 * torch.rand(2, 13, 3, device=device, dtype=dtype)
    # Distinct axis steps also have the same projection duplication pattern.
    steps = centers.new_tensor([.01, .02, .03])
    queries = stencil_queries(centers, steps)
    results = []
    for reuse in (False, True):
        model.zero_grad(set_to_none=True)
        values = predict_coordinates(model, boundary, queries.flatten(-3, -2),
                                     geometries, central_stencil=reuse).reshape(queries.shape)
        loss = elliptic_grid_scale_invariant_pde_loss(values, steps)
        loss.backward()
        grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        results.append((values.detach(), loss.detach(), grads.detach().clone()))
    for reference, actual in zip(*results):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_coordinate_autograd_uses_general_projection():
    geometry = [box_boundary()]
    q = torch.rand(1, 19, 3, dtype=torch.float64, requires_grad=True)
    assert not can_reuse_central_projections(geometry, *q.unbind(-1))
    model = HMONet3D(3, 3, 3, hmonet_mapping_mode='CCAP').double()
    boundary = torch.tensor(sample_boundary(geometry[0], (3, 3, 3))[None], dtype=q.dtype)
    expected = predict_coordinates(model, boundary, q, geometry)
    actual = predict_coordinates(model, boundary, q, geometry, central_stencil=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    a = torch.autograd.grad(actual.square().sum(), q)[0]
    b = torch.autograd.grad(expected.square().sum(), q)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_missing_geometry_and_non_stencil_queries_fall_back():
    q = torch.rand(1, 20, 3)
    assert not can_reuse_central_projections([box_boundary()], *q.unbind(-1))
    assert not can_reuse_central_projections(None, *q[:, :19].unbind(-1))
