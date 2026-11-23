"""Numerical parity and queue behavior; CUDA tests skip on CPU-only hosts."""
import numpy as np
import pytest
import torch

from src.numerics.traditional.winslow_torch import solve_uniform_grid, solve_winslow_2d_torch, LinearSolveError
from src.numerics.traditional.winslow_vectorized_2d import solve_winslow_2d_vectorized
from src.data.geometry.boundary_generator_2d import four_convex_semicircle_boundary
from src.data.generation.gpu_samples import validate_gpu_options


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_2d_stencil_matches_cpu(device):
    b = four_convex_semicircle_boundary(arc_angle_deg=70)
    reference, info = solve_winslow_2d_vectorized(b,13,11,tolerance=1e-9,return_info=True)
    actual, gpu = solve_winslow_2d_torch(b,13,11,tolerance=1e-9,return_info=True,device=device)
    assert info['converged'] and gpu['converged']
    np.testing.assert_allclose(actual,reference,atol=2e-9,rtol=0)
    assert gpu['max_linear_relative_residual'] < 1e-10


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_3d_mixed_stencil_matches_cpu(device):
    # Prescribed warped faces exercise all mixed derivatives, not only a box.
    from src.numerics.traditional.winslow_legacy import winslowSolver3D
    z,y,x = np.meshgrid(np.linspace(0,1,7),np.linspace(0,1,8),np.linspace(0,1,9),indexing='ij')
    grid=np.stack([x+.08*y*z, y+.07*x*z, z+.06*x*y],-1)
    grid[1:-1,1:-1,1:-1,0] += .015
    faces=dict(left=grid[:,:,0],right=grid[:,:,-1],front=grid[:,0],back=grid[:,-1],bottom=grid[0],top=grid[-1])
    cpu=winslowSolver3D(9,8,7,faces,initial_points=grid,max_iter=100,tol=1e-9)
    expected=cpu.solve()
    actual,info=solve_uniform_grid(grid,device=device,max_iter=100,tolerance=1e-9)
    assert info['converged']
    np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=0)
    out=actual.reshape(grid.shape)
    np.testing.assert_array_equal(out[0],grid[0])
    np.testing.assert_array_equal(out[:,:,-1],grid[:,:,-1])


def test_linear_failure_is_not_a_success():
    b=four_convex_semicircle_boundary(arc_angle_deg=70)
    with pytest.raises(LinearSolveError,match='linear solve failed'):
        solve_winslow_2d_torch(b,13,11,device='cpu',linear_max_iter=1)


def test_backend_validation():
    assert validate_gpu_options('cpu',None)==[]
    with pytest.raises(ValueError,match='requires'):
        validate_gpu_options('cpu',[0])
    with pytest.raises(ValueError,match='distinct'):
        validate_gpu_options('cuda',[0,0])


def test_queue_reassigns_free_gpu_and_keeps_indices(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from src.data.generation import gpu_samples
    import time
    # Simulate two devices without requiring two physical GPUs in CI.
    def pool(**kwargs):
        return ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(gpu_samples,'ProcessPoolExecutor',pool)
    def work(payload,options):
        time.sleep(.12 if payload==0 else .01)
        return payload,options['device']
    output=list(gpu_samples.gpu_sample_map(work,list(range(4)),[0,1]))
    assert sorted(i for i,_ in output)==list(range(4))
    result=dict(output)
    assert result[0]==(0,'cuda:0')
    assert result[1]==(1,'cuda:1')
    assert result[2]==(2,'cuda:1') and result[3]==(3,'cuda:1')


def test_queue_failure_cancels_following_tasks(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from src.data.generation import gpu_samples
    monkeypatch.setattr(gpu_samples,'ProcessPoolExecutor',lambda **kwargs: ThreadPoolExecutor(max_workers=1))
    seen=[]
    def work(payload,options):
        seen.append(payload)
        raise ValueError('bad boundary')
    with pytest.raises(RuntimeError,match='GPU 0, sample 0: bad boundary'):
        list(gpu_samples.gpu_sample_map(work,[0,1,2],[0]))
    assert seen==[0]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_spawned_gpu_dataset_is_readable_and_ordered(tmp_path):
    import json
    from src.data.generation.make_dataset_parametric_3d import generate_dataset
    from src.data.dataset import ParametricDataset3D
    path=generate_dataset('gpu_boxes','box',[dict(xmax=1.),dict(xmax=2.)],
                          grid_shape=(5,5,5),output_root=tmp_path,
                          solver_backend='cuda',gpus=[0],timing_samples=0)
    data=ParametricDataset3D(path)
    assert len(data)==2
    assert np.max(data.mesh_points[0,:,0])==1.
    assert np.max(data.mesh_points[1,:,0])==2.
    metadata=json.loads((path/'metadata.json').read_text())
    assert metadata['solver_backend']=='cuda'
    assert metadata['gpu_worker_count']==1
