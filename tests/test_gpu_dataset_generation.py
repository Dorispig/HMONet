"""Numerical parity and queue behavior; CUDA tests skip on CPU-only hosts."""
import numpy as np
import pytest
import torch

from src.numerics.traditional.cudss_solver import CudssSolver, load_cudss, check_residual, LinearSolveError
from src.numerics.traditional.winslow_vectorized_2d import solve_winslow_2d_vectorized
from src.data.geometry.boundary_generator_2d import four_convex_semicircle_boundary
from src.data.generation.gpu_datasets import validate_gpu_options


def direct_solver(device):
    if device == 'cpu':
        from scipy.sparse.linalg import spsolve
        return lambda a,b: spsolve(a,b,permc_spec='MMD_AT_PLUS_A')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from importlib.metadata import distribution, PackageNotFoundError
    try:
        distribution('nvidia-cudss-cu12')
    except PackageNotFoundError:
        pytest.skip('Optional cuDSS not installed')
    return CudssSolver(device)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_2d_stencil_matches_cpu(device):
    b = four_convex_semicircle_boundary(arc_angle_deg=70)
    reference, info = solve_winslow_2d_vectorized(b,13,11,tolerance=1e-9,return_info=True)
    direct = direct_solver(device)
    actual, gpu = solve_winslow_2d_vectorized(b,13,11,tolerance=1e-9,return_info=True,linear_solver=direct)
    assert info['converged'] and gpu['converged']
    np.testing.assert_allclose(actual,reference,atol=2e-9,rtol=0)
    assert gpu['iterations'] == info['iterations']
    if device != 'cpu':
        assert direct.max_residual < 1e-10


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_3d_mixed_stencil_matches_cpu(device):
    # Prescribed warped faces exercise all mixed derivatives, not only a box.
    from src.numerics.traditional.winslow_legacy import winslowSolver3D
    z,y,x = np.meshgrid(np.linspace(0,1,7),np.linspace(0,1,8),np.linspace(0,1,9),indexing='ij')
    grid=np.stack([x+.08*y*z, y+.07*x*z, z+.06*x*y],-1)
    grid[1:-1,1:-1,1:-1,0] += .015
    faces=dict(left=grid[:,:,0],right=grid[:,:,-1],front=grid[:,0],back=grid[:,-1],bottom=grid[0],top=grid[-1])
    cpu=winslowSolver3D(9,8,7,faces,initial_points=grid,max_iter=100,tol=1e-9)
    expected=cpu.solve()
    other=winslowSolver3D(9,8,7,faces,initial_points=grid,max_iter=100,tol=1e-9)
    actual=other.solve(linear_solver=direct_solver(device))
    assert other.last_relative_difference < 1e-9
    assert other.last_iterations == cpu.last_iterations
    np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=0)
    out=actual.reshape(grid.shape)
    np.testing.assert_array_equal(out[0],grid[0])
    np.testing.assert_array_equal(out[:,:,-1],grid[:,:,-1])


def test_linear_failure_is_not_a_success():
    from scipy.sparse import eye
    with pytest.raises(LinearSolveError,match='relative residual'):
        check_residual(eye(2),np.zeros((2,2)),np.ones((2,2)))


def test_backend_validation():
    assert validate_gpu_options('cpu',None)==[]
    with pytest.raises(ValueError,match='requires'):
        validate_gpu_options('cpu',[0])
    with pytest.raises(ValueError,match='distinct'):
        validate_gpu_options('cuda',[0,0])


def test_queue_tasks_are_whole_datasets(monkeypatch, tmp_path):
    from src.data.generation import gpu_datasets
    captured={}
    monkeypatch.setattr(gpu_datasets,'validate_gpu_options',lambda *args: [2,3])
    def run(tasks,gpus,**kwargs):
        captured.update(tasks=tasks,gpus=gpus,options=kwargs)
    monkeypatch.setattr(gpu_datasets,'run_gpu_queue',run)
    jobs=[('sidewall_deformed_block','extrapolation',2,21,21,11),
          ('sidewall_deformed_block','interpolation',2,21,21,11)]
    outputs=gpu_datasets.run_registered_gpu_datasets(jobs,size=500,gpus=[2,3],output_root=tmp_path,queue_log_root=tmp_path/'logs')
    assert len(captured['tasks'])==2 and len(outputs)==2
    assert captured['gpus']==[2,3]
    for task in captured['tasks']:
        command=task.command
        assert command[command.index('--size')+1]=='500'
        assert command[command.index('--gpus')+1]=='0'
        assert task.environment['HMONET_DATASET_GPU_WORKER']=='1'
        assert task.environment['PYTHONIOENCODING']=='utf-8'
        assert task.log_relative_path.parts==('3D',task.name,'generation.log')
        assert task.clear_log_directory


def test_generation_log_failure_and_skip_preserve_history(tmp_path, capsys):
    from src.data.generation.generation_logging import log_gpu_generation
    @log_gpu_generation
    def fake(name,output_root,solver_backend='cuda',overwrite=False,
             queue_log_root=None,gpus=None,grid_shape=(3,3,3),timing_samples=3,fail=False):
        if (output_root/name).exists() and not overwrite:
            print('[skip] existing')
            return
        print('Progress: 1/1 (100%)')
        if fail: raise ValueError('deliberate failure')
    kwargs=dict(name='example',output_root=tmp_path,queue_log_root=tmp_path/'logs')
    fake(**kwargs)
    log=tmp_path/'logs/3D/example/generation.log'
    original=log.read_text()
    assert 'Progress:' in original and 'COMPLETED' in original
    assert 'Progress:' not in capsys.readouterr().out
    (tmp_path/'example').mkdir()
    fake(**kwargs)
    assert log.read_text()==original
    with pytest.raises(ValueError,match='deliberate failure'):
        fake(**kwargs,overwrite=True,fail=True)
    assert 'Traceback' in log.read_text() and 'FAILED/INTERRUPTED' in log.read_text()
    assert 'COMPLETED' not in log.read_text()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_single_gpu_dataset_logs_and_is_readable(tmp_path, capsys):
    direct_solver('cuda:0')
    import json
    from src.data.generation.make_dataset_parametric_3d import generate_dataset
    from src.data.dataset import ParametricDataset3D
    path=generate_dataset('gpu_boxes','box',[dict(xmax=float(i+1)) for i in range(20)],
                          grid_shape=(5,5,5),output_root=tmp_path,
                          solver_backend='cuda',gpus=[0],timing_samples=3,queue_log_root=tmp_path/'logs')
    data=ParametricDataset3D(path)
    assert len(data)==20
    assert np.max(data.mesh_points[0,:,0])==1.
    assert np.max(data.mesh_points[1,:,0])==2.
    metadata=json.loads((path/'metadata.json').read_text())
    assert metadata['solver_backend']=='cuda'
    assert metadata['gpu_worker_count']==1

    logfile=tmp_path/'logs/3D/gpu_boxes/generation.log'
    content=logfile.read_text(encoding='utf-8')
    assert content.count('Sample ')==3
    assert content.count('Progress:')==10
    assert 'Progress: 20/20 (100%)' in content
    assert 'Progress:' not in capsys.readouterr().out
