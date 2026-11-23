"""Dynamic GPU scheduling with one complete dataset as the atomic task."""
from pathlib import Path
import sys

from src.utils.gpu_queue import GpuTask, run_gpu_queue
from .generation_logging import DEFAULT_LOG_ROOT


def validate_gpu_options(solver_backend, gpus):
    if solver_backend not in ('cpu', 'cuda'):
        raise ValueError('solver_backend must be cpu or cuda')
    if solver_backend == 'cpu':
        if gpus:
            raise ValueError('--gpus requires --solver-backend cuda')
        return []
    import torch
    requested = [0] if gpus is None else list(gpus)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError('Specify one or more distinct GPU indices')
    count = torch.cuda.device_count()
    if not torch.cuda.is_available() or any(isinstance(g, bool) or not isinstance(g, int) or g < 0 or g >= count for g in requested):
        raise ValueError(f'Invalid CUDA devices {requested}; {count} visible device(s).')
    from src.numerics.traditional.cudss_solver import load_cudss
    load_cudss()
    return requested


def run_registered_gpu_datasets(jobs, **options):
    """Each job includes cap solves, all samples, validation, examples and save."""
    if not jobs:
        return []
    from .make_dataset_parametric_3d import ROOT, DEFAULT_OUTPUT_ROOT, build_dataset_name
    gpu_ids = validate_gpu_options('cuda', options.get('gpus'))
    destination = Path(options.get('output_root', DEFAULT_OUTPUT_ROOT)).resolve()
    size = options.get('size', 500)
    tasks, outputs = [], []
    for family, split, version, nx, ny, nz in jobs:
        name = build_dataset_name(family, split, version, size, nx, ny, nz)
        command = [sys.executable, '-u', str(ROOT/'src/data/generation/make_dataset_parametric_3d.py'),
                   '--family', family, '--split', split, '--version', str(version),
                   '--size', str(size), '--grid-shape', f'{nx}x{ny}x{nz}',
                   '--output-root', str(destination), '--solver-backend', 'cuda', '--gpus', '0',
                   '--workers', '1', '--timing-samples', str(options.get('timing_samples', 3)),
                   '--winslow-max-iter', str(options.get('winslow_max_iter', 1000)),
                   '--winslow-tolerance', str(options.get('winslow_tolerance', 1e-6))]
        if options.get('overwrite'):
            command.append('--force')
        for low, high in options.get('parameter_ranges') or []:
            command.extend(['--parameter-range', str(low), str(high)])
        tasks.append(GpuTask(name=name, command=command,
                     environment={'HMONET_DATASET_GPU_WORKER': '1', 'PYTHONIOENCODING': 'utf-8'},
                     log_relative_path=Path('3D')/name/'generation.log',
                     clear_log_directory=True,
                     metadata={'dataset': name, 'task_scope': 'complete_dataset'}))
        outputs.append(destination/name)
    run_gpu_queue(tasks, gpu_ids, cwd=ROOT,
                  log_root=Path(options.get('queue_log_root') or DEFAULT_LOG_ROOT).resolve(),
                  validate_devices=False)
    return outputs
