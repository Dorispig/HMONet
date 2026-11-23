"""Per-dataset GPU logs; CPU generation retains its terminal output."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from functools import wraps
from inspect import signature
from pathlib import Path
import time
import traceback
import os

DEFAULT_LOG_ROOT = Path(__file__).resolve().parents[3] / 'outputs/gpu_queue/generation'


def log_gpu_generation(function):
    """Wrap the complete solve/save/audit operation, including failure traces."""
    parameters = signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = parameters.bind(*args, **kwargs)
        bound.apply_defaults()
        options = bound.arguments
        if options['solver_backend'] != 'cuda' or os.environ.get('HMONET_DATASET_GPU_WORKER') == '1':
            return function(*args, **kwargs)
        name = options['name']
        if not name or Path(name).name != name or name in {'.', '..'}:
            raise ValueError('name must be a single nonempty directory name')
        # A skipped dataset must not erase the log of its successful creation.
        target = Path(options['output_root']) / name
        if target.exists() and not options['overwrite']:
            return function(*args, **kwargs)
        root = Path(options['queue_log_root'] or DEFAULT_LOG_ROOT).resolve()
        directory = root / '3D' / name
        directory.resolve().relative_to(root)
        directory.mkdir(parents=True, exist_ok=True)
        for path in [directory/'generation.log', *directory.glob('gpu_*.log')]:
            if path.is_symlink():
                raise ValueError(f'Refusing symlink log file: {path}')
            if path.is_file():
                path.unlink()
        logfile = directory/'generation.log'
        print(f'[GPU generation] start {name}; log={logfile}', flush=True)
        started = time.perf_counter()
        try:
            with logfile.open('w', encoding='utf-8', buffering=1) as stream:
                with redirect_stdout(stream), redirect_stderr(stream):
                    print(f'Dataset: {name}; started: {datetime.now().isoformat(timespec="seconds")}')
                    print(f'GPUs: {options["gpus"] or [0]}; grid: {options["grid_shape"]}; '
                          f'timing_samples: {options["timing_samples"]}')
                    print('Scheduling: one complete dataset on one GPU; samples generated sequentially.')
                    try:
                        result = function(*args, **kwargs)
                    except BaseException:
                        traceback.print_exc()
                        print(f'FAILED/INTERRUPTED after {time.perf_counter()-started:.2f} s', flush=True)
                        raise
                    print(f'COMPLETED in {time.perf_counter()-started:.2f} s', flush=True)
        except BaseException as error:
            print(f'[GPU generation] failed/interrupted {name}: {error}; log={logfile}', flush=True)
            raise
        print(f'[GPU generation] completed {name}; log={logfile}', flush=True)
        return result

    return wrapped
