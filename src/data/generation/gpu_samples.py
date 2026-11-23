"""One spawned process per GPU; dynamically schedule independent mesh samples.

Only the parent writes dataset files. Failures/interrupts terminate the whole
owned queue; other user training processes are never touched.
"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
import signal


def validate_gpu_options(solver_backend, gpus):
    if solver_backend not in ('cpu','cuda'):
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
    if not torch.cuda.is_available() or any(isinstance(g,bool) or not isinstance(g,int) or g < 0 or g >= count for g in requested):
        raise ValueError(f'Invalid CUDA devices {requested}; {count} visible device(s). CPU remains available with --solver-backend cpu.')
    return requested


def _initialize(device):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_device(device)


def _shutdown(pools, abort):
    if abort:
        for pool in pools:
            for process in list((getattr(pool,'_processes',None) or {}).values()):
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
    for pool in pools:
        pool.shutdown(wait=True, cancel_futures=True)


def gpu_sample_map(function, payloads, gpus, *, linear_tolerance=1e-12, linear_max_iter=4000):
    """Yield (original_index, result) as completed; at most one job per GPU."""
    pools = []
    pending = {}
    tasks = iter(enumerate(payloads))
    aborted = True
    def submit(pool, gpu):
        item = next(tasks, None)
        if item is None:
            return
        index, payload = item
        options = dict(solver_backend='cuda',device=f'cuda:{gpu}',
                       linear_tolerance=linear_tolerance,linear_max_iter=linear_max_iter)
        print(f'[GPU generation] start GPU {gpu}: sample {index+1}/{len(payloads)}',flush=True)
        future = pool.submit(function,payload,options)
        pending[future] = (pool,gpu,index)
    try:
        for gpu in gpus:
            pool = ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn'),
                                       initializer=_initialize,initargs=(gpu,))
            pools.append(pool)
            submit(pool,gpu)
        while pending:
            done,_ = wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                pool,gpu,index = pending.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    raise RuntimeError(f'GPU {gpu}, sample {index}: {error}') from error
                print(f'[GPU generation] GPU {gpu}: sample {index+1}/{len(payloads)} completed',flush=True)
                yield index,result
                submit(pool,gpu)
        aborted = False
    finally:
        _shutdown(pools,aborted)
