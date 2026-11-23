"""Optional float64 sparse direct backend for CPU-assembled Winslow systems.

The small ctypes binding targets the cuDSS 0.4 C ABI (see the optional pinned
requirements). PyTorch owns device buffers; cuDSS owns only its LU workspace.
No discretization, row scaling, boundary elimination or nonlinear stopping
criterion is implemented here. General nonsymmetric matrices are supported.
"""
import ctypes as ct
from functools import lru_cache
from importlib.metadata import distribution, PackageNotFoundError
import os
from pathlib import Path
import sys

import numpy as np


class LinearSolveError(RuntimeError):
    """A failed direct solve must not be silently accepted as a mesh label."""


@lru_cache(maxsize=1)
def load_cudss():
    import torch
    if not str(torch.version.cuda).startswith('12.'):
        raise RuntimeError('This cuDSS backend requires CUDA 12 PyTorch; '
                           'use CPU generation or a compatible CUDA 12 environment.')
    try:
        package = distribution('nvidia-cudss-cu12')
    except PackageNotFoundError as error:
        raise RuntimeError('CUDA dataset generation requires cuDSS: '
                           'python -m pip install -r requirements-gpu-generation.txt; '
                           'CPU generation remains available.') from error
    if not package.version.startswith('0.4.0.2'):
        raise RuntimeError(f'Untested cuDSS ABI {package.version}; install '
                           'requirements-gpu-generation.txt (cuDSS 0.4).')
    root = Path(package.locate_file('nvidia/cu12'))
    handles = []
    if os.name == 'nt':
        paths = [root/'bin', Path(torch.__file__).parent/'lib',
                 Path(sys.prefix)/'bin', Path(sys.prefix)/'Library/bin']
        # Keep directory handles alive for delayed dependent-DLL loading.
        handles = [os.add_dll_directory(str(p)) for p in paths if p.is_dir()]
        filename = root/'bin/cudss64_0.dll'
        loader = ct.WinDLL
    else:
        filename = root/'lib/libcudss.so.0'
        if not filename.exists():
            filename = root/'lib/libcudss.so'
        loader = ct.CDLL
    try:
        lib = loader(str(filename))
    except OSError as error:
        raise RuntimeError(f'Cannot load cuDSS at {filename}. Install the CUDA 12 '
                           'dependencies from requirements-gpu-generation.txt. '
                           f'Loader error: {error}') from error
    lib._directory_handles = handles
    P, I, L, S = ct.c_void_p, ct.c_int, ct.c_int64, ct.c_size_t
    signatures = {
        'cudssCreate': [ct.POINTER(P)], 'cudssDestroy': [P],
        'cudssConfigCreate': [ct.POINTER(P)], 'cudssConfigDestroy': [P],
        'cudssDataCreate': [P, ct.POINTER(P)], 'cudssDataDestroy': [P, P],
        'cudssSetStream': [P, P], 'cudssGetProperty': [I, ct.POINTER(I)],
        'cudssConfigSet': [P, I, P, S],
        'cudssMatrixCreateCsr': [ct.POINTER(P), L, L, L, P, P, P, P, I, I, I, I, I],
        'cudssMatrixCreateDn': [ct.POINTER(P), L, L, L, P, I, I],
        'cudssMatrixDestroy': [P],
        'cudssExecute': [P, I, P, P, P, P, P],
        'cudssDataGet': [P, P, I, P, S, ct.POINTER(S)],
    }
    for name, args in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = args, I
    version = []
    for prop in (0, 1):
        value = I()
        _check(lib.cudssGetProperty(prop, ct.byref(value)), 'version query')
        version.append(value.value)
    if version != [0, 4]:
        raise RuntimeError(f'cuDSS library ABI {version} differs from supported 0.4')
    return lib


def _check(status, operation):
    if status:
        raise LinearSolveError(f'cuDSS {operation} failed (status={status})')


def check_residual(matrix, solution, rhs):
    """Check each RHS against the original system, without row normalization."""
    residual = np.linalg.norm(matrix @ solution - rhs, axis=0)
    relative = residual / np.maximum(np.linalg.norm(rhs, axis=0), np.finfo(float).tiny)
    maximum = float(np.max(relative))
    if not np.isfinite(solution).all() or not np.isfinite(maximum) or maximum > 1e-9:
        raise LinearSolveError(f'cuDSS original-system relative residual {maximum:.3e} exceeds 1e-9')
    return maximum


class CudssSolver:
    """Callable SciPy-CSR to NumPy solve; factor once for all coordinate RHSs."""
    def __init__(self, device='cuda:0'):
        import torch
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('cuDSS requires a CUDA device')
        self.lib = load_cudss()
        self.max_residual = 0.0
        self.solve_count = 0

    def diagnostics(self):
        return dict(solver_backend='cuda', solver_engine='cudss', dtype='float64',
                    linear_solver='cudss_lu', linear_solves=self.solve_count,
                    max_linear_relative_residual=self.max_residual)

    def __call__(self, matrix, rhs):
        import torch
        matrix = matrix.astype(np.float64).tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.sort_indices()
        rhs = np.asarray(rhs, dtype=np.float64)
        vector = rhs.ndim == 1
        if vector:
            rhs = rhs[:, None]
        n = matrix.shape[0]
        if matrix.shape != (n, n) or rhs.ndim != 2 or rhs.shape[0] != n:
            raise ValueError('Expected square A and matching one/two-dimensional RHS')
        if not np.isfinite(matrix.data).all() or not np.isfinite(rhs).all():
            raise LinearSolveError('Nonfinite linear system')
        if max(n, matrix.nnz) > np.iinfo(np.int32).max:
            raise ValueError('cuDSS adapter currently uses int32 CSR indices')
        lib = self.lib
        handle, config, data, a, b, x = [ct.c_void_p() for _ in range(6)]
        with torch.cuda.device(self.device):
            rows = torch.as_tensor(matrix.indptr.astype(np.int32), device=self.device)
            columns = torch.as_tensor(matrix.indices.astype(np.int32), device=self.device)
            values = torch.as_tensor(matrix.data, device=self.device)
            # Transposed contiguous storage is column-major (n, nrhs).
            bv = torch.as_tensor(np.ascontiguousarray(rhs.T), device=self.device)
            xv = torch.zeros_like(bv)
            stream = torch.cuda.current_stream(self.device)
            try:
                _check(lib.cudssCreate(ct.byref(handle)), 'create')
                _check(lib.cudssSetStream(handle, stream.cuda_stream), 'set stream')
                _check(lib.cudssConfigCreate(ct.byref(config)), 'create config')
                _check(lib.cudssDataCreate(handle, ct.byref(data)), 'create data')
                # BTF/COLAMD reordering for general nonsymmetric systems.
                algorithm = ct.c_int(1)
                _check(lib.cudssConfigSet(config, 0, ct.byref(algorithm), ct.sizeof(algorithm)), 'set reordering')
                # CUDA_R_32I=10, CUDA_R_64F=1; GENERAL/FULL/BASE_ZERO=0.
                _check(lib.cudssMatrixCreateCsr(ct.byref(a), n, n, matrix.nnz,
                    rows.data_ptr(), None, columns.data_ptr(), values.data_ptr(),
                    10, 1, 0, 0, 0), 'create CSR')
                for target, buffer in ((b, bv), (x, xv)):
                    _check(lib.cudssMatrixCreateDn(ct.byref(target), n, rhs.shape[1], n,
                        buffer.data_ptr(), 1, 0), 'create dense')
                for phase in (1, 2, 8):  # Analysis, factorization, solve.
                    _check(lib.cudssExecute(handle, phase, config, data, a, x, b), f'phase {phase}')
                stream.synchronize()
                info, written = ct.c_int(), ct.c_size_t()
                _check(lib.cudssDataGet(handle, data, 0, ct.byref(info),
                                      ct.sizeof(info), ct.byref(written)), 'get numerical status')
                if info.value:
                    raise LinearSolveError(f'cuDSS numerical failure (info={info.value})')
                solution = xv.cpu().numpy().T.copy()
            finally:
                # Work and buffers must finish before descriptors/workspace die.
                try:
                    stream.synchronize()
                finally:
                    for descriptor in (x, b, a):
                        if descriptor.value:
                            lib.cudssMatrixDestroy(descriptor)
                    if data.value:
                        lib.cudssDataDestroy(handle, data)
                    if config.value:
                        lib.cudssConfigDestroy(config)
                    if handle.value:
                        lib.cudssDestroy(handle)
        self.max_residual = max(self.max_residual, check_residual(matrix, solution, rhs))
        self.solve_count += 1
        return solution[:, 0] if vector else solution
