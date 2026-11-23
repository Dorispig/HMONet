"""Float64 Torch backend for the existing uniform Winslow discretizations.

Keep the 9/19-point mixed-derivative stencils and nonlinear update unchanged.
Dirichlet unknowns are eliminated; Jacobi-preconditioned BiCGSTAB solves each
coordinate RHS. A true linear residual is required before accepting a solve.
The CPU Torch device is useful for numerical regression against SciPy.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import math
import numpy as np
import torch

_OPTIONS = ContextVar('winslow_generation_options', default=None)


@contextmanager
def generation_solver_context(options):
    token = _OPTIONS.set(options)
    try:
        yield
    finally:
        _OPTIONS.reset(token)


def current_solver_options():
    return _OPTIONS.get() or {}


class LinearSolveError(RuntimeError):
    """An unconverged GPU solve must never become a silently accepted label."""


def _bicgstab(matvec, rhs, initial, tolerance, max_iter):
    # Rows have already been divided by the diagonal (Jacobi preconditioning).
    x = initial.clone()
    norm_b = torch.linalg.vector_norm(rhs)
    target = tolerance * max(float(norm_b), torch.finfo(rhs.dtype).tiny)
    iterations = 0
    # Restarts recover from loss of biorthogonality and verify the true residual.
    for restart in range(8):
        residual = rhs - matvec(x)
        if float(torch.linalg.vector_norm(residual)) <= target:
            return x, iterations, float(torch.linalg.vector_norm(residual)) / max(float(norm_b), 1e-300)
        shadow = residual.clone()
        p = torch.zeros_like(x)
        v = torch.zeros_like(x)
        rho_old = alpha = omega = torch.ones((), dtype=x.dtype, device=x.device)
        for _ in range(max_iter - iterations):
            iterations += 1
            rho = torch.dot(shadow, residual)
            if not bool(torch.isfinite(rho)) or float(rho.abs()) < 1e-290:
                break
            beta = (rho / rho_old) * (alpha / omega)
            p = residual + beta * (p - omega * v)
            v = matvec(p)
            denominator = torch.dot(shadow, v)
            if not bool(torch.isfinite(denominator)) or float(denominator.abs()) < 1e-290:
                break
            alpha = rho / denominator
            s = residual - alpha * v
            if float(torch.linalg.vector_norm(s)) <= target:
                x = x + alpha * p
                break  # Recompute the true residual before accepting.
            t = matvec(s)
            tt = torch.dot(t, t)
            if not bool(torch.isfinite(tt)) or float(tt) < 1e-290:
                break
            omega = torch.dot(t, s) / tt
            x = x + alpha * p + omega * s
            residual = s - omega * t
            if not bool(torch.isfinite(x).all()) or float(omega.abs()) < 1e-290:
                break
            if float(torch.linalg.vector_norm(residual)) <= target:
                break
            rho_old = rho
        if iterations >= max_iter:
            break
    relative = float(torch.linalg.vector_norm(rhs - matvec(x))) / max(float(norm_b), 1e-300)
    if math.isfinite(relative) and relative <= tolerance:
        return x, iterations, relative
    raise LinearSolveError(f'Winslow linear solve failed: relative residual={relative:.3e}, '
                           f'tolerance={tolerance:.3e}, iterations={iterations}/{max_iter}')


def _topology(shape, device):
    ids = torch.arange(math.prod(shape), device=device).reshape(shape)
    interior = ids[tuple(slice(1, -1) for _ in shape)].reshape(-1)
    strides = [1, shape[-1]]
    if len(shape) == 3:
        strides.append(shape[-1] * shape[-2])
    offsets = [0]
    for step in strides:
        offsets += [-step, step]
    for a in range(len(strides)):
        for b in range(a + 1, len(strides)):
            u, v = strides[a], strides[b]
            offsets += [u + v, u - v, -u + v, -u - v]
    neighbors = interior[:, None] + torch.tensor(offsets, device=device)
    lookup = torch.full((ids.numel(),), -1, dtype=torch.long, device=device)
    lookup[interior] = torch.arange(interior.numel(), device=device)
    local = lookup[neighbors]
    return interior, neighbors, local, strides


def _coefficients(points, ids, strides, sizes):
    steps = [1.0 / (n - 1) for n in sizes]
    derivatives = [(points[ids + s] - points[ids - s]) / (2 * h)
                   for s, h in zip(strides, steps)]
    dot = lambda a, b: (a * b).sum(-1)
    if len(strides) == 2:
        u, v = derivatives
        diagonal = [dot(v, v) / steps[0]**2, dot(u, u) / steps[1]**2]
        cross = [-dot(u, v) / (2 * steps[0] * steps[1])]
    else:
        u, v, w = derivatives
        g11, g22, g33 = dot(u, u), dot(v, v), dot(w, w)
        g12, g13, g23 = dot(u, v), dot(u, w), dot(v, w)
        a11, a22, a33 = g22*g33-g23*g23, g11*g33-g13*g13, g11*g22-g12*g12
        scale = (a11+a22+a33).clamp_min(1e-14)
        diagonal = [a11/scale/steps[0]**2, a22/scale/steps[1]**2, a33/scale/steps[2]**2]
        cross = [(g13*g23-g12*g33)/scale/(2*steps[0]*steps[1]),
                 (g12*g23-g13*g22)/scale/(2*steps[0]*steps[2]),
                 (g12*g13-g11*g23)/scale/(2*steps[1]*steps[2])]
    values = [-2 * sum(diagonal)]
    for value in diagonal:
        values += [value, value]
    for value in cross:
        values += [value, -value, -value, value]
    return torch.stack(values, -1)


@torch.no_grad()
def solve_uniform_grid(initial, *, device='cuda:0', max_iter=1000,
                       tolerance=1e-6, linear_tolerance=1e-12, linear_max_iter=4000):
    """Solve a (ny,nx,2) or (nz,ny,nx,3) grid with fixed boundary nodes."""
    grid = np.asarray(initial, dtype=np.float64)
    dim = grid.shape[-1]
    shape = grid.shape[:-1]
    if dim not in (2, 3) or len(shape) != dim or min(shape) < 3:
        raise ValueError('Expected 2-D/3-D structured coordinates, each dimension >=3')
    if not np.isfinite(grid).all():
        raise ValueError('Nonfinite initial grid')
    if (max_iter < 1 or linear_max_iter < 1 or not 0 < tolerance < 1
            or not 0 < linear_tolerance < 1):
        raise ValueError('Positive iteration limits and tolerances in (0,1) required')
    points = torch.as_tensor(grid.copy(), dtype=torch.float64, device=device).reshape(-1, dim)
    ids, neighbors, local, strides = _topology(shape, points.device)
    interior_mask = local >= 0
    safe_local = local.clamp_min(0)
    fixed_neighbors = points[neighbors].clone()
    total_linear_iterations = 0
    max_residual = 0.
    for iteration in range(1, max_iter + 1):
        coefficients = _coefficients(points, ids, strides, list(reversed(shape)))
        diag = coefficients[:, 0]
        if not bool(torch.isfinite(coefficients).all()) or bool((diag.abs() < 1e-290).any()):
            raise LinearSolveError('Nonfinite or singular Winslow stencil')
        normalized = coefficients / diag[:, None]
        weights = normalized * interior_mask
        rhs = -(normalized[..., None] * (~interior_mask)[..., None] * fixed_neighbors).sum(1)
        def matvec(x):
            return (weights * x[safe_local]).sum(1)
        old = points.clone()
        for component in range(dim):
            result, count, residual = _bicgstab(matvec, rhs[:, component],
                                               old[ids, component], linear_tolerance, linear_max_iter)
            # Check the unpreconditioned system too, not only the solver norm.
            original_r = (matvec(result) - rhs[:, component]) * diag
            original_b = rhs[:, component] * diag
            original_relative = float(torch.linalg.vector_norm(original_r)) / max(float(torch.linalg.vector_norm(original_b)), 1e-300)
            if not math.isfinite(original_relative) or original_relative > max(1e-10, 100*linear_tolerance):
                raise LinearSolveError(f'Original linear residual too large: {original_relative:.3e}')
            points[ids, component] = result
            total_linear_iterations += count
            max_residual = max(max_residual, original_relative)
        tiny = 1e-14 if dim == 3 else np.finfo(float).tiny
        difference = float(torch.linalg.vector_norm(points-old) / torch.linalg.vector_norm(old).clamp_min(tiny))
        if difference < tolerance:
            break
    info = dict(iterations=iteration, relative_difference=difference,
                converged=bool(difference < tolerance), solver_backend=points.device.type, solver_engine='torch',
                device=str(points.device), dtype='float64', linear_solver='bicgstab_jacobi',
                linear_tolerance=linear_tolerance, linear_max_iter=linear_max_iter,
                linear_iterations=total_linear_iterations, max_linear_relative_residual=max_residual)
    return points.cpu().numpy(), info


def solve_winslow_2d_torch(boundary, nx, ny, initial_grid=None, max_iter=1000,
                            tolerance=1e-6, return_info=False, **options):
    from .tfi_2d import tfi_from_curves
    from src.data.mesh.topology import GenerateQuadMeshBoundaryIndex
    grid = np.asarray(tfi_from_curves(boundary,nx=nx,ny=ny) if initial_grid is None else initial_grid,
                      dtype=np.float64).reshape(-1,2).copy()
    grid[GenerateQuadMeshBoundaryIndex(ny,nx)] = boundary.sample_closed(dict(bottom=nx,top=nx,left=ny,right=ny))[:-1]
    result, info = solve_uniform_grid(grid.reshape(ny,nx,2),max_iter=max_iter,tolerance=tolerance,**options)
    return (result,info) if return_info else result
