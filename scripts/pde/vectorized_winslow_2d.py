"""Vectorized assembly of the legacy uniform-grid Winslow iteration.

The nine-point stencil, TFI initialization and convergence definition are the
same as winslowSolver2D. Coordinates share one scalar matrix and two RHSs.
This avoids duplicated assembly/factorization during large dataset generation.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from src.data.mesh.topology import GenerateQuadMeshBoundaryIndex
from scripts.tfi.tfi_2d import tfi_from_curves


def solve_winslow_2d_vectorized(
    boundary,
    nx,
    ny,
    initial_grid=None,
    max_iter=1000,
    tolerance=1e-6,
    return_info=False,
):
    if nx < 3 or ny < 3 or max_iter < 1 or tolerance <= 0:
        raise ValueError("Require nx,ny>=3, max_iter>=1 and positive tolerance")
    solution = (
        np.asarray(
            (
                tfi_from_curves(boundary, nx=nx, ny=ny)
                if initial_grid is None
                else initial_grid
            ),
            dtype=float,
        )
        .reshape(-1, 2)
        .copy()
    )
    boundary_ids = GenerateQuadMeshBoundaryIndex(ny, nx)
    points = boundary.sample_closed(dict(bottom=nx, top=nx, left=ny, right=ny))[:-1]
    solution[boundary_ids] = points
    count = nx * ny
    ids = np.arange(count).reshape(ny, nx)[1:-1, 1:-1].ravel()
    offsets = np.array([0, -1, 1, -nx, nx, -nx - 1, nx + 1, nx - 1, -nx + 1])
    rows = np.concatenate((np.repeat(ids, 9), boundary_ids))
    columns = np.concatenate(((ids[:, None] + offsets).ravel(), boundary_ids))
    rhs = np.zeros((count, 2))
    rhs[boundary_ids] = points
    dx, dy = 1 / (nx - 1), 1 / (ny - 1)
    for iteration in range(max_iter):
        xi = (solution[ids + 1] - solution[ids - 1]) / (2 * dx)
        eta = (solution[ids + nx] - solution[ids - nx]) / (2 * dy)
        alpha = np.sum(eta * eta, axis=1) / (dx * dx)
        gamma = np.sum(xi * xi, axis=1) / (dy * dy)
        beta = np.sum(xi * eta, axis=1) / (2 * dx * dy)
        coefficients = np.stack(
            (
                -2 * (alpha + gamma),
                alpha,
                alpha,
                gamma,
                gamma,
                -beta,
                -beta,
                beta,
                beta,
            ),
            axis=1,
        )
        matrix = coo_matrix(
            (
                np.concatenate((coefficients.ravel(), np.ones(len(boundary_ids)))),
                (rows, columns),
            ),
            shape=(count, count),
        ).tocsc()
        # Eliminate Dirichlet nodes instead of solving identity boundary rows.
        # This keeps boundary coordinates exact and avoids a convergence floor
        # from roundoff when interior coefficients grow on fine grids.
        interior_rows = matrix.tocsr()[ids]
        interior_matrix = interior_rows[:, ids].tocsc()
        interior_rhs = -(interior_rows[:, boundary_ids] @ points)
        new = solution.copy()
        new[ids] = spsolve(interior_matrix, interior_rhs, permc_spec="MMD_AT_PLUS_A")
        new[boundary_ids] = points
        difference = float(
            np.linalg.norm(new - solution)
            / max(np.linalg.norm(solution), np.finfo(float).tiny)
        )
        solution = new
        if difference < tolerance:
            break
        solution[boundary_ids] = points
    info = {
        "iterations": iteration + 1,
        "relative_difference": difference,
        "converged": bool(np.isfinite(solution).all() and difference < tolerance),
    }
    return (solution, info) if return_info else solution
