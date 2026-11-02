"""Sparse Q1 solver for steady scalar diffusion on generated 2-D meshes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.sparse import linalg as sparse_linalg

from .boundary_conditions import DirichletBoundary
from .q1_fem import assemble_q1_system


@dataclass
class PoissonResult:
    solution: np.ndarray
    success: bool
    solver: str
    assembly_seconds: float
    solve_seconds: float
    total_seconds: float
    residual_norm: float
    relative_residual: float
    iterations: int
    free_dof_count: int
    condition_estimate: float | None
    minimum_element_jacobian: float
    maximum_element_jacobian: float

    def metrics(self):
        result = asdict(self)
        result.pop("solution")
        return result


def _condition_number_estimate(matrix):
    """Estimate the SPD spectral condition number from extreme eigenvalues."""
    if matrix.shape[0] < 2:
        return 1.0
    largest = float(
        sparse_linalg.eigsh(matrix, k=1, which="LA", return_eigenvectors=False)[0]
    )
    smallest = float(
        sparse_linalg.eigsh(matrix, k=1, which="SA", return_eigenvectors=False)[0]
    )
    return largest / smallest if smallest > 0 else float("inf")


def solve_poisson_2d(
    points,
    shape,
    boundary: DirichletBoundary,
    *,
    conductivity=1.0,
    source=0.0,
    solver="direct",
    tolerance=1.0e-10,
    max_iterations=None,
    estimate_condition=False,
    determinant_tolerance=1.0e-12,
):
    """Solve ``-div(k grad(u)) = source`` with strong Dirichlet conditions."""
    started = time.perf_counter()
    assembly_started = time.perf_counter()
    matrix, load, diagnostics = assemble_q1_system(
        points,
        shape,
        conductivity=conductivity,
        source=source,
        determinant_tolerance=determinant_tolerance,
    )
    assembly_seconds = time.perf_counter() - assembly_started
    node_count = int(np.prod(shape))
    constrained = np.zeros(node_count, dtype=bool)
    constrained[boundary.indices] = True
    free = np.flatnonzero(~constrained)
    if len(free) == 0:
        raise ValueError("At least one unconstrained degree of freedom is required")
    reduced = matrix[free][:, free].tocsr()
    rhs = load[free] - matrix[free][:, boundary.indices] @ boundary.values
    condition = _condition_number_estimate(reduced) if estimate_condition else None

    solve_started = time.perf_counter()
    iterations = 0
    if solver == "direct":
        free_solution = sparse_linalg.spsolve(reduced, rhs)
    elif solver == "cg":
        def callback(_):
            nonlocal iterations
            iterations += 1

        arguments = dict(maxiter=max_iterations, callback=callback)
        try:
            # SciPy >= 1.12 distinguishes relative and absolute tolerances.
            free_solution, info = sparse_linalg.cg(
                reduced, rhs, rtol=float(tolerance), atol=0.0, **arguments
            )
        except TypeError:
            # Retain compatibility with older project environments.
            free_solution, info = sparse_linalg.cg(
                reduced, rhs, tol=float(tolerance), **arguments
            )
        if info != 0:
            raise RuntimeError(
                f"Conjugate-gradient solve did not converge (info={info}, iterations={iterations})"
            )
    else:
        raise ValueError("solver must be 'direct' or 'cg'")
    solve_seconds = time.perf_counter() - solve_started
    if not np.isfinite(free_solution).all():
        raise FloatingPointError("The downstream solution contains non-finite values")

    solution = np.zeros(node_count, dtype=float)
    solution[boundary.indices] = boundary.values
    solution[free] = free_solution
    residual = reduced @ free_solution - rhs
    residual_norm = float(np.linalg.norm(residual))
    rhs_norm = float(np.linalg.norm(rhs))
    return PoissonResult(
        solution=solution,
        success=True,
        solver=solver,
        assembly_seconds=assembly_seconds,
        solve_seconds=solve_seconds,
        total_seconds=time.perf_counter() - started,
        residual_norm=residual_norm,
        relative_residual=residual_norm / max(rhs_norm, np.finfo(float).eps),
        iterations=iterations,
        free_dof_count=len(free),
        condition_estimate=condition,
        minimum_element_jacobian=diagnostics.minimum_determinant,
        maximum_element_jacobian=diagnostics.maximum_determinant,
    )
