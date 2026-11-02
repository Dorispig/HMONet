"""Downstream numerical solvers that consume generated structured meshes."""

from .boundary_conditions import DirichletBoundary, build_dirichlet_boundary
from .poisson_2d import PoissonResult, solve_poisson_2d
from .q1_fem import MeshAdmissibilityError, assemble_q1_system

__all__ = [
    "DirichletBoundary",
    "MeshAdmissibilityError",
    "PoissonResult",
    "assemble_q1_system",
    "build_dirichlet_boundary",
    "solve_poisson_2d",
]
