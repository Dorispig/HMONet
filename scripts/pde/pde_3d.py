"""3-D Winslow PDE interface for parameterized surfaces."""

import numpy as np
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.pde.pde_legacy import winslowSolver3D, NonuniformWinslowSolver3D


def solve_winslow_3d(
    boundary,
    nx,
    ny,
    nz,
    initial_grid=None,
    max_iter=1000,
    tolerance=1e-6,
    parameter_grid=None,
    return_info=False,
):
    counts = {
        "x_min": (ny, nz),
        "x_max": (ny, nz),
        "y_min": (nx, nz),
        "y_max": (nx, nz),
        "z_min": (nx, ny),
        "z_max": (nx, ny),
    }
    if parameter_grid is None:
        faces = {
            name: boundary.surfaces()[name].sample(*counts[name]).transpose(1, 0, 2)
            for name in boundary.names
        }
    else:
        faces = {}
        for name in boundary.names:
            u, v = parameter_grid[name]
            U, V = np.meshgrid(np.asarray(u), np.asarray(v), indexing="ij")
            faces[name] = boundary.surfaces()[name].evaluate(U, V).transpose(1, 0, 2)
    legacy = {
        "left": faces["x_min"],
        "right": faces["x_max"],
        "front": faces["y_min"],
        "back": faces["y_max"],
        "bottom": faces["z_min"],
        "top": faces["z_max"],
    }
    solver = winslowSolver3D(
        nx,
        ny,
        nz,
        legacy,
        initial_points=initial_grid,
        max_iter=max_iter,
        tol=tolerance,
    )
    result = solver.solve()
    if return_info:
        difference = float(solver.last_relative_difference)
        return result, {
            "iterations": int(solver.last_iterations),
            "relative_difference": difference,
            "converged": bool(np.isfinite(result).all() and difference < tolerance),
        }
    return result
