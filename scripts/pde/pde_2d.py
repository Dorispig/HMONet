"""Winslow PDE interface supporting nonuniform boundary samples."""

import numpy as np
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.pde.pde_legacy import winslowSolver2D, NonuniformWinslowSolver2D


def solve_winslow_2d(
    boundary,
    nx,
    ny,
    initial_grid=None,
    max_iter=1000,
    tolerance=1e-6,
    parameter_grid=None,
    return_info=False,
):
    """Solve sampled boundaries, optionally returning iteration diagnostics."""
    if parameter_grid is None:
        samples = boundary.sample({"bottom": nx, "right": ny, "top": nx, "left": ny})
    else:
        samples = {
            "bottom": boundary.bottom.evaluate(np.asarray(parameter_grid["bottom"])),
            "right": boundary.right.evaluate(np.asarray(parameter_grid["right"])),
            "top": boundary.top.evaluate(np.asarray(parameter_grid["top"])),
            "left": boundary.left.evaluate(np.asarray(parameter_grid["left"])),
        }
    points = (
        boundary.sample_closed({"bottom": nx, "right": ny, "top": nx, "left": ny})
        if parameter_grid is None
        else np.vstack(
            [
                samples["bottom"],
                samples["right"][1:],
                samples["top"][::-1][1:],
                samples["left"][::-1][1:],
            ]
        )
    )
    solver = winslowSolver2D(
        nx, ny, points, initial_points=initial_grid, max_iter=max_iter, tol=tolerance
    )
    result = solver.solve()
    if return_info:
        difference = solver.last_relative_difference
        return result, {
            "iterations": int(solver.iterations_performed),
            "relative_difference": float(difference) if difference is not None else None,
            "converged": bool(
                difference is not None
                and np.isfinite(difference)
                and difference < tolerance
                and np.isfinite(result).all()
            ),
        }
    return result
