"""Parameter-curve aware 2-D transfinite interpolation."""

import numpy as np


def tfi_from_curves(boundary, xi=None, eta=None, nx=None, ny=None, parameter_grid=None):
    if parameter_grid is not None:
        g = np.asarray(parameter_grid)
        U, V = g[..., 0], g[..., 1]
    else:
        if xi is None:
            xi = np.linspace(0, 1, nx)
        if eta is None:
            eta = np.linspace(0, 1, ny)
        xi = np.asarray(xi)
        eta = np.asarray(eta)
        U, V = np.meshgrid(xi, eta, indexing="xy")
    left = boundary.left.evaluate(V)
    right = boundary.right.evaluate(V)
    bottom = boundary.bottom.evaluate(U)
    top = boundary.top.evaluate(U)
    p00 = boundary.bottom.evaluate(0)
    p10 = boundary.bottom.evaluate(1)
    p01 = boundary.top.evaluate(0)
    p11 = boundary.top.evaluate(1)
    return (
        (1 - U[..., None]) * left
        + U[..., None] * right
        + (1 - V[..., None]) * bottom
        + V[..., None] * top
        - ((1 - U) * (1 - V))[..., None] * p00
        - (U * (1 - V))[..., None] * p10
        - ((1 - U) * V)[..., None] * p01
        - (U * V)[..., None] * p11
    )

def tfi_from_parameter_grid(boundary, parameter_grid):
    """Evaluate TFI queries supplied as an array ending in ``(xi, eta)``."""
    grid = np.asarray(parameter_grid)
    if grid.shape[-1] != 2:
        raise ValueError("parameter_grid must end in 2")
    return tfi_from_curves(boundary, parameter_grid=grid)
