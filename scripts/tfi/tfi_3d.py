"""Exact Gordon--Hall interpolation from six compatible parameter surfaces."""

import numpy as np


def tfi_from_surfaces(boundary, xi=None, eta=None, zeta=None, parameter_grid=None):
    """Blend six faces minus twelve exact edges plus eight corners.

    Integer axis arguments generate a uniform grid. Arrays specify arbitrary
    tensor-product logical coordinates; parameter_grid accepts (..., 3) queries.
    """
    if parameter_grid is None:
        axes = [
            np.linspace(0, 1, int(a)) if np.isscalar(a) else np.asarray(a, dtype=float)
            for a in (xi, eta, zeta)
        ]
        w, v, u = np.meshgrid(*axes[::-1], indexing="ij")
    else:
        queries = np.asarray(parameter_grid, dtype=float)
        if queries.shape[-1] != 3:
            raise ValueError("parameter_grid must have shape (..., 3)")
        u, v, w = np.moveaxis(queries, -1, 0)
    if any(not np.isfinite(a).all() or np.any((a < 0) | (a > 1)) for a in (u, v, w)):
        raise ValueError("Logical parameters must be finite and in [0, 1]")
    weights = [(1 - a, a) for a in (u, v, w)]
    faces = boundary.evaluate_faces(u, v, w)
    result = sum(
        weights[axis][side][..., None] * faces[2 * axis + side]
        for axis in range(3)
        for side in (0, 1)
    )
    # Retain curved edges: corner-only interpolation breaks curved boundaries.
    for i in (0, 1):
        for j in (0, 1):
            face = boundary.x_min if i == 0 else boundary.x_max
            result -= (weights[0][i] * weights[1][j])[..., None] * face.evaluate(
                np.full_like(w, j), w
            )
    for i in (0, 1):
        for k in (0, 1):
            face = boundary.x_min if i == 0 else boundary.x_max
            result -= (weights[0][i] * weights[2][k])[..., None] * face.evaluate(
                v, np.full_like(v, k)
            )
    for j in (0, 1):
        for k in (0, 1):
            face = boundary.y_min if j == 0 else boundary.y_max
            result -= (weights[1][j] * weights[2][k])[..., None] * face.evaluate(
                u, np.full_like(u, k)
            )
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                face = boundary.z_min if k == 0 else boundary.z_max
                result += (weights[0][i] * weights[1][j] * weights[2][k])[
                    ..., None
                ] * face.evaluate(float(i), float(j))
    return result
