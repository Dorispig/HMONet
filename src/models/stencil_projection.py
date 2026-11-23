"""Exact projection reuse for ordered 3-D central finite-difference stencils.

Only geometry values are reused; network features and CCAP weights are still
evaluated at every query. This route must not be used for arbitrary queries or
coordinate autograd: equal coordinates need not be the same independent input.
"""

from functools import lru_cache

import torch

from src.numerics.differential import stencil_offsets
from src.data.backend.parametric_torch import (
    evaluate_surfaces_torch,
)

_NAMES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
_FACE_AXES = ((1, 2), (1, 2), (0, 2), (0, 2), (0, 1), (0, 1))
_EDGE_AXES = (2,) * 4 + (1,) * 4 + (0,) * 4
# Fixed incident faces for the twelve exact CCAP edges.  This is the same
# assignment as the general selected-face evaluator, without a GPU mask scan.
_EDGE_GROUPS = ((0, (0, 2, 4, 6)), (1, (1, 3, 5, 7)),
                (2, (8, 10)), (3, (9, 11)))


@lru_cache(maxsize=64)
def _projection_map(axes, device):
    offsets = stencil_offsets(3, device="cpu", dtype=torch.int64).tolist()
    keys = [tuple(row[axis] for axis in axes) for row in offsets]
    unique = list(dict.fromkeys(keys))
    return (
        torch.tensor([keys.index(key) for key in unique], device=device),
        torch.tensor([unique.index(key) for key in keys], device=device),
    )


def can_reuse_central_projections(boundaries, xi, eta, zeta):
    """Structural guards; the caller must explicitly identify a central stencil."""
    return (
        boundaries is not None
        and not hasattr(boundaries, "project_torch")
        and xi.ndim == 2 and xi.shape == eta.shape == zeta.shape
        and xi.shape[1] > 0 and xi.shape[1] % 19 == 0
        and not any(q.requires_grad for q in (xi, eta, zeta))
    )


def central_stencil_faces(boundaries, xi, eta, zeta):
    """Evaluate nine distinct tangential positions per face and center."""
    batch, count = xi.shape
    if not isinstance(boundaries, (list, tuple)):
        boundaries = [boundaries] * batch
    if len(boundaries) != batch:
        raise ValueError("Boundary objects must match the numerical batch")
    coordinates = [q.reshape(batch, -1, 19) for q in (xi, eta, zeta)]
    surfaces, us, vs = [], [], []
    for name, axes in zip(_NAMES, _FACE_AXES):
        indices, _ = _projection_map(axes, xi.device)
        us.append(coordinates[axes[0]].index_select(2, indices).flatten(1))
        vs.append(coordinates[axes[1]].index_select(2, indices).flatten(1))
        surfaces.extend(getattr(boundary, name) for boundary in boundaries)
    values = evaluate_surfaces_torch(surfaces, torch.cat(us), torch.cat(vs))
    values = values.reshape(6, batch, count // 19, 9, 3)
    return torch.stack([
        values[i].index_select(2, _projection_map(axes, xi.device)[1])
        .reshape(batch, count, 3)
        for i, axes in enumerate(_FACE_AXES)
    ], dim=2)


def central_stencil_edges(boundaries, xi, eta, zeta):
    """Evaluate three positions per edge with fixed incident-face batches."""
    batch, count = xi.shape
    if not isinstance(boundaries, (list, tuple)):
        boundaries = [boundaries] * batch
    if len(boundaries) != batch:
        raise ValueError("Boundary objects must match the numerical batch")
    coordinates = [q.reshape(batch, -1, 19) for q in (xi, eta, zeta)]
    varying = [coordinates[axis].index_select(
        2, _projection_map((axis,), xi.device)[0]).flatten(1)
        for axis in (0, 1, 2)]
    x, y, z = varying
    zero, one = torch.zeros_like(x), torch.ones_like(x)
    parameters = (
        (zero, z), (zero, z), (one, z), (one, z),
        (y, zero), (y, zero), (y, one), (y, one),
        (x, zero), (x, zero), (x, one), (x, one),
    )
    values = [None] * 12
    for face_index, indices in _EDGE_GROUPS:
        surfaces = [getattr(boundary, _NAMES[face_index]) for boundary in boundaries]
        u = torch.cat([parameters[i][0] for i in indices], dim=1)
        v = torch.cat([parameters[i][1] for i in indices], dim=1)
        group = evaluate_surfaces_torch(surfaces, u, v).reshape(
            batch, len(indices), -1, 3
        )
        for offset, edge_index in enumerate(indices):
            values[edge_index] = group[:, offset]
    return torch.stack([
        values[i].reshape(batch, -1, 3, 3).index_select(
            2, _projection_map((axis,), xi.device)[1]
        )
        .reshape(batch, count, 3)
        for i, axis in enumerate(_EDGE_AXES)
    ], dim=2)
