"""Logical-edge boundary conditions for structured quadrilateral grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EDGE_NAMES = ("bottom", "right", "top", "left")


@dataclass(frozen=True)
class DirichletBoundary:
    indices: np.ndarray
    values: np.ndarray
    edge_indices: dict[str, np.ndarray]


def logical_edge_indices(shape):
    """Return each edge in increasing local logical coordinate order."""
    nx, ny = map(int, shape)
    if nx < 2 or ny < 2:
        raise ValueError("Boundary extraction requires nx, ny >= 2")
    return {
        "bottom": np.arange(nx, dtype=np.int64),
        "right": np.arange(nx - 1, nx * ny, nx, dtype=np.int64),
        "top": np.arange((ny - 1) * nx, ny * nx, dtype=np.int64),
        "left": np.arange(0, nx * ny, nx, dtype=np.int64),
    }


def build_dirichlet_boundary(shape, edge_values, *, corner_policy="max"):
    """Build nodal Dirichlet data from named logical edge values.

    Values may be scalars or callables of one local edge coordinate in [0, 1].
    Shared-corner conflicts use ``corner_policy`` (``max``, ``min``, ``mean``
    or ``error``).
    """
    unknown = set(edge_values) - set(EDGE_NAMES)
    if unknown:
        raise ValueError(f"Unknown logical edges: {sorted(unknown)}")
    if not edge_values:
        raise ValueError("At least one Dirichlet edge is required")
    edges = logical_edge_indices(shape)
    assigned: dict[int, list[float]] = {}
    for name, prescribed in edge_values.items():
        indices = edges[name]
        parameter = np.linspace(0.0, 1.0, len(indices))
        values = prescribed(parameter) if callable(prescribed) else prescribed
        values = np.broadcast_to(np.asarray(values, dtype=float), (len(indices),))
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite Dirichlet values on edge {name}")
        for index, value in zip(indices, values):
            assigned.setdefault(int(index), []).append(float(value))

    reducers = {"max": np.max, "min": np.min, "mean": np.mean}
    if corner_policy not in reducers and corner_policy != "error":
        raise ValueError("corner_policy must be max, min, mean or error")
    indices = np.array(sorted(assigned), dtype=np.int64)
    result = []
    for index in indices:
        values = np.asarray(assigned[int(index)])
        if corner_policy == "error":
            if not np.allclose(values, values[0]):
                raise ValueError(f"Conflicting Dirichlet values at node {index}: {values}")
            result.append(float(values[0]))
        else:
            result.append(float(reducers[corner_policy](values)))
    return DirichletBoundary(indices, np.asarray(result), edges)
