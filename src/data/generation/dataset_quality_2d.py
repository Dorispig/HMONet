"""Vectorized validity audit for stored, structured quadrilateral grids.

Strictly positive corner determinants certify convex positive cells. A spatial
broad phase followed by the separating-axis theorem finds interior overlaps
among those cells, including adjacent cells. Boundary contacts/seams are not
interior overlaps. Nonconvex cells are already illegal locally and excluded
from the convex SAT test; their overlap count is not claimed as exhaustive.
"""

import numpy as np
from scipy.spatial import cKDTree


def simple_boundary(grid, tolerance):
    """Check nonadjacent perimeter segments for any crossing or contact.

    Positive convex cells plus a simple boundary certify a degree-one planar
    grid, so a global cell-pair scan is unnecessary in this particular case.
    Seams/duplicate boundary coordinates fail this certificate and use SAT.
    """
    points = np.concatenate((grid[0], grid[1:, -1], grid[-1, -2::-1], grid[-2:0:-1, 0]))
    a, b = points, np.roll(points, -1, axis=0)
    center = (a + b) / 2
    radius = np.linalg.norm(b - a, axis=1) / 2
    if np.any(radius <= tolerance) or not np.isfinite(points).all():
        return False
    pairs = cKDTree(center).query_pairs(
        2 * radius.max() + tolerance, output_type="ndarray"
    )
    if not len(pairs):
        return True
    delta = np.abs(pairs[:, 0] - pairs[:, 1])
    pairs = pairs[(delta != 1) & (delta != len(points) - 1)]
    if not len(pairs):
        return True
    i, j = pairs[:, 0], pairs[:, 1]
    lower = np.maximum(np.minimum(a[i], b[i]), np.minimum(a[j], b[j]))
    upper = np.minimum(np.maximum(a[i], b[i]), np.maximum(a[j], b[j]))
    pairs = pairs[(upper - lower >= -tolerance).all(axis=1)]
    if not len(pairs):
        return True
    i, j = pairs[:, 0], pairs[:, 1]

    def side(p, q, r):
        e, f = q - p, r - p
        return e[:, 0] * f[:, 1] - e[:, 1] * f[:, 0]

    def straddles(u, v):
        return ((u <= tolerance) & (v >= -tolerance)) | (
            (v <= tolerance) & (u >= -tolerance)
        )

    contacts = straddles(side(a[i], b[i], a[j]), side(a[i], b[i], b[j])) & straddles(
        side(a[j], b[j], a[i]), side(a[j], b[j], b[i])
    )
    return not bool(contacts.any())


def audit_grid(points, nx, ny, tolerance=1e-12):
    grid = np.asarray(points, dtype=float).reshape(ny, nx, 2)
    cells = np.stack(
        (grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]), axis=2
    ).reshape(-1, 4, 2)
    finite = np.isfinite(cells).all(axis=(1, 2))
    edges = np.roll(cells, -1, axis=1) - cells
    nxt = np.roll(edges, -1, axis=1)
    det = edges[:, :, 0] * nxt[:, :, 1] - edges[:, :, 1] * nxt[:, :, 0]
    local_bad = ~finite | (det <= tolerance).any(axis=1)

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    def crossing(a, b, c, d):
        return (cross(b - a, c - a) * cross(b - a, d - a) < 0) & (
            cross(d - c, a - c) * cross(d - c, b - c) < 0
        )

    a, b, c, d = (cells[:, i] for i in range(4))
    self_cross = crossing(a, b, c, d) | crossing(b, c, d, a)
    overlap = np.zeros(len(cells), dtype=bool)
    good = np.flatnonzero(~local_bad)
    certified = not local_bad.any() and simple_boundary(grid, tolerance)
    if len(good) > 1 and not certified:
        polygons = cells[good]
        centers = polygons.mean(axis=1)
        radius = np.linalg.norm(polygons - centers[:, None], axis=2).max(axis=1)
        pairs = cKDTree(centers).query_pairs(2 * radius.max(), output_type="ndarray")
        lo, hi = polygons.min(axis=1), polygons.max(axis=1)
        for start in range(0, len(pairs), 20000):
            pair = pairs[start : start + 20000]
            keep = (
                np.minimum(hi[pair[:, 0]], hi[pair[:, 1]])
                - np.maximum(lo[pair[:, 0]], lo[pair[:, 1]])
                > tolerance
            ).all(axis=1)
            pair = pair[keep]
            if not len(pair):
                continue
            p, q = polygons[pair[:, 0]], polygons[pair[:, 1]]
            intersects = np.ones(len(pair), dtype=bool)
            for source in (p, q):
                e = np.roll(source, -1, axis=1) - source
                for k in range(4):
                    axis = np.stack((-e[:, k, 1], e[:, k, 0]), axis=1)
                    pp, qq = np.einsum("nki,ni->nk", p, axis), np.einsum(
                        "nki,ni->nk", q, axis
                    )
                    intersects &= np.minimum(
                        pp.max(axis=1), qq.max(axis=1)
                    ) - np.maximum(
                        pp.min(axis=1), qq.min(axis=1)
                    ) > tolerance * np.linalg.norm(
                        axis, axis=1
                    )
            overlap[good[pair[intersects].ravel()]] = True
    return {
        "cell_count": len(cells),
        "local_invalid_cells": int(local_bad.sum()),
        "self_intersecting_cells": int(self_cross.sum()),
        "overlapping_convex_cells": int(overlap.sum()),
        "invalid_cells": int((local_bad | overlap).sum()),
        "minimum_corner_determinant": float(np.nanmin(det)),
        "all_cells_legal": bool(not np.any(local_bad | overlap)),
        "nonconvex_overlap_tested": False,
        "global_check": (
            "positive_cells_and_simple_boundary"
            if certified
            else "convex_separating_axes"
        ),
    }
