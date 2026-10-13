"""Lightweight NumPy boundary projection utilities."""

from __future__ import annotations

import numpy as np


class BoundaryProjection:
    """Project a predicted structured mesh onto an ordered physical boundary."""

    def __init__(
        self,
        points,
        boundary_points,
        boundary_indices,
        projection_method="InverseDistance",
    ):
        self.points = np.asarray(points, dtype=float)
        self.boundary_points = np.asarray(boundary_points, dtype=float)
        self.boundary_indices = np.asarray(boundary_indices, dtype=int)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        if self.boundary_points.shape != (len(self.boundary_indices), 2):
            raise ValueError(
                "boundary_points and boundary_indices must have matching lengths"
            )
        if self.boundary_indices.size and (
            self.boundary_indices.min() < 0
            or self.boundary_indices.max() >= len(self.points)
        ):
            raise ValueError("boundary_indices contains an out-of-range index")
        self.projection_method = projection_method

    def project(self, power=2.0):
        if self.projection_method == "InverseDistance":
            return self.inverse_distance_projection(power=power)
        if self.projection_method in {"Direct", "Nearest"}:
            return self.direct_projection()
        raise ValueError("projection_method must be 'InverseDistance' or 'Direct'")

    def direct_projection(self):
        result = self.points.copy()
        result[self.boundary_indices] = self.boundary_points
        return result

    def inverse_distance_projection(self, power=2.0, eps=1e-10, chunk_size=4096):
        if power <= 0:
            raise ValueError("power must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        result = self.points.copy()
        boundary_source = self.points[self.boundary_indices]
        boundary_vectors = self.boundary_points - boundary_source
        result[self.boundary_indices] = self.boundary_points
        interior_mask = np.ones(len(self.points), dtype=bool)
        interior_mask[self.boundary_indices] = False
        interior_indices = np.flatnonzero(interior_mask)
        for start in range(0, len(interior_indices), chunk_size):
            indices = interior_indices[start : start + chunk_size]
            differences = self.points[indices, None, :] - boundary_source[None, :, :]
            distances = np.maximum(np.linalg.norm(differences, axis=-1), eps)
            weights = distances ** (-power)
            weights /= weights.sum(axis=1, keepdims=True)
            result[indices] = self.points[indices] + weights @ boundary_vectors
        return result

