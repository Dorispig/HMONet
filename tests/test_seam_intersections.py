"""Protect seam endpoint contacts without hiding real intersections."""

import numpy as np
from src.utils.metrics import (
    QuadMeshQualityEvaluator,
    _polygons_overlap,
    CheckSelfIntersection,
)
from src.data.mesh.topology import GenerateQuadMeshIndex


class TestSeamIntersection:
    def test_duplicate_endpoint_ids(self):
        points = np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1], [1, 1], [2, 1], [2, 2], [1, 2]], float
        )
        evaluator = QuadMeshQualityEvaluator(points, np.arange(8).reshape(2, 4))
        assert not evaluator.IntersectingCells().any()

    def test_duplicate_full_edge_is_boundary_contact(self):
        left = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        right = np.array([[1, 0], [2, 0], [2, 1], [1, 1]], float)
        points = np.vstack((left, right))
        evaluator = QuadMeshQualityEvaluator(points, np.arange(8).reshape(2, 4))
        assert not _polygons_overlap(left, right)
        assert not evaluator.IntersectingCells().any()

    def test_float32_shared_edge_roundoff_is_boundary_contact(self):
        left = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        right = np.array(
            [
                [1 + 2e-7, 1e-7],
                [2, 0],
                [2, 1],
                [1 - 1e-7, 1 - 2e-7],
            ],
            dtype=np.float32,
        )
        evaluator = QuadMeshQualityEvaluator(
            np.vstack((left, right)), np.arange(8).reshape(2, 4)
        )
        assert not _polygons_overlap(left, right)
        assert not evaluator.IntersectingCells().any()

    def test_float32_periodic_seam_is_not_an_intersection(self):
        angles = np.linspace(0, 2 * np.pi, 41, dtype=np.float32)
        radii = np.linspace(1, 2, 5, dtype=np.float32)
        x = radii[:, None] * np.cos(angles)
        y = radii[:, None] * np.sin(angles)
        points = np.stack((x, y), axis=-1).reshape(-1, 2)
        faces = GenerateQuadMeshIndex(len(radii), len(angles))
        evaluator = QuadMeshQualityEvaluator(points, faces)
        assert not evaluator.IntersectingCells().any()

    def test_float32_real_overlap_is_retained(self):
        first = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float32)
        second = first + np.array([0.25, 0.25], dtype=np.float32)
        evaluator = QuadMeshQualityEvaluator(
            np.vstack((first, second)), np.arange(8).reshape(2, 4)
        )
        assert _polygons_overlap(first, second)
        assert evaluator.IntersectingCells().all()

    def test_real_intersections_are_retained(self):
        square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
        cases = [
            square + [1, 1],
            square.copy(),
            np.array([[0, 0], [1, 0.2], [1, 1], [0.2, 1]]),
            np.array([[1, 0], [3, 0], [3, 1], [1, 1]]),
            np.array([[1, 2], [2, 3], [1, 4], [0, 3]]),
        ]
        for other in cases:
            assert _polygons_overlap(square, other)
            assert _polygons_overlap(other, square)
        assert CheckSelfIntersection([[0, 0], [1, 1], [0, 1], [1, 0]])
