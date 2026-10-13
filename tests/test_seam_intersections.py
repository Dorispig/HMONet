"""Protect seam endpoint contacts without hiding real intersections."""

import unittest
import numpy as np
from src.utils.metrics import (
    QuadMeshQualityEvaluator,
    _polygons_overlap,
    CheckSelfIntersection,
)


class SeamIntersectionTests(unittest.TestCase):
    def test_duplicate_endpoint_ids(self):
        points = np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1], [1, 1], [2, 1], [2, 2], [1, 2]], float
        )
        evaluator = QuadMeshQualityEvaluator(points, np.arange(8).reshape(2, 4))
        self.assertFalse(evaluator.IntersectingCells().any())

    def test_duplicate_full_edge_is_boundary_contact(self):
        left = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        right = np.array([[1, 0], [2, 0], [2, 1], [1, 1]], float)
        points = np.vstack((left, right))
        evaluator = QuadMeshQualityEvaluator(points, np.arange(8).reshape(2, 4))
        self.assertFalse(_polygons_overlap(left, right))
        self.assertFalse(evaluator.IntersectingCells().any())

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
            with self.subTest(other=other.tolist()):
                self.assertTrue(_polygons_overlap(square, other))
                self.assertTrue(_polygons_overlap(other, square))
        self.assertTrue(CheckSelfIntersection([[0, 0], [1, 1], [0, 1], [1, 0]]))


if __name__ == "__main__":
    unittest.main()
