"""Validity auditing must distinguish contact, folding and global overlap."""

import unittest
import numpy as np
from src.data.generation.dataset_quality_2d import audit_grid


class DatasetQualityTests(unittest.TestCase):
    def test_regular_grid_shared_edges_are_legal(self):
        x, y = np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 7))
        result = audit_grid(np.stack((x, y), axis=-1), 9, 7)
        self.assertTrue(result["all_cells_legal"])

    def test_bow_tie_is_illegal(self):
        points = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        result = audit_grid(points, 2, 2)
        self.assertEqual(result["self_intersecting_cells"], 1)
        self.assertFalse(result["all_cells_legal"])

    def test_double_wrapped_annulus_detects_global_overlap(self):
        angle = np.linspace(0, 4 * np.pi, 41)
        radius = np.linspace(1, 2, 4)
        # Reverse angular direction for positive orientation with increasing radius.
        x = radius[:, None] * np.cos(-angle)
        y = radius[:, None] * np.sin(-angle)
        result = audit_grid(np.stack((x, y), axis=-1), 41, 4)
        self.assertEqual(result["local_invalid_cells"], 0)
        self.assertGreater(result["overlapping_convex_cells"], 0)

    def test_single_wrapped_annulus_seam_is_contact(self):
        angle = np.linspace(0, 2 * np.pi, 41)
        radius = np.linspace(1, 2, 4)
        x = radius[:, None] * np.cos(-angle)
        y = radius[:, None] * np.sin(-angle)
        self.assertTrue(audit_grid(np.stack((x, y), axis=-1), 41, 4)["all_cells_legal"])


if __name__ == "__main__":
    unittest.main()
