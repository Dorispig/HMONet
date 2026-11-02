"""Tests for curvature-distance adaptive HMONet sampling."""

import unittest

import torch

from src.data.mesh.sampling import (
    sample_curvature_distance_adaptive_points,
    sample_inward_boundary_band_points_2d,
)
from src.utils.math import compute_discrete_boundary_curvature


class AdaptiveSamplingTests(unittest.TestCase):
    def test_vectorized_shape_and_open_unit_square(self):
        torch.manual_seed(4)
        boundary = torch.tensor([
            [[0., 0.], [0.5, 0.], [1., 0.], [1., 1.], [0., 1.]],
            [[0., 0.], [0.5, 0.3], [1., 0.], [1., 1.], [0., 1.]],
        ])
        computational = torch.tensor(
            [[0., 0.], [0.5, 0.], [1., 0.], [1., 1.], [0., 1.]]
        )
        curvature = compute_discrete_boundary_curvature(boundary)
        points = sample_curvature_distance_adaptive_points(
            curvature, computational, 32, candidate_multiplier=3
        )
        self.assertEqual(tuple(points.shape), (2, 32, 2))
        self.assertTrue(bool(((points > 0) & (points < 1)).all()))

    def test_high_curvature_boundary_attracts_samples(self):
        torch.manual_seed(12)
        computational = torch.tensor(
            [[0., 0.], [0.5, 0.], [1., 0.], [1., 1.], [0., 1.]]
        )
        curvature = torch.tensor([[0., 1., 0., 0., 0.]])
        points = sample_curvature_distance_adaptive_points(
            curvature, computational, 1000, candidate_multiplier=10,
            sigma_min=0.08, sigma_max=0.15, uniform_probability=0.001,
        )
        distance = torch.linalg.vector_norm(points[0] - torch.tensor([0.5, 0.]), dim=1)
        self.assertLess(float(distance.mean()), 0.35)

    def test_finite_difference_margin_limits_adaptive_candidates(self):
        computational = torch.tensor(
            [[0., 0.], [0.5, 0.], [1., 0.], [1., 1.], [0., 1.]]
        )
        curvature = torch.tensor([[0., 1., 0., 0., 0.]])
        points = sample_curvature_distance_adaptive_points(
            curvature, computational, 128, candidate_multiplier=3, margin=0.08
        )
        self.assertTrue(bool(((points >= 0.08) & (points <= 0.92)).all()))

    def test_inward_boundary_band_respects_normal_and_tangent_ranges(self):
        torch.manual_seed(19)
        points = sample_inward_boundary_band_points_2d(
            3,
            500,
            min_distance=0.01,
            max_distance=0.04,
            tangent_margin=0.01,
        )
        self.assertEqual(tuple(points.shape), (3, 500, 2))
        self.assertTrue(bool(((points >= 0.01) & (points <= 0.99)).all()))
        distance = torch.minimum(points, 1.0 - points).amin(dim=-1)
        self.assertTrue(bool(((distance >= 0.01) & (distance <= 0.04)).all()))


if __name__ == "__main__":
    unittest.main()
