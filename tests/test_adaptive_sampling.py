"""Tests for curvature-distance adaptive HMONet sampling."""

import torch

from src.data.mesh.sampling import (
    sample_curvature_distance_adaptive_points,
    sample_inward_boundary_band_points_2d,
    sample_surface_curvature_adaptive_points,
    structured_boundary_surface_curvature,
)
from src.utils.math import compute_discrete_boundary_curvature


class TestAdaptiveSampling:
    def test_vectorized_shape_and_open_unit_square(self):
        torch.manual_seed(4)
        boundary = torch.tensor(
            [
                [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.5, 0.3], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            ]
        )
        computational = torch.tensor(
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        )
        curvature = compute_discrete_boundary_curvature(boundary)
        points = sample_curvature_distance_adaptive_points(
            curvature, computational, 32, candidate_multiplier=3
        )
        assert (tuple(points.shape)) == (2, 32, 2)
        assert bool(((points > 0) & (points < 1)).all())

    def test_high_curvature_boundary_attracts_samples(self):
        torch.manual_seed(12)
        computational = torch.tensor(
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        )
        curvature = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
        points = sample_curvature_distance_adaptive_points(
            curvature,
            computational,
            1000,
            candidate_multiplier=10,
            sigma_min=0.08,
            sigma_max=0.15,
            uniform_probability=0.001,
        )
        distance = torch.linalg.vector_norm(points[0] - torch.tensor([0.5, 0.0]), dim=1)
        assert (float(distance.mean())) < (0.35)

    def test_finite_difference_margin_limits_adaptive_candidates(self):
        computational = torch.tensor(
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        )
        curvature = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
        points = sample_curvature_distance_adaptive_points(
            curvature, computational, 128, candidate_multiplier=3, margin=0.08
        )
        assert bool(((points >= 0.08) & (points <= 0.92)).all())

    def test_inward_boundary_band_respects_normal_and_tangent_ranges(self):
        torch.manual_seed(19)
        points = sample_inward_boundary_band_points_2d(
            3,
            500,
            min_distance=0.01,
            max_distance=0.04,
            tangent_margin=0.01,
        )
        assert (tuple(points.shape)) == (3, 500, 2)
        assert bool(((points >= 0.01) & (points <= 0.99)).all())
        distance = torch.minimum(points, 1.0 - points).amin(dim=-1)
        assert bool(((distance >= 0.01) & (distance <= 0.04)).all())

    def test_3d_surface_sampler_is_batched_and_respects_fd_margin(self):
        nx, ny, nz = 4, 5, 3
        zeta, eta, xi = torch.meshgrid(
            torch.linspace(0, 1, nz),
            torch.linspace(0, 1, ny),
            torch.linspace(0, 1, nx),
            indexing="ij",
        )
        full = torch.stack((xi, eta, zeta), -1)
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        mask[[0, -1], :, :] = True
        mask[:, [0, -1], :] = True
        mask[:, :, [0, -1]] = True
        boundary = full[mask].unsqueeze(0).repeat(2, 1, 1)
        curvature, computational = structured_boundary_surface_curvature(
            boundary, (nx, ny, nz)
        )
        assert curvature.shape[0] == 2
        assert computational.shape == (curvature.shape[1], 3)
        torch.testing.assert_close(curvature, torch.zeros_like(curvature))

        torch.manual_seed(23)
        points = sample_surface_curvature_adaptive_points(
            curvature,
            computational,
            64,
            margin=torch.tensor([0.02, 0.03, 0.04]),
        )
        assert points.shape == (2, 64, 3)
        assert bool((points[..., 0] >= 0.02).all())
        assert bool((points[..., 0] <= 0.98).all())
        assert bool((points[..., 1] >= 0.03).all())
        assert bool((points[..., 1] <= 0.97).all())
        assert bool((points[..., 2] >= 0.04).all())
        assert bool((points[..., 2] <= 0.96).all())
