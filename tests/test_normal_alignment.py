"""Numerical and optimization checks for inward unit-normal alignment."""

import unittest

import torch

from scripts.train.training_common import boundary_normal_loss, normal_alignment_loss
from src.data.geometry.boundary_generator_2d import rectangular_boundary
from src.data.geometry.boundary_generator_3d import box_boundary
from src.data.mesh.sampling import logical_grid, sample_boundary


class AffineModel(torch.nn.Module):
    def __init__(self, dimension, shear=0.0):
        super().__init__()
        matrix = torch.eye(dimension, dtype=torch.float64) * 2
        matrix[0, 1] = shear
        self.matrix = torch.nn.Parameter(matrix)

    def forward(self, *args, **kwargs):
        dimension = self.matrix.shape[0]
        q = torch.stack(args[dimension:], dim=-1)
        return (q @ self.matrix.T - 1).split(1, dim=-1)


class NormalAlignmentTests(unittest.TestCase):
    def test_angles_scaling_and_degenerate_prediction(self):
        n = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        for v, expected in (
            ([3.0, 0.0], 0.0),
            ([0.0, 2.0], 1.0),
            ([-2.0, 0.0], 4.0),
            ([0.0, 0.0], 1.0),
        ):
            self.assertAlmostEqual(
                normal_alignment_loss(torch.tensor([v]), n).item(), expected
            )
        with self.assertRaises(ValueError):
            normal_alignment_loss(n, torch.zeros_like(n))

    def test_curved_boundary_uses_exact_normal(self):
        from src.data.geometry.parametric_curve_2d import (
            BoundaryCurveSet2D,
            CircleArc,
            LineSegment,
        )
        import math
        import numpy as np

        geometry = BoundaryCurveSet2D(
            CircleArc((0, 0), 1, math.pi, 0),
            LineSegment((1, 0), (2, 0)),
            CircleArc((0, 0), 2, math.pi, 0),
            LineSegment((-1, 0), (-2, 0)),
        )
        shape = (7, 5)
        q = logical_grid(shape)
        angle = math.pi * (1 - q[:, 0])
        mesh = torch.tensor(
            np.stack(
                ((1 + q[:, 1]) * np.cos(angle), (1 + q[:, 1]) * np.sin(angle)), -1
            ),
            dtype=torch.float64,
        )[None]
        boundary = torch.tensor(sample_boundary(geometry, shape), dtype=torch.float64)[
            None
        ]

        class RadialModel(torch.nn.Module):
            def forward(self, bx, by, xi, eta, **kwargs):
                angle = math.pi * (1 - xi)
                return ((1 + eta) * angle.cos())[..., None], ((1 + eta) * angle.sin())[
                    ..., None
                ]

        # A tiny inward step also makes the chord at the radial side faces
        # approach their perpendicular tangent direction.
        torch.manual_seed(4)
        exact = boundary_normal_loss(
            RadialModel(), boundary, mesh, shape, 2, "cpu", 256, 1e-4, geometry
        )
        torch.manual_seed(4)
        sampled = boundary_normal_loss(
            RadialModel(), boundary, mesh, shape, 2, "cpu", 256, 1e-4
        )
        self.assertLess(exact.item(), 1e-12)
        self.assertGreater(sampled.item(), 1e-6)

    def test_exact_and_sampled_normals_in_both_dimensions(self):
        for dimension in (2, 3):
            shape = (5,) * dimension
            geometry = rectangular_boundary() if dimension == 2 else box_boundary()
            boundary = torch.tensor(
                sample_boundary(geometry, shape), dtype=torch.float64
            )[None]
            mesh = torch.tensor(logical_grid(shape) * 2 - 1, dtype=torch.float64)[None]
            for exact in (geometry, None):
                with self.subTest(dimension=dimension, exact=exact is not None):
                    model = AffineModel(dimension)

                    def loss():
                        torch.manual_seed(7)
                        return boundary_normal_loss(
                            model,
                            boundary,
                            mesh,
                            shape,
                            dimension,
                            "cpu",
                            128,
                            0.01,
                            geometry=exact,
                        )

                    self.assertLess(loss().item(), 1e-20)
                    with torch.no_grad():
                        model.matrix[0, 1] = 1.0
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
                    initial = loss().item()
                    for _ in range(15):
                        optimizer.zero_grad()
                        value = loss()
                        value.backward()
                        self.assertTrue(torch.isfinite(model.matrix.grad).all())
                        optimizer.step()
                    self.assertLess(loss().item(), initial * 0.8)


if __name__ == "__main__":
    unittest.main()
