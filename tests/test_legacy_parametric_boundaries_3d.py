"""Regression tests for parametric counterparts of legacy 3-D O-grids."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.backend.parametric_torch import evaluate_boundary_surfaces_torch
from src.data.geometry.boundary_generator_3d import (
    BOUNDARY_GENERATORS,
    cylindrical_domain_with_airfoil_hole,
    stadium_cylinder_with_circular_hole,
)
from src.data.geometry.parametric_curve_2d import ParametricCurve, StadiumCurve
from src.data.geometry.parametric_geometry_3d import BoundarySurfaceSet3D


def legacy_stadium(angle, half_length, radius):
    cosine, sine = np.cos(angle), np.sin(angle)
    horizontal_radius = radius / max(abs(sine), 1e-300)
    if abs(sine) > 1e-14 and abs(horizontal_radius * cosine) <= half_length:
        return horizontal_radius * np.array([cosine, sine])
    radial = half_length * abs(cosine) + np.sqrt(
        max(radius**2 - half_length**2 * sine**2, 0.0)
    )
    return radial * np.array([cosine, sine])


def legacy_airfoil(q, camber, position, thickness, chord):
    q = float(q) % 1.0
    upper = q <= 0.5
    x = 1.0 - 2.0 * q if upper else 2.0 * q - 1.0
    yt = 5.0 * thickness * (
        0.2969 * np.sqrt(max(x, 0.0))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )
    if x < position:
        yc = camber / position**2 * (2 * position * x - x**2)
        slope = 2 * camber / position**2 * (position - x)
    else:
        yc = camber / (1 - position) ** 2 * (
            1 - 2 * position + 2 * position * x - x**2
        )
        slope = 2 * camber / (1 - position) ** 2 * (position - x)
    angle = np.arctan(slope)
    sign = 1.0 if upper else -1.0
    return chord * np.array(
        [x - sign * yt * np.sin(angle), yc + sign * yt * np.cos(angle)]
    ) - np.array([0.5 * chord, 0.0])


class LegacyParametricBoundary3DTests(unittest.TestCase):
    def test_generators_are_distinct_and_registered(self):
        self.assertIs(
            BOUNDARY_GENERATORS["cylindrical_domain_with_airfoil_hole"],
            cylindrical_domain_with_airfoil_hole,
        )
        self.assertIs(
            BOUNDARY_GENERATORS["stadium_cylinder_with_circular_hole"],
            stadium_cylinder_with_circular_hole,
        )

    def test_stadium_curve_matches_legacy_formula_and_round_trips(self):
        curve = StadiumCurve(0.8, 0.7)
        q = np.linspace(0.0, 1.0, 41)
        expected = np.stack([legacy_stadium(2 * np.pi * value, 0.8, 0.7) for value in q])
        np.testing.assert_allclose(curve.evaluate(q), expected, atol=2e-14)
        restored = ParametricCurve.from_dict(curve.to_dict())
        np.testing.assert_allclose(restored.evaluate(q), expected, atol=2e-14)
        self.assertTrue(np.all(np.isfinite(curve.derivative(q))))

    def test_airfoil_wall_matches_legacy_formula(self):
        boundary = cylindrical_domain_with_airfoil_hole()
        q = np.array([0.0, 0.13, 0.5, 0.77, 1.0])
        zeta = np.full_like(q, 0.4)
        actual = boundary.x_min.evaluate(q, zeta)
        expected_xy = np.stack(
            [legacy_airfoil(value, 0.02, 0.4, 0.12, 1.0) for value in q]
        )
        expected = np.column_stack((expected_xy, np.full(q.shape, 1.0)))
        np.testing.assert_allclose(actual, expected, atol=2e-12)

    def test_faces_are_consistent_serializable_and_differentiable(self):
        boundaries = [
            cylindrical_domain_with_airfoil_hole(thickness=0.16),
            stadium_cylinder_with_circular_hole(straight_half_length=1.0),
        ]
        q = np.linspace(0, 1, 17)
        for boundary in boundaries:
            np.testing.assert_allclose(
                boundary.y_min.evaluate(q, q),
                boundary.y_max.evaluate(q, q),
                atol=1e-13,
            )
            np.testing.assert_allclose(
                boundary.z_min.evaluate(np.zeros_like(q), q),
                boundary.x_min.evaluate(q, np.zeros_like(q)),
                atol=1e-13,
            )
            np.testing.assert_allclose(
                boundary.z_min.evaluate(np.ones_like(q), q),
                boundary.x_max.evaluate(q, np.zeros_like(q)),
                atol=1e-13,
            )
            restored = BoundarySurfaceSet3D.from_dict(boundary.to_dict())
            for name in boundary.names:
                np.testing.assert_allclose(
                    getattr(restored, name).sample(9, 11),
                    getattr(boundary, name).sample(9, 11),
                    atol=2e-12,
                )

        query = torch.rand(2, 19, 3, dtype=torch.float64, requires_grad=True)
        values = evaluate_boundary_surfaces_torch(boundaries, *query.unbind(-1))
        self.assertEqual(values.shape, (2, 19, 6, 3))
        values.square().mean().backward()
        self.assertTrue(torch.isfinite(query.grad).all())


if __name__ == "__main__":
    unittest.main()
