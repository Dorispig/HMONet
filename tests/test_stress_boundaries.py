"""Geometry, serialization and differentiability checks for stress boundaries."""

import unittest
import numpy as np
import torch
from src.data.geometry.boundary_generator_2d import generate_boundary
from src.data.geometry.parametric_curve_2d import BoundaryCurveSet2D, PiecewiseCurve

NAMES = [
    "smooth_constricted_channel",
    "variable_width_s_channel",
    "gear_spline",
    "rounded_u_notch",
]


class StressBoundaryTests(unittest.TestCase):
    def test_values_derivatives_and_roundtrip(self):
        for name in NAMES:
            b = generate_boundary(name)
            restored = BoundaryCurveSet2D.from_dict(b.to_dict())
            b.validate()
            for key, curve in b.curves().items():
                t = torch.linspace(
                    0.013, 0.987, 37, dtype=torch.float64, requires_grad=True
                )
                values = curve(t)
                np.testing.assert_allclose(
                    values.detach(), curve(t.detach().numpy()), atol=1e-9
                )
                np.testing.assert_allclose(
                    restored.curves()[key](t).detach(), values.detach(), atol=1e-9
                )
                derivative = torch.stack(
                    [
                        torch.autograd.grad(values[:, i].sum(), t, retain_graph=True)[0]
                        for i in range(2)
                    ],
                    -1,
                )
                np.testing.assert_allclose(
                    curve.derivative(t).detach(), derivative.detach(), atol=1e-7
                )
                h = 1e-6
                numerical = (
                    curve(t.detach().numpy() + h) - curve(t.detach().numpy() - h)
                ) / (2 * h)
                np.testing.assert_allclose(
                    derivative.detach(), numerical, atol=1e-5, rtol=1e-4
                )
                for endpoint in (0.0, 1.0):
                    self.assertTrue(
                        np.isfinite(curve.derivative(np.array([endpoint]))).all()
                    )
                if isinstance(curve, PiecewiseCurve):
                    joins = torch.tensor(
                        curve.breakpoints[1:-1], dtype=torch.float64, requires_grad=True
                    )
                    y = curve(joins)
                    grad = torch.stack(
                        [
                            torch.autograd.grad(
                                y[:, i].sum(), joins, retain_graph=True
                            )[0]
                            for i in range(2)
                        ],
                        -1,
                    )
                    np.testing.assert_allclose(
                        grad.detach(), curve.derivative(joins).detach(), atol=1e-7
                    )

    def test_batch_gradient(self):
        for name in NAMES:
            b = BoundaryCurveSet2D.batch(
                [generate_boundary(name), generate_boundary(name)], dtype=torch.float64
            )
            t = (
                torch.linspace(0.03, 0.97, 17, dtype=torch.float64)
                .repeat(2, 1)
                .requires_grad_()
            )
            points = b.project_torch(t, t)
            g = torch.autograd.grad(points.square().sum(), t, create_graph=True)[0]
            self.assertTrue(torch.isfinite(g).all())
            self.assertTrue(torch.isfinite(torch.autograd.grad(g.sum(), t)[0]).all())

    def test_default_boundaries_have_no_crossings(self):
        for name in NAMES:
            b = generate_boundary(name)
            t = np.linspace(0, 1, 301)
            points = np.concatenate(
                [
                    b.bottom(t)[:-1],
                    b.right(t)[:-1],
                    b.top(t[::-1])[:-1],
                    b.left(t[::-1])[:-1],
                ]
            )
            starts = points
            ends = np.roll(points, -1, axis=0)

            def cross(a, b):
                return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

            direction = ends - starts
            delta = starts[None, :, :] - starts[:, None, :]
            a = cross(direction[:, None, :], delta)
            b = cross(direction[:, None, :], ends[None, :, :] - starts[:, None, :])
            c = cross(direction[None, :, :], -delta)
            d = cross(direction[None, :, :], ends[:, None, :] - starts[None, :, :])
            self.assertFalse(np.any((a * b < -1e-20) & (c * d < -1e-20)), name)

    def test_gap_and_waves(self):
        t = np.linspace(0, 1, 1001)
        for name in NAMES[:2]:
            b = generate_boundary(name)
            self.assertTrue(np.all(b.top(t)[:, 1] > b.bottom(t)[:, 1]))
        b = generate_boundary("smooth_constricted_channel")
        self.assertAlmostEqual(float(b.top(0.5)[1] - b.bottom(0.5)[1]), 0.2)
        b = generate_boundary("gear_spline")
        self.assertGreater(np.ptp(b.left(t)[:, 0]), 0.1)
        self.assertGreater(np.ptp(b.right(t)[:, 0]), 0.1)


if __name__ == "__main__":
    unittest.main()
