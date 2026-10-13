"""Check analytic tangents, vectorized batches, endpoint rules and model gradients."""

import unittest
from unittest.mock import patch
import numpy as np
import torch
from src.data.geometry.parametric_curve_2d import (
    ParametricCurve,
    LineSegment,
    CircleArc,
    EllipseArc,
    NACA4Curve,
    MappedCurve,
    PiecewiseCurve,
    BezierCurve,
    BSplineCurve,
    FittedCurve,
    FunctionalCurve,
    BoundaryCurveSet2D,
)
from src.data.geometry.parametric_geometry_3d import LineSegment3D
from src.data.mesh.sampling import sample_boundary
from tests.test_inference_geometry import boundary, small_model


class TensorCurveTests(unittest.TestCase):
    def curves(self):
        return [
            LineSegment([0, 0], [1, 2]),
            CircleArc([0.2, 0.3], 1.4, 0.1, 2.8),
            EllipseArc([0, 0], 2, 1, 0.2, 2.7, rotation=0.4),
            NACA4Curve(t=0.3),
            BezierCurve([[0, 0], [0.3, 1], [0.8, -0.2], [1, 0]]),
            BSplineCurve([[0, 0], [0.2, 1], [0.4, 0.3], [0.7, -0.2], [1, 0]]),
            FittedCurve([[0, 0], [0.4, 1], [1, 0]]),
            MappedCurve(CircleArc([0, 0], 1), 0.8, 0.2),
            PiecewiseCurve(
                [LineSegment([0, 0], [1, 0]), LineSegment([1, 0], [1, 2])],
                breakpoints=[0, 0.3, 1],
            ),
            LineSegment3D([0, 0, 0], [1, 2, 3]),
        ]

    def test_values_and_analytic_derivatives_cpu_cuda(self):
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            for curve in self.curves():
                with self.subTest(device=device, curve=type(curve).__name__):
                    t = torch.tensor(
                        [0.13, 0.27, 0.72, 0.91],
                        device=device,
                        dtype=torch.float64,
                        requires_grad=True,
                    )
                    batch = ParametricCurve.batch([curve, curve], device, torch.float64)
                    value = batch(t[None].expand(2, -1))
                    np.testing.assert_allclose(
                        value[0].detach().cpu(),
                        curve(t.detach().cpu().numpy()),
                        rtol=1e-9,
                        atol=1e-9,
                    )
                    derivative = curve.derivative(t)
                    fd = (
                        curve(t.detach().cpu().numpy() + 1e-6)
                        - curve(t.detach().cpu().numpy() - 1e-6)
                    ) / 2e-6
                    np.testing.assert_allclose(
                        derivative.detach().cpu(), fd, rtol=1e-5, atol=1e-6
                    )
                    for axis in range(curve.dimension):
                        grad = torch.autograd.grad(
                            value[0, :, axis].sum(), t, retain_graph=True
                        )[0]
                        torch.testing.assert_close(
                            grad, derivative[:, axis], rtol=1e-8, atol=1e-8
                        )
                    np.testing.assert_allclose(
                        curve.derivative(t.detach().cpu().numpy()),
                        derivative.detach().cpu(),
                        rtol=1e-8,
                        atol=1e-8,
                    )

    def test_endpoints_and_piecewise_global_mean(self):
        c = CircleArc([0, 0], 1, 0, 1)
        np.testing.assert_allclose(
            c.derivative(np.array([0.0, 1.0])),
            [[0, 1], [-np.sin(1), np.cos(1)]],
            atol=1e-12,
        )
        f = FunctionalCurve(lambda t: np.stack((t, 2 * t), -1))
        np.testing.assert_allclose(
            f.derivative(np.array([0.0, 1.0])), [[1, 2], [1, 2]], atol=1e-9
        )
        p = PiecewiseCurve(
            [LineSegment([0, 0], [1, 0]), LineSegment([1, 0], [1, 2])],
            breakpoints=[0, 0.25, 1],
        )
        np.testing.assert_allclose(p.derivative(np.array([0.25])), [[2, 4 / 3]])
        for c in (
            BezierCurve([[0, 0], [1, 2], [3, 0]]),
            BSplineCurve([[0, 0], [1, 2], [2, 1], [3, 0]]),
        ):
            t = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
            y = c(t)
            for axis in range(2):
                g = torch.autograd.grad(y[:, axis].sum(), t, retain_graph=True)[0]
                self.assertTrue(torch.isfinite(g).all())
                torch.testing.assert_close(g, c.derivative(t)[:, axis])

    def test_polynomial_endpoint_second_derivatives(self):
        for c in (
            BezierCurve([[0, 0], [1, 2], [3, 0]]),
            BSplineCurve([[0, 0], [1, 2], [2, 1], [3, 0]]),
            FittedCurve([[0, 0], [1, 2], [3, 0]]),
        ):
            t = torch.tensor(
                [0.0, 0.25, 0.5, 1.0], dtype=torch.float64, requires_grad=True
            )
            first = torch.autograd.grad(c(t).sum(), t, create_graph=True)[0]
            second = torch.autograd.grad(first.sum(), t)[0]
            self.assertTrue(torch.isfinite(second).all())

    def test_analytic_normal_loss_matches_autograd_reference(self):
        from scripts.train.training_common import boundary_normal_loss

        model, shape = small_model(2, "current")
        model = model.double().train()
        objects = [boundary(2, 0.2), boundary(2, 0.3)]
        packed = BoundaryCurveSet2D.batch(objects, dtype=torch.float64)
        b = torch.tensor(
            np.stack([sample_boundary(g, shape) for g in objects]), dtype=torch.float64
        )
        from src.data.mesh.sampling import logical_grid

        q = torch.tensor(logical_grid(shape), dtype=torch.float64)[None].expand(
            2, -1, -1
        )
        with torch.no_grad():
            mesh = torch.cat(
                model(*b.unbind(-1), *q.unbind(-1), boundary_geometry=packed), -1
            )
        torch.manual_seed(23)
        old = boundary_normal_loss(model, b, mesh, shape, 2, "cpu", 64, 0.01, objects)
        torch.manual_seed(23)
        new = boundary_normal_loss(model, b, mesh, shape, 2, "cpu", 64, 0.01, packed)
        torch.testing.assert_close(old, new, rtol=1e-9, atol=1e-9)
        parameter = next(model.parameters())
        old_gradient = torch.autograd.grad(old, parameter)[0]
        new_gradient = torch.autograd.grad(new, parameter)[0]
        torch.testing.assert_close(old_gradient, new_gradient, rtol=1e-8, atol=1e-8)

    def test_four_formulas_bypass_grouping_and_train(self):
        for mapping in ("current", "coons_residual"):
            model, shape = small_model(2, mapping)
            model = model.double().train()
            objects = [boundary(2, 0.2), boundary(2, 0.3)]
            packed = BoundaryCurveSet2D.batch(objects, dtype=torch.float64)
            b = torch.tensor(
                np.stack([sample_boundary(g, shape) for g in objects]),
                dtype=torch.float64,
            )
            q = torch.rand(2, 12, 2, dtype=torch.float64, requires_grad=True)
            with patch(
                "src.data.backend.parametric_torch._grouped",
                side_effect=AssertionError("query-time grouping"),
            ):
                values = torch.cat(
                    model(*b.unbind(-1), *q.unbind(-1), boundary_geometry=packed), -1
                )
                jac = torch.autograd.grad(values.sum(), q, create_graph=True)[0]
                loss = jac.square().mean() + values.square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(q.grad).all())
                self.assertTrue(
                    any(
                        p.grad is not None
                        and torch.isfinite(p.grad).all()
                        and p.grad.abs().sum() > 0
                        for p in model.parameters()
                    )
                )


if __name__ == "__main__":
    unittest.main()
