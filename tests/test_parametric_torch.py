"""Regression tests for exact batched geometry evaluation and model gradients."""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.geometry.boundary_generator_2d import (
    circle_airfoil_boundary,
    GenerateRectangleWithTopSemiEllipseBoundary,
    rectangular_boundary,
)
from src.data.geometry.boundary_generator_3d import box_boundary, cylindrical_domain_with_airfoil_hole
from src.data.generation.make_dataset_parametric_2d import to_computational_boundary
from src.data.geometry.parametric_curve_2d import (
    BezierCurve,
    BSplineCurve,
    CircleArc,
    EllipseArc,
    FittedCurve,
    FunctionalCurve,
    LineSegment,
    MappedCurve,
    NACA4Curve,
    PiecewiseCurve,
)
from src.data.geometry.parametric_geometry_3d import (
    BilinearSurface3D,
    FunctionalSurface3D,
    PiecewiseSurface3D,
)
from src.data.backend.parametric_torch import (
    evaluate_boundary_curves_torch,
    evaluate_boundary_surfaces_torch,
    evaluate_curves_torch,
    evaluate_surfaces_torch,
)
from src.models.model import HMONet2D, HMONet3D


def numpy_projection(boundaries, query):
    """Independent reference: evaluate each boundary using its NumPy formula."""
    q = query.detach().cpu().numpy()
    if q.shape[-1] == 2:
        sides = (("left", (1,)), ("right", (1,)), ("bottom", (0,)), ("top", (0,)))
    else:
        sides = (
            ("x_min", (1, 2)),
            ("x_max", (1, 2)),
            ("y_min", (0, 2)),
            ("y_max", (0, 2)),
            ("z_min", (0, 1)),
            ("z_max", (0, 1)),
        )
    values = np.stack(
        [
            np.stack(
                [
                    getattr(boundary, side).evaluate(*[q[i, :, axis] for axis in axes])
                    for side, axes in sides
                ],
                axis=-2,
            )
            for i, boundary in enumerate(boundaries)
        ]
    )
    return torch.as_tensor(values, dtype=query.dtype, device=query.device)


class ParametricTorchTests(unittest.TestCase):
    def test_analytic_curves_match_numpy_with_mixed_types_and_shapes(self):
        airfoil = NACA4Curve(m=0.03, p=0.35, t=0.17, c=1.3, center=(-0.3, 0.1))
        curves = [
            LineSegment((0.1, -0.2), (0.9, 0.7)),
            CircleArc((0.2, -0.1), 1.7, 0.4, 3.1, clockwise=True),
            EllipseArc((-0.1, 0.3), 1.1, 0.6, 0.3, 4.2, rotation=0.7),
            airfoil,
            MappedCurve(airfoil, 0.6, 0.9),
            PiecewiseCurve(
                [LineSegment((0, 0), (1, 0)), LineSegment((1, 0), (1, 2))],
                breakpoints=[0, 0.3, 1],
            ),
        ]
        t = torch.tensor([0, 0.1, 0.3, 0.5, 0.8, 1], dtype=torch.float64)
        expected = torch.tensor(np.stack([c.evaluate(t.numpy()) for c in curves]))
        actual = evaluate_curves_torch(curves, t[None].expand(len(curves), -1))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        for curve in curves:
            for shape in [(), (6,), (2, 3)]:
                q = t[2] if not shape else t.reshape(shape)
                result = curve.evaluate(q)
                self.assertEqual(result.shape, q.shape + (2,))
                np.testing.assert_allclose(
                    result.numpy(), curve.evaluate(q.numpy()), atol=1e-12
                )

    def test_varying_piecewise_parameters_endpoints_and_gradients(self):
        boundaries = [
            to_computational_boundary(circle_airfoil_boundary(t=t, m=m, p=p, c=c))
            for t, m, p, c in [
                (0.05, 0.01, 0.3, 0.8),
                (0.21, 0.03, 0.4, 1),
                (0.4, 0.04, 0.6, 1.2),
            ]
        ]
        q = torch.tensor([0, 0.17, 0.499, 0.5, 0.501, 0.87, 1], dtype=torch.float64)
        q = torch.stack((q, 1 - q), -1)[None].expand(3, -1, -1)
        expected = numpy_projection(boundaries, q)
        actual = evaluate_boundary_curves_torch(boundaries, q[..., 0], q[..., 1])
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        curves = [b.top for b in boundaries]
        query = torch.tensor(
            [[0.17, 0.33, 0.71]] * 3, dtype=torch.float64, requires_grad=True
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x: evaluate_curves_torch(curves, x), (query,)
            )
        )

    def test_mixed_boundary_order_and_single_boundary_broadcast(self):
        boundaries = [
            rectangular_boundary(),
            to_computational_boundary(circle_airfoil_boundary()),
            GenerateRectangleWithTopSemiEllipseBoundary(),
        ]
        q = torch.rand(3, 19, 2, dtype=torch.float64)
        out = evaluate_boundary_curves_torch(boundaries, q[..., 0], q[..., 1])
        torch.testing.assert_close(
            out, numpy_projection(boundaries, q), atol=1e-12, rtol=1e-12
        )
        out = evaluate_boundary_curves_torch(boundaries[1], q[..., 0], q[..., 1])
        torch.testing.assert_close(
            out, numpy_projection([boundaries[1]] * 3, q), atol=1e-12, rtol=1e-12
        )
        with self.assertRaises(ValueError):
            evaluate_boundary_curves_torch(boundaries[:2], q[..., 0], q[..., 1])

    def test_numpy_custom_fallback_and_explicit_tensor_hook(self):
        seen = []

        def function(t):
            self.assertIsInstance(t, np.ndarray)
            seen.append(t.copy())
            return np.stack((t, np.sin(t)), -1)

        curves = [FunctionalCurve(function), FunctionalCurve(function)]
        t = torch.rand(2, 7, dtype=torch.float64)
        out = evaluate_curves_torch(curves, t)
        torch.testing.assert_close(out, torch.stack((t, t.sin()), -1))
        self.assertEqual(len(seen), 2)

        class TensorCurve(FunctionalCurve):
            def evaluate_torch(self, t):
                return torch.stack((t, t.square()), -1)

        t.requires_grad_(True)
        out = evaluate_curves_torch([TensorCurve(function)] * 2, t)
        out.sum().backward()
        torch.testing.assert_close(t.grad, 1 + 2 * t)
        bad = FunctionalCurve(
            lambda t: (_ for _ in ()).throw(ValueError("invalid geometry"))
        )
        # Coordinate-gradient requests must fail before a NumPy conversion.
        with self.assertRaisesRegex(TypeError, "differentiable tensor evaluator"):
            evaluate_curves_torch(curves, t)
        self.assertEqual(len(seen), 2)
        # Ordinary value evaluation still propagates the geometry's own error.
        with self.assertRaisesRegex(ValueError, "invalid geometry"):
            evaluate_curves_torch([bad], t[:1].detach())

    def test_spline_compatibility_uses_same_curve_on_cpu_and_gpu(self):
        control = [[0, 0], [0.2, 1], [0.8, -0.2], [1, 0.4]]
        curves = [BSplineCurve(control), FittedCurve(control), BezierCurve(control)]
        t = torch.linspace(0, 1, 21, dtype=torch.float64)
        for curve in curves:
            torch.testing.assert_close(
                curve.evaluate(t),
                torch.tensor(curve.evaluate(t.numpy())),
                atol=1e-12,
                rtol=1e-12,
            )
        expected = sum(
            factor * np.asarray(point)
            for factor, point in zip([0.125, 0.375, 0.375, 0.125], control)
        )
        np.testing.assert_allclose(curves[-1].evaluate(np.array(0.5)), expected)

    def test_3d_surfaces_match_numpy_and_support_gradients(self):
        boundaries = [
            box_boundary(),
            cylindrical_domain_with_airfoil_hole(thickness=0.08),
            cylindrical_domain_with_airfoil_hole(thickness=0.3),
        ]
        q = torch.rand(3, 23, 3, dtype=torch.float64)
        expected = numpy_projection(boundaries, q)
        actual = evaluate_boundary_surfaces_torch(boundaries, *q.unbind(-1))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        q.requires_grad_(True)
        evaluate_boundary_surfaces_torch(boundaries, *q.unbind(-1)).sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        u, v = torch.rand(2, 3, dtype=torch.float64), torch.tensor(
            0.3, dtype=torch.float64
        )
        for s in boundaries[2].surfaces().values():
            np.testing.assert_allclose(
                s.evaluate(u, v).numpy(),
                s.evaluate(u.numpy(), np.full(u.shape, 0.3)),
                atol=1e-12,
            )

    def test_piecewise_and_custom_surfaces(self):
        lower = BilinearSurface3D([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0])
        upper = BilinearSurface3D([1, 0, 0], [2, 0, 0.5], [1, 1, 0], [2, 1, 0.5])
        surface = PiecewiseSurface3D([lower, upper], [0, 0.4, 1])
        u = torch.tensor([[0, 0.2, 0.4, 0.7, 1]], dtype=torch.float64)
        v = torch.full_like(u, 0.3)
        values = evaluate_surfaces_torch([surface], u, v)
        np.testing.assert_allclose(
            values[0].numpy(), surface.evaluate(u[0].numpy(), v[0].numpy())
        )
        custom = FunctionalSurface3D(lambda u, v: np.stack((u, v, u * v), -1))
        torch.testing.assert_close(
            custom.evaluate(u, v), torch.stack((u, v, u * v), -1)
        )

    def test_model_predictions_and_parameter_gradients_match_reference(self):
        options = dict(
            branch_dim=4,
            adjust_hidden_dim=4,
            branch_hidden_dim=4,
            trunk_hidden_dim=4,
            adjust_hidden_layer_num=1,
            branch_hidden_layer_num=1,
            trunk_hidden_layer_num=1,
            output_dim=4,
            branch_mlp_hidden_dim=4,
            branch_mlp_hidden_layer_num=1,
        )
        for dimension in (2, 3):
            boundaries = (
                [
                    to_computational_boundary(circle_airfoil_boundary(t=t))
                    for t in (0.09, 0.23)
                ]
                if dimension == 2
                else [box_boundary(), cylindrical_domain_with_airfoil_hole()]
            )
            klass = HMONet2D if dimension == 2 else HMONet3D
            for mode in ("current", "coons_residual"):
                with self.subTest(dimension=dimension, mode=mode):
                    model = klass(
                        *([3] * dimension),
                        trunk_dim=dimension,
                        hmonet_mapping_mode=mode,
                        **options
                    ).double()
                    reference = copy.deepcopy(model)
                    count = 8 if dimension == 2 else 26
                    b = torch.rand(2, count, dimension, dtype=torch.float64)
                    q = torch.rand(2, 13, dimension, dtype=torch.float64)
                    q[:, 0] = 0
                    q[:, -1] = 1
                    args = (*b.unbind(-1), *q.unbind(-1))

                    def reference_project(*args, boundary_geometry=None):
                        queries = torch.stack(args[dimension:], -1)
                        return numpy_projection(boundary_geometry, queries)

                    reference.project = reference_project
                    projected = model.project(*args, boundary_geometry=boundaries)
                    single_args = tuple(value[0] for value in args)
                    single = model.project(
                        *single_args, boundary_geometry=boundaries[0]
                    )
                    torch.testing.assert_close(single, projected[0])
                    single_list = model.project(
                        *single_args, boundary_geometry=boundaries[:1]
                    )
                    torch.testing.assert_close(single_list, projected[0])
                    actual = torch.cat(model(*args, boundary_geometry=boundaries), -1)
                    expected = torch.cat(
                        reference(*args, boundary_geometry=boundaries), -1
                    )
                    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
                    actual.square().sum().backward()
                    expected.square().sum().backward()
                    for p, ref in zip(model.parameters(), reference.parameters()):
                        if p.grad is not None:
                            torch.testing.assert_close(
                                p.grad, ref.grad, atol=1e-10, rtol=1e-10
                            )
                    self.assertEqual(
                        set(model.state_dict()), set(reference.state_dict())
                    )
                    self.assertFalse(any("cache" in name for name in vars(model)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_native_projection_has_no_host_query_transfers(self):
        b2 = [
            to_computational_boundary(circle_airfoil_boundary(t=t))
            for t in (0.05, 0.12, 0.3)
        ]
        b3 = [
            box_boundary(),
            cylindrical_domain_with_airfoil_hole(),
            cylindrical_domain_with_airfoil_hole(thickness=0.2),
        ]
        q2 = torch.rand(3, 31, 2, device="cuda")
        q3 = torch.rand(3, 31, 3, device="cuda")
        ref2, ref3 = numpy_projection(b2, q2), numpy_projection(b3, q3)
        with patch.object(
            torch.Tensor, "cpu", side_effect=AssertionError("host transfer")
        ):
            with patch.object(
                torch.Tensor, "numpy", side_effect=AssertionError("numpy conversion")
            ):
                out2 = evaluate_boundary_curves_torch(b2, *q2.unbind(-1))
                out3 = evaluate_boundary_surfaces_torch(b3, *q3.unbind(-1))
        torch.testing.assert_close(out2, ref2, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(out3, ref3, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

