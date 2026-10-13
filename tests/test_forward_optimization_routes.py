"""Numerical, gradient and no-preparation contracts for the four inference routes."""

import unittest
from unittest.mock import patch
import torch
from src.data.backend.inference_options import inference_routes
from src.data.backend.curve_kernels import kernel_status
from src.data.mesh.sampling import sample_boundary, logical_grid
from src.data.geometry.parametric_curve_2d import (
    BoundaryCurveSet2D,
    CircleArc,
    EllipseArc,
    LineSegment,
    NACA4Curve,
)
from src.models.inference_optimizations import blend_current_2d, fuse_coordinates
from tests.test_inference_geometry import boundary, small_model


class ForwardOptimizationTests(unittest.TestCase):
    def test_feature_contraction_for_batched_and_unbatched_inputs(self):
        for batch_shape in ((), (3,)):
            trunk = torch.randn(*batch_shape, 1200, 17, dtype=torch.float64)
            bx = torch.randn(*batch_shape, 1, 17, dtype=torch.float64)
            by = torch.randn(*batch_shape, 1, 17, dtype=torch.float64)
            bias = torch.randn(*batch_shape, 1, 2, dtype=torch.float64)
            expected = (
                torch.cat(
                    (
                        (bx * trunk).sum(-1, keepdim=True),
                        (by * trunk).sum(-1, keepdim=True),
                    ),
                    -1,
                )
                + bias
            )
            torch.testing.assert_close(fuse_coordinates(trunk, bx, by, bias), expected)

    def test_training_first_and_second_derivatives_unchanged(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            model, shape = small_model(2, "current")
            model = model.to(device=device, dtype=torch.float64).train()
            g = boundary(2, 0.3)
            b = torch.as_tensor(
                sample_boundary(g, shape), device=device, dtype=torch.float64
            )
            results = []
            for enabled in (False, True):
                q = torch.tensor(
                    [[0.23, 0.37], [0.67, 0.81]],
                    device=device,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                with inference_routes(*([enabled] * 4)):
                    out = torch.cat(
                        model(*b.unbind(-1), *q.unbind(-1), boundary_geometry=g), -1
                    )
                    jac = torch.autograd.grad(out.sum(), q, create_graph=True)[0]
                    loss = out.square().mean() + jac.square().mean()
                    grad = torch.autograd.grad(loss, model.weight_head.weight)[0]
                results.append((out, jac, grad))
            for old, new in zip(*results):
                torch.testing.assert_close(old, new, atol=1e-10, rtol=1e-10)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_all_routes_batched_and_unbatched_both_mappings(self):
        for dtype in (torch.float32, torch.float64):
            for mapping in ("current", "coons_residual"):
                model, shape = small_model(2, mapping)
                model = model.to(device="cuda", dtype=dtype)
                g = boundary(2, 0.3)
                b = torch.as_tensor(
                    sample_boundary(g, shape), device="cuda", dtype=dtype
                )
                q = torch.as_tensor(logical_grid((7, 6)), device="cuda", dtype=dtype)
                for batch in (None, 2):
                    bb = b if batch is None else b[None].expand(batch, -1, -1)
                    qq = q if batch is None else q[None].expand(batch, -1, -1)
                    geometry = g if batch is None else [g, boundary(2, 0.4)]
                    with torch.inference_mode():
                        with inference_routes(False, False, False, False):
                            expected = model(
                                *bb.unbind(-1),
                                *qq.unbind(-1),
                                boundary_geometry=geometry
                            )
                        actual = model(
                            *bb.unbind(-1), *qq.unbind(-1), boundary_geometry=geometry
                        )
                    for old, new in zip(expected, actual):
                        tolerance = 2e-6 if dtype == torch.float32 else 2e-13
                        torch.testing.assert_close(
                            old, new, atol=tolerance, rtol=tolerance
                        )
        self.assertFalse(kernel_status()["failures"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_single_boundary_has_no_grouping_or_parameter_tensor_upload(self):
        g = boundary(2, 0.3)
        q = torch.linspace(0, 1, 31, device="cuda").reshape(1, -1)
        with torch.inference_mode(), patch(
            "src.data.backend.parametric_torch._grouped",
            side_effect=AssertionError("unexpected grouping"),
        ), patch(
            "src.data.backend.parametric_torch._parameters",
            side_effect=AssertionError("unexpected parameter upload"),
        ):
            first = g.project_torch(q, q)
            for segment in g.top.segments:
                segment.base.thickness = 0.5
            second = g.project_torch(q, q)
        self.assertGreater(float((first - second).abs().max()), 0.01)
        self.assertTrue(torch.isfinite(second).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_scalar_curve_formulas_match_reference_and_gradients(self):
        curves = [
            LineSegment((0, 1), (2, 3)),
            CircleArc((0.2, 0.3), 1, 0.1, 5.8),
            EllipseArc((0.1, 0.2), 2, 1, 0.2, 4, rotation=0.3),
            NACA4Curve(t=0.3),
        ]
        for curve in curves:
            q = torch.linspace(0.01, 0.99, 41, device="cuda", dtype=torch.float64)
            with torch.inference_mode():
                with inference_routes(scalar=False):
                    expected = curve(q)
                actual = curve(q)
            torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_corner_override_ignores_nonfinite_raw_values(self):
        q = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device="cuda"
        )
        projected = torch.randn(4, 4, 2, device="cuda")
        raw = torch.full((4, 2), float("nan"), device="cuda")
        weight = torch.full((1, 4), 0.6, device="cuda")
        with torch.inference_mode():
            actual = blend_current_2d(q[:, 0], q[:, 1], weight, raw, projected)
        expected = torch.stack(
            (projected[0, 0], projected[1, 1], projected[2, 0], projected[3, 1])
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
