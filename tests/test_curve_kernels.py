"""Exact geometry and gradient regression checks for fused CUDA evaluation."""

import unittest
from unittest.mock import patch
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from src.data.geometry.boundary_generator_2d import circle_airfoil_boundary
from src.data.geometry.boundary_generator_3d import cylindrical_domain_with_airfoil_hole
from src.data.backend.curve_kernels import curve_kernel_mode, kernel_status
from src.data.backend import curve_kernels
from src.data.generation.make_dataset_parametric_2d import to_computational_boundary
from src.data.geometry.parametric_curve_2d import CircleArc, EllipseArc, LineSegment, NACA4Curve
from src.data.geometry.parametric_geometry_3d import LineSegment3D
from src.data.mesh.sampling import logical_grid, sample_boundary
from src.data.backend.parametric_torch import evaluate_curves_torch
from tests.test_inference_geometry import small_model


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for fused kernels")
class CurveKernelTests(unittest.TestCase):
    def assert_reference(self, function, tolerance=2e-6):
        with curve_kernel_mode(False):
            expected = function()
        with curve_kernel_mode(True):
            actual = function()
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        return actual, expected

    def test_primitives_batched_float32_and_float64(self):
        families = [
            [LineSegment((0, 1), (2, 3)), LineSegment((1, 1), (4, -2))],
            [
                LineSegment3D((0, 1, 2), (3, 4, 5)),
                LineSegment3D((2, 1, 0), (-1, -2, -3)),
            ],
            [CircleArc((0, 0), 1, 0, 6.2), CircleArc((2, 3), 4, 0.2, 3.8)],
            [
                EllipseArc((0, 0), 2, 1, 0, 4.2),
                EllipseArc((2, 3), 1, 0.5, -1, 2, rotation=0.4),
            ],
            [NACA4Curve(t=0.2), NACA4Curve(m=0.07, p=0.35, t=0.35, c=2)],
        ]
        for dtype in (torch.float32, torch.float64):
            t = torch.linspace(0, 1, 131, dtype=dtype, device="cuda")[None].expand(
                2, -1
            )
            for curves in families:
                with self.subTest(dtype=dtype, curve=type(curves[0]).__name__):
                    self.assert_reference(
                        lambda: evaluate_curves_torch(curves, t),
                        2e-6 if dtype == torch.float32 else 2e-14,
                    )
        self.assertFalse(kernel_status()["failures"])

    def test_piecewise_endpoints_junctions_and_mutation(self):
        boundaries = [
            to_computational_boundary(circle_airfoil_boundary(t=t)) for t in (0.15, 0.3)
        ]
        curves = [boundary.top for boundary in boundaries]
        t = torch.tensor(
            [[0, 0.1, 0.499999, 0.5, 0.500001, 0.9, 1]], device="cuda"
        ).expand(2, -1)
        before, _ = self.assert_reference(lambda: evaluate_curves_torch(curves, t))
        for segment in curves[1].segments:
            segment.base.thickness = 0.45
        after, _ = self.assert_reference(lambda: evaluate_curves_torch(curves, t))
        self.assertGreater(float((after - before).abs().max()), 0.01)

    def test_coordinate_gradients_keep_reference_path(self):
        curves = [NACA4Curve(t=0.2), NACA4Curve(t=0.3)]
        t = (
            torch.linspace(0.01, 0.99, 22, device="cuda")
            .reshape(2, 11)
            .requires_grad_()
        )
        count = kernel_status()["calls"]
        actual, expected = self.assert_reference(
            lambda: evaluate_curves_torch(curves, t), 0
        )
        actual_gradient = torch.autograd.grad(actual.sum(), t, create_graph=True)[0]
        reference_gradient = torch.autograd.grad(expected.sum(), t, create_graph=True)[
            0
        ]
        torch.testing.assert_close(actual_gradient, reference_gradient, atol=0, rtol=0)
        actual_second = torch.autograd.grad(actual_gradient.sum(), t)[0]
        reference_second = torch.autograd.grad(reference_gradient.sum(), t)[0]
        # CUDA may accumulate second-derivative branches in a different order.
        # The call counter below verifies that neither graph used fused code.
        torch.testing.assert_close(
            actual_second, reference_second, atol=2e-6, rtol=2e-6
        )
        self.assertEqual(count, kernel_status()["calls"])

    def test_models_batches_mappings_and_weight_gradients(self):
        for dimension in (2, 3):
            geometries = (
                [
                    to_computational_boundary(circle_airfoil_boundary(t=t))
                    for t in (0.2, 0.3)
                ]
                if dimension == 2
                else [cylindrical_domain_with_airfoil_hole(thickness=t) for t in (0.2, 0.3)]
            )
            for mapping in ("current", "coons_residual"):
                with self.subTest(dimension=dimension, mapping=mapping):
                    model, shape = small_model(dimension, mapping)
                    model = model.cuda().train()
                    b = torch.tensor(
                        np.stack([sample_boundary(g, shape) for g in geometries]),
                        dtype=torch.float32,
                        device="cuda",
                    )
                    q = torch.tensor(
                        logical_grid((5,) * dimension),
                        dtype=torch.float32,
                        device="cuda",
                    )[None].expand(2, -1, -1)

                    def forward():
                        return torch.cat(
                            model(
                                *b.unbind(-1),
                                *q.unbind(-1),
                                boundary_geometry=geometries,
                            ),
                            -1,
                        )

                    actual, expected = self.assert_reference(forward)
                    weights = list(model.parameters())
                    actual_gradients = torch.autograd.grad(
                        actual.square().mean(), weights, allow_unused=True
                    )
                    expected_gradients = torch.autograd.grad(
                        expected.square().mean(), weights, allow_unused=True
                    )
                    for a, e in zip(actual_gradients, expected_gradients):
                        if a is None or e is None:
                            self.assertIs(a, e)
                        else:
                            torch.testing.assert_close(a, e, rtol=2e-5, atol=2e-6)

    def test_direct_geometry_tracks_changed_queries(self):
        for dimension in (2, 3):
            geometry = (
                to_computational_boundary(circle_airfoil_boundary(t=0.3))
                if dimension == 2
                else cylindrical_domain_with_airfoil_hole(thickness=0.3)
            )
            model, shape = small_model(dimension, "current")
            model = model.cuda()
            b = torch.tensor(
                sample_boundary(geometry, shape), device="cuda", dtype=torch.float32
            )
            q = torch.tensor(
                logical_grid((5,) * dimension), device="cuda", dtype=torch.float32
            )
            with torch.inference_mode():

                def forward():
                    return model.project(
                        *b.unbind(-1), *q.unbind(-1), boundary_geometry=geometry
                    )

                self.assert_reference(forward)
                q.mul_(0.9).add_(0.03)
                self.assert_reference(forward)

    def test_missing_compiler_falls_back_without_repeated_attempts(self):
        curves = [NACA4Curve(t=0.3)]
        t = torch.linspace(0, 1, 21, device="cuda")[None]
        with curve_kernel_mode(False):
            expected = evaluate_curves_torch(curves, t)

        def unavailable(*args, **kwargs):
            raise RuntimeError("nvrtc unavailable in test")

        with patch.dict(curve_kernels._FAILED, {}, clear=True):
            with patch.dict(curve_kernels._FUNCTIONS, {"naca": unavailable}):
                actual = evaluate_curves_torch(curves, t)
                repeated = evaluate_curves_torch(curves, t)
                self.assertEqual(len(curve_kernels._FAILED), 1)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(repeated, expected, atol=0, rtol=0)

    @unittest.skipUnless(os.name == "nt", "Windows cache workaround")
    def test_cache_finalization_only_touches_owned_unpublished_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            owned = root / f"exact_test_arch8.9_nvrtc12.1_hash_tmp_{os.getpid()}"
            published = root / "exact_test_arch8.9_nvrtc12.1_hash"
            unrelated = root / "exact_test_arch8.9_nvrtc12.1_other_tmp_0"
            owned.write_bytes(b"compiled test kernel")
            unrelated.write_bytes(b"another process")
            with patch.dict(
                os.environ,
                {"PYTORCH_KERNEL_CACHE_PATH": folder, "USE_PYTORCH_KERNEL_CACHE": "1"},
            ):
                curve_kernels._persist_windows_kernel("exact_test")
                self.assertEqual(published.read_bytes(), b"compiled test kernel")
                self.assertTrue(unrelated.exists())
                owned.write_bytes(b"must not overwrite")
                curve_kernels._persist_windows_kernel("exact_test")
                self.assertEqual(published.read_bytes(), b"compiled test kernel")
                self.assertTrue(owned.exists())


if __name__ == "__main__":
    unittest.main()

