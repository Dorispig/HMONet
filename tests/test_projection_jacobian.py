"""Regression tests for complete projection Jacobians and averaged curve joins."""

import unittest

import numpy as np
import torch

from scripts.train.training_common import jacobian_positive_loss
from src.data.backend.curve_kernels import kernel_status
from src.data.geometry.parametric_curve_2d import (
    FunctionalCurve,
    LineSegment,
    MappedCurve,
    ParametricCurve,
    PiecewiseCurve,
)
from src.data.geometry.parametric_geometry_3d import ExtrudedCurveSurface3D
from src.data.mesh.sampling import sample_boundary
from src.data.backend.parametric_torch import evaluate_curves_torch
from tests.test_inference_geometry import boundary, small_model


def coordinate_jacobian(output, query):
    """Differentiate independent query outputs, retaining mixed derivatives."""
    return torch.stack(
        [
            torch.autograd.grad(
                output[..., axis].sum(), query, create_graph=True, retain_graph=True
            )[0]
            for axis in range(output.shape[-1])
        ],
        dim=-2,
    )


def finite_difference(function, query, step=1e-6):
    columns = []
    with torch.no_grad():
        for axis in range(query.shape[-1]):
            shift = torch.zeros_like(query)
            shift[..., axis] = step
            columns.append(
                (function(query + shift) - function(query - shift)) / (2 * step)
            )
    return torch.stack(columns, dim=-1)


class QuadraticCurve(ParametricCurve):
    """Test-only polynomial with a native differentiable evaluation hook."""

    def __init__(self, second=False):
        super().__init__()
        self.second = second

    def evaluate(self, t):
        if torch.is_tensor(t):
            return self.evaluate_torch(t)
        t = np.asarray(t)
        return np.stack((1 + t, 1 + 2 * t + 3 * t**2) if self.second else (t, t**2), -1)

    def evaluate_torch(self, t):
        return torch.stack(
            (1 + t, 1 + 2 * t + 3 * t**2) if self.second else (t, t**2), -1
        )


class ProjectionJacobianTests(unittest.TestCase):
    def devices(self):
        return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

    def test_piecewise_join_uses_global_parameter_and_preserves_values(self):
        curve = PiecewiseCurve(
            [LineSegment((0, 0), (1, 0)), LineSegment((1, 0), (1, 3))],
            breakpoints=[0, 0.25, 1],
        )
        for device in self.devices():
            with self.subTest(device=device):
                t = torch.tensor(
                    [[0, 0.25 - 1e-7, 0.25, 0.25 + 1e-7, 1]],
                    dtype=torch.float64,
                    device=device,
                    requires_grad=True,
                )
                value = evaluate_curves_torch([curve], t)
                reference = evaluate_curves_torch([curve], t.detach())
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
                derivative = coordinate_jacobian(value, t).transpose(-1, -2)
                expected = t.new_tensor([[[4, 0], [4, 0], [2, 2], [0, 4], [0, 4]]])
                torch.testing.assert_close(derivative, expected, rtol=0, atol=1e-12)
                for mapped, parameter, expected_tangent in (
                    (curve.reverse(), 0.75, [-2, -2]),
                    (MappedCurve(curve, 0.0, 0.5), 0.5, [1, 1]),
                ):
                    q = t.new_tensor([parameter]).requires_grad_()
                    result = mapped(q)
                    tangent = torch.stack(
                        [
                            torch.autograd.grad(
                                result[..., i].sum(), q, retain_graph=True
                            )[0]
                            for i in range(2)
                        ],
                        -1,
                    )
                    torch.testing.assert_close(
                        tangent, t.new_tensor([expected_tangent])
                    )

    def test_piecewise_higher_derivatives_and_batched_knots(self):
        curves = [
            PiecewiseCurve(
                [QuadraticCurve(), QuadraticCurve(True)], breakpoints=[0, knot, 1]
            )
            for knot in (0.25, 0.5)
        ]
        for device in self.devices():
            t = torch.tensor(
                [[0.25], [0.5]], dtype=torch.float64, device=device, requires_grad=True
            )
            value = evaluate_curves_torch(curves, t)
            first = torch.autograd.grad(value[..., 1].sum(), t, create_graph=True)[0]
            second = torch.autograd.grad(first.sum(), t)[0]
            expected_first = t.new_tensor([[0.5 * (2 / 0.25 + 2 / 0.75)], [4]])
            expected_second = t.new_tensor([[0.5 * (2 / 0.25**2 + 6 / 0.75**2)], [16]])
            torch.testing.assert_close(first, expected_first)
            torch.testing.assert_close(second, expected_second)

    def test_extruded_surface_inherits_join_derivative(self):
        curve = PiecewiseCurve(
            [LineSegment((0, 0), (1, 0)), LineSegment((1, 0), (1, 3))],
            breakpoints=[0, 0.25, 1],
        )
        surface = ExtrudedCurveSurface3D(curve, direction=(0, 0, 2))
        q = torch.tensor([[0.25, 0.3]], dtype=torch.float64, requires_grad=True)
        result = surface.evaluate(q[..., 0], q[..., 1])
        jacobian = coordinate_jacobian(result, q)
        torch.testing.assert_close(jacobian, q.new_tensor([[[2, 0], [2, 0], [0, 2]]]))

    def test_numpy_fallback_cannot_silently_detach_jacobian(self):
        curve = FunctionalCurve(lambda t: np.stack((t, t**2), -1))
        query = torch.tensor([0.3], requires_grad=True)
        with self.assertRaisesRegex(TypeError, "differentiable tensor evaluator"):
            curve(query)
        torch.testing.assert_close(curve(query.detach()), torch.tensor([[0.3, 0.09]]))

    def test_full_mapping_and_projection_match_finite_differences(self):
        for device in self.devices():
            for dimension in (2, 3):
                thicknesses = (0.62, 0.65) if dimension == 2 else (0.12, 0.15)
                geometries = [boundary(dimension, t) for t in thicknesses]
                for mapping in ("current", "coons_residual"):
                    model, shape = small_model(dimension, mapping)
                    model = model.to(device=device, dtype=torch.float64).train()
                    b = torch.tensor(
                        np.stack([sample_boundary(g, shape) for g in geometries]),
                        device=device,
                        dtype=torch.float64,
                    )
                    values = [
                        [0.173, 0.317, 0.413],
                        [0.731, 0.613, 0.277],
                        [0.381, 0.823, 0.713],
                    ]
                    q = (
                        torch.tensor(
                            [v[:dimension] for v in values],
                            dtype=torch.float64,
                            device=device,
                        )[None]
                        .repeat(2, 1, 1)
                        .requires_grad_()
                    )
                    for exact in (True, False):
                        with self.subTest(
                            device=device,
                            dimension=dimension,
                            mapping=mapping,
                            exact=exact,
                        ):
                            kwargs = {"boundary_geometry": geometries} if exact else {}

                            def forward(query):
                                return torch.cat(
                                    model(*b.unbind(-1), *query.unbind(-1), **kwargs),
                                    -1,
                                )

                            predicted = forward(q)
                            actual = coordinate_jacobian(predicted, q)
                            expected = finite_difference(forward, q.detach())
                            torch.testing.assert_close(
                                actual, expected, atol=2e-6, rtol=2e-5
                            )

                            # Flatten the face/coordinate axes to check every projection.
                            def project(query):
                                return model.project(
                                    *b.unbind(-1), *query.unbind(-1), **kwargs
                                ).flatten(-2)

                            actual_projection = coordinate_jacobian(project(q), q)
                            expected_projection = finite_difference(project, q.detach())
                            torch.testing.assert_close(
                                actual_projection,
                                expected_projection,
                                atol=2e-6,
                                rtol=2e-5,
                            )

    def test_airfoil_join_averages_child_endpoint_conventions(self):
        curve = boundary(2, 0.65).top
        for device in self.devices():
            q = torch.tensor(
                [0.5], dtype=torch.float64, device=device, requires_grad=True
            )
            point = curve(q)
            tangent = torch.stack(
                [
                    torch.autograd.grad(point[..., axis].sum(), q, retain_graph=True)[0]
                    for axis in range(2)
                ],
                -1,
            )
            sides = []
            for i, local in enumerate((1.0, 0.0)):
                s = q.new_tensor([local]).requires_grad_()
                endpoint = curve.segments[i](s)
                derivative = torch.stack(
                    [
                        torch.autograd.grad(
                            endpoint[..., axis].sum(), s, retain_graph=True
                        )[0]
                        for axis in range(2)
                    ],
                    -1,
                )
                sides.append(
                    derivative / (curve.breakpoints[i + 1] - curve.breakpoints[i])
                )
            torch.testing.assert_close(tangent, 0.5 * (sides[0] + sides[1]))
            self.assertTrue(torch.isfinite(tangent).all())
            torch.testing.assert_close(point, curve(q.detach()), atol=0, rtol=0)

    def test_jacobian_loss_parameter_gradient_and_optimization(self):
        for device in self.devices():
            for dimension in (2, 3):
                for mapping in ("current", "coons_residual"):
                    with self.subTest(
                        device=device, dimension=dimension, mapping=mapping
                    ):
                        model, shape = small_model(dimension, mapping)
                        model = model.to(device=device, dtype=torch.float64).train()
                        thicknesses = (0.62, 0.65) if dimension == 2 else (0.12, 0.15)
                        geometries = [boundary(dimension, t) for t in thicknesses]
                        b = torch.tensor(
                            np.stack([sample_boundary(g, shape) for g in geometries]),
                            dtype=torch.float64,
                            device=device,
                        )
                        q = (
                            torch.tensor(
                                [
                                    [0.173, 0.317, 0.413],
                                    [0.5, 0.73, 0.277],
                                    [0.831, 0.613, 0.713],
                                ],
                                device=device,
                                dtype=torch.float64,
                            )[..., :dimension][None]
                            .repeat(2, 1, 1)
                            .requires_grad_()
                        )

                        def loss():
                            prediction = torch.cat(
                                model(
                                    *b.unbind(-1),
                                    *q.unbind(-1),
                                    boundary_geometry=geometries,
                                ),
                                -1,
                            )
                            # A deliberately active margin tests backward even when all
                            # sampled cells already exceed the production 1e-4 threshold.
                            return jacobian_positive_loss(prediction, q, minimum=10.0)

                        before_calls = kernel_status()["calls"]
                        initial = loss()
                        initial.backward()
                        parameters = [
                            p for p in model.parameters() if p.grad is not None
                        ]
                        self.assertTrue(parameters)
                        self.assertTrue(
                            all(torch.isfinite(p.grad).all() for p in parameters)
                        )
                        norm = torch.sqrt(
                            sum(p.grad.square().sum() for p in parameters)
                        )
                        self.assertGreater(norm.item(), 1e-10)
                        # Check the derivative with respect to a real model weight by
                        # perturbing the parameter with the largest gradient component.
                        parameter = max(
                            parameters, key=lambda p: p.grad.abs().max().item()
                        )
                        index = parameter.grad.abs().reshape(-1).argmax().item()
                        expected = parameter.grad.reshape(-1)[index].item()
                        original = parameter.detach().clone()
                        step = 1e-5
                        with torch.no_grad():
                            parameter.view(-1)[index] += step
                        upper = loss().item()
                        with torch.no_grad():
                            parameter.copy_(original)
                            parameter.view(-1)[index] -= step
                        lower = loss().item()
                        with torch.no_grad():
                            parameter.copy_(original)
                        numerical = (upper - lower) / (2 * step)
                        self.assertAlmostEqual(
                            expected, numerical, delta=2e-4 * max(1.0, abs(expected))
                        )
                        saved = [p.detach().clone() for p in parameters]
                        gradients = [p.grad.detach().clone() for p in parameters]
                        improved = False
                        for rate in (1e-3, 1e-4, 1e-5, 1e-6):
                            with torch.no_grad():
                                for p, old, g in zip(parameters, saved, gradients):
                                    p.copy_(old - rate * g / norm)
                            final = loss().item()
                            if final < initial.item():
                                improved = True
                                break
                        self.assertTrue(improved, (initial.item(), final))
                        if device == "cuda":
                            self.assertEqual(before_calls, kernel_status()["calls"])
                        print(
                            f"Jacobian optimization {device} {dimension}D {mapping}: {initial.item():.9g} -> {final:.9g}; weight gradient {expected:.9g}, FD {numerical:.9g}"
                        )


if __name__ == "__main__":
    unittest.main()

