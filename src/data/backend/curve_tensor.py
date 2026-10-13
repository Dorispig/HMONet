"""Resident, differentiable batches of curve coefficients.

Object traversal happens only in construction. Query methods broadcast over
batch/point axes; they never regroup objects or upload coefficients. Mixed
primitives use a prebuilt binary dispatch tree instead of a query-time loop.
"""

import math
import numpy as np
import torch
from src.data.backend.curve_kernels import evaluate_fused


class TensorCurveBatch:
    """Callable curve batch with analytic derivatives in global parameters."""

    def __init__(self, curves, device="cpu", dtype=torch.float32):
        from src.data.geometry.parametric_curve_2d import (
            LineSegment,
            CircleArc,
            EllipseArc,
            NACA4Curve,
            MappedCurve,
            PiecewiseCurve,
            BezierCurve,
            BSplineCurve,
            FittedCurve,
            StadiumCurve,
        )
        from src.data.geometry.parametric_geometry_3d import LineSegment3D

        if not curves:
            raise ValueError("At least one curve is required")
        self.count = len(curves)
        self.dimension = curves[0].dimension
        if any(c.dimension != self.dimension for c in curves):
            raise ValueError("Curve dimensions must agree")
        self.device = torch.device(device)
        self.dtype = dtype
        self.objects = tuple(curves)

        def tensor(value, integer=False):
            return torch.as_tensor(
                np.asarray(value), device=device, dtype=torch.long if integer else dtype
            )

        def key(c):
            return (
                type(c),
                (
                    len(c.segments)
                    if isinstance(c, PiecewiseCurve)
                    else (
                        len(c.control)
                        if isinstance(c, (BezierCurve, BSplineCurve))
                        else len(c.points) if isinstance(c, FittedCurve) else None
                    )
                ),
                getattr(c, "degree", None),
                c.closed if isinstance(c, FittedCurve) else None,
            )

        keys = [key(c) for c in curves]
        if any(k != keys[0] for k in keys):
            selected = np.array([k == keys[0] for k in keys])
            self.first_ids = tensor(np.flatnonzero(selected), True)
            self.second_ids = tensor(np.flatnonzero(~selected), True)
            self.first = TensorCurveBatch(
                [c for c, flag in zip(curves, selected) if flag], device, dtype
            )
            self.second = TensorCurveBatch(
                [c for c, flag in zip(curves, selected) if not flag], device, dtype
            )
            self.value_formula = self._mixed_value
            self.derivative_formula = self._mixed_derivative
            return
        kind = type(curves[0])
        if kind in (LineSegment, LineSegment3D):
            self.start = tensor([c.start for c in curves])[:, None]
            self.delta = tensor([c.end - c.start for c in curves])[:, None]
            self.fused_parameters = tensor([[c.start, c.end] for c in curves]).flatten(
                1
            )
            self.value_formula, self.derivative_formula = (
                self._line,
                self._line_derivative,
            )
        elif kind in (CircleArc, EllipseArc):
            self.center = tensor([c.center for c in curves])[:, None]
            self.axes = tensor(
                [
                    (c.radius, c.radius) if kind is CircleArc else (c.a, c.b)
                    for c in curves
                ]
            )[:, None]
            self.start = tensor([c.start_angle for c in curves])[:, None]
            self.sweep = tensor(
                [
                    (
                        -abs(c.end_angle - c.start_angle)
                        if c.clockwise
                        else c.end_angle - c.start_angle
                    )
                    for c in curves
                ]
            )[:, None]
            rotation = tensor([0 if kind is CircleArc else c.rotation for c in curves])[
                :, None
            ]
            self.cosine, self.sine = rotation.cos(), rotation.sin()
            self.fused_parameters = torch.cat(
                (self.center[:, 0], self.axes[:, 0], self.start, self.sweep, rotation),
                -1,
            )
            self.value_formula, self.derivative_formula = (
                self._arc,
                self._arc_derivative,
            )
        elif kind is StadiumCurve:
            self.parameters = tensor(
                [
                    [*curve.center, curve.straight_half_length, curve.radius]
                    for curve in curves
                ]
            )[:, None]
            self.value_formula, self.derivative_formula = (
                self._stadium,
                self._stadium_derivative,
            )
        elif kind is NACA4Curve:
            self.parameters = tensor(
                [[c.m, c.p, c.thickness, c.c, *c.center] for c in curves]
            )[:, None]
            self.value_formula, self.derivative_formula = (
                self._naca,
                self._naca_derivative,
            )
        elif kind is MappedCurve:
            self.start = tensor([c.start for c in curves])[:, None]
            self.delta = tensor([c.end - c.start for c in curves])[:, None]
            self.fused_parameters = tensor([[c.start, c.end] for c in curves]).flatten(
                1
            )
            self.child = TensorCurveBatch([c.base for c in curves], device, dtype)
            self.value_formula, self.derivative_formula = (
                self._mapped,
                self._mapped_derivative,
            )
        elif kind is PiecewiseCurve:
            self.knots = tensor([c.breakpoints for c in curves])
            self.segment_count = len(curves[0].segments)
            self.segment_ids = torch.arange(self.segment_count, device=device)[
                :, None, None
            ]
            self.child = TensorCurveBatch(
                [c.segments[i] for i in range(self.segment_count) for c in curves],
                device,
                dtype,
            )
            self.value_formula, self.derivative_formula = (
                self._piecewise,
                self._piecewise_derivative,
            )
        elif kind is BezierCurve:
            self.control = tensor([c.control for c in curves])
            self.degree = self.control.shape[1] - 1
            self.indices = torch.arange(self.degree + 1, device=device)
            self.binomial = tensor(
                [math.comb(self.degree, i) for i in range(self.degree + 1)]
            )
            self.dcontrol = self.degree * (self.control[:, 1:] - self.control[:, :-1])
            self.dindices = torch.arange(max(self.degree, 1), device=device)
            self.dbinomial = (
                tensor([math.comb(self.degree - 1, i) for i in range(self.degree)])
                if self.degree
                else tensor([1])
            )
            self.value_formula, self.derivative_formula = (
                self._bezier,
                self._bezier_derivative,
            )
        elif kind in (BSplineCurve, FittedCurve):
            # Compile spline pieces once. Evaluation uses tensor gather and powers,
            # so neither SciPy nor a Cox-de Boor degree loop runs on query calls.
            if kind is BSplineCurve:
                from scipy.interpolate import BSpline, PPoly

                n = len(curves[0].control)
                degree = min(curves[0].degree, n - 1)
                knots = np.r_[
                    np.zeros(degree), np.linspace(0, 1, n - degree + 1), np.ones(degree)
                ]
                splines = [
                    [
                        PPoly.from_spline(BSpline(knots, c.control[:, j], degree))
                        for j in range(self.dimension)
                    ]
                    for c in curves
                ]
                self.knots = tensor([row[0].x for row in splines])
                coeff = np.stack([np.stack([p.c for p in row], -1) for row in splines])
                self.coefficients = tensor(coeff.transpose(0, 2, 1, 3))
            else:
                knots = np.stack([c.parameters for c in curves])
                points = np.stack([c.points for c in curves])
                if np.any(np.diff(knots, axis=1) <= 0):
                    raise ValueError("Fitted curve parameters must strictly increase")
                slope = np.diff(points, axis=1) / np.diff(knots, axis=1)[..., None]
                # np.interp keeps a closed curve's final sample constant until 1.
                if knots[0, -1] < 1:
                    knots = np.concatenate((knots, np.ones((len(curves), 1))), axis=1)
                    slope = np.concatenate((slope, np.zeros_like(slope[:, :1])), axis=1)
                    intercept = points
                else:
                    intercept = points[:, :-1]
                self.knots = tensor(knots)
                self.coefficients = tensor(np.stack((slope, intercept), axis=2))
            self.powers = torch.arange(
                self.coefficients.shape[2] - 1, -1, -1, device=device
            )
            self.value_formula, self.derivative_formula = (
                self._polynomial,
                self._polynomial_derivative,
            )
        else:
            # User functions may opt in with an explicit differentiable hook.
            if len(curves) != 1:
                mid = len(curves) // 2
                self.first_ids = tensor(np.arange(mid), True)
                self.second_ids = tensor(np.arange(mid, len(curves)), True)
                self.first = TensorCurveBatch(curves[:mid], device, dtype)
                self.second = TensorCurveBatch(curves[mid:], device, dtype)
                self.value_formula, self.derivative_formula = (
                    self._mixed_value,
                    self._mixed_derivative,
                )
            else:
                self.value_formula = self._custom
                self.derivative_formula = self._custom_derivative

    def __call__(self, t):
        return self.value_formula(t.clamp(0, 1))

    def derivative(self, t):
        return self.derivative_formula(t.clamp(0, 1))

    def _line(self, t):
        fused = evaluate_fused(
            "line2" if self.dimension == 2 else "line3", t, self.fused_parameters
        )
        if fused is not None:
            return fused
        return self.start + t[..., None] * self.delta

    def _line_derivative(self, t):
        return self.delta.expand(self.count, t.shape[1], self.dimension)

    def _rotate(self, x, y):
        return torch.stack(
            (self.cosine * x - self.sine * y, self.sine * x + self.cosine * y), -1
        )

    def _arc(self, t):
        fused = evaluate_fused("ellipse", t, self.fused_parameters)
        if fused is not None:
            return fused
        angle = self.start + self.sweep * t
        return self.center + self._rotate(
            self.axes[..., 0] * angle.cos(), self.axes[..., 1] * angle.sin()
        )

    def _arc_derivative(self, t):
        angle = self.start + self.sweep * t
        return self._rotate(
            -self.axes[..., 0] * angle.sin() * self.sweep,
            self.axes[..., 1] * angle.cos() * self.sweep,
        )

    def _stadium_terms(self, t):
        cx, cy, half_length, radius = self.parameters.unbind(-1)
        angle = 2.0 * torch.pi * t
        cosine, sine = angle.cos(), angle.sin()
        absolute_sine = sine.abs()
        safe_sine = absolute_sine.clamp_min(torch.finfo(t.dtype).eps)
        straight_radial = radius / safe_sine
        on_straight = (absolute_sine > 1e-14) & (
            (straight_radial * cosine).abs() <= half_length
        )
        discriminant = (radius.square() - half_length.square() * sine.square()).clamp_min(0)
        root = discriminant.sqrt()
        cap_radial = half_length * cosine.abs() + root
        radial = torch.where(on_straight, straight_radial, cap_radial)
        return cx, cy, half_length, radius, cosine, sine, radial, on_straight, root, safe_sine

    def _stadium(self, t):
        cx, cy, _, _, cosine, sine, radial, _, _, _ = self._stadium_terms(t)
        return torch.stack((cx + radial * cosine, cy + radial * sine), -1)

    def _stadium_derivative(self, t):
        _, _, half_length, radius, cosine, sine, radial, on_straight, root, safe_sine = self._stadium_terms(t)
        straight_rate = -radius * sine.sign() * cosine / safe_sine.square()
        cap_rate = (
            -half_length * cosine.sign() * sine
            - half_length.square() * sine * cosine / root.clamp_min(torch.finfo(t.dtype).eps)
        )
        radial_rate = torch.where(on_straight, straight_rate, cap_rate)
        return 2.0 * torch.pi * torch.stack(
            (
                radial_rate * cosine - radial * sine,
                radial_rate * sine + radial * cosine,
            ),
            -1,
        )
    def _naca_terms(self, t):
        m, p, thickness, chord, cx, cy = self.parameters.unbind(-1)
        upper = t <= 0.5
        x = torch.where(upper, 2 * t, 2 * (1 - t))
        positive = x > 0
        safe = torch.where(positive, x, torch.ones_like(x))
        root = torch.where(positive, safe.sqrt(), 0.0)
        yt = (
            5
            * thickness
            * chord
            * (
                0.2969 * root
                - 0.126 * x
                - 0.3516 * x**2
                + 0.2843 * x**3
                - 0.1036 * x**4
            )
        )
        denominator = torch.where(x < p, p**2, (1 - p) ** 2)
        yc = torch.where(
            x < p,
            m / p**2 * (2 * p * x - x**2),
            m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2),
        )
        slope = 2 * m / denominator * (p - x)
        angle = slope.atan()
        sign = torch.where(upper, 1.0, -1.0)
        return x, yt, yc, slope, angle, sign, safe, positive, denominator

    def _naca(self, t):
        fused = evaluate_fused("naca", t, self.parameters[:, 0])
        if fused is not None:
            return fused
        x, yt, yc, slope, angle, sign, safe, positive, denominator = self._naca_terms(t)
        m, p, thickness, chord, cx, cy = self.parameters.unbind(-1)
        return torch.stack(
            (
                x * chord - sign * yt * angle.sin() + cx,
                yc + sign * yt * angle.cos() + cy,
            ),
            -1,
        )

    def _naca_derivative(self, t):
        x, yt, yc, slope, angle, sign, safe, positive, denominator = self._naca_terms(t)
        m, p, thickness, chord, cx, cy = self.parameters.unbind(-1)
        # sqrt(x) has no finite derivative at x=0. Keep the established finite
        # endpoint convention used by tensor evaluation, rather than NaN gradients.
        root_derivative = torch.where(positive, 0.5 / safe.sqrt(), 0.0)
        dyt = (
            5
            * thickness
            * chord
            * (
                0.2969 * root_derivative
                - 0.126
                - 0.7032 * x
                + 0.8529 * x**2
                - 0.4144 * x**3
            )
        )
        dangle = (-2 * m / denominator) / (1 + slope**2)
        dx = chord - sign * (dyt * angle.sin() + yt * angle.cos() * dangle)
        dy = slope + sign * (dyt * angle.cos() - yt * angle.sin() * dangle)
        return torch.stack((dx, dy), -1) * (2 * sign)[..., None]

    def _mapped(self, t):
        return self.child(self.start + self.delta * t)

    def _mapped_derivative(self, t):
        return (
            self.child.derivative(self.start + self.delta * t) * self.delta[..., None]
        )

    def _piecewise_arguments(self, t):
        start, end = self.knots[:, :-1].T[..., None], self.knots[:, 1:].T[..., None]
        ids = (
            torch.searchsorted(self.knots.contiguous(), t.contiguous(), right=True) - 1
        ).clamp(0, self.segment_count - 1)
        selected = self.segment_ids == ids[None]
        join = (t[None] == end) & (self.segment_ids < self.segment_count - 1)
        local = torch.where(selected | join, (t[None] - start) / (end - start), 0.5)
        return local, selected, join, end - start

    def _piecewise(self, t):
        local, selected, join, width = self._piecewise_arguments(t)
        values = self.child(local.reshape(-1, t.shape[1])).reshape(
            self.segment_count, self.count, t.shape[1], self.dimension
        )
        result = torch.where(selected[..., None], values, 0.0).sum(0)
        if torch.is_grad_enabled() and t.requires_grad:
            left = torch.where(join[..., None], values, 0.0).sum(0)
            average = 0.5 * (left + result)
            joined = result.detach() + (average - average.detach())
            result = torch.where(join.any(0)[..., None], joined, result)
        return result

    def _piecewise_derivative(self, t):
        local, selected, join, width = self._piecewise_arguments(t)
        values = (
            self.child.derivative(local.reshape(-1, t.shape[1])).reshape(
                self.segment_count, self.count, t.shape[1], self.dimension
            )
            / width[..., None]
        )
        right = torch.where(selected[..., None], values, 0.0).sum(0)
        left = torch.where(join[..., None], values, 0.0).sum(0)
        return torch.where(join.any(0)[..., None], 0.5 * (left + right), right)

    @staticmethod
    def _power_table(t, count):
        """Integer powers using products, avoiding 0**negative in higher gradients."""
        return torch.cat(
            (torch.ones_like(t[..., None]), t[..., None].expand(*t.shape, count - 1)),
            dim=-1,
        ).cumprod(-1)

    def _bezier(self, t):
        basis = (
            self.binomial
            * self._power_table(t, self.degree + 1)
            * self._power_table(1 - t, self.degree + 1).flip(-1)
        )
        return torch.einsum("bnk,bkd->bnd", basis, self.control)

    def _bezier_derivative(self, t):
        if self.degree == 0:
            return torch.zeros(
                (*t.shape, self.dimension), device=t.device, dtype=t.dtype
            )
        basis = (
            self.dbinomial
            * self._power_table(t, self.degree)
            * self._power_table(1 - t, self.degree).flip(-1)
        )
        return torch.einsum("bnk,bkd->bnd", basis, self.dcontrol)

    def _polynomial_parts(self, t):
        ids = (
            torch.searchsorted(self.knots.contiguous(), t.contiguous(), right=True) - 1
        ).clamp(0, self.coefficients.shape[1] - 1)
        coeff = self.coefficients.gather(
            1,
            ids[..., None, None].expand(
                -1, -1, self.coefficients.shape[2], self.dimension
            ),
        )
        local = t - self.knots.gather(1, ids)
        return coeff, local

    def _polynomial(self, t):
        coeff, local = self._polynomial_parts(t)
        powers = self._power_table(local, self.coefficients.shape[2]).flip(-1)
        return (coeff * powers[..., None]).sum(-2)

    def _polynomial_derivative(self, t):
        coeff, local = self._polynomial_parts(t)
        powers = self._power_table(local, self.coefficients.shape[2])
        powers = powers.index_select(-1, (self.powers - 1).clamp_min(0))
        return (coeff * self.powers[None, None, :, None] * powers[..., None]).sum(-2)

    def _mixed_value(self, t):
        a = self.first(t.index_select(0, self.first_ids))
        b = self.second(t.index_select(0, self.second_ids))
        return (
            a.new_empty((self.count, t.shape[1], self.dimension))
            .index_copy(0, self.first_ids, a)
            .index_copy(0, self.second_ids, b)
        )

    def _mixed_derivative(self, t):
        a = self.first.derivative(t.index_select(0, self.first_ids))
        b = self.second.derivative(t.index_select(0, self.second_ids))
        return (
            a.new_empty((self.count, t.shape[1], self.dimension))
            .index_copy(0, self.first_ids, a)
            .index_copy(0, self.second_ids, b)
        )

    def _custom(self, t):
        from src.data.geometry.parametric_curve_2d import ParametricCurve

        obj = self.objects[0]
        if type(obj).evaluate_torch is not ParametricCurve.evaluate_torch:
            return obj.evaluate_torch(t[0])[None]
        if t.requires_grad:
            raise TypeError(
                f"{type(obj).__name__} has no differentiable tensor evaluator; implement evaluate_torch"
            )
        return torch.as_tensor(
            obj.evaluate(t[0].detach().cpu().numpy()), device=t.device, dtype=t.dtype
        )[None]

    def _custom_derivative(self, t):
        obj = self.objects[0]
        if getattr(obj, "derivative_fn", None) is not None:
            value = obj.derivative_fn(t[0])
            return torch.as_tensor(value, device=t.device, dtype=t.dtype)[None]
        lo = (t - 1e-6).clamp(0, 1)
        hi = (t + 1e-6).clamp(0, 1)
        return (self._custom(hi) - self._custom(lo)) / (hi - lo)[..., None]


class TensorBoundaryCurves2D:
    """Four resident curve batches; query-time projection is four formula calls."""

    supports_coordinate_gradients = True

    def __init__(self, boundaries, device="cpu", dtype=torch.float32):
        self.left = TensorCurveBatch([b.left for b in boundaries], device, dtype)
        self.right = TensorCurveBatch([b.right for b in boundaries], device, dtype)
        self.bottom = TensorCurveBatch([b.bottom for b in boundaries], device, dtype)
        self.top = TensorCurveBatch([b.top for b in boundaries], device, dtype)

    def tangents_torch(self, xi, eta):
        """Analytic edge tangents in each edge's global [0,1] parameter."""
        left = self.left.derivative(eta)
        right = self.right.derivative(eta)
        bottom = self.bottom.derivative(xi)
        top = self.top.derivative(xi)
        return torch.stack((left, right, bottom, top), dim=-2)

    def project_torch(self, xi, eta):
        left = self.left(eta)
        right = self.right(eta)
        bottom = self.bottom(xi)
        top = self.top(xi)
        return torch.stack((left, right, bottom, top), dim=-2)


