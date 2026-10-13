from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence, Any
import json
import numpy as np
import torch


def _clip(t):
    if hasattr(t, "clamp"):
        return t.clamp(0, 1)
    return np.clip(np.asarray(t), 0.0, 1.0)


def _stack(values, axis=-1):
    if hasattr(values[0], "new_zeros"):
        import torch

        return torch.stack(values, dim=axis)
    return np.stack(values, axis=axis)


class ParametricCurve(ABC):
    dimension = 2
    exact = True
    closed = False

    def __init__(self, dimension=2, closed=False, exact=True, metadata=None):
        self.dimension, self.closed, self.exact = (
            int(dimension),
            bool(closed),
            bool(exact),
        )
        self.metadata = metadata or {}

    @abstractmethod
    def evaluate(self, t): ...

    def evaluate_torch(self, t):
        """Evaluate native tensor formulas without the legacy grouping dispatcher."""
        return self.to_tensor(device=t.device, dtype=t.dtype)(t.reshape(1, -1)).reshape(
            *t.shape, self.dimension
        )

    def to_tensor(self, device="cpu", dtype=torch.float32):
        """Build resident coefficients once for repeated, differentiable queries."""
        from src.data.backend.curve_tensor import TensorCurveBatch

        return TensorCurveBatch([self], device, dtype)

    @staticmethod
    def batch(curves, device="cpu", dtype=torch.float32):
        """Pack like or mixed curves once; queries have shape [batch, points]."""
        from src.data.backend.curve_tensor import TensorCurveBatch

        return TensorCurveBatch(curves, device, dtype)

    def _analytic_derivative(self, t):
        """Return the formula derivative with the same array backend as t."""
        is_tensor = torch.is_tensor(t)
        query = t if is_tensor else torch.as_tensor(np.asarray(t), dtype=torch.float64)
        result = self.to_tensor(query.device, query.dtype).derivative(
            query.reshape(1, -1)
        )
        result = result.reshape(*query.shape, self.dimension)
        return result if is_tensor else result.detach().numpy()

    def __call__(self, t):
        return self.evaluate(t)

    def sample(self, count, endpoint=True):
        t = np.linspace(0.0, 1.0, int(count), endpoint=endpoint)
        return self.evaluate(t)

    def derivative(self, t):
        """Bounded central differences, with one-sided differences at endpoints."""
        t = _clip(t)
        lower, upper = _clip(t - 1e-6), _clip(t + 1e-6)
        return (self.evaluate(upper) - self.evaluate(lower)) / (upper - lower)[
            ..., None
        ]

    def reverse(self):
        # Keep the base curve description so the reversed curve can be saved.
        return MappedCurve(self, 1.0, 0.0)

    def validate(self, tol=1e-7):
        p0, p1 = np.asarray(self.evaluate(0.0)), np.asarray(self.evaluate(1.0))
        if p0.shape != (self.dimension,) or p1.shape != (self.dimension,):
            raise ValueError("curve output dimension mismatch")
        if self.closed and np.linalg.norm(p0 - p1) > tol:
            raise ValueError("closed curve endpoints do not coincide")
        return True

    def to_dict(self):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data):
        typ = data["type"]
        if typ not in CURVE_REGISTRY:
            raise ValueError(f"unknown curve type: {typ}")
        return CURVE_REGISTRY[typ].from_dict(data)


class FunctionalCurve(ParametricCurve):
    def __init__(
        self,
        evaluator,
        dimension=2,
        closed=False,
        exact=True,
        derivative=None,
        metadata=None,
    ):
        super().__init__(dimension, closed, exact, metadata)
        self.evaluator = evaluator
        self.derivative_fn = derivative

    def evaluate(self, t):
        if torch.is_tensor(t):
            return self.evaluate_torch(t)
        return self.evaluator(_clip(t))

    def derivative(self, t):
        return (
            self.derivative_fn(_clip(t))
            if self.derivative_fn
            else super().derivative(t)
        )

    def to_dict(self):
        return {
            "type": "functional",
            "dimension": self.dimension,
            "closed": self.closed,
            "exact": self.exact,
            "metadata": self.metadata,
        }


class MappedCurve(ParametricCurve):
    """Exact reparameterization of a base curve over a subinterval."""

    def __init__(self, base, start, end):
        self.base, self.start, self.end = base, float(start), float(end)
        closed = base.closed and abs(self.end - self.start) == 1.0
        super().__init__(base.dimension, closed, base.exact)

    def derivative(self, t):
        """Analytic global-parameter derivative for NumPy or tensor queries."""
        return self._analytic_derivative(t)

    def evaluate(self, t):
        if torch.is_tensor(t):
            return self.evaluate_torch(t)
        t = _clip(t)
        return self.base.evaluate(self.start + (self.end - self.start) * t)

    def reverse(self):
        return MappedCurve(self.base, self.end, self.start)

    def to_dict(self):
        return {
            "type": "mapped",
            "base": self.base.to_dict(),
            "start": self.start,
            "end": self.end,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(ParametricCurve.from_dict(d["base"]), d["start"], d["end"])


class NACA4Curve(ParametricCurve):
    """Analytic four-digit airfoil curve, upper surface then reversed lower."""

    def __init__(self, m=0.02, p=0.4, t=0.12, c=1.0, center=(0.0, 0.0)):
        self.m, self.p, self.thickness, self.c = float(m), float(p), float(t), float(c)
        self.center = np.asarray(center, float)
        super().__init__(2, closed=True, exact=True, metadata={"family": "naca4"})

    def evaluate_torch(self, t):
        """Evaluate this curve directly; retain analytic autograd when requested."""
        from src.data.backend.curve_kernels import evaluate_scalar_formula

        name = "naca"
        coefficients = dict(
            m=self.m,
            position=self.p,
            thickness=self.thickness,
            chord=self.c,
            cx=float(self.center[0]),
            cy=float(self.center[1]),
        )
        result = evaluate_scalar_formula(name, t, coefficients)
        if result is not None:
            return result
        from src.data.backend.parametric_torch import _evaluate_curve_group

        return _evaluate_curve_group([self], t.reshape(1, -1)).reshape(
            *t.shape, self.dimension
        )

    def derivative(self, t):
        """Analytic global-parameter derivative for NumPy or tensor queries."""
        return self._analytic_derivative(t)

    def evaluate(self, q):
        """Vectorized upper/lower NACA formula; preserve the existing geometry."""
        if torch.is_tensor(q):
            return self.evaluate_torch(q)
        q = _clip(q)
        upper = q <= 0.5
        xx = np.where(upper, 2 * q, 2 * (1 - q))
        yt = (
            5
            * self.thickness
            * self.c
            * (
                0.2969 * np.sqrt(xx)
                - 0.1260 * xx
                - 0.3516 * xx**2
                + 0.2843 * xx**3
                - 0.1036 * xx**4
            )
        )
        before = xx < self.p
        denominator = np.where(before, self.p**2, (1 - self.p) ** 2)
        yc = np.where(
            before,
            self.m / self.p**2 * (2 * self.p * xx - xx**2),
            self.m / (1 - self.p) ** 2 * ((1 - 2 * self.p) + 2 * self.p * xx - xx**2),
        )
        angle = np.arctan(2 * self.m / denominator * (self.p - xx))
        signed_yt = np.where(upper, yt, -yt)
        return np.stack(
            (
                xx * self.c - signed_yt * np.sin(angle) + self.center[0],
                yc + signed_yt * np.cos(angle) + self.center[1],
            ),
            -1,
        )

    def to_dict(self):
        return {
            "type": "naca4",
            "m": self.m,
            "p": self.p,
            "t": self.thickness,
            "c": self.c,
            "center": self.center.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d.get("m", 0.02),
            d.get("p", 0.4),
            d.get("t", 0.12),
            d.get("c", 1.0),
            d.get("center", (0.0, 0.0)),
        )


class StadiumCurve(ParametricCurve):
    """Closed x-aligned stadium sampled by polar angle about its center.

    The normalized parameter is the polar angle divided by ``2*pi``. Thus
    ``t=0`` and ``t=1`` both identify the rightmost point, matching the
    circumferential correspondence in the legacy O-grid generator.
    """

    def __init__(self, straight_half_length=0.8, radius=0.7, center=(0.0, 0.0)):
        if straight_half_length < 0 or radius <= 0:
            raise ValueError(
                "straight_half_length must be non-negative and radius positive"
            )
        self.straight_half_length = float(straight_half_length)
        self.radius = float(radius)
        self.center = np.asarray(center, dtype=float)
        if self.center.shape != (2,):
            raise ValueError("center must contain two coordinates")
        super().__init__(2, closed=True, exact=True, metadata={"family": "stadium"})

    def evaluate(self, t):
        if torch.is_tensor(t):
            return self.evaluate_torch(t)
        angle = 2.0 * np.pi * np.asarray(_clip(t), dtype=float)
        cosine, sine = np.cos(angle), np.sin(angle)
        absolute_sine = np.abs(sine)
        safe_sine = np.maximum(absolute_sine, np.finfo(float).eps)
        straight_radial = self.radius / safe_sine
        on_straight = (absolute_sine > 1e-14) & (
            np.abs(straight_radial * cosine) <= self.straight_half_length
        )
        discriminant = np.maximum(
            self.radius**2 - self.straight_half_length**2 * sine**2, 0.0
        )
        cap_radial = self.straight_half_length * np.abs(cosine) + np.sqrt(
            discriminant
        )
        radial = np.where(on_straight, straight_radial, cap_radial)
        return self.center + np.stack((radial * cosine, radial * sine), axis=-1)

    def evaluate_torch(self, t):
        q = t.clamp(0, 1)
        angle = 2.0 * torch.pi * q
        cosine, sine = angle.cos(), angle.sin()
        absolute_sine = sine.abs()
        safe_sine = absolute_sine.clamp_min(torch.finfo(t.dtype).eps)
        straight_radial = self.radius / safe_sine
        on_straight = (absolute_sine > 1e-14) & (
            (straight_radial * cosine).abs() <= self.straight_half_length
        )
        discriminant = (
            self.radius**2 - self.straight_half_length**2 * sine.square()
        ).clamp_min(0)
        cap_radial = self.straight_half_length * cosine.abs() + discriminant.sqrt()
        radial = torch.where(on_straight, straight_radial, cap_radial)
        center = torch.as_tensor(self.center, dtype=t.dtype, device=t.device)
        return center + torch.stack((radial * cosine, radial * sine), dim=-1)

    def derivative(self, t):
        """Return the analytic derivative with respect to normalized angle."""
        if torch.is_tensor(t):
            q = t.clamp(0, 1)
            angle = 2.0 * torch.pi * q
            cosine, sine = angle.cos(), angle.sin()
            absolute_sine = sine.abs()
            safe_sine = absolute_sine.clamp_min(torch.finfo(t.dtype).eps)
            straight_radial = self.radius / safe_sine
            on_straight = (absolute_sine > 1e-14) & (
                (straight_radial * cosine).abs() <= self.straight_half_length
            )
            discriminant = (
                self.radius**2 - self.straight_half_length**2 * sine.square()
            ).clamp_min(0)
            root = discriminant.sqrt()
            safe_root = root.clamp_min(torch.finfo(t.dtype).eps)
            cap_radial = self.straight_half_length * cosine.abs() + root
            radial = torch.where(on_straight, straight_radial, cap_radial)
            straight_rate = -self.radius * sine.sign() * cosine / safe_sine.square()
            cap_rate = (
                -self.straight_half_length * cosine.sign() * sine
                - self.straight_half_length**2 * sine * cosine / safe_root
            )
            radial_rate = torch.where(on_straight, straight_rate, cap_rate)
            return 2.0 * torch.pi * torch.stack(
                (
                    radial_rate * cosine - radial * sine,
                    radial_rate * sine + radial * cosine,
                ),
                dim=-1,
            )
        query = np.asarray(_clip(t), dtype=float)
        angle = 2.0 * np.pi * query
        cosine, sine = np.cos(angle), np.sin(angle)
        absolute_sine = np.abs(sine)
        safe_sine = np.maximum(absolute_sine, np.finfo(float).eps)
        straight_radial = self.radius / safe_sine
        on_straight = (absolute_sine > 1e-14) & (
            np.abs(straight_radial * cosine) <= self.straight_half_length
        )
        discriminant = np.maximum(
            self.radius**2 - self.straight_half_length**2 * sine**2, 0.0
        )
        root = np.sqrt(discriminant)
        safe_root = np.maximum(root, np.finfo(float).eps)
        cap_radial = self.straight_half_length * np.abs(cosine) + root
        radial = np.where(on_straight, straight_radial, cap_radial)
        straight_rate = -self.radius * np.sign(sine) * cosine / safe_sine**2
        cap_rate = (
            -self.straight_half_length * np.sign(cosine) * sine
            - self.straight_half_length**2 * sine * cosine / safe_root
        )
        radial_rate = np.where(on_straight, straight_rate, cap_rate)
        return 2.0 * np.pi * np.stack(
            (
                radial_rate * cosine - radial * sine,
                radial_rate * sine + radial * cosine,
            ),
            axis=-1,
        )

    def to_dict(self):
        return {
            "type": "stadium",
            "straight_half_length": self.straight_half_length,
            "radius": self.radius,
            "center": self.center.tolist(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            data.get("straight_half_length", 0.8),
            data.get("radius", 0.7),
            data.get("center", (0.0, 0.0)),
        )

class LineSegment(ParametricCurve):
    def __init__(self, start, end):
        self.start = np.asarray(start, float)
        self.end = np.asarray(end, float)
        super().__init__(self.start.size, False, True)

    def evaluate_torch(self, t):
        """Evaluate this curve directly; retain analytic autograd when requested."""
        from src.data.backend.curve_kernels import evaluate_scalar_formula

        name = "line2" if self.dimension == 2 else "line3"
        keys = (
            ("sx", "sy", "ex", "ey")
            if self.dimension == 2
            else ("sx", "sy", "sz", "ex", "ey", "ez")
        )
        coefficients = dict(zip(keys, map(float, (*self.start, *self.end))))
        result = evaluate_scalar_formula(name, t, coefficients)
        if result is not None:
            return result
        from src.data.backend.parametric_torch import _evaluate_curve_group

        return _evaluate_curve_group([self], t.reshape(1, -1)).reshape(
            *t.shape, self.dimension
        )

    def derivative(self, t):
        """Analytic global-parameter derivative for NumPy or tensor queries."""
        return self._analytic_derivative(t)

    def evaluate(self, t):
        if torch.is_tensor(t):
            return self.evaluate_torch(t)
        t = _clip(t)
        a, b = self.start, self.end
        if hasattr(t, "new_zeros"):
            return (1 - t)[..., None] * a + t[..., None] * b
        return (1 - t)[..., None] * a + t[..., None] * b

    def reverse(self):
        return LineSegment(self.end, self.start)

    def to_dict(self):
        return {"type": "line", "start": self.start.tolist(), "end": self.end.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["start"], d["end"])


class CircleArc(ParametricCurve):
    def __init__(
        self, center, radius, start_angle=0.0, end_angle=2 * np.pi, clockwise=False
    ):
        self.center = np.asarray(center, float)
        self.radius = float(radius)
        self.start_angle = float(start_angle)
        self.end_angle = float(end_angle)
        self.clockwise = bool(clockwise)
        super().__init__(2, abs(abs(end_angle - start_angle) - 2 * np.pi) < 1e-8, True)

    def evaluate_torch(self, t):
        """Evaluate this curve directly; retain analytic autograd when requested."""
        from src.data.backend.curve_kernels import evaluate_scalar_formula

        name = "circle"
        sweep = self.end_angle - self.start_angle
        if self.clockwise:
            sweep = -abs(sweep)
        coefficients = dict(
            cx=float(self.center[0]),
            cy=float(self.center[1]),
            a=self.radius,
            b=self.radius,
            start=self.start_angle,
            sweep=sweep,
            rotation=0.0,
        )
        result = evaluate_scalar_formula(name, t, coefficients)
        if result is not None:
            return result
        from src.data.backend.parametric_torch import _evaluate_curve_group

        return _evaluate_curve_group([self], t.reshape(1, -1)).reshape(
            *t.shape, self.dimension
        )

    def derivative(self, t):
        """Analytic global-parameter derivative for NumPy or tensor queries."""
        return self._analytic_derivative(t)

    def evaluate(self, t):
        if torch.is_tensor(t):
            return self.evaluate_torch(t)
        t = _clip(t)
        d = self.end_angle - self.start_angle
        if self.clockwise:
            d = -abs(d)
        th = self.start_angle + t * d
        return _stack(
            [
                self.center[0] + self.radius * np.cos(th),
                self.center[1] + self.radius * np.sin(th),
            ]
        )

    def reverse(self):
        sweep = self.end_angle - self.start_angle
        if self.clockwise:
            sweep = -abs(sweep)
        return CircleArc(
            self.center,
            self.radius,
            self.start_angle + sweep,
            self.start_angle,
        )

    def to_dict(self):
        return {
            "type": "circle_arc",
            "center": self.center.tolist(),
            "radius": self.radius,
            "start_angle": self.start_angle,
            "end_angle": self.end_angle,
            "clockwise": self.clockwise,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d["center"],
            d["radius"],
            d["start_angle"],
            d["end_angle"],
            d.get("clockwise", False),
        )


class EllipseArc(ParametricCurve):
    def __init__(
        self,
        center,
        a,
        b,
        start_angle=0.0,
        end_angle=2 * np.pi,
        rotation=0.0,
        clockwise=False,
    ):
        self.center = np.asarray(center, float)
        self.a = float(a)
        self.b = float(b)
        self.start_angle = float(start_angle)
        self.end_angle = float(end_angle)
        self.rotation = float(rotation)
        self.clockwise = bool(clockwise)
        super().__init__(2, abs(abs(end_angle - start_angle) - 2 * np.pi) < 1e-8, True)

    def evaluate_torch(self, t):
        """Evaluate this curve directly; retain analytic autograd when requested."""
        from src.data.backend.curve_kernels import evaluate_scalar_formula

        name = "ellipse"
        sweep = self.end_angle - self.start_angle
        if self.clockwise:
            sweep = -abs(sweep)
        coefficients = dict(
            cx=float(self.center[0]),
            cy=float(self.center[1]),
            a=self.a,
            b=self.b,
            start=self.start_angle,
            sweep=sweep,
            rotation=self.rotation,
        )
        result = evaluate_scalar_formula(name, t, coefficients)
        if result is not None:
            return result
        from src.data.backend.parametric_torch import _evaluate_curve_group

        return _evaluate_curve_group([self], t.reshape(1, -1)).reshape(
            *t.shape, self.dimension
        )

    def derivative(self, t):
        """Analytic global-parameter derivative for NumPy or tensor queries."""
        return self._analytic_derivative(t)

    def evaluate(self, t):
        if torch.is_tensor(t):
            return self.evaluate_torch(t)
        t = _clip(t)
        d = self.end_angle - self.start_angle
        if self.clockwise:
            d = -abs(d)
        th = self.start_angle + t * d
        x = self.a * np.cos(th)
        y = self.b * np.sin(th)
        c, s = np.cos(self.rotation), np.sin(self.rotation)
        return _stack([self.center[0] + c * x - s * y, self.center[1] + s * x + c * y])

    def reverse(self):
        sweep = self.end_angle - self.start_angle
        if self.clockwise:
            sweep = -abs(sweep)
        return EllipseArc(
            self.center,
            self.a,
            self.b,
            self.start_angle + sweep,
            self.start_angle,
            self.rotation,
        )

    def to_dict(self):
        return {
            "type": "ellipse_arc",
            "center": self.center.tolist(),
            "a": self.a,
            "b": self.b,
            "start_angle": self.start_angle,
            "end_angle": self.end_angle,
            "rotation": self.rotation,
            "clockwise": self.clockwise,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d["center"],
            d["a"],
            d["b"],
            d["start_angle"],
            d["end_angle"],
            d.get("rotation", 0.0),
            d.get("clockwise", False),
        )


class BezierCurve(ParametricCurve):
    def __init__(self, control_points, closed=False):
        self.control = np.asarray(control_points, float)
        super().__init__(self.control.shape[1], closed, True)

    def derivative(self, t):
        """Analytic global-parameter derivative for NumPy or tensor queries."""
        return self._analytic_derivative(t)

    def evaluate(self, t):
        """Evaluate Bernstein basis over all query points and controls at once."""
        if torch.is_tensor(t):
            return self.evaluate_torch(t)
        from scipy.special import comb

        t = _clip(t)
        degree = len(self.control) - 1
        indices = np.arange(degree + 1)
        basis = (
            comb(degree, indices)
            * t[..., None] ** indices
            * (1 - t[..., None]) ** (degree - indices)
        )
        return basis @ self.control

    def to_dict(self):
        return {
            "type": "bezier",
            "control_points": self.control.tolist(),
            "closed": self.closed,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["control_points"], d.get("closed", False))


class BSplineCurve(ParametricCurve):
    def __init__(self, control_points, degree=3, closed=False):
        self.control = np.asarray(control_points, float)
        self.degree = int(degree)
        super().__init__(self.control.shape[1], closed, True)

    def derivative(self, t):
        """Analytic global-parameter derivative for NumPy or tensor queries."""
        return self._analytic_derivative(t)

    def evaluate(self, t):
        """Evaluate all coordinate components together using the same knots."""
        if torch.is_tensor(t):
            return self.evaluate_torch(t)
        from scipy.interpolate import BSpline

        n = len(self.control)
        degree = min(self.degree, n - 1)
        knots = np.r_[
            np.zeros(degree), np.linspace(0, 1, n - degree + 1), np.ones(degree)
        ]
        return BSpline(knots, self.control, degree)(_clip(t))

    def to_dict(self):
        return {
            "type": "bspline",
            "control_points": self.control.tolist(),
            "degree": self.degree,
            "closed": self.closed,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["control_points"], d.get("degree", 3), d.get("closed", False))


class FittedCurve(ParametricCurve):
    def __init__(
        self, points, closed=False, parameterization="centripetal", method="linear"
    ):
        self.points = np.asarray(points, float)
        self.parameterization = parameterization
        self.method = method
        if len(self.points) < 2:
            raise ValueError("at least two points required")
        self.parameters = self._parameters(self.points, parameterization, closed)
        super().__init__(
            self.points.shape[1],
            closed,
            False,
            {"method": method, "parameterization": parameterization},
        )

    @staticmethod
    def _parameters(p, mode, closed):
        d = np.linalg.norm(
            np.diff(np.vstack([p, p[:1]]) if closed else p, axis=0), axis=1
        )
        if mode == "centripetal":
            d = np.sqrt(np.maximum(d, 0))
        s = np.r_[0, np.cumsum(d)]
        s = s / s[-1] if s[-1] > 0 else np.linspace(0, 1, len(s))
        return s[:-1] if closed else s

    def derivative(self, t):
        """Analytic global-parameter derivative for NumPy or tensor queries."""
        return self._analytic_derivative(t)

    def evaluate(self, t):
        if torch.is_tensor(t):
            return self.evaluate_torch(t)
        t = np.asarray(_clip(t))
        flat = t.ravel()
        out = np.empty((flat.size, self.dimension))
        for j in range(self.dimension):
            out[:, j] = np.interp(flat, self.parameters, self.points[:, j])
        return out.reshape(t.shape + (self.dimension,))

    def to_dict(self):
        return {
            "type": "fitted",
            "points": self.points.tolist(),
            "closed": self.closed,
            "parameterization": self.parameterization,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d["points"],
            d.get("closed", False),
            d.get("parameterization", "centripetal"),
            d.get("method", "linear"),
        )


class PiecewiseCurve(ParametricCurve):
    """Exact segments with an averaged autograd tangent at internal joins.

    At an internal breakpoint, tensor evaluation keeps the right segment's
    position and uses the mean of both derivatives with respect to the global
    parameter. This is a chosen corner derivative, not a smoothness guarantee;
    each constituent evaluator still defines its own endpoint convention.
    """

    def __init__(self, segments, breakpoints=None, closed=False, allocation="length"):
        if not segments:
            raise ValueError("segments cannot be empty")
        self.segments = list(segments)
        self.closed = closed
        dim = self.segments[0].dimension
        if any(s.dimension != dim for s in self.segments):
            raise ValueError("dimension mismatch")
        self.breakpoints = np.asarray(
            breakpoints if breakpoints is not None else self._length_breakpoints(),
            float,
        )
        if (
            len(self.breakpoints) != len(self.segments) + 1
            or not np.isfinite(self.breakpoints).all()
            or not np.all(np.diff(self.breakpoints) > 0)
            or abs(self.breakpoints[0]) > 1e-8
            or abs(self.breakpoints[-1] - 1) > 1e-8
        ):
            raise ValueError("invalid breakpoints")
        super().__init__(
            dim, closed, all(s.exact for s in self.segments), {"allocation": allocation}
        )
        self.validate()

    def _length_breakpoints(self):
        lengths = []
        for s in self.segments:
            q = s.sample(256)
            lengths.append(float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()))
        total = sum(lengths) or 1.0
        return np.r_[0, np.cumsum(lengths) / total]

    def evaluate_torch(self, t):
        """Use the selected airfoil segment in one kernel, or the general differentiable rule."""
        from src.data.backend.curve_kernels import eligible, evaluate_scalar_formula

        if eligible(t) and len(self.segments) == 2:
            first, second = self.segments
            if (
                type(first) is MappedCurve
                and type(second) is MappedCurve
                and type(first.base) is NACA4Curve
                and type(second.base) is NACA4Curve
            ):
                a, b = first.base, second.base
                av = (a.m, a.p, a.thickness, a.c, *a.center)
                bv = (b.m, b.p, b.thickness, b.c, *b.center)
                if av == bv:
                    values = (
                        *av,
                        *self.breakpoints,
                        first.start,
                        first.end,
                        second.start,
                        second.end,
                    )
                    keys = (
                        "m",
                        "position",
                        "thickness",
                        "chord",
                        "cx",
                        "cy",
                        "k0",
                        "k1",
                        "k2",
                        "start0",
                        "end0",
                        "start1",
                        "end1",
                    )
                    result = evaluate_scalar_formula(
                        "piecewise_naca", t, dict(zip(keys, map(float, values)))
                    )
                    if result is not None:
                        return result
        from src.data.backend.parametric_torch import _piecewise_curves

        return _piecewise_curves([self], t.reshape(1, -1)).reshape(
            *t.shape, self.dimension
        )

    def derivative(self, t):
        """Analytic global-parameter derivative for NumPy or tensor queries."""
        return self._analytic_derivative(t)

    def evaluate(self, t):
        if torch.is_tensor(t):
            return self.evaluate_torch(t)
        t = _clip(t)
        arr = np.asarray(t)
        out = np.empty(arr.shape + (self.dimension,))
        flat = arr.ravel()
        for i, s in enumerate(self.segments):
            mask = (flat >= self.breakpoints[i]) & (
                (flat <= self.breakpoints[i + 1])
                if i == len(self.segments) - 1
                else (flat < self.breakpoints[i + 1])
            )
            if np.any(mask):
                a, b = self.breakpoints[i : i + 2]
                out.reshape(-1, self.dimension)[mask] = s.evaluate(
                    (flat[mask] - a) / (b - a)
                )
        return out

    def sample_uniform(self, count, include_junctions=True):
        t = np.linspace(0, 1, int(count))
        if include_junctions:
            t = np.unique(np.r_[t, self.breakpoints])
        return t, self.evaluate(t)

    def junction_parameters(self):
        return self.breakpoints[1:-1].copy()

    def check_junctions(self, tol=1e-7):
        for a, b in zip(self.segments[:-1], self.segments[1:]):
            if np.linalg.norm(a.evaluate(1) - b.evaluate(0)) > tol:
                raise ValueError("piecewise junction is discontinuous")

    def reverse(self):
        return PiecewiseCurve(
            [s.reverse() for s in self.segments[::-1]],
            1 - self.breakpoints[::-1],
            self.closed,
        )

    def to_dict(self):
        return {
            "type": "piecewise",
            "segments": [s.to_dict() for s in self.segments],
            "breakpoints": self.breakpoints.tolist(),
            "closed": self.closed,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            [ParametricCurve.from_dict(x) for x in d["segments"]],
            d["breakpoints"],
            d.get("closed", False),
        )


class BoundaryCurveSet2D:
    """Four logical sides, stored in computational or perimeter directions."""

    names = ("bottom", "right", "top", "left")

    def __init__(
        self, bottom, right, top, left, closed=True, metadata=None, validate=True
    ):
        self.bottom, self.right, self.top, self.left = bottom, right, top, left
        self.closed = closed
        self.metadata = metadata or {}
        if validate:
            self.validate()

    supports_coordinate_gradients = True

    def project_torch(self, xi, eta):
        """Evaluate four complete query arrays directly; never cache projection points."""
        from src.data.backend.inference_options import ROUTES

        if not ROUTES.get()[1]:
            from src.data.backend.parametric_torch import evaluate_boundary_curves_torch

            return evaluate_boundary_curves_torch([self] * xi.shape[0], xi, eta)
        left = self.left(eta)
        right = self.right(eta)
        bottom = self.bottom(xi)
        top = self.top(xi)
        return torch.stack((left, right, bottom, top), dim=-2)

    def curves(self):
        return {n: getattr(self, n) for n in self.names}

    @property
    def parameter_order(self):
        """Read the orientation convention, including old circle/airfoil files."""
        order = self.metadata.get("parameter_order")
        if order is None:
            # Old descriptors encoded the direction convention in topology.
            order = (
                "perimeter"
                if self.metadata.get("topology") == "circle_airfoil"
                else "computational"
            )
        if order not in {"computational", "perimeter"}:
            raise ValueError("Unknown boundary parameter_order")
        return order

    def validate(self, tol=1e-6):
        c = self.curves()
        if any(v.dimension != 2 for v in c.values()):
            raise ValueError("2D boundary required")
        if self.parameter_order == "perimeter":
            expected = [
                (c["bottom"].evaluate(1), c["right"].evaluate(0)),
                (c["right"].evaluate(1), c["top"].evaluate(0)),
                (c["top"].evaluate(1), c["left"].evaluate(0)),
                (c["left"].evaluate(1), c["bottom"].evaluate(0)),
            ]
        else:
            expected = [
                (c["bottom"].evaluate(0), c["left"].evaluate(0)),
                (c["bottom"].evaluate(1), c["right"].evaluate(0)),
                (c["top"].evaluate(0), c["left"].evaluate(1)),
                (c["top"].evaluate(1), c["right"].evaluate(1)),
            ]
        for a, b in expected:
            if not np.isfinite([a, b]).all():
                raise ValueError("Boundary endpoints must be finite")
            if np.linalg.norm(np.asarray(a) - np.asarray(b)) > tol:
                raise ValueError("boundary corner mismatch")
        return True

    def sample(self, counts):
        return {n: getattr(self, n).sample(counts[n]) for n in self.names}

    def sample_closed(self, counts):
        if self.parameter_order == "perimeter":
            parts = [
                self.bottom.sample(counts["bottom"]),
                self.right.sample(counts["right"]),
                self.top.sample(counts["top"]),
                self.left.sample(counts["left"]),
            ]
        else:
            parts = [
                self.bottom.sample(counts["bottom"]),
                self.right.sample(counts["right"]),
                self.top.reverse().sample(counts["top"]),
                self.left.reverse().sample(counts["left"]),
            ]
        out = parts[0]
        for i, part in enumerate(parts[1:], 1):
            # The final left endpoint is the initial bottom endpoint; retain
            # it only once when returning a closed loop.
            tail = part[1:-1] if i == len(parts) - 1 else part[1:]
            out = np.vstack([out, tail])
        return np.vstack([out, out[0]]) if self.closed else out

    def to_tensor(self, device="cpu", dtype=torch.float32):
        """Build four resident, differentiable edge batches."""
        return self.batch([self], device, dtype)

    @staticmethod
    def batch(boundaries, device="cpu", dtype=torch.float32):
        """Prepare geometry coefficients, never projected point coordinates."""
        from src.data.backend.curve_tensor import TensorBoundaryCurves2D

        return TensorBoundaryCurves2D(boundaries, device, dtype)

    def to_dict(self):
        return {
            "type": "boundary_curve_set_2d",
            "closed": self.closed,
            "metadata": self.metadata,
            "curves": {n: getattr(self, n).to_dict() for n in self.names},
        }

    @classmethod
    def from_dict(cls, d):
        c = {n: ParametricCurve.from_dict(d["curves"][n]) for n in cls.names}
        return cls(**c, closed=d.get("closed", True), metadata=d.get("metadata"))


CURVE_REGISTRY = {
    "functional": FunctionalCurve,
    "line": LineSegment,
    "circle_arc": CircleArc,
    "ellipse_arc": EllipseArc,
    "bezier": BezierCurve,
    "bspline": BSplineCurve,
    "fitted": FittedCurve,
    "piecewise": PiecewiseCurve,
    "naca4": NACA4Curve,
    "mapped": MappedCurve,
    "stadium": StadiumCurve,
}

