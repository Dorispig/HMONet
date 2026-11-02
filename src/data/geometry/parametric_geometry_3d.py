from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import torch
from src.data.geometry.parametric_curve_2d import ParametricCurve, FunctionalCurve, LineSegment


class ParametricCurve3D(ParametricCurve):
    def __init__(self, *args, **kwargs):
        super().__init__(dimension=3, *args, **kwargs)


class FunctionalCurve3D(ParametricCurve3D):
    def __init__(self, evaluator, closed=False, exact=True, metadata=None):
        super().__init__(closed=closed, exact=exact, metadata=metadata)
        self.evaluator = evaluator

    def evaluate(self, t):
        if torch.is_tensor(t):
            return self.evaluate_torch(t)
        return self.evaluator(np.clip(t, 0, 1))


class LineSegment3D(ParametricCurve3D):
    def __init__(self, start, end):
        self.start = np.asarray(start, float)
        self.end = np.asarray(end, float)
        super().__init__()

    def derivative(self, t):
        """Exact constant tangent, preserving the query array backend."""
        return self._analytic_derivative(t)

    def evaluate(self, t):
        if torch.is_tensor(t):
            return self.evaluate_torch(t)
        t = np.clip(t, 0, 1)
        return (1 - t)[..., None] * self.start + t[..., None] * self.end

    def reverse(self):
        return LineSegment3D(self.end, self.start)

    def to_dict(self):
        return {
            "type": "line3d",
            "start": self.start.tolist(),
            "end": self.end.tolist(),
        }


class ParametricSurface3D(ABC):
    dimension = 3
    exact = True

    @abstractmethod
    def evaluate(self, u, v): ...

    def evaluate_torch(self, u, v):
        """Evaluate on the tensor device; custom surfaces may override this."""
        from src.data.backend.parametric_torch import evaluate_surface_torch

        return evaluate_surface_torch(self, u, v)

    def sample(self, nu, nv):
        u = np.linspace(0, 1, nu)
        v = np.linspace(0, 1, nv)
        U, V = np.meshgrid(u, v, indexing="ij")
        return self.evaluate(U, V)

    def to_dict(self):
        raise NotImplementedError


class FunctionalSurface3D(ParametricSurface3D):
    def __init__(self, evaluator, exact=True, metadata=None):
        self.evaluator = evaluator
        self.exact = exact
        self.metadata = metadata or {}

    def evaluate(self, u, v):
        if torch.is_tensor(u) or torch.is_tensor(v):
            ref = u if torch.is_tensor(u) else v
            u = torch.as_tensor(u, dtype=ref.dtype, device=ref.device)
            v = torch.as_tensor(v, dtype=ref.dtype, device=ref.device)
            return self.evaluate_torch(u, v)
        return self.evaluator(np.clip(u, 0, 1), np.clip(v, 0, 1))

    def to_dict(self):
        return {
            "type": "functional_surface_3d",
            "exact": self.exact,
            "metadata": self.metadata,
        }


class CylindricalSurface3D(ParametricSurface3D):
    def __init__(
        self,
        center=(0.0, 0.0),
        radius=1.0,
        length=1.0,
        angle_start=0.0,
        angle_end=2 * np.pi,
    ):
        self.center = np.asarray(center, float)
        self.radius = float(radius)
        self.length = float(length)
        self.angle_start = float(angle_start)
        self.angle_end = float(angle_end)

    def evaluate(self, u, v):
        if torch.is_tensor(u) or torch.is_tensor(v):
            ref = u if torch.is_tensor(u) else v
            u = torch.as_tensor(u, dtype=ref.dtype, device=ref.device)
            v = torch.as_tensor(v, dtype=ref.dtype, device=ref.device)
            return self.evaluate_torch(u, v)
        th = self.angle_start + np.asarray(u) * (self.angle_end - self.angle_start)
        return np.stack(
            [
                self.center[0] + self.radius * np.cos(th),
                self.center[1] + self.radius * np.sin(th),
                np.asarray(v) * self.length,
            ],
            -1,
        )

    def to_dict(self):
        return {
            "type": "cylindrical_surface_3d",
            "center": self.center.tolist(),
            "radius": self.radius,
            "length": self.length,
            "angle_start": self.angle_start,
            "angle_end": self.angle_end,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d["center"],
            d["radius"],
            d["length"],
            d.get("angle_start", 0.0),
            d.get("angle_end", 2 * np.pi),
        )


class BilinearSurface3D(ParametricSurface3D):
    def __init__(self, p00, p10, p01, p11):
        self.p00 = np.asarray(p00, float)
        self.p10 = np.asarray(p10, float)
        self.p01 = np.asarray(p01, float)
        self.p11 = np.asarray(p11, float)

    def evaluate(self, u, v):
        if torch.is_tensor(u) or torch.is_tensor(v):
            ref = u if torch.is_tensor(u) else v
            u = torch.as_tensor(u, dtype=ref.dtype, device=ref.device)
            v = torch.as_tensor(v, dtype=ref.dtype, device=ref.device)
            return self.evaluate_torch(u, v)
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        return (
            (1 - u)[..., None] * (1 - v)[..., None] * self.p00
            + u[..., None] * (1 - v)[..., None] * self.p10
            + (1 - u)[..., None] * v[..., None] * self.p01
            + u[..., None] * v[..., None] * self.p11
        )

    def to_dict(self):
        return {
            "type": "bilinear_surface_3d",
            "p00": self.p00.tolist(),
            "p10": self.p10.tolist(),
            "p01": self.p01.tolist(),
            "p11": self.p11.tolist(),
        }


class ExtrudedCurveSurface3D(ParametricSurface3D):
    """Surface S(u,v)=curve(u)+v*direction, useful for axial extrusion."""

    def __init__(self, curve, direction, origin=(0.0, 0.0, 0.0)):
        self.curve = curve
        self.direction = np.asarray(direction, float)
        self.origin = np.asarray(origin, float)

    def evaluate(self, u, v):
        if torch.is_tensor(u) or torch.is_tensor(v):
            ref = u if torch.is_tensor(u) else v
            u = torch.as_tensor(u, dtype=ref.dtype, device=ref.device)
            v = torch.as_tensor(v, dtype=ref.dtype, device=ref.device)
            return self.evaluate_torch(u, v)
        base = np.asarray(self.curve.evaluate(u), dtype=float)
        shift = np.asarray(v)[..., None] * self.direction
        if base.shape[-1] == 2:
            base = np.concatenate([base, np.zeros(base.shape[:-1] + (1,))], axis=-1)
        return base + shift + self.origin

    def to_dict(self):
        return {
            "type": "extruded_curve_surface_3d",
            "curve": self.curve.to_dict(),
            "direction": self.direction.tolist(),
            "origin": self.origin.tolist(),
        }


class CurveBlendSurface3D(ParametricSurface3D):
    """Radially blend two matched closed curves at a constant axial position.

    ``u`` is the radial coordinate from the inner curve to the outer curve;
    ``v`` is the shared circumferential curve parameter.  This ordering agrees
    with the ``z_min`` and ``z_max`` faces of ``BoundarySurfaceSet3D``.
    """

    def __init__(self, inner_curve, outer_curve, z):
        if inner_curve.dimension != 2 or outer_curve.dimension != 2:
            raise ValueError("CurveBlendSurface3D requires two planar curves")
        self.inner_curve = inner_curve
        self.outer_curve = outer_curve
        self.z = float(z)
        self.exact = bool(inner_curve.exact and outer_curve.exact)

    def evaluate(self, u, v):
        if torch.is_tensor(u) or torch.is_tensor(v):
            reference = u if torch.is_tensor(u) else v
            u = torch.as_tensor(u, dtype=reference.dtype, device=reference.device)
            v = torch.as_tensor(v, dtype=reference.dtype, device=reference.device)
            return self.evaluate_torch(u, v)
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        inner = self.inner_curve.evaluate(v)
        outer = self.outer_curve.evaluate(v)
        xy = (1 - u)[..., None] * inner + u[..., None] * outer
        return np.concatenate([xy, np.full(u.shape + (1,), self.z)], axis=-1)

    def to_dict(self):
        return {
            "type": "curve_blend_surface_3d",
            "inner_curve": self.inner_curve.to_dict(),
            "outer_curve": self.outer_curve.to_dict(),
            "z": self.z,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            ParametricCurve.from_dict(data["inner_curve"]),
            ParametricCurve.from_dict(data["outer_curve"]),
            data["z"],
        )

class AnnulusCapSurface3D(ParametricSurface3D):
    """End cap between a closed outer curve and a closed inner curve."""

    def __init__(self, outer_curve, inner_curve, z, outer_at_zero=False):
        self.outer_curve = outer_curve
        self.inner_curve = inner_curve
        self.z = float(z)
        self.outer_at_zero = bool(outer_at_zero)

    def evaluate(self, u, v):
        if torch.is_tensor(u) or torch.is_tensor(v):
            ref = u if torch.is_tensor(u) else v
            u = torch.as_tensor(u, dtype=ref.dtype, device=ref.device)
            v = torch.as_tensor(v, dtype=ref.dtype, device=ref.device)
            return self.evaluate_torch(u, v)
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        inner = self.inner_curve.evaluate(u)
        outer = self.outer_curve.evaluate(u)
        if self.outer_at_zero:
            v = 1 - v
        xy = (1 - v)[..., None] * inner + v[..., None] * outer
        return np.concatenate([xy, np.full(u.shape + (1,), self.z)], axis=-1)

    def to_dict(self):
        return {
            "type": "annulus_cap_surface_3d",
            "outer_curve": self.outer_curve.to_dict(),
            "inner_curve": self.inner_curve.to_dict(),
            "z": self.z,
            "outer_at_zero": self.outer_at_zero,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            ParametricCurve.from_dict(d["outer_curve"]),
            ParametricCurve.from_dict(d["inner_curve"]),
            d["z"],
            outer_at_zero=d.get("outer_at_zero", False),
        )


class PiecewiseSurface3D(ParametricSurface3D):
    def __init__(self, surfaces, breakpoints, axis="u"):
        self.surfaces = surfaces
        self.breakpoints = np.asarray(breakpoints, float)
        self.axis = axis
        self.exact = all(s.exact for s in surfaces)

    def evaluate(self, u, v):
        if torch.is_tensor(u) or torch.is_tensor(v):
            ref = u if torch.is_tensor(u) else v
            u = torch.as_tensor(u, dtype=ref.dtype, device=ref.device)
            v = torch.as_tensor(v, dtype=ref.dtype, device=ref.device)
            return self.evaluate_torch(u, v)
        q = u if self.axis == "u" else v
        out = np.zeros(np.broadcast(u, v).shape + (3,))
        flat = np.asarray(q).ravel()
        for i, s in enumerate(self.surfaces):
            mask = (flat >= self.breakpoints[i]) & (
                (flat <= self.breakpoints[i + 1])
                if i == len(self.surfaces) - 1
                else (flat < self.breakpoints[i + 1])
            )
            if np.any(mask):
                a, b = self.breakpoints[i : i + 2]
                uu = (
                    (flat[mask] - a) / (b - a)
                    if self.axis == "u"
                    else np.asarray(u).ravel()[mask]
                )
                vv = (
                    np.asarray(v).ravel()[mask]
                    if self.axis == "u"
                    else (flat[mask] - a) / (b - a)
                )
                vals = s.evaluate(uu, vv)
                out.reshape(-1, 3)[mask] = vals
        return out

    def to_dict(self):
        return {
            "type": "piecewise_surface_3d",
            "axis": self.axis,
            "breakpoints": self.breakpoints.tolist(),
            "surfaces": [s.to_dict() for s in self.surfaces],
        }



VOLUME_MAP_PARAMETER_ORDER = {
    "annular_nozzle_diffuser": (
        "inner_radius_in", "inner_radius_out", "outer_radius_in",
        "outer_radius_out", "length", "z_min",
    ),
    "twisted_elliptic_annulus": (
        "inner_a", "inner_b", "outer_a", "outer_b",
        "twist_angle_deg", "length", "z_min",
    ),
    "bent_annular_duct": (
        "inner_radius", "outer_radius", "bend_radius",
        "bend_angle_deg", "z_min",
    ),
    "lobed_annular_duct": (
        "inner_radius", "outer_radius", "amplitude", "lobes",
        "phase_twist_deg", "length", "z_min",
    ),
}


def _evaluate_volume_map_numpy(mapping, xi, eta, zeta, parameters):
    """Evaluate one injective annular reference map with NumPy arrays."""
    xi, eta, zeta = np.broadcast_arrays(
        np.asarray(xi, float), np.asarray(eta, float), np.asarray(zeta, float)
    )
    xi, eta, zeta = (np.clip(value, 0.0, 1.0) for value in (xi, eta, zeta))
    theta = 2.0 * np.pi * eta
    p = parameters
    if mapping == "annular_nozzle_diffuser":
        inner = (1.0 - zeta) * p["inner_radius_in"] + zeta * p["inner_radius_out"]
        outer = (1.0 - zeta) * p["outer_radius_in"] + zeta * p["outer_radius_out"]
        radius = (1.0 - xi) * inner + xi * outer
        return np.stack((radius * np.cos(theta), radius * np.sin(theta), p["z_min"] + p["length"] * zeta), -1)
    if mapping == "twisted_elliptic_annulus":
        a = (1.0 - xi) * p["inner_a"] + xi * p["outer_a"]
        b = (1.0 - xi) * p["inner_b"] + xi * p["outer_b"]
        local_x, local_y = a * np.cos(theta), b * np.sin(theta)
        angle = np.deg2rad(p["twist_angle_deg"]) * zeta
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.stack((cosine * local_x - sine * local_y, sine * local_x + cosine * local_y, p["z_min"] + p["length"] * zeta), -1)
    if mapping == "bent_annular_duct":
        radius = (1.0 - xi) * p["inner_radius"] + xi * p["outer_radius"]
        bend = np.deg2rad(p["bend_angle_deg"]) * zeta
        cosine, sine = np.cos(bend), np.sin(bend)
        radial = radius * np.cos(theta)
        return np.stack((p["bend_radius"] * sine - radial * sine, -radius * np.sin(theta), p["z_min"] + p["bend_radius"] * (1.0 - cosine) + radial * cosine), -1)
    if mapping == "lobed_annular_duct":
        phase = np.deg2rad(p["phase_twist_deg"]) * zeta
        outer = p["outer_radius"] * (1.0 + p["amplitude"] * np.cos(p["lobes"] * theta + phase))
        radius = (1.0 - xi) * p["inner_radius"] + xi * outer
        return np.stack((radius * np.cos(theta), radius * np.sin(theta), p["z_min"] + p["length"] * zeta), -1)
    raise ValueError(f"unknown volume map: {mapping}")


class MappedVolumeFace3D(ParametricSurface3D):
    """One face of a serializable analytic map from the unit cube.

    Every face is restricted from the same volume map. This guarantees exact
    shared edges and gives the TFI/PDE pipeline six mutually compatible faces.
    """

    def __init__(self, mapping, fixed_axis, fixed_value, parameters):
        if mapping not in VOLUME_MAP_PARAMETER_ORDER:
            raise ValueError(f"unknown volume map: {mapping}")
        if fixed_axis not in (0, 1, 2) or fixed_value not in (0, 1, 0.0, 1.0):
            raise ValueError("fixed_axis must be 0, 1 or 2 and fixed_value 0 or 1")
        required = VOLUME_MAP_PARAMETER_ORDER[mapping]
        if set(parameters) != set(required):
            raise ValueError(f"{mapping} requires parameters {required}")
        self.mapping = str(mapping)
        self.fixed_axis = int(fixed_axis)
        self.fixed_value = float(fixed_value)
        self.parameters = {name: float(parameters[name]) for name in required}

    def _logical(self, u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        fixed = np.full_like(u, self.fixed_value)
        if self.fixed_axis == 0:
            return fixed, u, v
        if self.fixed_axis == 1:
            return u, fixed, v
        return u, v, fixed

    def evaluate(self, u, v):
        if torch.is_tensor(u) or torch.is_tensor(v):
            reference = u if torch.is_tensor(u) else v
            u = torch.as_tensor(u, dtype=reference.dtype, device=reference.device)
            v = torch.as_tensor(v, dtype=reference.dtype, device=reference.device)
            return self.evaluate_torch(u, v)
        return _evaluate_volume_map_numpy(self.mapping, *self._logical(u, v), self.parameters)

    def evaluate_torch(self, u, v):
        u, v = torch.broadcast_tensors(u, v)
        return self.evaluate_batch_torch([self], u.reshape(1, -1), v.reshape(1, -1)).reshape(*u.shape, 3)

    @classmethod
    def evaluate_batch_torch(cls, surfaces, u, v):
        mapping = surfaces[0].mapping
        axis = surfaces[0].fixed_axis
        value = surfaces[0].fixed_value
        if any((s.mapping, s.fixed_axis, s.fixed_value) != (mapping, axis, value) for s in surfaces):
            raise ValueError("mapped surface batches must share mapping and fixed face")
        names = VOLUME_MAP_PARAMETER_ORDER[mapping]
        p = torch.as_tensor(
            [[surface.parameters[name] for name in names] for surface in surfaces],
            dtype=u.dtype,
            device=u.device,
        )
        fixed = torch.full_like(u, value)
        xi, eta, zeta = ((fixed, u, v) if axis == 0 else (u, fixed, v) if axis == 1 else (u, v, fixed))
        xi, eta, zeta = xi.clamp(0, 1), eta.clamp(0, 1), zeta.clamp(0, 1)
        theta = 2.0 * torch.pi * eta
        values = {name: p[:, index:index + 1] for index, name in enumerate(names)}
        if mapping == "annular_nozzle_diffuser":
            inner = (1 - zeta) * values["inner_radius_in"] + zeta * values["inner_radius_out"]
            outer = (1 - zeta) * values["outer_radius_in"] + zeta * values["outer_radius_out"]
            radius = (1 - xi) * inner + xi * outer
            return torch.stack((radius * theta.cos(), radius * theta.sin(), values["z_min"] + values["length"] * zeta), -1)
        if mapping == "twisted_elliptic_annulus":
            a = (1 - xi) * values["inner_a"] + xi * values["outer_a"]
            b = (1 - xi) * values["inner_b"] + xi * values["outer_b"]
            local_x, local_y = a * theta.cos(), b * theta.sin()
            angle = values["twist_angle_deg"] * (torch.pi / 180.0) * zeta
            return torch.stack((angle.cos() * local_x - angle.sin() * local_y, angle.sin() * local_x + angle.cos() * local_y, values["z_min"] + values["length"] * zeta), -1)
        if mapping == "bent_annular_duct":
            radius = (1 - xi) * values["inner_radius"] + xi * values["outer_radius"]
            bend = values["bend_angle_deg"] * (torch.pi / 180.0) * zeta
            radial = radius * theta.cos()
            return torch.stack((values["bend_radius"] * bend.sin() - radial * bend.sin(), -radius * theta.sin(), values["z_min"] + values["bend_radius"] * (1 - bend.cos()) + radial * bend.cos()), -1)
        if mapping == "lobed_annular_duct":
            phase = values["phase_twist_deg"] * (torch.pi / 180.0) * zeta
            outer = values["outer_radius"] * (1 + values["amplitude"] * (values["lobes"] * theta + phase).cos())
            radius = (1 - xi) * values["inner_radius"] + xi * outer
            return torch.stack((radius * theta.cos(), radius * theta.sin(), values["z_min"] + values["length"] * zeta), -1)
        raise ValueError(f"unknown volume map: {mapping}")

    def to_dict(self):
        return {
            "type": "mapped_volume_face_3d",
            "mapping": self.mapping,
            "fixed_axis": self.fixed_axis,
            "fixed_value": self.fixed_value,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data["mapping"], data["fixed_axis"], data["fixed_value"], data["parameters"])

class BoundarySurfaceSet3D:
    names = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")

    def __init__(
        self, x_min, x_max, y_min, y_max, z_min, z_max, edge_curves=None, metadata=None
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.edge_curves = edge_curves or {}
        self.metadata = metadata or {}

    def surfaces(self):
        return {n: getattr(self, n) for n in self.names}

    def evaluate_faces(self, xi, eta, zeta):
        return (
            self.x_min.evaluate(eta, zeta),
            self.x_max.evaluate(eta, zeta),
            self.y_min.evaluate(xi, zeta),
            self.y_max.evaluate(xi, zeta),
            self.z_min.evaluate(xi, eta),
            self.z_max.evaluate(xi, eta),
        )

    def to_dict(self):
        return {
            "type": "boundary_surface_set_3d",
            "metadata": self.metadata,
            "surfaces": {n: getattr(self, n).to_dict() for n in self.names},
            "edge_curves": self.edge_curves,
        }

    @classmethod
    def from_dict(cls, d):
        surfaces = {n: surface_from_dict(d["surfaces"][n]) for n in cls.names}
        return cls(
            **surfaces, edge_curves=d.get("edge_curves", {}), metadata=d.get("metadata")
        )


def surface_from_dict(data):
    typ = data["type"]
    if typ not in SURFACE_REGISTRY:
        raise ValueError(f"unknown surface type: {typ}")
    klass = SURFACE_REGISTRY[typ]
    if hasattr(klass, "from_dict"):
        return klass.from_dict(data)
    if typ == "bilinear_surface_3d":
        return klass(data["p00"], data["p10"], data["p01"], data["p11"])
    if typ == "extruded_curve_surface_3d":
        return klass(
            ParametricCurve.from_dict(data["curve"]),
            data["direction"],
            origin=data.get("origin", (0.0, 0.0, 0.0)),
        )
    raise ValueError(f"surface type {typ} requires a registered constructor")


SURFACE_REGISTRY = {
    "functional_surface_3d": FunctionalSurface3D,
    "bilinear_surface_3d": BilinearSurface3D,
    "extruded_curve_surface_3d": ExtrudedCurveSurface3D,
    "piecewise_surface_3d": PiecewiseSurface3D,
    "cylindrical_surface_3d": CylindricalSurface3D,
    "annulus_cap_surface_3d": AnnulusCapSurface3D,
    "curve_blend_surface_3d": CurveBlendSurface3D,
    "mapped_volume_face_3d": MappedVolumeFace3D,
}
