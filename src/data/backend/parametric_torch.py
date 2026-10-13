"""Batched exact geometry evaluation on the query tensor's device.

Geometry objects remain serializable Python objects. Ordinary calls pack small
parameter arrays on demand; evaluation adapters may supply resident parameters.
Query points and evaluated coordinates stay on the GPU for supported geometries.
Projection coordinates are calculated on every forward, never memoized here.
"""

from __future__ import annotations

import numpy as np
import torch

from src.data.backend.curve_kernels import eligible as fused_eligible, evaluate_fused

from src.data.geometry.parametric_curve_2d import (
    CircleArc,
    EllipseArc,
    LineSegment,
    MappedCurve,
    NACA4Curve,
    ParametricCurve,
    PiecewiseCurve,
    StadiumCurve,
)
from src.data.geometry.parametric_geometry_3d import (
    AnnulusCapSurface3D,
    BilinearSurface3D,
    CylindricalSurface3D,
    CurveBlendSurface3D,
    ExtrudedCurveSurface3D,
    LineSegment3D,
    ParametricSurface3D,
    PiecewiseSurface3D,
)


def _parameters(rows, query):
    """Upload one small, contiguous parameter array for an entire batch."""
    values = np.asarray(rows, dtype=np.float64)
    return torch.as_tensor(values, dtype=query.dtype, device=query.device)


def _group_key(obj):
    """Group by exact class and segment count without overriding subclasses."""
    kind = type(obj)
    if kind is PiecewiseCurve:
        return kind, len(obj.segments)
    if kind is PiecewiseSurface3D:
        return kind, len(obj.surfaces), obj.axis
    return kind


def _grouped(objects, queries, evaluator):
    """Evaluate homogeneous batches directly; scatter mixed groups in order."""
    if not objects or len(objects) != queries[0].shape[0]:
        raise ValueError("Geometry objects must match the query batch size")
    keys = [_group_key(obj) for obj in objects]
    if all(key == keys[0] for key in keys):
        return evaluator(objects, *queries)
    groups = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    result = None
    for indices in groups.values():
        ids = torch.tensor(indices, device=queries[0].device, dtype=torch.long)
        values = evaluator(
            [objects[i] for i in indices],
            *[query.index_select(0, ids) for query in queries],
        )
        if result is None:
            result = values.new_empty((len(objects), *values.shape[1:]))
        result = result.index_copy(0, ids, values)
    return result


def evaluate_curve_torch(curve, t):
    """Evaluate one curve at scalar, vector or mesh-shaped tensor parameters."""
    return evaluate_curves_torch([curve], t.reshape(1, -1)).reshape(
        *t.shape, curve.dimension
    )


def evaluate_curves_torch(curves, t):
    """Return [batch, points, coordinates] for [batch, points] parameters."""
    if t.ndim != 2:
        raise ValueError("Batched curve parameters must have shape [batch, points]")
    return _grouped(curves, (t,), _evaluate_curve_group)


def _evaluate_curve_group(curves, t, prepared=None):
    kind = type(curves[0])
    if kind in (LineSegment, LineSegment3D):
        p = (
            prepared.parameters["main"]
            if prepared
            else _parameters([[c.start, c.end] for c in curves], t)
        )
        fused = evaluate_fused(
            "line2" if p.shape[-1] == 2 else "line3",
            t,
            p.flatten(1),
        )
        if fused is not None:
            return fused
        q = t.clamp(0, 1)[..., None]
        return (1 - q) * p[:, None, 0] + q * p[:, None, 1]
    if kind in (CircleArc, EllipseArc):
        if prepared:
            p = prepared.parameters["main"]
        else:
            rows = []
            for c in curves:
                sweep = c.end_angle - c.start_angle
                if c.clockwise:
                    sweep = -abs(sweep)
                a, b = (c.radius, c.radius) if kind is CircleArc else (c.a, c.b)
                rotation = 0.0 if kind is CircleArc else c.rotation
                rows.append([*c.center, a, b, c.start_angle, sweep, rotation])
            p = _parameters(rows, t)
        fused = evaluate_fused(
            "circle" if kind is CircleArc else "ellipse",
            t,
            p,
        )
        if fused is not None:
            return fused
        cx, cy, a, b, start, sweep, rotation = p.unbind(-1)
        angle = start[:, None] + sweep[:, None] * t.clamp(0, 1)
        x = a[:, None] * angle.cos()
        y = b[:, None] * angle.sin()
        if kind is EllipseArc:
            cosine, sine = rotation.cos()[:, None], rotation.sin()[:, None]
            x, y = cosine * x - sine * y, sine * x + cosine * y
        return torch.stack((cx[:, None] + x, cy[:, None] + y), -1)
    if kind is NACA4Curve:
        p = (
            prepared.parameters["main"]
            if prepared
            else _parameters(
                [[c.m, c.p, c.thickness, c.c, *c.center] for c in curves], t
            )
        )
        fused = evaluate_fused("naca", t, p)
        if fused is not None:
            return fused
        m, position, thickness, chord, cx, cy = p.unbind(-1)
        m, position, thickness, chord, cx, cy = [
            v[:, None] for v in (m, position, thickness, chord, cx, cy)
        ]
        upper = t <= 0.5
        xx = torch.where(upper, 2 * t, 2 * (1 - t))
        # Preserve sqrt(0) exactly and use a finite endpoint derivative. This
        # also prevents masked, inactive piecewise branches from producing NaN.
        positive = xx > 0
        root = torch.where(
            positive, torch.where(positive, xx, torch.ones_like(xx)).sqrt(), 0.0
        )
        yt = (
            5
            * thickness
            * chord
            * (
                0.2969 * root
                - 0.1260 * xx
                - 0.3516 * xx**2
                + 0.2843 * xx**3
                - 0.1036 * xx**4
            )
        )
        before = xx < position
        yc = torch.where(
            before,
            m / position**2 * (2 * position * xx - xx**2),
            m / (1 - position) ** 2 * ((1 - 2 * position) + 2 * position * xx - xx**2),
        )
        slope = torch.where(
            before,
            2 * m / position**2 * (position - xx),
            2 * m / (1 - position) ** 2 * (position - xx),
        )
        angle = slope.atan()
        signed_yt = torch.where(upper, yt, -yt)
        return torch.stack(
            (
                xx * chord - signed_yt * angle.sin() + cx,
                yc + signed_yt * angle.cos() + cy,
            ),
            -1,
        )
    if kind is StadiumCurve:
        parameters = (
            prepared.parameters["main"]
            if prepared
            else _parameters(
                [
                    [*curve.center, curve.straight_half_length, curve.radius]
                    for curve in curves
                ],
                t,
            )
        )
        cx, cy, half_length, radius = [value[:, None] for value in parameters.unbind(-1)]
        angle = 2.0 * torch.pi * t.clamp(0, 1)
        cosine, sine = angle.cos(), angle.sin()
        absolute_sine = sine.abs()
        safe_sine = absolute_sine.clamp_min(torch.finfo(t.dtype).eps)
        straight_radial = radius / safe_sine
        on_straight = (absolute_sine > 1e-14) & (
            (straight_radial * cosine).abs() <= half_length
        )
        discriminant = (radius.square() - half_length.square() * sine.square()).clamp_min(0)
        cap_radial = half_length * cosine.abs() + discriminant.sqrt()
        radial = torch.where(on_straight, straight_radial, cap_radial)
        return torch.stack((cx + radial * cosine, cy + radial * sine), -1)
    if kind is MappedCurve:
        endpoints = (
            prepared.parameters["main"]
            if prepared
            else _parameters([[c.start, c.end] for c in curves], t)
        )
        start, end = endpoints[:, 0:1], endpoints[:, 1:2]
        if prepared:
            return prepared.children[0].evaluate(start + (end - start) * t.clamp(0, 1))
        return evaluate_curves_torch(
            [c.base for c in curves], start + (end - start) * t.clamp(0, 1)
        )
    if kind is PiecewiseCurve:
        return _piecewise_curves(curves, t.clamp(0, 1), prepared)
    from src.data.geometry.parametric_curve_2d import BezierCurve, BSplineCurve, FittedCurve

    if kind in (BezierCurve, BSplineCurve, FittedCurve):
        from src.data.backend.curve_tensor import TensorCurveBatch

        return TensorCurveBatch(curves, t.device, t.dtype)(t)
    return _fallback(curves, (t,), ParametricCurve)


def _piecewise_curves(curves, t, prepared=None):
    """Preserve positions and average global-parameter derivatives at joins.

    Only exact internal knots use the average. Neighboring points retain their
    own segment derivative. The local-parameter transformation includes the
    inverse interval length, so unequal parameter intervals are respected.
    """
    fused = _fused_piecewise_naca(curves, t, prepared)
    if fused is not None:
        return fused
    knots = (
        prepared.parameters["main"]
        if prepared
        else _parameters([c.breakpoints for c in curves], t)
    )
    starts, ends = knots[:, :-1], knots[:, 1:]
    # [segments, batch, points] lets matching child curves share one GPU call.
    local = (t[None] - starts.T[..., None]) / (ends - starts).T[..., None]
    ids = torch.searchsorted(knots.contiguous(), t.contiguous(), right=True) - 1
    ids = ids.clamp(0, starts.shape[1] - 1)
    segments = (
        prepared.parameters["segments"]
        if prepared
        else torch.arange(starts.shape[1], device=t.device)[:, None, None]
    )
    selected = segments == ids[None]
    needs_grad = torch.is_grad_enabled() and t.requires_grad
    if needs_grad:
        left_at_join = (t[None] == ends.T[..., None]) & (segments < starts.shape[1] - 1)
        active = selected | left_at_join
    else:
        active = selected
    # Inactive branches use an interior parameter to avoid undefined endpoint
    # derivatives contaminating a masked backward pass.
    local = torch.where(active, local, 0.5)
    if prepared:
        values = prepared.children[0].evaluate(local.reshape(-1, t.shape[1]))
    else:
        children = [c.segments[i] for i in range(starts.shape[1]) for c in curves]
        values = evaluate_curves_torch(children, local.reshape(-1, t.shape[1]))
    values = values.reshape(starts.shape[1], *t.shape, -1)
    result = torch.where(selected[..., None], values, 0.0).sum(0)
    if needs_grad:
        left = torch.where(left_at_join[..., None], values, 0.0).sum(0)
        average = 0.5 * (left + result)
        # Preserve the right-segment position bit-for-bit. The averaged graph
        # also supports the higher derivatives used by Jacobian-loss backward.
        joined = result.detach() + (average - average.detach())
        result = torch.where(left_at_join.any(0)[..., None], joined, result)
    return result


def _fused_piecewise_naca(curves, t, prepared):
    """Evaluate only the selected branch for compatible two-part airfoils.

    Other piecewise geometries retain the general evaluator. Geometry coefficients
    may differ between batch samples, while the segment mapping must be shared.
    """
    if not fused_eligible(t):
        return None
    mapping = None
    rows = []
    for curve in curves:
        if len(curve.segments) != 2 or any(
            type(segment) is not MappedCurve or type(segment.base) is not NACA4Curve
            for segment in curve.segments
        ):
            return None
        first, second = curve.segments
        a, b = first.base, second.base
        coefficients = [a.m, a.p, a.thickness, a.c, *a.center]
        if not np.array_equal(coefficients, [b.m, b.p, b.thickness, b.c, *b.center]):
            return None
        current = (
            *curve.breakpoints,
            first.start,
            first.end,
            second.start,
            second.end,
        )
        if mapping is not None and current != mapping:
            return None
        mapping = current
        rows.append(coefficients)
    if not (mapping[0] < mapping[1] < mapping[2]):
        return None
    if prepared:
        mapped = prepared.children[0].groups[0]["group"]
        naca = mapped.children[0].groups[0]["group"]
        p = naca.parameters["main"][: len(curves)]
    else:
        p = _parameters(rows, t)
    constants = dict(
        zip(
            ("k0", "k1", "k2", "start0", "end0", "start1", "end1"),
            map(float, mapping),
        )
    )
    return evaluate_fused(
        "piecewise_naca",
        t,
        p,
        **constants,
    )


def _fallback(objects, queries, base_class):
    """Use explicit tensor hooks or one CPU round trip for a NumPy-only group.

    Custom classes can override evaluate_torch to retain coordinate gradients.
    NumPy-only implementations are allowed for ordinary value evaluation, but
    must not silently remove projection terms from a coordinate Jacobian.
    """
    implementation = getattr(type(objects[0]), "evaluate_torch", None)
    if implementation is not None and implementation is not base_class.evaluate_torch:
        return torch.stack(
            [
                obj.evaluate_torch(*[q[i] for q in queries])
                for i, obj in enumerate(objects)
            ]
        )
    if torch.is_grad_enabled() and any(q.requires_grad for q in queries):
        raise TypeError(
            f"{type(objects[0]).__name__} has no differentiable tensor evaluator. "
            "Implement evaluate_torch with Torch operations before enabling "
            "Jacobian loss; NumPy evaluation would detach coordinate gradients."
        )
    cpu_queries = [q.detach().cpu().numpy() for q in queries]
    values = np.stack(
        [obj.evaluate(*[q[i] for q in cpu_queries]) for i, obj in enumerate(objects)]
    )
    return torch.as_tensor(values, dtype=queries[0].dtype, device=queries[0].device)


def evaluate_boundary_curves_torch(boundaries, xi, eta):
    """Return left/right/bottom/top projections without a per-sample GPU loop."""
    if hasattr(boundaries, "project_torch"):
        return boundaries.project_torch(xi, eta)
    if not isinstance(boundaries, (list, tuple)):
        boundaries = [boundaries] * xi.shape[0]
    if len(boundaries) != xi.shape[0]:
        raise ValueError("Boundary objects must match the numerical batch")
    sides = ("left", "right", "bottom", "top")
    curves = [getattr(boundary, side) for side in sides for boundary in boundaries]
    queries = torch.cat((eta, eta, xi, xi), dim=0)
    values = evaluate_curves_torch(curves, queries)
    return values.reshape(4, *xi.shape, 2).permute(1, 2, 0, 3)


def evaluate_surface_torch(surface, u, v):
    """Evaluate a surface using broadcast-compatible tensor coordinates."""
    u, v = torch.broadcast_tensors(u, v)
    return evaluate_surfaces_torch(
        [surface], u.reshape(1, -1), v.reshape(1, -1)
    ).reshape(*u.shape, 3)


def evaluate_surfaces_torch(surfaces, u, v):
    """Evaluate an entire surface batch, including curve extrusions and caps."""
    if u.ndim != 2 or u.shape != v.shape:
        raise ValueError(
            "Batched surface parameters must have equal [batch, points] shapes"
        )
    return _grouped(surfaces, (u, v), _evaluate_surface_group)


def _evaluate_surface_group(surfaces, u, v, prepared=None):
    kind = type(surfaces[0])
    if kind is BilinearSurface3D:
        corners = (
            prepared.parameters["main"]
            if prepared
            else _parameters([[s.p00, s.p10, s.p01, s.p11] for s in surfaces], u)
        )
        u, v = u.clamp(0, 1)[..., None], v.clamp(0, 1)[..., None]
        return (
            (1 - u) * (1 - v) * corners[:, None, 0]
            + u * (1 - v) * corners[:, None, 1]
            + (1 - u) * v * corners[:, None, 2]
            + u * v * corners[:, None, 3]
        )
    if kind is CylindricalSurface3D:
        p = (
            prepared.parameters["main"]
            if prepared
            else _parameters(
                [
                    [*s.center, s.radius, s.length, s.angle_start, s.angle_end]
                    for s in surfaces
                ],
                u,
            )
        )
        cx, cy, radius, length, start, end = [x[:, None] for x in p.unbind(-1)]
        angle = start + (end - start) * u
        return torch.stack(
            (cx + radius * angle.cos(), cy + radius * angle.sin(), length * v), -1
        )
    if kind is ExtrudedCurveSurface3D:
        base = (
            prepared.children[0].evaluate(u)
            if prepared
            else evaluate_curves_torch([s.curve for s in surfaces], u)
        )
        if base.shape[-1] == 2:
            base = torch.cat((base, torch.zeros_like(u[..., None])), -1)
        direction = (
            prepared.parameters["direction"]
            if prepared
            else _parameters([s.direction for s in surfaces], u)
        )
        origin = (
            prepared.parameters["origin"]
            if prepared
            else _parameters([s.origin for s in surfaces], u)
        )
        return base + v[..., None] * direction[:, None] + origin[:, None]
    if kind is CurveBlendSurface3D:
        count = len(surfaces)
        if prepared:
            values = prepared.children[0].evaluate(torch.cat((v, v)))
        else:
            curves = [surface.inner_curve for surface in surfaces] + [
                surface.outer_curve for surface in surfaces
            ]
            values = evaluate_curves_torch(curves, torch.cat((v, v)))
        radial = u.clamp(0, 1)[..., None]
        xy = (1 - radial) * values[:count] + radial * values[count:]
        z = (
            prepared.parameters["z"]
            if prepared
            else _parameters([surface.z for surface in surfaces], u)
        )[:, None].expand_as(u)
        return torch.cat((xy, z[..., None]), -1)
    if kind is AnnulusCapSurface3D:
        count = len(surfaces)
        if prepared:
            values = prepared.children[0].evaluate(torch.cat((u, u)))
        else:
            curves = [s.inner_curve for s in surfaces] + [
                s.outer_curve for s in surfaces
            ]
            values = evaluate_curves_torch(curves, torch.cat((u, u)))
        reversed_v = (
            prepared.parameters["reverse"]
            if prepared
            else _parameters([s.outer_at_zero for s in surfaces], v)
        )[:, None]
        blend = v * (1 - reversed_v) + (1 - v) * reversed_v
        xy = (1 - blend)[..., None] * values[:count] + blend[..., None] * values[count:]
        z = (
            prepared.parameters["z"]
            if prepared
            else _parameters([s.z for s in surfaces], u)
        )[:, None].expand_as(u)
        return torch.cat((xy, z[..., None]), -1)
    if kind is PiecewiseSurface3D:
        q = u if surfaces[0].axis == "u" else v
        knots = (
            prepared.parameters["main"]
            if prepared
            else _parameters([s.breakpoints for s in surfaces], q)
        )
        ids = torch.searchsorted(knots.contiguous(), q.contiguous(), right=True) - 1
        ids = ids.clamp(0, knots.shape[1] - 2)
        result = u.new_zeros((*u.shape, 3))
        for i in range(knots.shape[1] - 1):
            selected = ids == i
            local = (q - knots[:, i : i + 1]) / (
                knots[:, i + 1 : i + 2] - knots[:, i : i + 1]
            )
            local = torch.where(selected, local, 0.5)
            uu, vv = (local, v) if surfaces[0].axis == "u" else (u, local)
            values = (
                prepared.children[i].evaluate(uu, vv)
                if prepared
                else evaluate_surfaces_torch([s.surfaces[i] for s in surfaces], uu, vv)
            )
            result = result + torch.where(selected[..., None], values, 0.0)
        return result
    return _fallback(surfaces, (u, v), ParametricSurface3D)


def evaluate_boundary_surfaces_torch(boundaries, xi, eta, zeta):
    """Return six face projections; equal face types share one tensor batch."""
    if hasattr(boundaries, "project_torch"):
        return boundaries.project_torch(xi, eta, zeta)
    if not isinstance(boundaries, (list, tuple)):
        boundaries = [boundaries] * xi.shape[0]
    if len(boundaries) != xi.shape[0]:
        raise ValueError("Boundary objects must match the numerical batch")
    names = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
    surfaces = [getattr(boundary, name) for name in names for boundary in boundaries]
    u = torch.cat((eta, eta, xi, xi, xi, xi))
    v = torch.cat((zeta, zeta, zeta, zeta, eta, eta))
    values = evaluate_surfaces_torch(surfaces, u, v)
    return values.reshape(6, *xi.shape, 3).permute(1, 2, 0, 3)


