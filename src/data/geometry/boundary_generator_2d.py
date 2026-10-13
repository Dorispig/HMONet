"""Analytic 2-D boundary generators for the parameterized dataset format."""

import numpy as np
from src.data.geometry.parametric_curve_2d import (
    LineSegment,
    FunctionalCurve,
    MappedCurve,
    BSplineCurve,
    BezierCurve,
    CircleArc,
    EllipseArc,
    NACA4Curve,
    PiecewiseCurve,
    BoundaryCurveSet2D,
)


def naca4_curve(m=0.02, p=0.4, t=0.12, c=1.0, center=(-0.5, 0.0)):
    return NACA4Curve(m=m, p=p, t=t, c=c, center=center)


def rectangular_boundary(x_min=-1, x_max=1, y_min=-1, y_max=1):
    return BoundaryCurveSet2D(
        LineSegment([x_min, y_min], [x_max, y_min]),
        LineSegment([x_max, y_min], [x_max, y_max]),
        LineSegment([x_min, y_max], [x_max, y_max]),
        LineSegment([x_min, y_min], [x_min, y_max]),
        closed=True,
    )


def circle_airfoil_boundary(
    radius=1.0, m=0.02, p=0.4, t=0.12, c=1.0, center=(0.0, 0.0)
):
    """Create the four-part circle/airfoil boundary topology.

    The four sides follow perimeter order: outer circle, connector to the
    lower trailing edge, airfoil through the leading tip to the upper trailing
    edge, then connector back to the circle. Convert the direction convention
    before using these curves for TFI or HMONet projection.
    """
    outer = CircleArc(center, radius, 0.0, 2 * np.pi, clockwise=False)
    airfoil = naca4_curve(m, p, t, c, center=(center[0] - c / 2, center[1]))
    # q=0 and q=1 are the coincident leading edge; q=0.5 is the
    # closed trailing edge shared by the lower and upper surfaces.
    lower_te = np.asarray(airfoil.evaluate(0.5))
    upper_te = lower_te.copy()
    circle_point = np.asarray(outer.evaluate(0.0))
    connector_lower = LineSegment(circle_point, lower_te)
    connector_upper = LineSegment(upper_te, circle_point)
    airfoil_path = PiecewiseCurve(
        [
            MappedCurve(airfoil, 0.5, 1.0),
            MappedCurve(airfoil, 0.0, 0.5),
        ],
        breakpoints=[0.0, 0.5, 1.0],
    )
    return BoundaryCurveSet2D(
        outer,
        connector_lower,
        airfoil_path,
        connector_upper,
        closed=True,
        metadata={
            "family": "circle_airfoil",
            "topology": "circle_airfoil",
            "parameter_order": "perimeter",
            "m": m,
            "p": p,
            "t": t,
            "c": c,
            "radius": radius,
            "center": list(center),
        },
    )


def GenerateRectangleWithTopSemiEllipseBoundary(
    length=1.0, height=1.0, semi_minor_axis=0.5
):
    """Three straight sides and an upper elliptical arc; lower-left is origin."""
    if length <= 0 or height <= 0 or semi_minor_axis < 0:
        raise ValueError(
            "Require positive rectangle dimensions and nonnegative semi_minor_axis"
        )
    top = EllipseArc((length / 2, height), length / 2, semi_minor_axis, np.pi, 0.0)
    return BoundaryCurveSet2D(
        LineSegment((0, 0), (length, 0)),
        LineSegment((length, 0), (length, height)),
        top,
        LineSegment((0, 0), (0, height)),
        metadata={
            "family": "rect_top_semiellipse",
            "length": length,
            "height": height,
            "semi_minor_axis": semi_minor_axis,
        },
    )


def EccentricAnnularSectorBoundary(
    outer_radius=0.5, inner_radius=0.1, inner_center_x=0.5
):
    """Upper half-annulus bounded by two semicircles and horizontal segments.

    Both arcs use theta(t) = pi * (1 - t), so their stored parameter
    direction is left to right. A closed boundary traversal reverses the
    top arc and the left segment: inner arc, right segment, outer arc,
    left segment. The outer arc's left endpoint is the origin.
    """
    if (
        not (0 < inner_radius < outer_radius)
        or abs(inner_center_x - outer_radius) + inner_radius >= outer_radius
    ):
        raise ValueError(
            "Inner semicircle must lie strictly inside the outer semicircle"
        )
    outer = CircleArc((outer_radius, 0.0), outer_radius, np.pi, 0.0)
    inner = CircleArc((inner_center_x, 0.0), inner_radius, np.pi, 0.0)
    outer_left, outer_right = outer.evaluate(0.0), outer.evaluate(1.0)
    inner_left, inner_right = inner.evaluate(0.0), inner.evaluate(1.0)
    return BoundaryCurveSet2D(
        inner,
        LineSegment(inner_right, outer_right),
        outer,
        LineSegment(inner_left, outer_left),
        metadata={
            "family": "eccentric_semiannulus",
            "outer_radius": outer_radius,
            "inner_radius": inner_radius,
            "inner_center_x": inner_center_x,
        },
    )


def four_convex_semicircle_boundary(chord=1.0, arc_angle_deg=90.0):
    """Build four outward circular arcs with a common chord and angle."""
    chord = float(chord)
    angle = float(arc_angle_deg)
    if chord <= 0:
        raise ValueError("chord must be positive")
    if angle < 90.0 or angle >= 360.0:
        raise ValueError("arc_angle_deg must satisfy 90 <= angle < 360")
    theta = np.deg2rad(angle)
    radius = chord / (2.0 * np.sin(theta / 2.0))
    offset = radius * np.cos(theta / 2.0)

    def arc(p0, p1, center, direction):
        a0 = np.arctan2(p0[1] - center[1], p0[0] - center[0])
        return CircleArc(center, radius, a0, a0 + direction * theta)

    c = chord
    bottom = arc((0, 0), (c, 0), (c / 2, offset), +1)
    right = arc((c, 0), (c, c), (c - offset, c / 2), +1)
    top = arc((0, c), (c, c), (c / 2, c - offset), -1)
    left = arc((0, 0), (0, c), (offset, c / 2), -1)
    return BoundaryCurveSet2D(
        bottom,
        right,
        top,
        left,
        metadata={
            "family": "four_convex_arcs",
            "chord": chord,
            "arc_angle_deg": angle,
            "radius": radius,
        },
    )


def VerticallyAsymmetricConvexTriBoundary(
    width=1.0, side_length=1.0, protrusion_distance=0.5
):
    """Vertical sides and outward/inward isosceles triangular top and bottom."""
    if (
        width <= 0
        or side_length <= 0
        or side_length + 2 * min(0.0, protrusion_distance) <= 0
    ):
        raise ValueError("Triangle boundaries must not overlap")
    bottom_apex = (width / 2, -protrusion_distance)
    top_apex = (width / 2, side_length + protrusion_distance)
    return BoundaryCurveSet2D(
        PiecewiseCurve(
            [LineSegment((0, 0), bottom_apex), LineSegment(bottom_apex, (width, 0))]
        ),
        LineSegment((width, 0), (width, side_length)),
        PiecewiseCurve(
            [
                LineSegment((0, side_length), top_apex),
                LineSegment(top_apex, (width, side_length)),
            ]
        ),
        LineSegment((0, 0), (0, side_length)),
        metadata={
            "family": "symmetric_convex_tri",
            "width": width,
            "side_length": side_length,
            "protrusion_distance": protrusion_distance,
        },
    )


def smooth_constricted_channel_boundary(length=3.0, height=1.0, depth=0.4):
    """Two smooth opposing degree-six Bezier bumps with an exact throat gap.

    The bump is 64*t**3*(1-t)**3, with zero first/second endpoint
    derivatives. The minimum channel gap is height - 2*depth.
    """
    if not (length > 0 and height > 0 and 0 <= depth < height / 2):
        raise ValueError("Require positive dimensions and 0 <= depth < height/2")
    x = np.linspace(0, length, 7)
    bump = np.zeros(7)
    bump[3] = 3.2 * depth
    return BoundaryCurveSet2D(
        BezierCurve(np.column_stack([x, bump])),
        LineSegment((length, 0), (length, height)),
        BezierCurve(np.column_stack([x, height - bump])),
        LineSegment((0, 0), (0, height)),
        metadata=dict(
            family="smooth_constricted_channel",
            length=length,
            height=height,
            depth=depth,
        ),
    )


def variable_width_s_channel_boundary(
    length=4.0, height=0.8, bend=2.0, constriction=0.65
):
    """Bezier channel with reversing bend and a positive varying vertical gap.

    Both walls share monotone x controls. Their difference is a Bezier
    polynomial with strictly positive controls, preventing wall crossings.
    bend controls centerline control offsets, not its maximum displacement.
    """
    if not (length > 0 and height > 0 and np.isfinite(bend) and 0 <= constriction < 1):
        raise ValueError("Require positive dimensions and 0 <= constriction < 1")
    x = np.linspace(0, length, 6)
    center = np.array([0, 0, bend, -bend, 0, 0])
    gap = height * np.array([1, 1, 1 - constriction, 1 - constriction, 1, 1])
    lower = center + (height - gap) / 2
    upper = center + (height + gap) / 2
    return BoundaryCurveSet2D(
        BezierCurve(np.column_stack([x, lower])),
        LineSegment((length, 0), (length, height)),
        BezierCurve(np.column_stack([x, upper])),
        LineSegment((0, 0), (0, height)),
        metadata=dict(
            family="variable_width_s_channel",
            length=length,
            height=height,
            bend=bend,
            constriction=constriction,
        ),
    )


def rounded_u_notch_boundary(
    width=3.0, height=2.0, notch_width=0.9, depth=1.65, radius=0.1
):
    """Rectangle with a deep top notch and four exact tangent circular fillets.

    The top is stored left-to-right. PiecewiseCurve allocates [0,1] by
    segment arc length and averages global-parameter tangents at joins.
    """
    margin = (width - notch_width) / 2
    if not (
        width > 0
        and height > 0
        and 0 < depth < height
        and 0 < notch_width < width
        and 0 < radius < min(margin, notch_width / 2, depth / 2)
    ):
        raise ValueError(
            "Notch dimensions must leave positive walls and fillet segments"
        )
    a, b = margin, width - margin
    low = height - depth
    top = PiecewiseCurve(
        [
            LineSegment((0, height), (a - radius, height)),
            CircleArc((a - radius, height - radius), radius, np.pi / 2, 0),
            LineSegment((a, height - radius), (a, low + radius)),
            CircleArc((a + radius, low + radius), radius, np.pi, 1.5 * np.pi),
            LineSegment((a + radius, low), (b - radius, low)),
            CircleArc((b - radius, low + radius), radius, -np.pi / 2, 0),
            LineSegment((b, low + radius), (b, height - radius)),
            CircleArc((b + radius, height - radius), radius, np.pi, np.pi / 2),
            LineSegment((b + radius, height), (width, height)),
        ]
    )
    return BoundaryCurveSet2D(
        LineSegment((0, 0), (width, 0)),
        LineSegment((width, 0), (width, height)),
        top,
        LineSegment((0, 0), (0, height)),
        metadata=dict(
            family="rounded_u_notch",
            width=width,
            height=height,
            notch_width=notch_width,
            depth=depth,
            radius=radius,
        ),
    )


def gear_spline_boundary(
    width=2.0, height=1.6, teeth=8, amplitude=0.16, controls_per_wave=16
):
    """Four C2 cubic spline walls with resolved, tapered wave controls.

    Controls scale with wave count to avoid aliasing. amplitude describes
    control offsets; the spline slightly attenuates the actual displacement.
    Endpoint control offsets vanish so every wall meets its adjacent wall.
    """
    if not (
        width > 0
        and height > 0
        and int(teeth) == teeth
        and teeth >= 1
        and int(controls_per_wave) == controls_per_wave
        and controls_per_wave >= 12
        and 0 <= amplitude < 0.2 * min(width, height)
    ):
        raise ValueError(
            "Require positive dimensions, integer teeth >= 1, controls_per_wave >= 12 and amplitude < 0.2*min(width,height)"
        )
    u = np.linspace(0, 1, int(teeth) * int(controls_per_wave) + 1)
    offset = amplitude * np.sin(2 * np.pi * teeth * u) * np.sin(np.pi * u) ** 2
    offset[[0, -1]] = 0
    return BoundaryCurveSet2D(
        BSplineCurve(np.column_stack([width * u, -offset]), degree=3),
        BSplineCurve(np.column_stack([width + offset, height * u]), degree=3),
        BSplineCurve(np.column_stack([width * u, height + offset]), degree=3),
        BSplineCurve(np.column_stack([-offset, height * u]), degree=3),
        metadata=dict(
            family="gear_spline",
            width=width,
            height=height,
            teeth=int(teeth),
            amplitude=amplitude,
            controls_per_wave=int(controls_per_wave),
        ),
    )


BOUNDARY_GENERATORS = {
    "rectangle": rectangular_boundary,
    "circle_airfoil": circle_airfoil_boundary,
    "rect_top_semiellipse": GenerateRectangleWithTopSemiEllipseBoundary,
    "eccentric_semiannulus": EccentricAnnularSectorBoundary,
    "four_convex_semicircle": four_convex_semicircle_boundary,
    "symmetric_convex_tri": VerticallyAsymmetricConvexTriBoundary,
    "gear_spline": gear_spline_boundary,
    "smooth_constricted_channel": smooth_constricted_channel_boundary,
    "variable_width_s_channel": variable_width_s_channel_boundary,
    "rounded_u_notch": rounded_u_notch_boundary,
}


def generate_boundary(boundary_type, **kwargs):
    if boundary_type not in BOUNDARY_GENERATORS:
        raise ValueError(f"unsupported 2D boundary: {boundary_type}")
    return BOUNDARY_GENERATORS[boundary_type](**kwargs)
