"""Analytic 3-D boundary-surface generators."""

import numpy as np

from src.data.geometry.parametric_curve_2d import (
    CircleArc,
    LineSegment,
    MappedCurve,
    NACA4Curve,
    PiecewiseCurve,
    StadiumCurve,
)
from src.data.geometry.parametric_geometry_3d import (
    BilinearSurface3D,
    BoundarySurfaceSet3D,
    CurveBlendSurface3D,
    ExtrudedCurveSurface3D,
    MappedVolumeFace3D,
)


def _legacy_airfoil_contour(camber, camber_position, thickness, chord):
    """Return the legacy trailing-upper-leading-lower-trailing contour."""
    base = NACA4Curve(m=camber, p=camber_position, t=thickness, c=chord, center=(-0.5 * chord, 0.0))
    return PiecewiseCurve(
        [MappedCurve(base, 0.5, 0.0), MappedCurve(base, 1.0, 0.5)],
        breakpoints=[0.0, 0.5, 1.0],
        closed=True,
        allocation="explicit",
    )


def _o_grid_extrusion(inner_curve, outer_curve, length, z_min, metadata):
    """Build six exact faces for a straight O-grid extrusion."""
    z0, z1 = float(z_min), float(z_min + length)
    direction = np.array([0.0, 0.0, float(length)], dtype=float)
    seam = LineSegment(inner_curve.evaluate(0.0), outer_curve.evaluate(0.0))
    return BoundarySurfaceSet3D(
        ExtrudedCurveSurface3D(inner_curve, direction, origin=(0, 0, z0)),
        ExtrudedCurveSurface3D(outer_curve, direction, origin=(0, 0, z0)),
        ExtrudedCurveSurface3D(seam, direction, origin=(0, 0, z0)),
        ExtrudedCurveSurface3D(seam, direction, origin=(0, 0, z0)),
        CurveBlendSurface3D(inner_curve, outer_curve, z0),
        CurveBlendSurface3D(inner_curve, outer_curve, z1),
        metadata=metadata,
    )


def _mapped_boundary(mapping, parameters, metadata):
    """Restrict one analytic unit-cube map to six exactly compatible faces."""
    faces = [
        MappedVolumeFace3D(mapping, axis, side, parameters)
        for axis in range(3)
        for side in (0, 1)
    ]
    return BoundarySurfaceSet3D(*faces, metadata=metadata)


def cylindrical_domain_with_airfoil_hole(
    outer_radius=1.5, chord=1.0, camber=0.02, camber_position=0.4,
    thickness=0.12, length=2.5, z_min=0.0,
):
    """Extrude a circular casing around a closed NACA four-digit hole."""
    if outer_radius <= 0 or length <= 0 or outer_radius <= 0.75 * chord:
        raise ValueError("outer radius/length must be positive and leave a safe airfoil gap")
    if not 0 <= camber < 0.2 or not 0 < camber_position < 1:
        raise ValueError("camber must be in [0, 0.2) and its position in (0, 1)")
    if not 0 < thickness < 0.4 or chord <= 0:
        raise ValueError("thickness must be in (0, 0.4) and chord positive")
    inner = _legacy_airfoil_contour(camber, camber_position, thickness, chord)
    outer = CircleArc((0.0, 0.0), outer_radius, 0.0, 2.0 * np.pi)
    return _o_grid_extrusion(inner, outer, length, z_min, {
        "family": "cylindrical_domain_with_airfoil_hole", "topology": "o_grid_extrusion",
        "outer_radius": float(outer_radius), "chord": float(chord), "camber": float(camber),
        "camber_position": float(camber_position), "thickness": float(thickness),
        "length": float(length), "z_min": float(z_min),
    })


def stadium_cylinder_with_circular_hole(
    straight_half_length=0.8, stadium_radius=0.7, hole_radius=0.25,
    length=2.5, z_min=0.0,
):
    """Extrude a stadium casing around a concentric circular hole."""
    if straight_half_length < 0 or stadium_radius <= 0 or hole_radius <= 0:
        raise ValueError("stadium dimensions must be non-negative/positive")
    if hole_radius >= stadium_radius or length <= 0:
        raise ValueError("hole_radius must be below stadium_radius and length positive")
    inner = CircleArc((0.0, 0.0), hole_radius, 0.0, 2.0 * np.pi)
    outer = StadiumCurve(straight_half_length, stadium_radius)
    return _o_grid_extrusion(inner, outer, length, z_min, {
        "family": "stadium_cylinder_with_circular_hole", "topology": "o_grid_extrusion",
        "straight_half_length": float(straight_half_length), "stadium_radius": float(stadium_radius),
        "hole_radius": float(hole_radius), "length": float(length), "z_min": float(z_min),
    })


def sidewall_deformed_block_boundary(
    length_x=3.0, width_y=2.0, length_z=2.5,
    sidewall_amplitude=0.15, z_min=0.0,
):
    """Box with two opposing concave sidewalls extruded along the z axis."""
    if min(length_x, width_y, length_z) <= 0:
        raise ValueError("block dimensions must be positive")
    if not 0.0 <= sidewall_amplitude <= 0.35:
        raise ValueError("sidewall_amplitude must lie in [0, 0.35]")
    parameters = {
        "length_x": float(length_x), "width_y": float(width_y),
        "length_z": float(length_z), "sidewall_amplitude": float(sidewall_amplitude),
        "z_min": float(z_min),
    }
    return _mapped_boundary("sidewall_deformed_block", parameters, {
        "family": "sidewall_deformed_block", "topology": "symmetric_constricted_block", **parameters,
    })


def rounded_u_groove_extrusion_boundary(
    width=3.0, height=2.0, groove_width=0.9, groove_depth=0.8,
    corner_smoothing=0.08, length=2.5, z_min=0.0,
):
    """Legacy smooth U-groove block retained for reproducibility (``ugc_d``)."""
    if min(width, height, groove_width, groove_depth, corner_smoothing, length) <= 0:
        raise ValueError("U-groove dimensions must be positive")
    if not (groove_depth < height and groove_width < width):
        raise ValueError("groove width/depth must remain below the outer block dimensions")
    parameters = {
        "width": float(width), "height": float(height), "groove_width": float(groove_width),
        "groove_depth": float(groove_depth), "corner_smoothing": float(corner_smoothing),
        "length": float(length), "z_min": float(z_min),
    }
    return _mapped_boundary("rounded_u_groove_extrusion", parameters, {
        "family": "rounded_u_groove_extrusion", "topology": "smooth_u_groove_extrusion", **parameters,
    })


def rounded_u_notch_extrusion_boundary(
    width=3.0, height=2.0, notch_width=0.9, depth=1.0,
    radius=0.25, length=2.5, z_min=0.0,
):
    """Extrude the exact 2-D ``rounded_u_notch`` (``run``) along ``z``."""
    margin = 0.5 * (width - notch_width)
    if min(width, height, notch_width, depth, radius, length) <= 0:
        raise ValueError("rounded-U-notch dimensions must be positive")
    if not (depth < height and notch_width < width):
        raise ValueError("notch width/depth must remain below the outer block dimensions")
    if not radius < min(margin, 0.5 * notch_width, 0.5 * depth):
        raise ValueError("radius must leave positive straight walls and fillet segments")
    parameters = {
        "width": float(width), "height": float(height), "notch_width": float(notch_width),
        "depth": float(depth), "radius": float(radius), "length": float(length), "z_min": float(z_min),
    }
    return _mapped_boundary("rounded_u_notch_extrusion", parameters, {
        "family": "rounded_u_notch_extrusion", "topology": "exact_rounded_u_notch_extrusion", **parameters,
    })


def tapered_elliptic_bore_block_boundary(
    inner_major_axis=0.32, inner_minor_axis_in=0.18, minor_axis_out=0.24,
    outer_half_width=1.5, outer_half_height=1.0, length=2.5, z_min=0.0,
):
    """Rectangular casing containing a z-tapered elliptical through-bore."""
    if min(inner_major_axis, inner_minor_axis_in, minor_axis_out, outer_half_width, outer_half_height, length) <= 0:
        raise ValueError("elliptic-bore dimensions must be positive")
    if inner_major_axis >= outer_half_width or max(inner_minor_axis_in, minor_axis_out) >= outer_half_height:
        raise ValueError("elliptic bore must remain strictly inside the rectangular casing")
    parameters = {
        "inner_major_axis": float(inner_major_axis),
        "inner_minor_axis_in": float(inner_minor_axis_in),
        "minor_axis_out": float(minor_axis_out),
        "outer_half_width": float(outer_half_width),
        "outer_half_height": float(outer_half_height),
        "length": float(length), "z_min": float(z_min),
    }
    return _mapped_boundary("tapered_elliptic_bore_block", parameters, {
        "family": "tapered_elliptic_bore_block", "topology": "rectangular_o_grid", **parameters,
    })


def annular_nozzle_diffuser_boundary(
    inner_radius_in=0.30, inner_radius_out=0.35,
    outer_radius_in=1.00, outer_radius_out=1.30,
    length=2.5, z_min=0.0,
):
    """Coaxial annular nozzle/diffuser with linearly varying wall radii."""
    radii = (inner_radius_in, inner_radius_out, outer_radius_in, outer_radius_out)
    if min(radii) <= 0 or length <= 0:
        raise ValueError("radii and length must be positive")
    if inner_radius_in >= outer_radius_in or inner_radius_out >= outer_radius_out:
        raise ValueError("inner radius must stay strictly below outer radius")
    parameters = {name: float(value) for name, value in zip(
        ("inner_radius_in", "inner_radius_out", "outer_radius_in", "outer_radius_out", "length", "z_min"),
        (*radii, length, z_min),
    )}
    return _mapped_boundary("annular_nozzle_diffuser", parameters, {
        "family": "annular_nozzle_diffuser", "topology": "analytic_o_grid", **parameters,
    })


def twisted_elliptic_annulus_boundary(
    inner_a=0.35, inner_b=0.22, outer_a=1.10, outer_b=0.75,
    twist_angle_deg=40.0, length=2.5, z_min=0.0,
):
    """Homothetic elliptic annulus whose two walls rotate together."""
    if min(inner_a, inner_b, outer_a, outer_b, length) <= 0:
        raise ValueError("ellipse axes and length must be positive")
    if inner_a >= outer_a or inner_b >= outer_b or abs(twist_angle_deg) >= 120:
        raise ValueError("ellipses must be nested and |twist_angle_deg| below 120")
    parameters = {name: float(value) for name, value in zip(
        ("inner_a", "inner_b", "outer_a", "outer_b", "twist_angle_deg", "length", "z_min"),
        (inner_a, inner_b, outer_a, outer_b, twist_angle_deg, length, z_min),
    )}
    return _mapped_boundary("twisted_elliptic_annulus", parameters, {
        "family": "twisted_elliptic_annulus", "topology": "analytic_o_grid", **parameters,
    })


def bent_annular_duct_boundary(
    inner_radius=0.25, outer_radius=0.60, bend_radius=3.0,
    bend_angle_deg=60.0, z_min=0.0,
):
    """Annular duct swept along a planar circular centreline."""
    if inner_radius <= 0 or inner_radius >= outer_radius:
        raise ValueError("require 0 < inner_radius < outer_radius")
    if bend_radius <= outer_radius or not 0 < bend_angle_deg < 150:
        raise ValueError("bend_radius must exceed outer_radius and angle lie in (0, 150)")
    if outer_radius / bend_radius > 0.35:
        raise ValueError("outer_radius/bend_radius must not exceed 0.35")
    parameters = {name: float(value) for name, value in zip(
        ("inner_radius", "outer_radius", "bend_radius", "bend_angle_deg", "z_min"),
        (inner_radius, outer_radius, bend_radius, bend_angle_deg, z_min),
    )}
    return _mapped_boundary("bent_annular_duct", parameters, {
        "family": "bent_annular_duct", "topology": "analytic_o_grid", **parameters,
    })


def lobed_annular_duct_boundary(
    inner_radius=0.35, outer_radius=1.0, amplitude=0.10, lobes=5,
    phase_twist_deg=45.0, length=2.5, z_min=0.0,
):
    """Star-shaped lobed annulus with a gradual axial phase rotation."""
    if inner_radius <= 0 or outer_radius <= 0 or length <= 0:
        raise ValueError("radii and length must be positive")
    if int(lobes) != lobes or lobes < 2 or not 0 <= amplitude <= 0.22:
        raise ValueError("lobes must be an integer >= 2 and amplitude in [0, 0.22]")
    if inner_radius >= outer_radius * (1.0 - amplitude):
        raise ValueError("inner wall must remain strictly inside every outer-wall lobe")
    parameters = {name: float(value) for name, value in zip(
        ("inner_radius", "outer_radius", "amplitude", "lobes", "phase_twist_deg", "length", "z_min"),
        (inner_radius, outer_radius, amplitude, lobes, phase_twist_deg, length, z_min),
    )}
    return _mapped_boundary("lobed_annular_duct", parameters, {
        "family": "lobed_annular_duct", "topology": "analytic_o_grid", **parameters,
    })


def box_boundary(xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1):
    """Axis-aligned six-face verification box."""
    if not (xmin < xmax and ymin < ymax and zmin < zmax):
        raise ValueError("box minimum coordinates must be below maximum coordinates")
    return BoundarySurfaceSet3D(
        BilinearSurface3D([xmin, ymin, zmin], [xmin, ymax, zmin], [xmin, ymin, zmax], [xmin, ymax, zmax]),
        BilinearSurface3D([xmax, ymin, zmin], [xmax, ymax, zmin], [xmax, ymin, zmax], [xmax, ymax, zmax]),
        BilinearSurface3D([xmin, ymin, zmin], [xmax, ymin, zmin], [xmin, ymin, zmax], [xmax, ymin, zmax]),
        BilinearSurface3D([xmin, ymax, zmin], [xmax, ymax, zmin], [xmin, ymax, zmax], [xmax, ymax, zmax]),
        BilinearSurface3D([xmin, ymin, zmin], [xmax, ymin, zmin], [xmin, ymax, zmin], [xmax, ymax, zmin]),
        BilinearSurface3D([xmin, ymin, zmax], [xmax, ymin, zmax], [xmin, ymax, zmax], [xmax, ymax, zmax]),
        metadata={"family": "box"},
    )


BOUNDARY_GENERATORS = {
    "cylindrical_domain_with_airfoil_hole": cylindrical_domain_with_airfoil_hole,
    "stadium_cylinder_with_circular_hole": stadium_cylinder_with_circular_hole,
    "sidewall_deformed_block": sidewall_deformed_block_boundary,
    "rounded_u_groove_extrusion": rounded_u_groove_extrusion_boundary,
    "rounded_u_notch_extrusion": rounded_u_notch_extrusion_boundary,
    "tapered_elliptic_bore_block": tapered_elliptic_bore_block_boundary,
    "annular_nozzle_diffuser": annular_nozzle_diffuser_boundary,
    "twisted_elliptic_annulus": twisted_elliptic_annulus_boundary,
    "bent_annular_duct": bent_annular_duct_boundary,
    "lobed_annular_duct": lobed_annular_duct_boundary,
    "box": box_boundary,
}


def generate_boundary(boundary_type, **kwargs):
    if boundary_type not in BOUNDARY_GENERATORS:
        raise ValueError(f"unsupported 3D boundary: {boundary_type}")
    return BOUNDARY_GENERATORS[boundary_type](**kwargs)
