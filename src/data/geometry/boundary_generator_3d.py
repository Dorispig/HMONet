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
)


def box_boundary(xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1):
    return BoundarySurfaceSet3D(
        BilinearSurface3D(
            [xmin, ymin, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmin, ymax, zmax],
        ),
        BilinearSurface3D(
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
        ),
        BilinearSurface3D(
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
        ),
        BilinearSurface3D(
            [xmin, ymax, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmax],
            [xmax, ymax, zmax],
        ),
        BilinearSurface3D(
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmin, ymax, zmin],
            [xmax, ymax, zmin],
        ),
        BilinearSurface3D(
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmin, ymax, zmax],
            [xmax, ymax, zmax],
        ),
        metadata={"family": "box"},
    )



def _legacy_airfoil_contour(camber, camber_position, thickness, chord):
    """Return the legacy trailing-upper-leading-lower-trailing contour."""
    base = NACA4Curve(
        m=camber,
        p=camber_position,
        t=thickness,
        c=chord,
        center=(-0.5 * chord, 0.0),
    )
    return PiecewiseCurve(
        [MappedCurve(base, 0.5, 0.0), MappedCurve(base, 1.0, 0.5)],
        breakpoints=[0.0, 0.5, 1.0],
        closed=True,
        allocation="explicit",
    )


def _o_grid_extrusion(inner_curve, outer_curve, length, z_min, metadata):
    """Build six exact faces for an O-grid extrusion.

    Computational coordinates are radial ``xi``, circumferential ``eta`` and
    axial ``zeta``.  The two eta faces are the same physical seam.
    """
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


def cylindrical_domain_with_airfoil_hole(
    outer_radius=1.5,
    chord=1.0,
    camber=0.02,
    camber_position=0.4,
    thickness=0.12,
    length=2.5,
    z_min=0.0,
):
    """Extrude a circular casing around a closed NACA four-digit hole.

    This is the exact parametric counterpart of the legacy generator. ``xi``
    runs from the airfoil to the outer circle, ``eta`` follows both closed
    contours, and ``zeta`` follows the extrusion axis.
    """
    if outer_radius <= 0 or length <= 0:
        raise ValueError("outer_radius and length must be positive")
    if outer_radius <= 0.75 * chord:
        raise ValueError("outer_radius is too small for the requested airfoil")
    if not 0 <= camber < 0.2:
        raise ValueError("camber must be in [0, 0.2)")
    if not 0 < camber_position < 1:
        raise ValueError("camber_position must be in (0, 1)")
    if not 0 < thickness < 0.4 or chord <= 0:
        raise ValueError("thickness must be in (0, 0.4) and chord positive")
    inner = _legacy_airfoil_contour(camber, camber_position, thickness, chord)
    outer = CircleArc((0.0, 0.0), outer_radius, 0.0, 2.0 * np.pi)
    return _o_grid_extrusion(
        inner,
        outer,
        length,
        z_min,
        metadata={
            "family": "cylindrical_domain_with_airfoil_hole",
            "topology": "o_grid_extrusion",
            "outer_radius": float(outer_radius),
            "chord": float(chord),
            "camber": float(camber),
            "camber_position": float(camber_position),
            "thickness": float(thickness),
            "length": float(length),
            "z_min": float(z_min),
        },
    )


def stadium_cylinder_with_circular_hole(
    straight_half_length=0.8,
    stadium_radius=0.7,
    hole_radius=0.25,
    length=2.5,
    z_min=0.0,
):
    """Extrude a stadium casing around a concentric circular hole.

    The inner circle and outer stadium share normalized polar angle, preserving
    the point correspondence and coincident O-grid seam of the legacy domain.
    """
    if straight_half_length < 0 or stadium_radius <= 0 or hole_radius <= 0:
        raise ValueError("stadium dimensions must be non-negative/positive")
    if hole_radius >= stadium_radius or length <= 0:
        raise ValueError(
            "hole_radius must be below stadium_radius and length must be positive"
        )
    inner = CircleArc((0.0, 0.0), hole_radius, 0.0, 2.0 * np.pi)
    outer = StadiumCurve(straight_half_length, stadium_radius)
    return _o_grid_extrusion(
        inner,
        outer,
        length,
        z_min,
        metadata={
            "family": "stadium_cylinder_with_circular_hole",
            "topology": "o_grid_extrusion",
            "straight_half_length": float(straight_half_length),
            "stadium_radius": float(stadium_radius),
            "hole_radius": float(hole_radius),
            "length": float(length),
            "z_min": float(z_min),
        },
    )


BOUNDARY_GENERATORS = {
    "box": box_boundary,
    "cylindrical_domain_with_airfoil_hole": cylindrical_domain_with_airfoil_hole,
    "stadium_cylinder_with_circular_hole": stadium_cylinder_with_circular_hole,
}


def generate_boundary(boundary_type, **kwargs):
    if boundary_type not in BOUNDARY_GENERATORS:
        raise ValueError(f"unsupported 3D boundary: {boundary_type}")
    return BOUNDARY_GENERATORS[boundary_type](**kwargs)


