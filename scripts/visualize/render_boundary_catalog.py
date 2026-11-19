"""Render the twelve canonical HMONet benchmark boundaries.

The script reads the active 2-D and 3-D dataset registries rather than
duplicating geometric constants.  Each panel is the lower or upper endpoint
of the *extrapolation* interval, so it is useful when reviewing a proposed
change to a family definition.

Run from the repository root:
    python scripts/visualize/render_boundary_catalog.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Direct ``python scripts/...`` execution places the script directory, rather
# than the repository root, at the beginning of ``sys.path``.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.generation.make_dataset_parametric_2d import (
    DATASET_FAMILIES as FAMILIES_2D,
    DATASET_GENERATORS as GENERATORS_2D,
)
from src.data.generation.make_dataset_parametric_3d import (
    DATASET_FAMILIES as FAMILIES_3D,
)
from src.data.geometry.boundary_generator_3d import generate_boundary as generate_boundary_3d


# These are the six 2-D families retained in the paper benchmark.  The active
# registry also contains optional exploratory families, intentionally omitted
# here so the figure stays aligned with the paper's 2-D-1 ... 2-D-6 notation.
CANONICAL_2D = (
    ("circle_airfoil_t", "2D-1  Airfoil hole"),
    ("eccentric_semiannulus", "2D-2  Semi-annulus"),
    ("rect_top_semiellipse", "2D-3  Semiellipse"),
    ("rounded_u_notch_compact", "2D-4  Deep U-notch"),
    ("smooth_constricted_channel", "2D-5  Constricted channel"),
    ("variable_width_s_channel", "2D-6  S/wavy channel"),
)

CANONICAL_3D = (
    ("cylindrical_domain_with_airfoil_hole", "3D-1  Cylinder + airfoil hole"),
    ("sidewall_deformed_block", "3D-2  Sidewall-deformed block"),
    ("rounded_u_notch_extrusion", "3D-3  Rounded U-notch extrusion"),
    ("tapered_elliptic_bore_block", "3D-4  Tapered elliptic-bore block"),
    ("bent_annular_duct", "3D-5  Bent annular duct"),
    ("lobed_annular_duct", "3D-6  Lobed annular duct"),
)


def extrapolation_endpoints(specification: dict) -> tuple[float, float]:
    """Return extrema across all extrapolation intervals in a family."""
    ranges = np.asarray(specification["extrapolation"], dtype=float)
    return float(ranges[:, 0].min()), float(ranges[:, 1].max())


def geometry_parameters(specification: dict, value: float) -> dict:
    """Create one exact geometry-parameter dictionary for a family endpoint."""
    return {
        **specification["fixed_parameters"],
        specification["parameter"]: float(value),
    }


def _format_value(value: float) -> str:
    return f"{value:g}"


def _plot_2d_family(axis, family: str, title: str, value: float, endpoint: str) -> None:
    specification = FAMILIES_2D[family]
    boundary = GENERATORS_2D[specification["function"]](
        **geometry_parameters(specification, value)
    )
    counts = {name: 401 for name in boundary.names}
    polygon = boundary.sample_closed(counts)
    axis.fill(polygon[:, 0], polygon[:, 1], color="#DCEAF7", alpha=0.72, zorder=0)
    axis.plot(polygon[:, 0], polygon[:, 1], color="#174F7C", linewidth=1.55, zorder=2)
    axis.scatter(polygon[0, 0], polygon[0, 1], color="#C83E4D", s=10, zorder=3)
    axis.set_aspect("equal", adjustable="box")
    axis.margins(0.10)
    axis.grid(True, linewidth=0.35, alpha=0.35)
    axis.tick_params(labelsize=7)
    axis.set_title(
        f"{title}\n{endpoint}: {specification['parameter']} = {_format_value(value)}",
        fontsize=10,
        fontweight="semibold",
        pad=7,
    )


def render_2d(output_path: Path) -> None:
    """Render min/max extrapolation boundaries for the six 2-D paper cases."""
    figure, axes = plt.subplots(len(CANONICAL_2D), 2, figsize=(12, 29), dpi=220)
    for row, (family, title) in enumerate(CANONICAL_2D):
        specification = FAMILIES_2D[family]
        lower, upper = extrapolation_endpoints(specification)
        _plot_2d_family(axes[row, 0], family, title, lower, "Extrapolation minimum")
        _plot_2d_family(axes[row, 1], family, title, upper, "Extrapolation maximum")
    figure.suptitle(
        "Canonical 2-D HMONet boundaries: extrapolation endpoints",
        fontsize=16,
        fontweight="bold",
        y=0.998,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.992), h_pad=2.0)
    figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_3d_surface(axis, points: np.ndarray, color: str) -> None:
    axis.plot_surface(
        points[..., 0],
        points[..., 1],
        points[..., 2],
        rstride=2,
        cstride=2,
        linewidth=0.20,
        edgecolor=color,
        color=color,
        alpha=0.14,
        shade=False,
        antialiased=True,
    )


def _set_3d_bounds(axis, point_sets: list[np.ndarray]) -> None:
    points = np.concatenate([array.reshape(-1, 3) for array in point_sets], axis=0)
    lower, upper = points.min(axis=0), points.max(axis=0)
    midpoint = 0.5 * (lower + upper)
    span = np.maximum(upper - lower, 1e-8)
    radius = 0.55 * float(span.max())
    axis.set_xlim(midpoint[0] - radius, midpoint[0] + radius)
    axis.set_ylim(midpoint[1] - radius, midpoint[1] + radius)
    axis.set_zlim(midpoint[2] - radius, midpoint[2] + radius)
    axis.set_box_aspect(tuple(span / span.max()))
    axis.set_axis_off()
    axis.set_proj_type("ortho")


def _plot_3d_family(axis, family: str, title: str, value: float, endpoint: str) -> None:
    specification = FAMILIES_3D[family]
    boundary = generate_boundary_3d(
        specification["function"], **geometry_parameters(specification, value)
    )
    colors = ("#236B8E", "#236B8E", "#C96F2D", "#C96F2D", "#627D5A", "#627D5A")
    point_sets = []
    for color, surface in zip(colors, boundary.surfaces().values()):
        points = surface.sample(33, 33)
        point_sets.append(points)
        _plot_3d_surface(axis, points, color)
    _set_3d_bounds(axis, point_sets)
    axis.view_init(elev=22, azim=-58)
    axis.set_title(
        f"{title}\n{endpoint}: {specification['parameter']} = {_format_value(value)}",
        fontsize=10,
        fontweight="semibold",
        pad=4,
    )


def render_3d(output_path: Path) -> None:
    """Render min/max extrapolation boundaries for the six 3-D paper cases."""
    figure = plt.figure(figsize=(14, 32), dpi=220)
    for row, (family, title) in enumerate(CANONICAL_3D):
        specification = FAMILIES_3D[family]
        lower, upper = extrapolation_endpoints(specification)
        left = figure.add_subplot(len(CANONICAL_3D), 2, 2 * row + 1, projection="3d")
        right = figure.add_subplot(len(CANONICAL_3D), 2, 2 * row + 2, projection="3d")
        _plot_3d_family(left, family, title, lower, "Extrapolation minimum")
        _plot_3d_family(right, family, title, upper, "Extrapolation maximum")
    figure.suptitle(
        "Canonical 3-D HMONet boundaries: extrapolation endpoints",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.99), h_pad=2.5, w_pad=0.5)
    figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/boundary_catalog"),
        help="Directory for the two rendered PNG catalogues.",
    )
    arguments = parser.parse_args()
    output_dir = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_2d = output_dir / "canonical_2d_extrapolation_endpoints.png"
    output_3d = output_dir / "canonical_3d_extrapolation_endpoints.png"
    render_2d(output_2d)
    render_3d(output_3d)
    print(f"Wrote {output_2d}")
    print(f"Wrote {output_3d}")


if __name__ == "__main__":
    main()
