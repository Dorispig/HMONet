"""Render unused 2-D generators and proposed industrial 3-D test geometries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.geometry.boundary_generator_2d import BOUNDARY_GENERATORS


def _draw_boundary(ax, boundary, color):
    for curve in boundary.curves().values():
        points = curve.sample(500)
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")


def render_unused_2d(output: Path):
    cases = (
        (
            "gear_spline",
            "Gear spline",
            ({"amplitude": 0.02, "teeth": 6}, {"amplitude": 0.18, "teeth": 6}),
            ("A=0.02", "A=0.18"),
        ),
        (
            "four_convex_semicircle",
            "Four convex arcs",
            ({"arc_angle_deg": 90.0}, {"arc_angle_deg": 150.0}),
            ("angle=90 deg", "angle=150 deg"),
        ),
        (
            "symmetric_convex_tri",
            "Sharp triangular tips",
            ({"protrusion_distance": 0.05}, {"protrusion_distance": 0.45}),
            ("tip=0.05", "tip=0.45"),
        ),
        (
            "rectangle",
            "Rectangle reference",
            ({}, {"x_min": -1.4, "x_max": 1.4, "y_min": -0.7, "y_max": 0.7}),
            ("square", "rectangle"),
        ),
    )
    colors = ("#2f6fbe", "#cb6b2f")
    fig, axes = plt.subplots(2, len(cases), figsize=(13.5, 6.2), constrained_layout=True)
    for column, (name, title, parameter_sets, labels) in enumerate(cases):
        for row in range(2):
            boundary = BOUNDARY_GENERATORS[name](**parameter_sets[row])
            _draw_boundary(axes[row, column], boundary, colors[row])
            axes[row, column].set_title(f"{title}\n{labels[row]}", fontsize=10)
    fig.suptitle(
        "Registered 2-D generators not used by the six default datasets",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _style_3d(ax, title):
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_axis_off()
    ax.view_init(elev=24, azim=-56)
    try:
        ax.set_box_aspect((1.25, 1.0, 1.0))
    except AttributeError:
        pass


def _plot_rings(ax, rings, color="#315f9b"):
    for ring in rings[::2]:
        ax.plot(*ring.T, color=color, linewidth=0.6, alpha=0.85)
    count = len(rings[0])
    for index in range(0, count, max(1, count // 12)):
        line = np.asarray([ring[index] for ring in rings])
        ax.plot(*line.T, color=color, linewidth=0.5, alpha=0.7)


def _rectangle_ring(theta, half_width, half_height):
    cosine, sine = np.cos(theta), np.sin(theta)
    scale = 1.0 / np.maximum(
        np.abs(cosine) / half_width,
        np.abs(sine) / half_height,
    )
    return scale * cosine, scale * sine


def _square_to_round(ax):
    theta = np.linspace(0, 2 * np.pi, 97)
    rings = []
    for zeta in np.linspace(0, 1, 29):
        square_x, square_y = _rectangle_ring(theta, 1.0, 0.75)
        circle_x, circle_y = 0.72 * np.cos(theta), 0.72 * np.sin(theta)
        blend = 3 * zeta**2 - 2 * zeta**3
        x = (1 - blend) * square_x + blend * circle_x
        y = (1 - blend) * square_y + blend * circle_y
        rings.append(np.column_stack((x, y, 3.0 * zeta * np.ones_like(theta))))
    _plot_rings(ax, rings)


def _s_duct(ax):
    theta = np.linspace(0, 2 * np.pi, 73)
    rings = []
    for zeta in np.linspace(0, 1, 33):
        center_x = 0.75 * np.sin(2 * np.pi * zeta)
        center_y = 0.18 * np.sin(np.pi * zeta)
        a = 0.55 - 0.13 * zeta
        b = 0.40 + 0.08 * zeta
        x = center_x + a * np.cos(theta)
        y = center_y + b * np.sin(theta)
        rings.append(np.column_stack((x, y, 3.2 * zeta * np.ones_like(theta))))
    _plot_rings(ax, rings, "#2f7d6d")


def _blade_passage(ax):
    chord = np.linspace(0, 1, 41)
    span = np.linspace(0, 1, 17)
    s, z = np.meshgrid(chord, span)
    camber = 0.24 * np.sin(np.pi * s)
    twist = 0.28 * (z - 0.5) * (2 * s - 1)
    thickness = 0.10 * np.sin(np.pi * s) + 0.025
    for sign, color in ((-1, "#a44a3f"), (1, "#315f9b")):
        x = 2.1 * s + 0.12 * (z - 0.5)
        y = camber + twist + sign * (0.42 + thickness)
        ax.plot_wireframe(
            x,
            y,
            1.4 * z,
            rstride=2,
            cstride=4,
            color=color,
            linewidth=0.45,
            alpha=0.85,
        )
    for level in (0.0, 1.4):
        ax.plot(2.1 * chord, camber[0] - 0.55, level, color="#64748b", lw=0.8)
        ax.plot(2.1 * chord, camber[0] + 0.55, level, color="#64748b", lw=0.8)


def _serpentine_channel(ax):
    theta = np.linspace(0, 2 * np.pi, 57)
    rings = []
    for t in np.linspace(0, 1, 45):
        center_x = 2.8 * t
        center_y = 0.75 * np.sin(2 * np.pi * t)
        center_z = 0.42 * np.sin(4 * np.pi * t)
        radius = 0.23 * (1.0 - 0.15 * np.cos(2 * np.pi * t))
        x = center_x * np.ones_like(theta)
        y = center_y + radius * np.cos(theta)
        z = center_z + radius * np.sin(theta)
        rings.append(np.column_stack((x, y, z)))
    _plot_rings(ax, rings, "#7c4d9e")


def _localized_dimple(ax):
    x = np.linspace(0, 2.5, 31)
    y = np.linspace(0, 1.7, 25)
    xx, yy = np.meshgrid(x, y)
    top = 1.2 - 0.62 * np.exp(-((xx - 1.25) ** 2 / 0.24 + (yy - 0.85) ** 2 / 0.16))
    ax.plot_wireframe(xx, yy, top, rstride=3, cstride=3, color="#b06a2b", linewidth=0.45)
    for edge_x in (0, -1):
        ax.plot(x if edge_x == 0 else x, np.full_like(x, y[edge_x]), top[edge_x], color="#64748b", lw=0.7)
    for column in (0, -1):
        ax.plot(np.full_like(y, x[column]), y, top[:, column], color="#64748b", lw=0.7)


def _tube(ax, start, end, radius, color):
    start, end = np.asarray(start, float), np.asarray(end, float)
    tangent = end - start
    tangent /= np.linalg.norm(tangent)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(seed, tangent)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    first = np.cross(tangent, seed)
    first /= np.linalg.norm(first)
    second = np.cross(tangent, first)
    theta = np.linspace(0, 2 * np.pi, 49)
    rings = []
    for t in np.linspace(0, 1, 18):
        center = (1 - t) * start + t * end
        rings.append(
            center
            + radius
            * (np.cos(theta)[:, None] * first + np.sin(theta)[:, None] * second)
        )
    _plot_rings(ax, rings, color)


def _y_manifold(ax):
    _tube(ax, (0, 0, 0), (0, 0, 1.5), 0.35, "#2f6fbe")
    _tube(ax, (0, 0, 1.35), (-0.9, 0, 2.5), 0.27, "#2f6fbe")
    _tube(ax, (0, 0, 1.35), (0.9, 0, 2.5), 0.27, "#2f6fbe")


def render_industrial_3d(output: Path):
    examples = (
        (_square_to_round, "Square-to-round transition"),
        (_s_duct, "Variable-area S-duct"),
        (_blade_passage, "Twisted blade passage"),
        (_serpentine_channel, "Serpentine cooling channel"),
        (_localized_dimple, "Localized dimple / cavity"),
        (_y_manifold, "Y-manifold (multi-block)"),
    )
    fig = plt.figure(figsize=(15.5, 8.4), constrained_layout=True)
    for index, (drawer, title) in enumerate(examples, start=1):
        ax = fig.add_subplot(2, 3, index, projection="3d")
        drawer(ax)
        _style_3d(ax, title)
    fig.suptitle(
        "Recommended industrial 3-D boundary families (schematic parameter surfaces)",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    render_unused_2d(args.output_dir / "unused_2d_boundary_examples.png")
    render_industrial_3d(args.output_dir / "recommended_industrial_3d_boundaries.png")


if __name__ == "__main__":
    main()
