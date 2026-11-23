"""Render three 3x3 atlases of the current 2-D/3-D boundary families.

Geometry only: no learned meshes or volume TFI/PDE solutions are plotted.
z_mid means the logical mid-span section zeta=0.5, shown in its own plane.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.visualize.render_boundary_family_catalog import (
    PAPER_2D, PAPER_3D, FAMILIES_2D, FAMILIES_3D,
    boundary_2d, boundary_3d, endpoints,
)

LABELS_2D = ["Airfoil in a circular domain", "Eccentric semi-annulus",
             "Rectangle with semi-elliptic top", "Rounded U-notch",
             "Smooth constricted channel", "Variable-width S-channel",
             "Triangular protrusions", "Gear-spline boundary", "Four convex arcs"]
LABELS_3D = ["Cylinder with airfoil hole", "Contracting sidewall block",
             "Extruded rounded U-notch", "Tapered elliptic bore block",
             "Bent annular duct", "Lobed annular duct", "Rectangular-to-round duct",
             "Variable-area S-duct", "Twisted inter-blade passage"]
BLUE, ORANGE = "#245c87", "#c56832"


def section_curves(boundary):
    q = np.linspace(0, 1, 801)
    mid = np.full_like(q, .5)
    names = ["x_min", "x_max", "y_min", "y_max"]
    if "o_grid" in boundary.metadata.get("topology", ""):
        names = ["x_min", "x_max"]  # computational seam is not a physical wall
    return [getattr(boundary, name).evaluate(q, mid) for name in names]


def planar_coordinates(curves):
    points = np.concatenate(curves)
    origin = points.mean(0)
    _, _, vectors = np.linalg.svd(points-origin, full_matrices=False)
    normal = vectors[-1]
    if normal[2] < 0:
        normal = -normal
    horizontal = np.array([1., 0., 0.])
    horizontal -= np.dot(horizontal, normal)*normal
    if np.linalg.norm(horizontal) < 1e-8:
        horizontal = vectors[0]
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(normal, horizontal)
    error = float(np.abs((points-origin) @ normal).max())
    if error > 1e-8 * max(float(np.ptp(points, axis=0).max()), 1):
        raise ValueError(f"Mid-span section is not planar: deviation={error}")
    basis = np.stack((horizontal, vertical), -1)
    return [(curve-origin) @ basis for curve in curves]


def title(ax, index, spec, label, value):
    # Keep row headings aligned even when equal aspect shrinks an axes box.
    box = ax.get_position(original=True)
    ax.figure.text((box.x0 + box.x1) / 2, box.y1 + .008,
                   f"({chr(97+index)})  {spec['output_name']}  |  {label}\n"
                   f"{spec['parameter']} = {value:g}", fontsize=11.5,
                   ha="center", va="bottom", color="#172d42", linespacing=1.6)


def draw_curves(ax, curves, color=BLUE):
    for curve in curves:
        ax.plot(curve[:, 0], curve[:, 1], color=color, lw=2.0,
                solid_capstyle="round", solid_joinstyle="round")
    points = np.concatenate(curves)
    low, high = points.min(0), points.max(0)
    span = max(float((high-low).max()), 1e-8)
    ax.set_xlim(low[0]-.08*span, high[0]+.08*span)
    ax.set_ylim(low[1]-.08*span, high[1]+.08*span)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def save(fig, output, dpi):
    fig.savefig(output.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", choices=["low", "high"], default="high")
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/boundary_overview")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    choice = 1 if args.endpoint == "high" else 0
    records = []
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 14.5))
    fig.subplots_adjust(left=.035, right=.965, bottom=.045, top=.865, hspace=.42, wspace=.22)
    fig.suptitle("Two-dimensional boundary families", fontsize=22, y=.972, color="#172d42")
    fig.text(.5, .938, f"{args.endpoint.capitalize()} endpoint of each extrapolation training range", ha="center", fontsize=12, color="#53697c")
    for i, (name, label, ax) in enumerate(zip(PAPER_2D, LABELS_2D, axes.flat)):
        spec = FAMILIES_2D[name]
        value = endpoints(spec)[choice]
        boundary = boundary_2d(name, value)
        curves = boundary.curves()
        # In the circular-airfoil O-grid the two coincident cut lines are not walls.
        selected = [curves["bottom"], curves["top"]] if name == "circle_airfoil_t" else curves.values()
        draw_curves(ax, [curve.sample(1001) for curve in selected])
        title(ax, i, spec, label, value)
        records.append(dict(dimension=2, family=name, prefix=spec["output_name"], parameter=spec["parameter"], value=value))
    save(fig, args.output_dir / "boundaries_2d_3x3", args.dpi)

    boundary_objects = []
    for name in PAPER_3D:
        spec = FAMILIES_3D[name]
        value = endpoints(spec)[choice]
        boundary_objects.append((spec, value, boundary_3d(name, value)))
        records.append(dict(dimension=3, family=name, prefix=spec["output_name"], parameter=spec["parameter"], value=value))
    fig = plt.figure(figsize=(16.5, 15))
    fig.subplots_adjust(left=.01, right=.99, bottom=.045, top=.865, hspace=.25, wspace=.03)
    fig.suptitle("Three-dimensional boundary families", fontsize=22, y=.973, color="#172d42")
    fig.text(.5, .939, f"{args.endpoint.capitalize()} extrapolation endpoint  |  Orange curves: logical mid-span boundary", ha="center", fontsize=12, color="#53697c")
    for i, ((spec, value, boundary), label) in enumerate(zip(boundary_objects, LABELS_3D)):
        ax = fig.add_subplot(3, 3, i+1, projection="3d")
        clouds = []
        for name, surface in boundary.surfaces().items():
            if "o_grid" in boundary.metadata.get("topology", "") and name in ("y_min", "y_max"):
                continue
            points = surface.sample(65, 65)
            clouds.append(points.reshape(-1, 3))
            ax.plot_surface(*np.moveaxis(points, -1, 0), rcount=17, ccount=17,
                            color="#a9cde4", alpha=.12, linewidth=0, shade=False,
                            antialiased=True, rasterized=True)
            ax.plot_wireframe(*np.moveaxis(points, -1, 0), rstride=8, cstride=8,
                              color=BLUE, linewidth=.48, alpha=.55)
            for edge in (points[0], points[-1], points[:, 0], points[:, -1]):
                ax.plot(*edge.T, color=BLUE, lw=.85, alpha=.85)
        for curve in section_curves(boundary):
            ax.plot(*curve.T, color=ORANGE, lw=1.35, alpha=.95)
        cloud = np.concatenate(clouds)
        lo, hi = cloud.min(0), cloud.max(0)
        spans = hi-lo
        ax.set_box_aspect(spans)
        for setter, a, b in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), lo, hi):
            setter(a, b)
        ax.set_proj_type("ortho")
        ax.view_init(elev=27, azim=-58)
        ax.set_axis_off()
        title(ax, i, spec, label, value)
    save(fig, args.output_dir / "boundaries_3d_3x3", args.dpi)

    fig, axes = plt.subplots(3, 3, figsize=(16.5, 14.5))
    fig.subplots_adjust(left=.035, right=.965, bottom=.06, top=.865, hspace=.42, wspace=.22)
    fig.suptitle("Three-dimensional families: z_mid section boundaries", fontsize=22, y=.972, color="#172d42")
    fig.text(.5, .938, r"Logical mid-span $\zeta=0.5$  |  Each section is shown in its own plane", ha="center", fontsize=12, color="#53697c")
    for i, ((spec, value, boundary), label, ax) in enumerate(zip(boundary_objects, LABELS_3D, axes.flat)):
        draw_curves(ax, planar_coordinates(section_curves(boundary)), ORANGE)
        title(ax, i, spec, label, value)
    fig.text(.5, .023, "Boundary geometry only; no interior mesh or learned prediction is shown.", ha="center", fontsize=11, color="#53697c")
    save(fig, args.output_dir / "boundaries_3d_zmid_3x3", args.dpi)
    (args.output_dir / "parameters.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
