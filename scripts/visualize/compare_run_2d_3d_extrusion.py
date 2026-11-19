"""Render a geometric cross-section check for 2-D run and 3-D run_e."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.geometry.boundary_generator_2d import rounded_u_notch_boundary
from src.data.geometry.boundary_generator_3d import rounded_u_notch_extrusion_boundary


def _section_3d(boundary, points=601):
    q = np.linspace(0.0, 1.0, points)
    face = boundary.z_min
    bottom = face.evaluate(q, np.zeros_like(q))[:, :2]
    right = face.evaluate(np.ones_like(q), q)[:, :2]
    top = face.evaluate(q[::-1], np.ones_like(q))[:, :2]
    left = face.evaluate(np.zeros_like(q), q[::-1])[:, :2]
    return np.vstack((bottom, right[1:], top[1:], left[1:]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=float, default=1.5)
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--output", default="outputs/validation/run_2d_vs_run_e_cross_section.png")
    args = parser.parse_args(argv)
    two_d = rounded_u_notch_boundary(depth=args.depth, radius=args.radius)
    three_d = rounded_u_notch_extrusion_boundary(depth=args.depth, radius=args.radius)
    curve_2d = two_d.sample_closed({"bottom": 601, "right": 601, "top": 601, "left": 601})
    curve_3d = _section_3d(three_d)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True, sharey=True)
    panels = (
        (axes[0], curve_2d, "2-D run: rounded_u_notch"),
        (axes[1], curve_3d, "3-D run_e: z = z_min cross-section"),
    )
    for axis, curve, title in panels:
        axis.plot(curve[:, 0], curve[:, 1], color="#1f77b4", linewidth=2)
        axis.fill(curve[:, 0], curve[:, 1], color="#1f77b4", alpha=0.10)
        axis.set_title(title)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
        axis.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[2].plot(curve_2d[:, 0], curve_2d[:, 1], label="2-D run", linewidth=2.5)
    axes[2].plot(curve_3d[:, 0], curve_3d[:, 1], "--", label="3-D run_e section", linewidth=1.5)
    axes[2].set_title("Overlay (identical profiles)")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].grid(alpha=0.25)
    axes[2].set_xlabel("x")
    axes[2].legend()
    figure.suptitle(f"Exact cross-section check, depth = {args.depth:g}")
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)
    print(output.resolve())


if __name__ == "__main__":
    main()
