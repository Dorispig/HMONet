"""Render reproducible low/high boundary examples for the dataset guide.

The script reads the registered family definitions rather than a generated
dataset.  It therefore documents a boundary class once, independent of how
many interpolation/extrapolation datasets currently exist on disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generation.make_dataset_parametric_2d import (
    DATASET_FAMILIES as FAMILIES_2D,
    DATASET_GENERATORS as GENERATORS_2D,
)
from src.data.generation.make_dataset_parametric_3d import (
    DATASET_FAMILIES as FAMILIES_3D,
)
from src.data.geometry.boundary_generator_3d import BOUNDARY_GENERATORS as GENERATORS_3D
from src.numerics.traditional.tfi_3d import tfi_from_surfaces


PAPER_2D = (
    "circle_airfoil_t",
    "eccentric_semiannulus",
    "rect_top_semiellipse",
    "rounded_u_notch_compact",
    "smooth_constricted_channel",
    "variable_width_s_channel",
    "symmetric_convex_tri",
    "gear_spline",
    "four_convex_arcs",
)
PAPER_3D = (
    "cylindrical_domain_with_airfoil_hole",
    "sidewall_deformed_block",
    "rounded_u_notch_extrusion",
    "tapered_elliptic_bore_block",
    "bent_annular_duct",
    "lobed_annular_duct",
    "square_to_round_transition",
    "variable_area_s_duct",
    "twisted_blade_passage",
)


def endpoints(spec: dict) -> tuple[float, float]:
    intervals = np.asarray(spec["extrapolation"], dtype=float)
    return float(intervals[:, 0].min()), float(intervals[:, 1].max())


def boundary_2d(name: str, value: float):
    spec = FAMILIES_2D[name]
    parameters = dict(spec["fixed_parameters"])
    parameters[spec["parameter"]] = value
    return GENERATORS_2D[spec["function"]](**parameters)


def render_2d(output: Path) -> None:
    fig, axes = plt.subplots(2, len(PAPER_2D), figsize=(27, 6.1), constrained_layout=True)
    for column, name in enumerate(PAPER_2D):
        spec = FAMILIES_2D[name]
        low, high = endpoints(spec)
        for row, (value, tag, color) in enumerate(((low, "low", "#2f6fbe"), (high, "high", "#cb6b2f"))):
            ax = axes[row, column]
            boundary = boundary_2d(name, value)
            for curve in boundary.curves().values():
                points = curve.sample(240)
                ax.plot(points[:, 0], points[:, 1], color=color, lw=1.4)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"{spec['output_name']} | {tag}\n{spec['parameter']}={value:g}",
                fontsize=9,
            )
            for spine in ax.spines.values():
                spine.set_color("#cbd5e1")
    fig.suptitle("Registered 2-D boundary families: extrapolation-end examples", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_fca_candidate(output: Path) -> None:
    """Show the requested broad FCA range without declaring it PDE-safe."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), constrained_layout=True)
    for ax, value, color in zip(
        axes, (30.0, 150.0), ("#2f6fbe", "#cb6b2f")
    ):
        boundary = boundary_2d("four_convex_arcs", value)
        for curve in boundary.curves().values():
            points = curve.sample(400)
            ax.plot(points[:, 0], points[:, 1], color=color, lw=2.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"arc_angle_deg = {value:g} deg")
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
    fig.suptitle("FCA candidate endpoints (geometry only)", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def boundary_3d(name: str, value: float):
    spec = FAMILIES_3D[name]
    parameters = dict(spec["fixed_parameters"])
    parameters[spec["parameter"]] = value
    return GENERATORS_3D[spec["function"]](**parameters)


def render_3d(output: Path) -> None:
    fig = plt.figure(figsize=(27, 6.4), constrained_layout=True)
    for column, name in enumerate(PAPER_3D):
        spec = FAMILIES_3D[name]
        low, high = endpoints(spec)
        for row, (value, tag, color) in enumerate(((low, "low", "#2f6fbe"), (high, "high", "#cb6b2f"))):
            ax = fig.add_subplot(2, len(PAPER_3D), row * len(PAPER_3D) + column + 1, projection="3d")
            boundary = boundary_3d(name, value)
            for surface in boundary.surfaces().values():
                points = surface.sample(25, 25)
                ax.plot_wireframe(points[..., 0], points[..., 1], points[..., 2], rstride=3, cstride=3, color=color, linewidth=0.35, alpha=0.72)
            ax.view_init(elev=23, azim=-57)
            ax.set_axis_off()
            ax.set_title(f"{spec['output_name']} | {tag}\n{spec['parameter']}={value:g}", fontsize=8)
    fig.suptitle("Registered 3-D boundary families: extrapolation-end examples", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_tbp_endpoints(output: Path) -> None:
    """Render the widened blade-passage range as a readable two-panel view."""
    fig = plt.figure(figsize=(8.4, 3.9), constrained_layout=True)
    for index, (value, color) in enumerate(
        ((0.0, "#2f6fbe"), (90.0, "#cb6b2f")), start=1
    ):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        boundary = boundary_3d("twisted_blade_passage", value)
        for surface in boundary.surfaces().values():
            points = surface.sample(31, 31)
            ax.plot_wireframe(
                points[..., 0], points[..., 1], points[..., 2],
                rstride=3, cstride=3, color=color, linewidth=0.45, alpha=0.78,
            )
        ax.view_init(elev=24, azim=-58)
        ax.set_axis_off()
        ax.set_title(f"twist_angle_deg = {value:g} deg")
    fig.suptitle("TBP extrapolation endpoints", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def render_3d_sections(output: Path) -> None:
    """Render one logical mid-span section for each 3-D solid example."""
    fig = plt.figure(figsize=(27, 6.4), constrained_layout=True)
    for column, name in enumerate(PAPER_3D):
        spec = FAMILIES_3D[name]
        low, high = endpoints(spec)
        for row, (value, tag, color) in enumerate(
            ((low, "low", "#2f6fbe"), (high, "high", "#cb6b2f"))
        ):
            ax = fig.add_subplot(
                2, len(PAPER_3D), row * len(PAPER_3D) + column + 1,
                projection="3d",
            )
            boundary = boundary_3d(name, value)
            grid = tfi_from_surfaces(boundary, xi=21, eta=21, zeta=21)
            section = grid[len(grid) // 2]
            ax.plot_wireframe(
                section[..., 0], section[..., 1], section[..., 2],
                rstride=1, cstride=1, color=color, linewidth=0.48, alpha=0.92,
            )
            spans = np.ptp(section.reshape(-1, 3), axis=0)
            reference = max(float(spans.max()), 1.0e-9)
            ax.set_box_aspect(tuple(np.maximum(spans, 0.22 * reference)))
            ax.view_init(elev=25, azim=-58)
            ax.set_axis_off()
            ax.set_title(
                f"{spec['output_name']} | {tag}\n{spec['parameter']}={value:g}",
                fontsize=8,
            )
    fig.suptitle("Registered 3-D boundary families: logical mid-span sections", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", choices=("2", "3", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    if args.dimension in {"2", "both"}:
        render_2d(args.output_dir / "boundary_families_2d.png")
        render_fca_candidate(args.output_dir / "fca_a_candidate_30_150.png")
    if args.dimension in {"3", "both"}:
        render_3d(args.output_dir / "boundary_families_3d.png")
        render_3d_sections(args.output_dir / "boundary_families_3d_sections.png")
        render_tbp_endpoints(args.output_dir / "tbp_a_endpoints_0_90.png")


if __name__ == "__main__":
    main()
