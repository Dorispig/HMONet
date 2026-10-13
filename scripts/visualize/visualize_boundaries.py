"""Visualize all currently registered parametric boundary generators."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

if __package__ in ("", None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.geometry.boundary_generator_2d import (
    BOUNDARY_GENERATORS as BOUNDARY_2D,
    generate_boundary as generate_2d,
)
from src.data.geometry.boundary_generator_3d import (
    BOUNDARY_GENERATORS as BOUNDARY_3D,
    generate_boundary as generate_3d,
)
from src.utils import visualize


def draw_2d(
    name,
    output_dir,
    nx=21,
    ny=21,
    four_arc_angle=90.0,
    parameters=None,
    filename=None,
    reference=None,
):
    """Generate one 2-D figure through the shared visualization utility."""
    parameters = dict(parameters or {})
    if name == "four_convex_semicircle":
        parameters.setdefault("arc_angle_deg", four_arc_angle)
    boundary = generate_2d(name, **parameters)
    fig, ax = plt.subplots(figsize=(6, 5))
    visualize.plot_boundary_curve_set_2d(
        boundary, nx=nx, ny=ny, ax=ax, show_samples=True
    )
    if reference is not None:
        original = generate_2d(name)
        counts = dict(bottom=nx, right=ny, top=nx, left=ny)
        outline = original.sample_closed(counts)
        ax.plot(
            outline[:, 0],
            outline[:, 1],
            "--",
            color="gray",
            alpha=0.65,
            label="default reference",
            linewidth=1,
        )
        ax.legend(loc="best")
    label = ", ".join(f"{k}={v}" for k, v in parameters.items()) or "default parameters"
    ax.set_title(f"{name}\n{label}", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = output_dir / (filename or f"{name}.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_3d(
    name,
    output_dir,
    nx=21,
    ny=21,
    nz=21,
    parameters=None,
    filename=None,
    reference=None,
):
    """Generate one 3-D figure through the shared visualization utility."""
    parameters = dict(parameters or {})
    boundary = generate_3d(name, **parameters)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    visualize.plot_boundary_surface_set_3d(boundary, nx=nx, ny=ny, nz=nz, ax=ax)
    all_points = np.concatenate(
        [
            boundary.surfaces()[face].sample(8, 8).reshape(-1, 3)
            for face in boundary.names
        ]
    )
    mins, maxs = all_points.min(axis=0), all_points.max(axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    label = ", ".join(f"{k}={v}" for k, v in parameters.items()) or "default parameters"
    ax.set_title(f"{name}\n{label}", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=25, azim=-55)
    ax.set_box_aspect(np.maximum(maxs - mins, 1e-6))
    fig.tight_layout()
    path = output_dir / (filename or f"{name}.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# Each variant changes exactly one constructor parameter from its default.
VARIANTS_2D = {
    "rectangle": {"x_max": 2.0},
    "circle_airfoil": {"t": 0.4, "m": 0.1, "p": 0.7, "c": 0.7, "radius": 1.4},
    "rect_top_semiellipse": {"length": 1.8, "height": 0.5, "semi_minor_axis": 0.7},
    "eccentric_semiannulus": {
        "inner_center_x": 0.25,
        "inner_radius": 0.25,
        "outer_radius": 0.7,
    },
    "four_convex_semicircle": {"arc_angle_deg": 140.0, "chord": 1.5},
    "symmetric_convex_tri": {
        "width": 1.8,
        "side_length": 1.5,
        "protrusion_distance": -0.35,
    },
    "smooth_constricted_channel": {"length": 1.8, "height": 1.5, "depth": 0.15},
    "variable_width_s_channel": {
        "length": 2.5,
        "height": 1.2,
        "bend": 4.0,
        "constriction": 0.05,
    },
    "gear_spline": {"width": 3.0, "height": 2.4, "teeth": 3, "amplitude": 0.04},
    "rounded_u_notch": {
        "width": 4.0,
        "height": 2.6,
        "notch_width": 1.5,
        "depth": 0.7,
        "radius": 0.3,
    },
}
VARIANTS_3D = {
    "box": {"xmax": 3.0, "ymax": 2.0, "zmax": 2.0},
    "cylindrical_domain_with_airfoil_hole": {
        "outer_radius": 1.8,
        "chord": 1.25,
        "camber": 0.06,
        "camber_position": 0.55,
        "thickness": 0.2,
        "length": 3.2,
    },
    "stadium_cylinder_with_circular_hole": {
        "straight_half_length": 1.2,
        "stadium_radius": 0.9,
        "hole_radius": 0.4,
        "length": 3.2,
    },
}


def draw_family(name, dimension, root, nx, ny, nz):
    """Write default and one-parameter variants with reproducible metadata."""
    folder = root / f"{dimension}D" / name
    folder.mkdir(parents=True, exist_ok=True)
    registry = BOUNDARY_2D if dimension == 2 else BOUNDARY_3D
    defaults = {
        key: value.default
        for key, value in inspect.signature(registry[name]).parameters.items()
    }
    variants = (VARIANTS_2D if dimension == 2 else VARIANTS_3D)[name]
    draw = draw_2d if dimension == 2 else draw_3d
    options = dict(nx=nx, ny=ny)
    if dimension == 3:
        options["nz"] = nz
    records = {"default.png": defaults}
    draw(name, folder, filename="default.png", **options)
    for key, value in variants.items():
        filename = f"{key}_{value}.png"
        draw(
            name,
            folder,
            parameters={key: value},
            filename=filename,
            reference=True,
            **options,
        )
        records[filename] = dict(defaults, **{key: value})
    (folder / "parameters.json").write_text(
        json.dumps(
            dict(
                boundary=name,
                dimension=dimension,
                nx=nx,
                ny=ny,
                nz=nz if dimension == 3 else None,
                images=records,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--names",
        nargs="+",
        choices=list(BOUNDARY_2D),
        help="Draw only the selected 2-D boundaries; skip 3-D.",
    )
    parser.add_argument("--output-root", default="outputs/boundaries")
    parser.add_argument("--nx", type=int, default=21)
    parser.add_argument("--ny", type=int, default=21)
    parser.add_argument("--nz", type=int, default=21)
    parser.add_argument(
        "--surface-grid",
        type=int,
        help="Optional equal per-axis sample count for 3-D previews",
    )
    parser.add_argument("--four-arc-angle", type=float, default=90.0)
    args = parser.parse_args()
    root = Path(args.output_root)
    out2d, out3d = root / "2D", root / "3D"
    out2d.mkdir(parents=True, exist_ok=True)
    out3d.mkdir(parents=True, exist_ok=True)
    # Keep one canonical circle+airfoil figure; legacy aliases are not redrawn.
    names_2d = [
        name
        for name in BOUNDARY_2D
        if name not in {"circle_airfoil_m", "circle_airfoil_t", "air_t", "air_m"}
    ]
    for name in args.names or names_2d:
        draw_family(name, 2, root, args.nx, args.ny, args.nz)
    # Draw the verified surface generators only when no 2-D selection is given.
    for name in (
        ()
        if args.names
        else (
            "box",
            "cylindrical_domain_with_airfoil_hole",
            "stadium_cylinder_with_circular_hole",
        )
    ):
        draw_family(
            name,
            3,
            root,
            args.surface_grid or args.nx,
            args.surface_grid or args.ny,
            args.surface_grid or args.nz,
        )
    if args.four_arc_angle != 90 and (
        not args.names or "four_convex_semicircle" in args.names
    ):
        draw_2d(
            "four_convex_semicircle",
            out2d / "four_convex_semicircle",
            args.nx,
            args.ny,
            args.four_arc_angle,
            filename=f"arc_angle_deg_{args.four_arc_angle}.png",
        )
    print(f"2D boundary figures: {out2d}")
    if not args.names:
        print(f"3D boundary figures: {out3d}")


if __name__ == "__main__":
    main()



