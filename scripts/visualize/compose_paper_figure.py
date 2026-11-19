"""Compose publication-ready mesh or quality-figure panels.

Two source modes are deliberately supported:

``--images``
    Compose existing raster figures without recomputing meshes.  This is useful
    for 2-D/3-D result images already produced by the evaluator.  A synthetic
    shared colorbar can be added for quality panels; crop pre-existing per-panel
    colorbars with ``--crop`` first.

``--mesh-files``
    Re-render 2-D saved ``mesh.npy`` arrays.  This preserves physical axes and
    permits exact-boundary overlays, controlled mesh lines, and a genuinely
    shared quality colorbar.  Optional ``quality_data.npz`` files are read from
    the evaluator; otherwise the requested quality metric is recomputed.

Examples
--------
Compose existing mesh figures::

    python scripts/visualize/compose_paper_figure.py \
      --images pde.png hmonet.png meshonet.png pimeshonet.png \
      --rows 1 --cols 4 --panel-titles PDE HMONet MeshONet PI-MeshONet \
      --output paper/figures/air_t_mesh_comparison --formats png pdf

Re-render four 2-D mesh arrays with the exact parameter boundary::

    python scripts/visualize/compose_paper_figure.py \
      --mesh-files pde/mesh.npy hmonet/mesh.npy meshonet/mesh.npy pimeshonet/mesh.npy \
      --shape 101 101 --dataset air_t_ext_1_500_21_21 --value 0.62 \
      --rows 1 --cols 4 --panel-titles PDE HMONet MeshONet PI-MeshONet \
      --show-boundary --show-grid-lines --no-show-axes \
      --output paper/figures/air_t_mesh_comparison --formats png pdf

Re-render a shared-scale quality panel figure::

    python scripts/visualize/compose_paper_figure.py \
      --mesh-files pde/mesh.npy hmonet/mesh.npy meshonet/mesh.npy pimeshonet/mesh.npy \
      --quality-data-files pde/grid_quality/quality_data.npz hmonet/grid_quality/quality_data.npz \
          meshonet/grid_quality/quality_data.npz pimeshonet/grid_quality/quality_data.npz \
      --shape 101 101 --mode quality --metric max_angle --dataset air_t_ext_1_500_21_21 --value 0.62 \
      --rows 1 --cols 4 --show-boundary --show-grid-lines --colorbar-label "Maximum angle (deg)" \
      --output paper/figures/air_t_max_angle --formats png pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.mesh.sampling import sample_boundary
from src.evaluation.configuration import create_geometry
from src.evaluation.quality_output import compute_quad_quality
from src.utils.paths import prepare_output_file, resolve_project_path
from src.utils.visualize import plot_quad_mesh, plot_quad_mesh_colored


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--images",
        nargs="+",
        type=Path,
        help="Existing PNG/JPG/TIFF/WebP raster panel files to compose. PDF is an output format only.",
    )
    source.add_argument(
        "--mesh-files",
        nargs="+",
        type=Path,
        help="Saved 2-D mesh.npy files to re-render as panels.",
    )
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--mode", choices=("mesh", "quality"), default="mesh")
    parser.add_argument(
        "--panel-titles", nargs="+", default=[], help="One title per panel."
    )
    parser.add_argument(
        "--panel-labels",
        nargs="+",
        default=[],
        help="Panel labels, for example '(a)' '(b)' '(c)'.",
    )
    parser.add_argument("--row-labels", nargs="+", default=[])
    parser.add_argument("--column-labels", nargs="+", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path or stem; a stem is combined with --formats.",
    )
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=["png"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figure-width", type=float)
    parser.add_argument("--figure-height", type=float)
    parser.add_argument("--panel-width", type=float, default=3.5)
    parser.add_argument("--panel-height", type=float, default=3.2)
    parser.add_argument("--wspace", type=float, default=0.08)
    parser.add_argument("--hspace", type=float, default=0.12)
    parser.add_argument("--font-size", type=float, default=10.0)
    parser.add_argument("--title-font-size", type=float, default=10.0)
    parser.add_argument("--label-font-size", type=float, default=11.0)
    parser.add_argument("--background", default="white")
    parser.add_argument("--transparent", action="store_true")

    # Existing-image controls.  Coordinates are fractions measured from the
    # top-left of each image and are intentionally shared by all panels.
    parser.add_argument(
        "--crop",
        nargs=4,
        type=float,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Normalized crop box for every image, each value in [0, 1].",
    )
    parser.add_argument(
        "--show-image-axes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show image-pixel axes in --images mode (normally disabled).",
    )

    # Re-rendered 2-D mesh controls.
    parser.add_argument(
        "--shape",
        nargs=2,
        type=int,
        metavar=("NX", "NY"),
        help="Logical mesh shape, required with --mesh-files.",
    )
    parser.add_argument(
        "--quality-data-files",
        nargs="+",
        type=Path,
        help="Optional evaluator quality_data.npz files, one per mesh file.",
    )
    parser.add_argument(
        "--metric",
        help="Quality field to plot in quality mode, e.g. max_angle or scaled_jacobian.",
    )
    parser.add_argument(
        "--dataset",
        help="Evaluation case name used to reconstruct the exact parametric boundary.",
    )
    parser.add_argument(
        "--value",
        type=float,
        help="The one variable geometry parameter for --dataset.",
    )
    parser.add_argument(
        "--evaluate-config",
        type=Path,
        default=ROOT / "configs/2D/evaluate.yaml",
    )
    parser.add_argument(
        "--show-boundary",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overlay the exact parametric boundary in mesh-file mode.",
    )
    parser.add_argument(
        "--show-grid-lines",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Draw mesh-cell edges in mesh-file mode.",
    )
    parser.add_argument(
        "--show-axes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show physical coordinate axes in mesh-file mode.",
    )
    parser.add_argument(
        "--shared-limits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use common physical x/y limits across re-rendered panels.",
    )
    parser.add_argument(
        "--view-window",
        nargs=4,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help=(
            "Physical 2-D local-view window for --mesh-files: XMIN XMAX YMIN YMAX. "
            "Overrides --shared-limits and is applied identically to every panel."
        ),
    )
    parser.add_argument("--mesh-linewidth", type=float, default=0.32)
    parser.add_argument("--mesh-color", default="#263238")
    parser.add_argument("--mesh-alpha", type=float, default=0.88)
    parser.add_argument("--boundary-linewidth", type=float, default=1.6)
    parser.add_argument("--boundary-color", default="#D62728")
    parser.add_argument("--quality-edge-color", default="#303030")
    parser.add_argument("--quality-alpha", type=float, default=1.0)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument(
        "--colorbar",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Add exactly one colorbar for a quality figure (default: enabled in quality mode).",
    )
    parser.add_argument("--colorbar-label")
    parser.add_argument("--colorbar-width", type=float, default=0.022)
    parser.add_argument("--colorbar-pad", type=float, default=0.025)
    return parser


def _validate_layout(args, panel_count: int) -> None:
    if args.rows < 1 or args.cols < 1:
        raise ValueError("--rows and --cols must be positive")
    if panel_count > args.rows * args.cols:
        raise ValueError(
            f"{panel_count} panels do not fit in {args.rows}x{args.cols}; increase rows or columns"
        )
    for name, values, maximum in (
        ("panel titles", args.panel_titles, panel_count),
        ("panel labels", args.panel_labels, panel_count),
        ("row labels", args.row_labels, args.rows),
        ("column labels", args.column_labels, args.cols),
    ):
        if values and len(values) != maximum:
            raise ValueError(f"--{name.replace(' ', '-')} requires exactly {maximum} entries")
    if args.quality_data_files and len(args.quality_data_files) != panel_count:
        raise ValueError("--quality-data-files must have one file per --mesh-files entry")
    if args.mesh_files and args.shape is None:
        raise ValueError("--shape NX NY is required with --mesh-files")
    if args.mode == "quality" and args.mesh_files and not args.metric:
        raise ValueError("--metric is required for --mode quality with --mesh-files")
    if args.crop and not all(0.0 <= value <= 1.0 for value in args.crop):
        raise ValueError("--crop values must be within [0, 1]")
    if args.crop and not (args.crop[0] < args.crop[2] and args.crop[1] < args.crop[3]):
        raise ValueError("--crop requires LEFT<RIGHT and TOP<BOTTOM")
    if args.view_window is not None:
        if args.images:
            raise ValueError("--view-window requires --mesh-files; use --crop for existing images")
        xmin, xmax, ymin, ymax = args.view_window
        if not xmin < xmax or not ymin < ymax:
            raise ValueError("--view-window requires XMIN<XMAX and YMIN<YMAX")


def _load_image(path: Path, crop) -> np.ndarray:
    path = resolve_project_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    image = np.asarray(mpimg.imread(path))
    if crop is None:
        return image
    height, width = image.shape[:2]
    left, top, right, bottom = crop
    x0, x1 = round(left * width), round(right * width)
    y0, y1 = round(top * height), round(bottom * height)
    return image[y0:y1, x0:x1]


def _load_mesh(path: Path, shape: tuple[int, int]) -> np.ndarray:
    path = resolve_project_path(path)
    values = np.asarray(np.load(path), dtype=float)
    nx, ny = shape
    if values.ndim == 3 and values.shape == (ny, nx, 2):
        values = values.reshape(-1, 2)
    if values.shape != (nx * ny, 2):
        raise ValueError(f"{path}: expected {(nx * ny, 2)} or {(ny, nx, 2)}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: mesh contains non-finite coordinates")
    return values


def _quality_values(mesh: np.ndarray, shape: tuple[int, int], metric: str, source: Path | None):
    if source is not None:
        source = resolve_project_path(source)
        with np.load(source) as archive:
            if metric not in archive:
                raise KeyError(f"{source}: quality field {metric!r} is unavailable")
            values = np.asarray(archive[metric], dtype=float).reshape(-1)
    else:
        values = np.asarray(
            compute_quad_quality(mesh, *shape, metrics=[metric])["fields"][metric],
            dtype=float,
        ).reshape(-1)
    expected = (shape[0] - 1) * (shape[1] - 1)
    if values.size != expected:
        raise ValueError(f"{metric}: expected {expected} cell values, got {values.size}")
    return values


def _exact_boundary(args, shape: tuple[int, int]):
    if not args.dataset or args.value is None:
        raise ValueError("--dataset and --value are required when --show-boundary is enabled")
    config_path = resolve_project_path(args.evaluate_config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
    try:
        case = config["cases"][args.dataset]
    except KeyError as error:
        raise KeyError(f"{args.dataset!r} is not a case in {config_path}") from error
    geometry, parameters = create_geometry(2, case, args.value)
    boundary = sample_boundary(geometry, shape)
    return np.vstack((boundary, boundary[:1])), parameters


def _common_limits(meshes: list[np.ndarray], boundary: np.ndarray | None):
    points = [*meshes]
    if boundary is not None:
        points.append(boundary)
    stacked = np.concatenate(points, axis=0)
    lower, upper = stacked.min(axis=0), stacked.max(axis=0)
    margin = max(float(np.max(upper - lower)), 1e-12) * 0.035
    return (lower[0] - margin, upper[0] + margin), (lower[1] - margin, upper[1] + margin)


def _view_window_limits(window: list[float] | tuple[float, ...]):
    """Convert a physical local-view window to the axis-limit convention."""
    xmin, xmax, ymin, ymax = map(float, window)
    return (xmin, xmax), (ymin, ymax)


def _decorate_axis(axis, args, boundary, limits):
    if boundary is not None:
        axis.plot(
            boundary[:, 0],
            boundary[:, 1],
            color=args.boundary_color,
            linewidth=args.boundary_linewidth,
            zorder=5,
        )
    if limits is not None:
        axis.set_xlim(limits[0])
        axis.set_ylim(limits[1])
    axis.set_aspect("equal", adjustable="box")
    if args.show_axes:
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.tick_params(labelsize=max(6, args.font_size - 2))
    else:
        axis.set_axis_off()


def _make_figure(args):
    width = args.figure_width or args.cols * args.panel_width
    height = args.figure_height or args.rows * args.panel_height
    figure, axes = plt.subplots(
        args.rows,
        args.cols,
        figsize=(width, height),
        squeeze=False,
        facecolor=args.background,
    )
    figure.subplots_adjust(wspace=args.wspace, hspace=args.hspace)
    return figure, axes.ravel()


def _apply_panel_text(figure, axes, args, panel_count: int):
    for index, axis in enumerate(axes[:panel_count]):
        if args.panel_titles:
            axis.set_title(args.panel_titles[index], fontsize=args.title_font_size, pad=7)
        if args.panel_labels:
            axis.text(
                0.015,
                0.985,
                args.panel_labels[index],
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=args.label_font_size,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
                zorder=10,
            )
    for row, label in enumerate(args.row_labels):
        axis = axes[row * args.cols]
        position = axis.get_position()
        figure.text(
            position.x0 - 0.025,
            (position.y0 + position.y1) / 2,
            label,
            ha="right",
            va="center",
            rotation=90,
            fontsize=args.label_font_size,
            fontweight="bold",
        )
    for column, label in enumerate(args.column_labels):
        axis = axes[column]
        position = axis.get_position()
        figure.text(
            (position.x0 + position.x1) / 2,
            position.y1 + 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=args.label_font_size,
            fontweight="bold",
        )


def _save(figure, args) -> list[Path]:
    output = resolve_project_path(args.output)
    paths = [output] if output.suffix else [output.with_suffix("." + kind) for kind in args.formats]
    for path in paths:
        path = prepare_output_file(path, context="Publication figure")
        figure.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor=args.background, transparent=args.transparent)
    plt.close(figure)
    return paths


def compose_images(args) -> list[Path]:
    images = [_load_image(path, args.crop) for path in args.images]
    _validate_layout(args, len(images))
    figure, axes = _make_figure(args)
    for axis, image in zip(axes, images):
        axis.imshow(image)
        if not args.show_image_axes:
            axis.set_axis_off()
    for axis in axes[len(images):]:
        axis.set_axis_off()
    if args.mode == "quality" and (args.colorbar if args.colorbar is not None else True):
        if args.crop is None:
            warnings.warn(
                "Existing quality images may already contain colorbars. Use --crop to remove them before adding the single shared colorbar.",
                stacklevel=2,
            )
        minimum = 0.0 if args.vmin is None else args.vmin
        maximum = 1.0 if args.vmax is None else args.vmax
        if maximum <= minimum:
            raise ValueError("--vmax must exceed --vmin")
        mappable = ScalarMappable(norm=Normalize(minimum, maximum), cmap=args.cmap)
        colorbar = figure.colorbar(
            mappable,
            ax=list(axes[: len(images)]),
            fraction=args.colorbar_width,
            pad=args.colorbar_pad,
        )
        colorbar.set_label(args.colorbar_label or args.metric or "Quality")
    _apply_panel_text(figure, axes, args, len(images))
    return _save(figure, args)


def compose_meshes(args) -> list[Path]:
    _validate_layout(args, len(args.mesh_files))
    shape = tuple(map(int, args.shape))
    meshes = [_load_mesh(path, shape) for path in args.mesh_files]
    show_boundary = True if args.show_boundary is None else args.show_boundary
    show_grid = True if args.show_grid_lines is None else args.show_grid_lines
    boundary = _exact_boundary(args, shape)[0] if show_boundary else None
    limits = (
        _view_window_limits(args.view_window)
        if args.view_window is not None
        else _common_limits(meshes, boundary) if args.shared_limits else None
    )
    figure, axes = _make_figure(args)
    color_values = []
    if args.mode == "quality":
        quality_sources = args.quality_data_files or [None] * len(meshes)
        color_values = [
            _quality_values(mesh, shape, args.metric, source)
            for mesh, source in zip(meshes, quality_sources)
        ]
        finite = np.concatenate([values[np.isfinite(values)] for values in color_values])
        if finite.size == 0:
            raise ValueError(f"{args.metric}: no finite quality values")
        vmin = float(np.min(finite) if args.vmin is None else args.vmin)
        vmax = float(np.max(finite) if args.vmax is None else args.vmax)
        if vmax <= vmin:
            vmax = vmin + np.finfo(float).eps
        norm = Normalize(vmin, vmax)
        mappable = ScalarMappable(norm=norm, cmap=args.cmap)
    else:
        norm = mappable = None
    for index, (axis, mesh) in enumerate(zip(axes, meshes)):
        if args.mode == "mesh":
            if show_grid:
                plot_quad_mesh(
                    mesh,
                    compute_quad_quality(mesh, *shape, metrics=[])["mesh"].faces,
                    ax=axis,
                    show_node_id=False,
                    edgecolor=args.mesh_color,
                    linewidth=args.mesh_linewidth,
                    alpha=args.mesh_alpha,
                )
            else:
                axis.scatter(mesh[:, 0], mesh[:, 1], s=0.1, alpha=0.0)
        else:
            quality = compute_quad_quality(mesh, *shape, metrics=[])["mesh"]
            plot_quad_mesh_colored(
                mesh,
                quality.faces,
                color_values[index],
                ax=axis,
                cmap=args.cmap,
                norm=norm,
                edgecolor=args.quality_edge_color if show_grid else "none",
                linewidth=args.mesh_linewidth if show_grid else 0.0,
                alpha=args.quality_alpha,
            )
        _decorate_axis(axis, args, boundary, limits)
    for axis in axes[len(meshes):]:
        axis.set_axis_off()
    if args.mode == "quality" and (args.colorbar if args.colorbar is not None else True):
        colorbar = figure.colorbar(
            mappable,
            ax=list(axes[: len(meshes)]),
            fraction=args.colorbar_width,
            pad=args.colorbar_pad,
        )
        colorbar.set_label(args.colorbar_label or args.metric)
    _apply_panel_text(figure, axes, args, len(meshes))
    return _save(figure, args)


def main(argv=None) -> list[Path]:
    args = build_parser().parse_args(argv)
    plt.rcParams.update({"font.size": args.font_size})
    outputs = compose_images(args) if args.images else compose_meshes(args)
    for path in outputs:
        print(f"saved={path}", flush=True)
    return outputs


if __name__ == "__main__":
    main()
