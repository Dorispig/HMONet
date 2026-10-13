"""Reproducible 2-D section and 3-D volume zooms of saved structured meshes."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import matplotlib.patheffects as path_effects
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

from src.data.mesh.topology import HexMeshXYZ
from src.evaluation.method_order import ordered_methods, method_label
from src.evaluation.quality_3d import (
    HIGHER,
    SURFACE_HIGHER_IS_BETTER,
    surface_quad_quality_fields,
)
from src.evaluation.quality_output import summarize_fields, quality_plot_style
from src.utils.metrics import evaluate_hexahedral_grid_quality

VERSION = 6
FACES = np.array(
    [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [3, 7, 6, 2], [0, 4, 7, 3], [1, 2, 6, 5]]
)
EDGES = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
)
DEFAULTS = dict(
    mode="slice",
    slice="z_mid",
    position=0.5,
    plot="both",
    cell_padding=2,
    color_scale="global",
    elevation=24.0,
    azimuth=-58.0,
    volume_style="surface",
    alpha=0.3,
    save_png=True,
    save_pdf=False,
    save_plots=True,
    show_axes=True,
    slice_coordinates="physical",
)
EXTRA_OPTIONS = {
    "metric",
    "i_range",
    "j_range",
    "k_range",
    "xlim",
    "ylim",
    "zlim",
    "center_on",
    "cell_index",
    "reference",
    "cut_axis",
    "cut_position",
    "name",
    "methods",
    "output_dir",
    "force",
    "index",
}


def normalize_options(options):
    unknown = set(options) - set(DEFAULTS) - EXTRA_OPTIONS
    if unknown:
        raise ValueError(f"Unknown zoom options: {sorted(unknown)}")
    opts = {**DEFAULTS, **options}
    for key, choices in {
        "mode": ("slice", "volume"),
        "plot": ("mesh", "quality", "both"),
        "color_scale": ("global", "local"),
        "volume_style": ("surface", "wireframe", "transparent"),
        "slice": ("x_mid", "y_mid", "z_mid"),
        "slice_coordinates": ("projected", "physical"),
    }.items():
        if opts[key] not in choices:
            raise ValueError(f"{key} must be one of {choices}")
    opts.setdefault(
        "metric",
        "scaled_jacobian" if opts["mode"] == "slice" else "minimum_scaled_jacobian",
    )
    if not 0 <= float(opts["position"]) <= 1:
        raise ValueError("position must be between 0 and 1")
    if int(opts["cell_padding"]) != opts["cell_padding"] or opts["cell_padding"] < 0:
        raise ValueError("cell_padding must be a non-negative integer")
    if not 0 < float(opts["alpha"]) <= 1:
        raise ValueError("alpha must be in (0, 1]")
    dimensions = 2 if opts["mode"] == "slice" else 3
    range_keys = ("i_range", "j_range", "k_range")[:dimensions]
    limit_keys = ("xlim", "ylim", "zlim")[:dimensions]
    if dimensions == 2 and (
        opts.get("k_range") is not None or opts.get("zlim") is not None
    ):
        raise ValueError("k_range and zlim apply only to volume mode")
    for keys in (range_keys, limit_keys):
        present = [opts.get(k) is not None for k in keys]
        if any(present) and not all(present):
            raise ValueError(f"Provide all of {keys}")
        for key in keys:
            if opts.get(key) is None:
                continue
            values = opts[key]
            if (
                len(values) != 2
                or not np.isfinite(values).all()
                or values[0] >= values[1]
            ):
                raise ValueError(f"{key} requires two finite increasing values")
            if key.endswith("range") and any(int(v) != v for v in values):
                raise ValueError(f"{key} must use integer cell indices")
    choices = [
        opts.get(range_keys[0]) is not None,
        opts.get(limit_keys[0]) is not None,
        opts.get("cell_index") is not None,
        opts.get("center_on") is not None,
    ]
    if sum(choices) > 1:
        raise ValueError(
            "Choose one selection: index ranges, physical limits, cell_index, or center_on"
        )
    if opts.get("center_on") not in (None, "worst"):
        raise ValueError("center_on must be worst")
    if opts.get("cut_axis") is not None:
        if opts["mode"] != "volume" or opts["cut_axis"] not in ("x", "y", "z"):
            raise ValueError("cut_axis requires volume mode and x/y/z")
        opts.setdefault("cut_position", 0.5)
        if not 0 < float(opts["cut_position"]) <= 1:
            raise ValueError("cut_position must be in (0, 1]")
    elif opts.get("cut_position") is not None:
        raise ValueError("cut_position requires cut_axis")
    if opts.get("name") and not re.fullmatch(r"[A-Za-z0-9_.-]+", opts["name"]):
        raise ValueError(
            "name must contain only letters, digits, underscores, hyphens and dots"
        )
    if opts.get("name") in (".", ".."):
        raise ValueError("name cannot be . or ..")
    return opts


def extract_section(grid, axis, position=0.5, index=None):
    """Interpolate node layers, never substitute a volume-cell quality slice."""
    dimension = {"x": 2, "y": 1, "z": 0}[axis]
    p = float(index) if index is not None else position * (grid.shape[dimension] - 1)
    if not 0 <= p <= grid.shape[dimension] - 1:
        raise ValueError("section index is outside the node grid")
    lo, hi = int(np.floor(p)), int(np.ceil(p))
    weight = p - lo
    return (
        np.take(grid, lo, axis=dimension) * (1 - weight)
        + np.take(grid, hi, axis=dimension) * weight
    )


def projection_basis(section):
    """One deterministic PCA frame from the reference method for all methods."""
    points = section.reshape(-1, 3)
    origin = points.mean(axis=0)
    _, _, vectors = np.linalg.svd(points - origin, full_matrices=False)
    basis = vectors[:2].T.copy()
    for column in range(2):
        if basis[np.argmax(np.abs(basis[:, column])), column] < 0:
            basis[:, column] *= -1
    return origin, basis


def quad_cells(section):
    rows, cols = section.shape[:2]
    ids = np.arange(rows * cols).reshape(rows, cols)
    return np.stack(
        (ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]), axis=-1
    ).reshape(-1, 4)


def worst_index(values, metric, mode):
    values = np.asarray(values)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        raise ValueError(f"No finite values for {metric}")
    higher = SURFACE_HIGHER_IS_BETTER if mode == "slice" else HIGHER
    if metric == "cell_area":
        return int(
            finite[np.argmax(np.abs(values[finite] - np.mean(values[finite])))]
        )
    operation = np.argmin if metric in higher else np.argmax
    return int(finite[operation(values[finite])])


def select_cells(cell_shape, vertices, cells, fields, reference, opts):
    """cell_shape is row-major (j,i) or (k,j,i); ranges are half-open."""
    dims = len(cell_shape)
    range_keys = ("i_range", "j_range", "k_range")[:dims]
    limit_keys = ("xlim", "ylim", "zlim")[:dims]
    count = int(np.prod(cell_shape))
    coordinates = np.indices(cell_shape).reshape(dims, -1)
    if opts.get(limit_keys[0]) is not None:
        box = np.asarray([opts[k] for k in limit_keys], dtype=float)
        selected = {}
        for method, points in vertices.items():
            polygons = points[cells]
            # Retain complete cells whose axis-aligned bounding boxes intersect.
            mask = (
                (polygons.max(axis=1) >= box[:, 0])
                & (polygons.min(axis=1) <= box[:, 1])
            ).all(axis=1)
            selected[method] = mask
        selection = {
            "kind": "physical_bounding_box",
            "bounds": box.tolist(),
            "rule": "retain complete cells with intersecting axis-aligned bounding boxes",
        }
    else:
        cell_index = opts.get("cell_index")
        if opts.get("center_on") == "worst":
            cell_index = worst_index(
                fields[reference][opts["metric"]], opts["metric"], opts["mode"]
            )
        if cell_index is not None:
            if int(cell_index) != cell_index or not 0 <= cell_index < count:
                raise ValueError(f"cell_index must be in [0, {count})")
            center = np.unravel_index(int(cell_index), cell_shape)
            pad = int(opts["cell_padding"])
            ranges = [
                (max(0, c - pad), min(n, c + pad + 1))
                for c, n in zip(center, cell_shape)
            ]
        elif opts.get(range_keys[0]) is not None:
            ranges = [tuple(map(int, opts[key])) for key in reversed(range_keys)]
        else:
            ranges = [(n // 4, max(n // 4 + 1, (3 * n + 3) // 4)) for n in cell_shape]
        for (low, high), n in zip(ranges, cell_shape):
            if not 0 <= low < high <= n:
                raise ValueError(f"Cell range [{low}, {high}) is outside [0, {n})")
        mask = np.ones(count, dtype=bool)
        for coordinate, (low, high) in zip(coordinates, ranges):
            mask &= (coordinate >= low) & (coordinate < high)
        selected = {method: mask.copy() for method in vertices}
        selection = {
            "kind": "logical_cell_ranges",
            "ranges_ijk": [[int(a), int(b)] for a, b in reversed(ranges)],
            "reference_method": reference,
            "reference_cell_index": cell_index,
            "index_convention": "zero-based half-open [start,end)",
        }
    if opts.get("cut_axis"):
        dimension = {"x": 2, "y": 1, "z": 0}[opts["cut_axis"]]
        cut = (coordinates[dimension] + 0.5) / cell_shape[dimension] <= opts[
            "cut_position"
        ]
        for method in selected:
            selected[method] &= cut
        selection["cut"] = {
            "logical_axis": opts["cut_axis"],
            "keep_centers_at_or_below": opts["cut_position"],
        }
    for method, mask in selected.items():
        if not mask.any():
            raise ValueError(f"The selected region has no cells for {method}")
    return {m: np.flatnonzero(mask) for m, mask in selected.items()}, selection


def exterior_faces(cells):
    faces = cells[:, FACES].reshape(-1, 4)
    owners = np.repeat(np.arange(len(cells)), 6)
    _, inverse, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    keep = counts[inverse] == 1
    return faces[keep], owners[keep]


def shared_limits(vertices, cells, selected=None):
    points = np.concatenate(
        [
            v if selected is None else v[np.unique(cells[selected[m]])]
            for m, v in vertices.items()
        ]
    )
    return points.min(axis=0), points.max(axis=0)


def draw_volume(axis, vertices, cells, values, limits, opts, norm, cmap, quality):
    if quality and opts["volume_style"] == "wireframe":
        edge_ids, inverse = np.unique(
            np.sort(cells[:, EDGES].reshape(-1, 2), axis=1), axis=0, return_inverse=True
        )
        totals = np.zeros(len(edge_ids))
        counts = np.zeros(len(edge_ids))
        np.add.at(totals, inverse, np.repeat(values, len(EDGES)))
        np.add.at(counts, inverse, 1)
        axis.add_collection3d(
            Line3DCollection(
                vertices[edge_ids], colors=cmap(norm(totals / counts)), linewidths=0.8
            )
        )
    elif quality:
        if opts["volume_style"] == "transparent":
            faces = cells[:, FACES].reshape(-1, 4)
            owners = np.repeat(np.arange(len(cells)), 6)
        else:
            faces, owners = exterior_faces(cells)
        artist = Poly3DCollection(
            vertices[faces], linewidths=0.25, edgecolors="#263238"
        )
        artist.set_facecolor(cmap(norm(values[owners])))
        if opts["volume_style"] == "transparent":
            artist.set_alpha(opts["alpha"])
        axis.add_collection3d(artist)
    if not quality:
        edge_ids = np.unique(np.sort(cells[:, EDGES].reshape(-1, 2), axis=1), axis=0)
        axis.add_collection3d(
            Line3DCollection(vertices[edge_ids], colors="#263238", linewidths=0.45)
        )
    low, high = limits
    extent = high - low
    margin = max(float(extent.max()), 1e-9) * 0.05
    for fn, a, b in zip((axis.set_xlim, axis.set_ylim, axis.set_zlim), low, high):
        fn(a - margin, b + margin)
    axis.set_box_aspect(np.maximum(extent, max(float(extent.max()), 1e-9) * 0.05))
    axis.view_init(elev=opts["elevation"], azim=opts["azimuth"])
    axis.set_proj_type("ortho")
    if opts["show_axes"]:
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        for coordinate in (axis.xaxis, axis.yaxis, axis.zaxis):
            coordinate.set_major_locator(MaxNLocator(nbins=4))
        axis.tick_params(labelsize=7)
    else:
        axis.set_axis_off()


def draw_section(axis, vertices, cells, values, limits, opts, norm, cmap, quality):
    polygons = vertices[cells]
    artist = PolyCollection(
        polygons,
        edgecolors="#263238",
        linewidths=0.4,
        facecolors=cmap(norm(values)) if quality else "none",
    )
    axis.add_collection(artist)
    low, high = limits
    margin = max(float((high - low).max()), 1e-9) * 0.04
    axis.set_xlim(low[0] - margin, high[0] + margin)
    axis.set_ylim(low[1] - margin, high[1] + margin)
    axis.set_aspect("equal", adjustable="box")
    if opts["show_axes"]:
        labels = (
            {"x_mid": ("Y", "Z"), "y_mid": ("X", "Z"), "z_mid": ("X", "Y")}[
                opts["slice"]
            ]
            if opts["slice_coordinates"] == "physical"
            else ("Projected U", "Projected V")
        )
        axis.set_xlabel(labels[0])
        axis.set_ylabel(labels[1])
    else:
        axis.set_axis_off()


def draw_selection(axis, bounds, volume):
    low, high = bounds
    if not volume:
        axis.add_patch(
            Rectangle(
                low, *(high - low), fill=False, edgecolor="#d62728", linewidth=1.8
            )
        )
    else:
        corners = np.array(
            [
                [low[0], low[1], low[2]],
                [high[0], low[1], low[2]],
                [high[0], high[1], low[2]],
                [low[0], high[1], low[2]],
                [low[0], low[1], high[2]],
                [high[0], low[1], high[2]],
                [high[0], high[1], high[2]],
                [low[0], high[1], high[2]],
            ]
        )
        # Overlay the projected selection box so opaque front faces cannot hide it.
        x, y, _ = proj3d.proj_transform(*corners.T, axis.get_proj())
        for start, end in EDGES:
            line = Line2D(
                x[[start, end]],
                y[[start, end]],
                color="#d62728",
                linewidth=1.4,
                linestyle="--",
                transform=axis.transData,
                zorder=10000,
            )
            line.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2.4, foreground="white"),
                    path_effects.Normal(),
                ]
            )
            axis.add_artist(line)


def render_figures(
    output, vertices, cells, fields, selected, opts, limits, local_limits, curved
):
    names = ordered_methods(vertices)
    metric = opts["metric"]
    values = (
        np.concatenate(
            [
                (
                    fields[m][metric][selected[m]]
                    if opts["color_scale"] == "local"
                    else fields[m][metric]
                )
                for m in names
            ]
        )
        if opts["plot"] != "mesh"
        else np.array([0.0, 1.0])
    )
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError(f"No finite {metric} values")
    cmap, norm = quality_plot_style(metric, values)
    low, high = norm.vmin, norm.vmax
    cmap = cmap.copy()
    cmap.set_bad("#888888")
    files = []
    if not opts["save_plots"] or not (opts["save_png"] or opts["save_pdf"]):
        return files, [low, high]
    kinds = ("mesh", "quality") if opts["plot"] == "both" else (opts["plot"],)
    volume = opts["mode"] == "volume"
    for kind in kinds:
        for layout in ("overview", "zoom", "overview_and_zoom"):
            rows = 2 if layout == "overview_and_zoom" else 1
            columns = len(names)
            figure = plt.figure(
                figsize=(5.0 * columns, 4.2 * rows), constrained_layout=True
            )
            axes = []
            for row in range(rows):
                for column, method in enumerate(names):
                    zoom = layout == "zoom" or (rows == 2 and row == 1)
                    axis = figure.add_subplot(
                        rows,
                        columns,
                        row * columns + column + 1,
                        projection="3d" if volume else None,
                    )
                    axes.append(axis)
                    indices = selected[method] if zoom else np.arange(len(cells))
                    bounds = local_limits if zoom else limits
                    drawer = draw_volume if volume else draw_section
                    drawer(
                        axis,
                        vertices[method],
                        cells[indices],
                        fields[method][metric][indices],
                        bounds,
                        opts,
                        norm,
                        cmap,
                        kind == "quality",
                    )
                    if zoom and not volume:
                        # Clip polygons at the requested rectangle, without cell-sized expansion.
                        axis.set_xlim(bounds[0][0], bounds[1][0])
                        axis.set_ylim(bounds[0][1], bounds[1][1])
                    if not zoom:
                        draw_selection(axis, local_limits, volume)
                    axis.set_title(
                        f"{method_label(method)} | {'local' if zoom else 'overview'}",
                        fontsize=10,
                    )
            title = (
                "3-D volume"
                if volume
                else f"{opts['slice'][0]} section: "
                + (
                    "curved surface projected to 2-D"
                    if curved
                    else "physical plane in 2-D"
                )
            )
            figure.suptitle(
                f"{title} | {metric if kind == 'quality' else 'mesh'}", fontsize=12
            )
            if kind == "quality":
                figure.colorbar(
                    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=axes,
                    shrink=0.7,
                    pad=0.03,
                    label=f"{metric} ({opts['color_scale']} scale)",
                )
            for extension in ("png", "pdf"):
                if opts[f"save_{extension}"]:
                    filename = output / f"{kind}_{layout}.{extension}"
                    figure.savefig(filename, dpi=200)
                    files.append(filename.name)
            plt.close(figure)
    return files, [low, high]


def load_meshes(directory):
    directory = Path(directory).resolve()
    comparison = directory / "comparison_mesh_data.npz"
    if comparison.exists():
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(comparison) as archive:
            names = ordered_methods(metadata.get("method_order", archive.files))
            meshes = {m: archive[m].copy() for m in names}
        return meshes, tuple(metadata["grid_shape"]), [comparison, metadata_path]
    source = directory / "mesh_data.npz"
    with np.load(source) as archive:
        shape = tuple(int(archive[k]) for k in ("nx", "ny", "nz"))
        method = next(
            (
                p.name
                for p in [directory, *directory.parents]
                if p.name
                in ("HMONet", "MeshONet", "PI-MeshONet", "PI-HMONet", "TFI", "PDE")
            ),
            directory.name,
        )
        return {method: archive["vertices"].copy()}, shape, [source]


def generate_zoom(directory, options=None):
    directory = Path(directory).resolve()
    opts = normalize_options(options or {})
    meshes, shape, sources = load_meshes(directory)
    if opts.get("methods"):
        unknown = set(opts["methods"]) - set(meshes)
        if unknown:
            raise ValueError(f"Missing methods: {sorted(unknown)}")
        meshes = {m: meshes[m] for m in ordered_methods(opts["methods"])}
    reference = opts.get("reference") or (
        "PDE" if "PDE" in meshes else next(iter(meshes))
    )
    if reference not in meshes:
        raise ValueError(f"Missing reference method: {reference}")
    opts["reference"] = reference
    for key in ("output_dir",):
        if opts.get(key) is not None:
            opts[key] = str(opts[key])
    signature_options = {k: v for k, v in opts.items() if k != "force"}
    signature = hashlib.sha256(
        json.dumps(
            {"version": VERSION, "options": signature_options}, sort_keys=True
        ).encode()
    )
    for source in sources:
        signature.update(source.read_bytes())
    digest = signature.hexdigest()
    category = (
        Path("slices") / opts["slice"] if opts["mode"] == "slice" else Path("volume")
    )
    output = (
        Path(opts["output_dir"])
        if opts.get("output_dir")
        else directory / "zoom_views" / category / (opts.get("name") or opts["metric"])
    )
    marker = output / "zoom_metadata.json"
    nx, ny, nz = shape
    grids = {m: np.asarray(v).reshape(nz, ny, nx, 3) for m, v in meshes.items()}
    if not all(np.isfinite(g).all() for g in grids.values()):
        raise ValueError("Meshes contain nonfinite coordinates")
    frame = None
    curved = False
    needs_quality = opts["plot"] != "mesh" or opts.get("center_on") == "worst"
    requested_metrics = [opts["metric"]] if needs_quality else []
    if opts["mode"] == "slice":
        surfaces = {
            m: extract_section(g, opts["slice"][0], opts["position"], opts.get("index"))
            for m, g in grids.items()
        }
        origin, basis = projection_basis(surfaces[reference])
        normal = np.cross(basis[:, 0], basis[:, 1])
        scale = max(
            float(np.ptp(surfaces[reference].reshape(-1, 3), axis=0).max()), 1e-12
        )
        deviation = max(
            float(np.abs((s.reshape(-1, 3) - origin) @ normal).max())
            for s in surfaces.values()
        )
        curved = deviation > scale * 1e-6
        frame = {
            "origin": origin.tolist(),
            "basis": basis.tolist(),
            "reference": reference,
            "curved_projection": bool(curved),
            "maximum_plane_deviation": deviation,
        }
        physical = {m: s.reshape(-1, 3) for m, s in surfaces.items()}
        if opts["slice_coordinates"] == "physical":
            components = {"x_mid": [1, 2], "y_mid": [0, 2], "z_mid": [0, 1]}[
                opts["slice"]
            ]
            vertices = {m: v[:, components] for m, v in physical.items()}
            frame["display_coordinates"] = "physical"
            frame["display_components"] = components
        else:
            vertices = {m: (v - origin) @ basis for m, v in physical.items()}
            frame["display_coordinates"] = "projected"
        cells = quad_cells(surfaces[reference])
        cell_shape = tuple(v - 1 for v in surfaces[reference].shape[:2])
        fields = {
            m: (
                surface_quad_quality_fields(s, metrics=requested_metrics)
                if needs_quality
                else {}
            )
            for m, s in surfaces.items()
        }
    else:
        physical = meshes
        vertices = meshes
        cells = HexMeshXYZ(nx, ny, nz).cells
        cell_shape = (nz - 1, ny - 1, nx - 1)
        fields = {
            m: (
                evaluate_hexahedral_grid_quality(g, metrics=requested_metrics)
                if needs_quality
                else {}
            )
            for m, g in grids.items()
        }
    if needs_quality and any(opts["metric"] not in f for f in fields.values()):
        raise ValueError(
            f"Unknown {opts['mode']} metric {opts['metric']!r}; choose {sorted(next(iter(fields.values())))}"
        )
    selected, selection = select_cells(
        cell_shape, vertices, cells, fields, reference, opts
    )
    limits = shared_limits(vertices, cells)
    local_limits = shared_limits(vertices, cells, selected)
    if opts["mode"] == "slice" and opts.get("xlim") is not None:
        box = np.asarray([opts["xlim"], opts["ylim"]], dtype=float)
        local_limits = (box[:, 0], box[:, 1])
        selection["display_rule"] = (
            "clip to exact rectangle; statistics retain complete intersecting cells"
        )
    output.mkdir(parents=True, exist_ok=True)
    summaries = {
        m: summarize_fields({k: v[selected[m]] for k, v in values.items()})
        for m, values in fields.items()
    }
    files = [
        "local_quality_summary.json",
        "local_quality_comparison.csv",
        "local_quality_cells.csv",
        "local_mesh_data.npz",
    ]
    if needs_quality:
        (output / files[0]).write_text(
            json.dumps(summaries, indent=2), encoding="utf-8"
        )
        with (output / files[1]).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "method",
                    "metric",
                    "worst",
                    "worst_direction",
                    "mean",
                    "affected_cell_count",
                ]
            )
            for m, values in summaries.items():
                for metric, summary in values.items():
                    writer.writerow(
                        [
                            m,
                            metric,
                            summary.get("maximum", summary.get("minimum", "")),
                            "maximum" if "maximum" in summary else "minimum",
                            summary.get("mean", ""),
                            summary.get("affected_cell_count", ""),
                        ]
                    )
        metrics = list(next(iter(fields.values())))
        with (output / files[2]).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["method", "original_cell_index", *metrics])
            for m, indices in selected.items():
                for index in indices:
                    writer.writerow(
                        [m, int(index), *[fields[m][k][index] for k in metrics]]
                    )
    arrays = {}
    for m, indices in selected.items():
        node_ids, inverse = np.unique(cells[indices], return_inverse=True)
        arrays.update(
            {
                f"{m}__vertices": physical[m][node_ids],
                f"{m}__cells": inverse.reshape(-1, cells.shape[1]),
                f"{m}__original_cell_indices": indices,
                f"{m}__original_node_indices": node_ids,
            }
        )
    np.savez_compressed(output / files[3], **arrays)
    if not needs_quality:
        # Avoid empty quality tables and stale quality figures in mesh-only views.
        files = ["local_mesh_data.npz"]
    plot_files, color_limits = render_figures(
        output, vertices, cells, fields, selected, opts, limits, local_limits, curved
    )
    files += plot_files
    previous = (
        json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
    )
    for name in set(previous.get("files", [])) - set(files):
        stale = (output / name).resolve()
        if not stale.is_relative_to(output.resolve()):
            raise ValueError("Invalid path in zoom metadata")
        if stale.is_file():
            stale.unlink()
    record = {
        "version": VERSION,
        "signature": digest,
        "sources": [str(p) for p in sources],
        "grid_shape": list(shape),
        "options": opts,
        "selection": selection,
        "projection": frame,
        "selected_cell_counts": {m: len(v) for m, v in selected.items()},
        "quality_source": (
            "computed from original physical mesh coordinates"
            if needs_quality
            else "disabled"
        ),
        "global_limits": [v.tolist() for v in limits],
        "local_limits": [v.tolist() for v in local_limits],
        "color_limits": color_limits,
        "files": files,
    }
    marker.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return output


def run_configured_zooms(
    comparison_root, jobs, *, save_plots=True, save_png=True, save_pdf=False
):
    if not isinstance(jobs, list):
        raise ValueError("evaluation.zoom_views must be a list")
    outputs = []
    for raw in jobs:
        job = dict(raw)
        if not job.pop("enabled", True):
            continue
        groups = job.pop("groups", None)
        job["save_plots"] = save_plots and job.get("save_plots", True)
        job.setdefault("save_png", save_png)
        job.setdefault("save_pdf", save_pdf)
        normalize_options(job)
        directories = sorted(Path(comparison_root).glob("*/comparison_mesh_data.npz"))
        if groups:
            missing = set(groups) - {p.parent.name for p in directories}
            if missing:
                raise ValueError(
                    f"Zoom comparison groups unavailable: {sorted(missing)}"
                )
        for source in directories:
            if groups is None or source.parent.name in groups:
                outputs.append(generate_zoom(source.parent, job))
    return outputs
