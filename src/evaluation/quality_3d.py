"""Cached 3-D quality analysis and shared-scale surface/slice figures."""

from __future__ import annotations
import csv
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from src.utils.metrics import evaluate_hexahedral_grid_quality
from src.utils.quality_selection import QUAD_METRICS, metric_names
from .method_order import ordered_methods, method_label

HIGHER = {
    "minimum_scaled_jacobian",
    "center_jacobian",
    "min_angle",
    "orthogonality",
}
VOLUME_QUALITY_CACHE_VERSION = 2
SURFACE_QUALITY_CACHE_VERSION = 3
SURFACE_NAMES = ("x_mid", "y_mid", "z_mid")
SURFACE_HIGHER_IS_BETTER = {
    "min_angle",
    "jacobian",
    "scaled_jacobian",
    "orthogonality",
}
SURFACE_AREA_CMAP = LinearSegmentedColormap.from_list(
    "surface_area_quality", ["#d7191c", "#2166ac", "#d7191c"]
)
SURFACE_METRIC_DEFINITIONS = {
    "max_angle": "Maximum physical corner angle of each surface quadrilateral (degrees).",
    "min_angle": "Minimum physical corner angle of each surface quadrilateral (degrees).",
    "jacobian": "Minimum oriented physical corner area scale relative to the average surface normal.",
    "scaled_jacobian": "Minimum corner Jacobian divided by adjacent edge lengths.",
    "skewness": "Maximum normalized deviation of corner angles from 90 degrees.",
    "aspect_ratio": "Longest divided by shortest physical edge length.",
    "orthogonality": "One minus the largest absolute cosine between adjacent edges.",
    "cell_area": "Physical quadrilateral area computed from two triangles.",
    "concave_vertex_count": "Number of reflex corner angles greater than 180 degrees.",
    "self_intersection": "Whether opposite edges intersect in a stable local tangent projection.",
    "intersecting_cell": "Alias of the surface self-intersection indicator.",
    "flipping_cell": "Whether the minimum oriented corner Jacobian is non-positive.",
    "degenerate_cell": "Whether area or an edge length is at or below tolerance.",
}


def boundary_surface_grids(grid):
    """Extract six structured boundary surfaces as ``(rows, columns, xyz)``."""
    grid = np.asarray(grid, dtype=float)
    return {
        "x_min": grid[:, :, 0, :],
        "x_max": grid[:, :, -1, :],
        "y_min": grid[:, 0, :, :],
        "y_max": grid[:, -1, :, :],
        "z_min": grid[0, :, :, :],
        "z_max": grid[-1, :, :, :],
    }


def middle_surface_grids(grid):
    """Physical surfaces at logical xi/eta/zeta=0.5; interpolate even grids."""
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 4 or grid.shape[-1] != 3 or min(grid.shape[:3]) < 2:
        raise ValueError("grid must have shape (nz>=2, ny>=2, nx>=2, 3)")
    result = {}
    for name, axis in zip(SURFACE_NAMES, (2, 1, 0)):
        position = (grid.shape[axis] - 1) / 2
        lo, hi = int(np.floor(position)), int(np.ceil(position))
        result[name] = (np.take(grid, lo, axis=axis) + np.take(grid, hi, axis=axis)) / 2
    return result


def _segments_intersect_2d(a, b, c, d, tolerance=1e-12):
    def orientation(p, q, r):
        first = q - p
        second = r - p
        return first[0] * second[1] - first[1] * second[0]

    first = orientation(a, b, c) * orientation(a, b, d)
    second = orientation(c, d, a) * orientation(c, d, b)
    return first < -tolerance and second < -tolerance


def surface_quad_quality_fields(surface, tolerance=1e-12, metrics=None):
    """Compute the complete 2-D metric set intrinsically on a 3-D surface.

    Jacobian is the oriented corner area scale relative to the surface's
    average normal. Angles and lengths use physical 3-D vectors. Intersection
    is checked after projection to a stable local tangent plane.
    """
    surface = np.asarray(surface, dtype=float)
    rows, columns, coordinate_dimension = surface.shape
    if coordinate_dimension != 3 or min(rows, columns) < 2:
        raise ValueError("surface must have shape (rows>=2, columns>=2, 3)")

    selected = set(metric_names(metrics, QUAD_METRICS))
    if not selected:
        return {}
    need_angles = bool(
        selected
        & {
            "max_angle",
            "min_angle",
            "skewness",
            "orthogonality",
            "concave_vertex_count",
        }
    )
    need_jac = need_angles or bool(
        selected & {"jacobian", "scaled_jacobian", "flipping_cell"}
    )
    need_scaled = "scaled_jacobian" in selected
    need_lengths = bool(selected & {"aspect_ratio", "degenerate_cell"})
    need_area = bool(selected & {"cell_area", "degenerate_cell"})
    need_intersection = bool(selected & {"self_intersection", "intersecting_cell"})
    cells = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            cells.append(
                np.stack(
                    (
                        surface[row, column],
                        surface[row, column + 1],
                        surface[row + 1, column + 1],
                        surface[row + 1, column],
                    )
                )
            )
    cells = np.asarray(cells)
    # A curved or closed surface (for example a cylinder) has no single valid
    # global normal: its cell normals can cancel to nearly zero. Construct one
    # oriented normal for every quad from its counter-clockwise vertex loop.
    # This prevents valid convex cells on the opposite side of a curved
    # surface from being interpreted as 360-angle reflex cells.
    if need_jac or need_intersection:
        cell_normals = np.cross(cells[:, 1] - cells[:, 0], cells[:, 2] - cells[:, 0])
        cell_normals += np.cross(cells[:, 2] - cells[:, 0], cells[:, 3] - cells[:, 0])
        normal_lengths = np.linalg.norm(cell_normals, axis=1)
        fallback_normals = np.cross(
            cells[:, 1] - cells[:, 0], cells[:, 3] - cells[:, 0]
        )
        use_fallback = normal_lengths <= tolerance
        cell_normals[use_fallback] = fallback_normals[use_fallback]
        normal_lengths = np.linalg.norm(cell_normals, axis=1)
        cell_normals = cell_normals / np.maximum(normal_lengths[:, None], tolerance)

    else:
        cell_normals = np.zeros((len(cells), 3))

    angles = []
    corner_jacobians = []
    corner_scaled_jacobians = []
    edge_lengths = []
    self_intersections = []
    for cell, reference_normal in zip(cells, cell_normals):
        cell_angles = []
        cell_jacobians = []
        cell_scaled = []
        if need_jac:
            for corner in range(4):
                current = cell[corner]
                next_vector = cell[(corner + 1) % 4] - current
                previous_vector = cell[(corner - 1) % 4] - current
                if need_angles or need_scaled:
                    denominator = np.linalg.norm(next_vector) * np.linalg.norm(
                        previous_vector
                    )
                    if need_angles:
                        cosine = np.dot(next_vector, previous_vector) / max(
                            denominator, tolerance
                        )
                        base_angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                signed_jacobian = float(
                    np.dot(np.cross(next_vector, previous_vector), reference_normal)
                )
                if need_angles:
                    cell_angles.append(
                        base_angle if signed_jacobian >= 0 else 360.0 - base_angle
                    )
                cell_jacobians.append(signed_jacobian)
                if need_scaled:
                    cell_scaled.append(signed_jacobian / max(denominator, tolerance))
        if need_lengths:
            lengths = np.linalg.norm(np.roll(cell, -1, axis=0) - cell, axis=1)
        angles.append(cell_angles)
        corner_jacobians.append(cell_jacobians)
        corner_scaled_jacobians.append(cell_scaled)
        if need_lengths:
            edge_lengths.append(lengths)

        if need_intersection:
            tangent_x = cell[1] - cell[0]
            tangent_x /= max(np.linalg.norm(tangent_x), tolerance)
            tangent_y = np.cross(reference_normal, tangent_x)
            projected = np.stack((cell @ tangent_x, cell @ tangent_y), axis=-1)
            self_intersections.append(
                _segments_intersect_2d(
                    projected[0], projected[1], projected[2], projected[3]
                )
                or _segments_intersect_2d(
                    projected[1], projected[2], projected[3], projected[0]
                )
            )

    if need_angles:
        angles = np.asarray(angles)
    if selected & {"jacobian", "flipping_cell"}:
        minimum_jacobian = np.asarray(corner_jacobians).min(axis=1)
    if need_scaled:
        corner_scaled_jacobians = np.asarray(corner_scaled_jacobians)
    if need_lengths:
        edge_lengths = np.asarray(edge_lengths)
    if need_area:
        cell_area = 0.5 * (
            np.linalg.norm(
                np.cross(cells[:, 1] - cells[:, 0], cells[:, 2] - cells[:, 0]), axis=1
            )
            + np.linalg.norm(
                np.cross(cells[:, 2] - cells[:, 0], cells[:, 3] - cells[:, 0]), axis=1
            )
        )
    if need_intersection:
        self_intersections = np.asarray(self_intersections, dtype=float)
    calculations = {
        "max_angle": lambda: angles.max(axis=1),
        "min_angle": lambda: angles.min(axis=1),
        "jacobian": lambda: minimum_jacobian,
        "scaled_jacobian": lambda: corner_scaled_jacobians.min(axis=1),
        "skewness": lambda: np.max(np.abs(angles - 90.0) / 90.0, axis=1),
        "aspect_ratio": lambda: edge_lengths.max(axis=1)
        / np.maximum(edge_lengths.min(axis=1), tolerance),
        "orthogonality": lambda: 1.0
        - np.max(np.abs(np.cos(np.radians(angles))), axis=1),
        "cell_area": lambda: cell_area,
        "concave_vertex_count": lambda: np.count_nonzero(angles > 180.0, axis=1).astype(
            float
        ),
        "self_intersection": lambda: self_intersections,
        "intersecting_cell": lambda: self_intersections.copy(),
        "flipping_cell": lambda: (minimum_jacobian <= 0).astype(float),
        "degenerate_cell": lambda: (
            (cell_area <= tolerance)
            | (edge_lengths.min(axis=1) <= tolerance)
            | ~np.isfinite(cells).all(axis=(1, 2))
        ).astype(float),
    }
    return {name: calculations[name]() for name in QUAD_METRICS if name in selected}


def compute_all_surface_quality(grid, *, boundary=False):
    """Return intrinsic quadrilateral quality on the three middle sections."""
    return {
        name: surface_quad_quality_fields(surface)
        for name, surface in (
            boundary_surface_grids(grid) if boundary else middle_surface_grids(grid)
        ).items()
    }


def load_or_compute_surface_analysis(path, meshes, nx, ny, nz, signature, reuse=True):
    """Load or compute cached intrinsic quality of the three middle sections."""
    metadata_path = path / "surface_quality_cache.json"
    archive_path = path / "surface_quality_cache.npz"
    if reuse and metadata_path.exists() and archive_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("signature") == signature
                and metadata.get("quality_version") == SURFACE_QUALITY_CACHE_VERSION
            ):
                archive = np.load(archive_path)
                result = {method: {} for method in meshes}
                for key in archive.files:
                    method, surface, metric = key.split("__", 2)
                    result.setdefault(method, {}).setdefault(surface, {})[metric] = (
                        archive[key]
                    )
                if all(
                    set(result.get(method, {})) == set(SURFACE_NAMES)
                    for method in meshes
                ):
                    return result, True
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    result = {
        method: compute_all_surface_quality(np.asarray(mesh).reshape(nz, ny, nx, 3))
        for method, mesh in meshes.items()
    }
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        archive_path,
        **{
            f"{method}__{surface}__{metric}": values
            for method, surfaces in result.items()
            for surface, fields in surfaces.items()
            for metric, values in fields.items()
        },
    )
    metadata_path.write_text(
        json.dumps(
            {
                "signature": signature,
                "shape": [nx, ny, nz],
                "quality_version": SURFACE_QUALITY_CACHE_VERSION,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result, False


def save_surface_quality_data(path, surface_fields):
    """Save per-cell NPZ/CSV and summaries for every middle section."""
    from src.evaluation.quality_output import summarize_fields

    path.mkdir(parents=True, exist_ok=True)
    (path / "metric_definitions.json").write_text(
        json.dumps(SURFACE_METRIC_DEFINITIONS, indent=2), encoding="utf-8"
    )
    for surface_name, fields in surface_fields.items():
        surface_path = path / surface_name
        surface_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(surface_path / "quality_data.npz", **fields)
        metrics = list(fields)
        with (surface_path / "quality_data.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["surface_cell_index", *metrics])
            for index in range(len(fields[metrics[0]])):
                writer.writerow([index, *[fields[name][index] for name in metrics]])
        (surface_path / "quality_summary.json").write_text(
            json.dumps(summarize_fields(fields), indent=2), encoding="utf-8"
        )


def _surface_metric_limits(method_surface_fields, metric):
    values = np.concatenate(
        [
            fields[metric]
            for surfaces in method_surface_fields.values()
            for fields in surfaces.values()
        ]
    )
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0, 1.0
    lower, upper = float(finite.min()), float(finite.max())
    if lower == upper:
        upper = lower + max(abs(lower) * 0.01, 1e-12)
    return lower, upper


def _surface_camera(surface):
    """Return a stable camera derived from one shared reference surface."""
    coordinates = np.asarray(surface["_coordinates"], dtype=float)
    edge_u = coordinates[:-1, 1:] - coordinates[:-1, :-1]
    edge_v = coordinates[1:, :-1] - coordinates[:-1, :-1]
    normal = np.cross(edge_u, edge_v).reshape(-1, 3).mean(axis=0)
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 1e-12:
        return 24.0, -58.0
    normal /= normal_norm
    tangent = edge_u.reshape(-1, 3).mean(axis=0)
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm > 1e-12:
        normal = normal + 0.28 * tangent / tangent_norm
        normal /= np.linalg.norm(normal)
    elevation = np.degrees(np.arcsin(np.clip(normal[2], -1, 1)))
    azimuth = np.degrees(np.arctan2(normal[1], normal[0]))
    return float(elevation), float(azimuth)


def _shared_surface_geometry(method_surface_fields, surface_name):
    """Use identical camera, limits, and aspect for every method in one face."""
    surfaces = [fields[surface_name] for fields in method_surface_fields.values()]
    points = np.concatenate(
        [np.asarray(surface["_coordinates"]).reshape(-1, 3) for surface in surfaces],
        axis=0,
    )
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    extent = maximum - minimum
    largest = max(float(extent.max()), 1e-12)
    margin = 0.04 * largest
    limits = (minimum - margin, maximum + margin)
    aspect = np.maximum(extent, 0.08 * largest)
    reference_name = (
        "PDE" if "PDE" in method_surface_fields else next(iter(method_surface_fields))
    )
    camera = _surface_camera(method_surface_fields[reference_name][surface_name])
    return limits, aspect, camera


def _draw_physical_surface_quality(
    axis,
    surface,
    metric,
    colormap,
    normalization,
    shared_geometry=None,
):
    """Draw cell values on the real curved 3-D quadrilateral surface."""
    coordinates = np.asarray(surface["_coordinates"], dtype=float)
    polygons = np.stack(
        (
            coordinates[:-1, :-1],
            coordinates[:-1, 1:],
            coordinates[1:, 1:],
            coordinates[1:, :-1],
        ),
        axis=2,
    ).reshape(-1, 4, 3)
    values = np.asarray(surface[metric]).reshape(-1)
    collection = Poly3DCollection(
        polygons,
        facecolors=colormap(normalization(values)),
        edgecolors="black",
        linewidths=0.28,
        antialiased=True,
    )
    collection.set_array(values)
    collection.set_cmap(colormap)
    collection.set_norm(normalization)
    axis.add_collection3d(collection)

    if shared_geometry is None:
        flat = coordinates.reshape(-1, 3)
        minimum, maximum = flat.min(axis=0), flat.max(axis=0)
        extent = maximum - minimum
        largest = max(float(extent.max()), 1e-12)
        margin = 0.04 * largest
        limits = (minimum - margin, maximum + margin)
        aspect = np.maximum(extent, 0.08 * largest)
        camera = _surface_camera(surface)
    else:
        limits, aspect, camera = shared_geometry
    axis.set_xlim(limits[0][0], limits[1][0])
    axis.set_ylim(limits[0][1], limits[1][1])
    axis.set_zlim(limits[0][2], limits[1][2])
    axis.set_box_aspect(aspect)
    axis.view_init(elev=camera[0], azim=camera[1])
    axis.set_proj_type("ortho")
    axis.set_axis_off()
    return collection


def plot_boundary_surface_quality_comparison(
    path, method_surface_fields, save_png=True, save_pdf=False
):
    """Plot middle sections with physical cells and shared scales across methods."""
    path.mkdir(parents=True, exist_ok=True)
    save_section_comparison_tables(path, method_surface_fields)
    boundary_faces_directory = path / "boundary_faces"
    boundary_faces_directory.mkdir(parents=True, exist_ok=True)
    (boundary_faces_directory / "metric_definitions.json").write_text(
        json.dumps(
            {
                "coordinate_system": "physical_3d",
                "description": (
                    "Each subdirectory contains multi-method comparisons on "
                    "one physical middle section at logical xi/eta/zeta=0.5. "
                    "Even-sized grids interpolate the two central layers."
                ),
                "surfaces": list(SURFACE_NAMES),
                "metrics": SURFACE_METRIC_DEFINITIONS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    methods = ordered_methods(method_surface_fields)
    shared_geometry = {
        surface_name: _shared_surface_geometry(method_surface_fields, surface_name)
        for surface_name in SURFACE_NAMES
    }
    metrics = [
        name
        for name in next(iter(next(iter(method_surface_fields.values())).values()))
        if not name.startswith("_")
    ]
    for metric in metrics:
        lower, upper = _surface_metric_limits(method_surface_fields, metric)
        if metric == "cell_area":
            combined = np.concatenate(
                [
                    fields[metric]
                    for surfaces in method_surface_fields.values()
                    for fields in surfaces.values()
                ]
            )
            finite = combined[np.isfinite(combined)]
            center = float(np.mean(finite)) if finite.size else (lower + upper) / 2
            center = min(
                max(center, np.nextafter(lower, upper)),
                np.nextafter(upper, lower),
            )
            colormap = SURFACE_AREA_CMAP
            normalization = TwoSlopeNorm(vmin=lower, vcenter=center, vmax=upper)
        else:
            colormap = "jet_r" if metric in SURFACE_HIGHER_IS_BETTER else "jet"
            normalization = Normalize(vmin=lower, vmax=upper)
        figure = plt.figure(figsize=(4.6 * len(methods), 3.8 * len(SURFACE_NAMES)))
        axes = np.empty((len(SURFACE_NAMES), len(methods)), dtype=object)
        colored = None
        for row, surface_name in enumerate(SURFACE_NAMES):
            for column, method in enumerate(methods):
                axis = figure.add_subplot(
                    len(SURFACE_NAMES),
                    len(methods),
                    row * len(methods) + column + 1,
                    projection="3d",
                )
                axes[row, column] = axis
                surface = method_surface_fields[method][surface_name]
                colored = _draw_physical_surface_quality(
                    axis,
                    surface,
                    metric,
                    plt.get_cmap(colormap) if isinstance(colormap, str) else colormap,
                    normalization,
                    shared_geometry[surface_name],
                )
                if row == 0:
                    axis.set_title(method_label(method))
                if column == 0:
                    axis.text2D(
                        -0.08,
                        0.5,
                        surface_name,
                        transform=axis.transAxes,
                        rotation=90,
                        ha="right",
                        va="center",
                    )
        figure.subplots_adjust(right=0.91, wspace=0.08, hspace=0.16)
        colorbar = figure.colorbar(
            colored, ax=axes.ravel().tolist(), fraction=0.012, pad=0.02
        )
        colorbar.set_label(metric)
        if metric == "cell_area":
            colorbar.ax.text(
                -2.1,
                0.5,
                "Better near mean",
                transform=colorbar.ax.transAxes,
                rotation=90,
                ha="center",
                va="center",
            )
        else:
            better_up = metric in SURFACE_HIGHER_IS_BETTER
            colorbar.ax.annotate(
                "Better quality",
                xy=(-2.0, 0.82 if better_up else 0.18),
                xytext=(-2.0, 0.18 if better_up else 0.82),
                xycoords="axes fraction",
                textcoords="axes fraction",
                rotation=90,
                ha="center",
                va="center",
                arrowprops={"arrowstyle": "->", "linewidth": 1.2},
            )
        output = path / f"surface_quality_{metric}"
        if save_png:
            figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
        if save_pdf:
            figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)

        # Save every physical face separately as a paper-ready method
        # comparison. The overview above remains useful for quick inspection.
        for surface_name in SURFACE_NAMES:
            face_directory = boundary_faces_directory / surface_name
            face_directory.mkdir(parents=True, exist_ok=True)
            face_figure = plt.figure(figsize=(4.6 * len(methods) + 0.8, 4.2))
            face_axes = []
            face_colored = None
            for column, method in enumerate(methods):
                axis = face_figure.add_subplot(
                    1, len(methods), column + 1, projection="3d"
                )
                face_axes.append(axis)
                surface = method_surface_fields[method][surface_name]
                face_colored = _draw_physical_surface_quality(
                    axis,
                    surface,
                    metric,
                    plt.get_cmap(colormap) if isinstance(colormap, str) else colormap,
                    normalization,
                    shared_geometry[surface_name],
                )
                axis.set_title(method_label(method))
            face_figure.subplots_adjust(right=0.90, wspace=0.08)
            face_colorbar = face_figure.colorbar(
                face_colored,
                ax=face_axes,
                fraction=0.018,
                pad=0.025,
            )
            face_colorbar.set_label(metric)
            face_figure.suptitle(surface_name)
            face_output = face_directory / f"quality_{metric}"
            if save_png:
                face_figure.savefig(
                    face_output.with_suffix(".png"), dpi=220, bbox_inches="tight"
                )
            if save_pdf:
                face_figure.savefig(
                    face_output.with_suffix(".pdf"), bbox_inches="tight"
                )
            plt.close(face_figure)
    (path / "surface_plot_version.json").write_text(
        json.dumps(
            {
                "version": 6,
                "coordinates": "physical_3d",
                "description": (
                    "Logical middle sections with physical quadrilateral edges; "
                    "individual comparisons are stored by surface below "
                    "boundary_faces/."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def attach_surface_shapes(surface_fields, grid):
    """Attach non-persisted topology sizes used only by the plotter."""
    surfaces = middle_surface_grids(grid)
    result = {}
    for name, fields in surface_fields.items():
        result[name] = dict(fields)
        result[name]["_shape"] = surfaces[name].shape[:2]
        result[name]["_coordinates"] = surfaces[name]
    return result


def save_section_comparison_tables(path, method_surface_fields):
    """Write intrinsic section summaries and paper tables for every method group."""
    from src.evaluation.quality_output import summarize_fields
    from .artifacts import save_quality_table

    for name in SURFACE_NAMES:
        target = path / "boundary_faces" / name
        target.mkdir(parents=True, exist_ok=True)
        summaries = {
            method: summarize_fields(
                {k: v for k, v in surfaces[name].items() if not k.startswith("_")}
            )
            for method, surfaces in method_surface_fields.items()
        }
        (target / "quality_comparison.json").write_text(
            json.dumps(summaries, indent=2), encoding="utf-8"
        )
        save_quality_table(target / "paper_surface_quality_table.csv", summaries)


def fields_for(mesh, nx, ny, nz):
    return evaluate_hexahedral_grid_quality(np.asarray(mesh).reshape(nz, ny, nx, 3))


def load_or_compute_analysis(path: Path, meshes, nx, ny, nz, signature, reuse=True):
    meta = path / "analysis_cache.json"
    archive = path / "analysis_cache.npz"
    if reuse and meta.exists() and archive.exists():
        try:
            metadata = json.loads(meta.read_text(encoding="utf-8"))
            if (
                metadata.get("signature") == signature
                and metadata.get("quality_version") == VOLUME_QUALITY_CACHE_VERSION
            ):
                z = np.load(archive)
                return {
                    m: {
                        k.split("__", 1)[1]: z[k]
                        for k in z.files
                        if k.startswith(m + "__")
                    }
                    for m in meshes
                }, True
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    result = {m: fields_for(v, nx, ny, nz) for m, v in meshes.items()}
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        archive, **{f"{m}__{k}": v for m, d in result.items() for k, v in d.items()}
    )
    meta.write_text(
        json.dumps(
            {
                "signature": signature,
                "shape": [nx, ny, nz],
                "methods": list(meshes),
                "quality_version": VOLUME_QUALITY_CACHE_VERSION,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result, False


def _views(values, nx, ny, nz):
    a = np.asarray(values).reshape(nz - 1, ny - 1, nx - 1)
    return [
        (a[:, :, -1], "Y", "Z"),
        (a[:, -1, :], "X", "Z"),
        (a[-1, :, :], "X", "Y"),
        (a[:, :, a.shape[2] // 2], "Y", "Z"),
        (a[:, a.shape[1] // 2, :], "X", "Z"),
        (a[a.shape[0] // 2, :, :], "X", "Y"),
    ]


def plot_surface_slice_comparison(
    path,
    method_fields,
    nx,
    ny,
    nz,
    save_png=True,
    save_pdf=False,
):
    """Rows are surface/slice views; columns are TFI, PDE, MeshONet, HMONet."""
    path.mkdir(parents=True, exist_ok=True)
    methods = ordered_methods(method_fields)
    metrics = list(next(iter(method_fields.values())))
    titles = [
        "+X surface",
        "+Y surface",
        "+Z surface",
        "X centre slice",
        "Y centre slice",
        "Z centre slice",
    ]
    for metric in metrics:
        allv = np.concatenate([np.asarray(method_fields[m][metric]) for m in methods])
        finite = allv[np.isfinite(allv)]
        lo, hi = (
            (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        )
        if lo == hi:
            hi = lo + 1
        fig, axes = plt.subplots(
            len(titles),
            len(methods),
            figsize=(4.6 * len(methods), 3.2 * len(titles)),
            squeeze=False,
        )
        im = None
        for col, m in enumerate(methods):
            for r, view in enumerate(_views(method_fields[m][metric], nx, ny, nz)):
                ax = axes[r, col]
                img, horizontal, vertical = view
                im = ax.imshow(
                    img,
                    origin="lower",
                    aspect="auto",
                    cmap="jet_r" if metric in HIGHER else "jet",
                    vmin=lo,
                    vmax=hi,
                )
                ax.set_title(method_label(m) if r == 0 else "")
                ax.set_xlabel(horizontal)
                ax.set_ylabel(vertical)
                if col == 0:
                    ax.text(
                        -0.28,
                        0.5,
                        titles[r],
                        transform=ax.transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                    )
        fig.subplots_adjust(right=0.92, wspace=0.2, hspace=0.25)
        cb = fig.colorbar(
            im, ax=axes.ravel().tolist(), fraction=0.012, pad=0.02, label=metric
        )
        cb.ax.annotate(
            "Better",
            xy=(-2, 0.82 if metric in HIGHER else 0.18),
            xytext=(-2, 0.18 if metric in HIGHER else 0.82),
            xycoords="axes fraction",
            textcoords="axes fraction",
            rotation=90,
            arrowprops={"arrowstyle": "->"},
            ha="center",
        )
        output_stem = path / f"quality_{metric}_surfaces_and_slices"
        if save_png:
            fig.savefig(output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
        if save_pdf:
            fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
