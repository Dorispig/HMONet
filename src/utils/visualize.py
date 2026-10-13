"""Documentation string."""

# ============================================================
# Implementation comment.
# ============================================================
import logging
import os
from datetime import datetime

# ============================================================
# Implementation comment.
# ============================================================
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ============================================================
# Implementation comment.
# ============================================================
import src.utils.metrics as metrics
from src.data.mesh.topology import QuadMeshXY

# ============================================================
# Implementation comment.
# ============================================================


__all__ = [
    "plot_boundary",
    "plot_boundary_curve_set_2d",
    "plot_boundary_surface_set_3d",
    "plot_quad_mesh",
    "plot_quad_mesh_colored",
    "plot_quad_meshXY",
    "plot_hex_mesh_3d",
    "plot_hex_mesh_views",
    "plot_hex_mesh_paper",
]


def plot_boundary_curve_set_2d(boundary, nx=21, ny=21, ax=None, show_samples=True):
    """Plot a 2-D parameter boundary using uniform samples including endpoints."""
    ax = ax or plt.gca()
    counts = {"bottom": int(nx), "right": int(ny), "top": int(nx), "left": int(ny)}
    sides = boundary.sample(counts)
    if boundary.parameter_order == "perimeter":
        parts = [sides["bottom"], sides["right"][1:], sides["top"][1:], sides["left"][1:]]
    else:
        parts = [sides["bottom"], sides["right"][1:], sides["top"][::-1][1:], sides["left"][::-1][1:]]
    points = np.vstack(parts)
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    ax.plot(points[:, 0], points[:, 1], color="tab:blue", linewidth=1.3)
    if show_samples:
        ax.scatter(points[:, 0], points[:, 1], s=8, color="tab:red", zorder=3)
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_boundary_surface_set_3d(boundary, nx=21, ny=21, nz=21, ax=None, alpha=0.35):
    """Plot all six 3-D boundary surfaces with endpoint-inclusive uniform grids."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if ax is None:
        ax = plt.figure().add_subplot(111, projection="3d")
    counts = {"x_min": (ny, nz), "x_max": (ny, nz), "y_min": (nx, nz),
              "y_max": (nx, nz), "z_min": (nx, ny), "z_max": (nx, ny)}
    colors = plt.cm.tab10(np.linspace(0, 1, len(boundary.names)))
    for color, name in zip(colors, boundary.names):
        points = boundary.surfaces()[name].sample(*counts[name])
        polygons = []
        for i in range(points.shape[0] - 1):
            for j in range(points.shape[1] - 1):
                polygons.append([points[i, j], points[i + 1, j], points[i + 1, j + 1], points[i, j + 1]])
        ax.add_collection3d(Poly3DCollection(polygons, facecolors=[color], edgecolors="k", linewidths=0.25, alpha=alpha))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    return ax


# ================================ 3D =================================
def plot_hex_mesh_3d(
    points,
    nx,
    ny,
    nz,
    ax=None,
    linewidth=0.22,
    color="0.25",
    show_axes=False,
    limits=None,
):
    """Draw every hexahedral edge with a common orthographic camera."""
    points = np.asarray(points, dtype=float).reshape(nz, ny, nx, 3)
    if ax is None:
        ax = plt.figure().add_subplot(111, projection="3d")

    x_edges = np.stack((points[:, :, :-1], points[:, :, 1:]), axis=-2)
    y_edges = np.stack((points[:, :-1, :], points[:, 1:, :]), axis=-2)
    z_edges = np.stack((points[:-1, :, :], points[1:, :, :]), axis=-2)
    all_edges = np.concatenate(
        [
            x_edges.reshape(-1, 2, 3),
            y_edges.reshape(-1, 2, 3),
            z_edges.reshape(-1, 2, 3),
        ],
        axis=0,
    )
    ax.add_collection3d(
        Line3DCollection(
            all_edges,
            colors=color,
            linewidths=linewidth,
            alpha=0.28,
        )
    )

    boundary_x = np.zeros((nz, ny, nx - 1), dtype=bool)
    boundary_x[[0, -1], :, :] = True
    boundary_x[:, [0, -1], :] = True
    boundary_y = np.zeros((nz, ny - 1, nx), dtype=bool)
    boundary_y[[0, -1], :, :] = True
    boundary_y[:, :, [0, -1]] = True
    boundary_z = np.zeros((nz - 1, ny, nx), dtype=bool)
    boundary_z[:, [0, -1], :] = True
    boundary_z[:, :, [0, -1]] = True
    exterior_edges = np.concatenate(
        [
            x_edges[boundary_x].reshape(-1, 2, 3),
            y_edges[boundary_y].reshape(-1, 2, 3),
            z_edges[boundary_z].reshape(-1, 2, 3),
        ],
        axis=0,
    )
    ax.add_collection3d(
        Line3DCollection(
            exterior_edges,
            colors="0.08",
            linewidths=max(0.45, 1.8 * linewidth),
            alpha=0.88,
        )
    )

    flat = points.reshape(-1, 3)
    if limits is None:
        minimum, maximum = flat.min(axis=0), flat.max(axis=0)
    else:
        minimum, maximum = (np.asarray(value, dtype=float) for value in limits)
    extent = np.maximum(maximum - minimum, np.finfo(float).eps)
    margin = 0.035 * np.max(extent)
    ax.set_xlim(minimum[0] - margin, maximum[0] + margin)
    ax.set_ylim(minimum[1] - margin, maximum[1] + margin)
    ax.set_zlim(minimum[2] - margin, maximum[2] + margin)
    ax.set_box_aspect(extent)
    ax.view_init(elev=24, azim=-58)
    ax.set_proj_type("ortho")
    if show_axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.tick_params(labelsize=8, pad=1)
        ax.grid(True, alpha=0.25)
    else:
        ax.set_axis_off()
    return ax


def plot_hex_mesh_views(
    points, nx, ny, nz, title="3D structured grid", show_axes=False
):
    """Return perspective plus front (xz), side (yz), and top (xy) views."""
    grid = np.asarray(points).reshape(nz, ny, nx, 3)
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(141, projection="3d")
    plot_hex_mesh_3d(grid, nx, ny, nz, ax, show_axes=show_axes)
    ax.set_title(title)
    flat = grid.reshape(-1, 3)
    for position, (a, b, label) in enumerate(
        ((0, 2, "Front"), (1, 2, "Side"), (0, 1, "Top")), 2
    ):
        axis = fig.add_subplot(1, 4, position)
        axis.scatter(flat[:, a], flat[:, b], s=2)
        axis.set_aspect("equal")
        axis.set_title(label)
        if show_axes:
            axis.set_xlabel(("x", "y", "x")[position - 2])
            axis.set_ylabel(("z", "z", "y")[position - 2])
        else:
            axis.set_axis_off()
    fig.tight_layout()
    return fig


def plot_hex_mesh_paper(
    points,
    nx,
    ny,
    nz,
    ax=None,
    elevation=24,
    azimuth=-58,
    maximum_lines_per_direction=14,
    show_axes=False,
    limits=None,
):
    """Draw a complete paper-oriented hexahedral wireframe at a fixed camera."""
    grid = np.asarray(points, dtype=float).reshape(nz, ny, nx, 3)
    if ax is None:
        ax = plt.figure(figsize=(7, 6)).add_subplot(111, projection="3d")
    plot_hex_mesh_3d(
        grid,
        nx,
        ny,
        nz,
        ax=ax,
        show_axes=show_axes,
        limits=limits,
    )
    ax.view_init(elev=elevation, azim=azimuth)
    return ax


# ================================ 2D =================================


def plot_boundary(
    points,
    ax=None,
    closed=True,
    show_node_id=True,
    edgecolor="black",
    facecolor="none",
    linewidth=0.8,
    alpha=1.0,
):
    """Documentation string."""
    ax = ax or plt.gca()
    # Implementation comment.
    if closed:
        points = np.vstack([points, points[0]])
    line_collection = ax.plot(
        points[:, 0], points[:, 1], color=edgecolor, linewidth=linewidth, alpha=alpha
    )
    ax.set_aspect("equal")
    if show_node_id:
        for idx, (x, y) in enumerate(points):
            ax.text(x, y, str(idx), fontsize=8)
    return ax, line_collection


# Implementation comment.
# Implementation comment.
# Implementation comment.
def plot_quad_mesh_general(
    Nx,
    Ny,
    points,
    values=None,
    ax=None,
    xlim: tuple = None,
    ylim: tuple = None,
    title: str = None,
    cmap="viridis",
    norm=None,
    show_node_id=False,
    show_axis=True,
    linewidth=0.1,
    edgecolor="black",
    facecolor="none",
):
    """Documentation string."""
    points_matrix = points.reshape(Ny, Nx, -1)
    X = points_matrix[:, :, 0]
    Y = points_matrix[:, :, 1]
    if ax is None:
        fig, ax = plt.subplots()
    if values is None:
        ax.pcolormesh(
            X,
            Y,
            np.zeros_like(X[:-1, :-1]),
            edgecolor=edgecolor,
            facecolor=facecolor,
            linewidth=linewidth,
        )
    else:
        ax.pcolormesh(
            X,
            Y,
            values.reshape(Ny - 1, Nx - 1),
            cmap=cmap,
            norm=norm,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )

    ax.autoscale()
    ax.set_aspect("equal")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title is not None:
        ax.set_title(title)
    if show_node_id:
        for idx, (x, y) in enumerate(points):
            ax.text(x, y, str(idx), fontsize=8)
    if not show_axis:
        ax.axis("off")
    return ax


def plot_quad_mesh(
    points,
    quads,
    ax=None,
    show_node_id=True,
    edgecolor="black",
    facecolor="none",
    linewidth=0.8,
    alpha=1.0,
):
    """Documentation string."""
    ax = ax or plt.gca()
    patches = [Polygon(points[q], closed=True) for q in quads]
    pc = PatchCollection(
        patches,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
    )
    # Implementation comment.
    if show_node_id:
        for idx, (x, y) in enumerate(points):
            ax.text(x, y, str(idx), fontsize=8)
    ax.add_collection(pc)
    ax.autoscale()  # Implementation comment.
    ax.set_aspect("equal")
    return ax, pc


def plot_quad_mesh_colored(
    points,
    quads,
    values,
    ax=None,
    cmap="viridis",
    edgecolor="k",
    linewidth=0.3,
    alpha=1.0,
    norm=None,
):
    """Documentation string."""
    ax = ax or plt.gca()
    patches = [Polygon(points[q], closed=True) for q in quads]
    pc = PatchCollection(
        patches,
        cmap=cmap,
        norm=norm,
        edgecolor=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
    )
    pc.set_array(np.asarray(values))
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect("equal")
    return ax, pc


def plot_quad_mesh_colored_smooth(
    points,
    quads,
    values,
    ax=None,
    cmap="viridis",
    edgecolor="k",
    linewidth=0.25,
    alpha=1.0,
    norm=None,
):
    """Smoothly display cell values after averaging them onto mesh vertices.

    This changes visualization only. The saved per-cell quality values and all
    statistics remain untouched. Each quad is split into two triangles for
    Gouraud shading, then the original quad edges are drawn on top.
    """
    from matplotlib.tri import Triangulation

    points = np.asarray(points, dtype=float)
    quads = np.asarray(quads, dtype=int)
    values = np.asarray(values, dtype=float)
    if values.shape != (len(quads),):
        raise ValueError("values must contain one scalar per quadrilateral")
    sums = np.zeros(len(points), dtype=float)
    counts = np.zeros(len(points), dtype=float)
    finite_values = np.where(np.isfinite(values), values, 0.0)
    for local_vertex in range(4):
        np.add.at(sums, quads[:, local_vertex], finite_values)
        np.add.at(counts, quads[:, local_vertex], np.isfinite(values))
    node_values = np.divide(
        sums, counts, out=np.full_like(sums, np.nan), where=counts > 0
    )
    triangles = np.concatenate((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]), axis=0)
    triangulation = Triangulation(points[:, 0], points[:, 1], triangles)
    ax = ax or plt.gca()
    colored = ax.tripcolor(
        triangulation,
        node_values,
        shading="gouraud",
        cmap=cmap,
        norm=norm,
        alpha=alpha,
    )
    # Retain exact cell topology without overpowering the smooth quality field.
    segments = np.concatenate((points[quads], points[quads[:, :1]]), axis=1)
    from matplotlib.collections import LineCollection

    ax.add_collection(LineCollection(segments, colors=edgecolor, linewidths=linewidth))
    ax.autoscale()
    ax.set_aspect("equal")
    return ax, colored


def plot_quad_meshXY(
    quad_mesh: QuadMeshXY,
    ax=None,
    show_node_id=True,
    edgecolor="black",
    facecolor="none",
    linewidth=0.8,
    alpha=1.0,
):
    """Documentation string."""
    return plot_quad_mesh(
        quad_mesh.vertices,
        quad_mesh.faces,
        ax=ax,
        show_node_id=show_node_id,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=linewidth,
        alpha=alpha,
    )


def plot_quad_meshXY_colored(
    quad_mesh: QuadMeshXY,
    values,
    ax=None,
    cmap="viridis",
    edgecolor="k",
    linewidth=0.3,
    alpha=1.0,
):
    """Documentation string."""
    return plot_quad_mesh_colored(
        quad_mesh.vertices,
        quad_mesh.faces,
        values,
        ax=ax,
        cmap=cmap,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
    )


if __name__ == "__main__":

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    points = np.array(
        [[0, 0], [1, 0], [2, 0], [0, 0.4], [1.2, 0.6], [2, 0.5], [0, 1], [1, 1], [2, 1]]
    )
    quads = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]])
    mesh = QuadMeshXY(Nx=3, Ny=3, vertices=points, faces=None)
    # plot_quad_mesh(points, quads, ax=axs[0], edgecolor='k', facecolor='none', linewidth=0.6)
    # # plot_quad_meshXY(mesh, ax=axs[0], edgecolor='k', facecolor='none', linewidth=0.6)
    # # axs[0].plot(points[:,0], points[:,1],  label='Physical Domain')
    # # axs[0].legend()
    # axs[0].set_title('Quad mesh (edges)')
    # axs[0].autoscale()
    # Implementation comment.
    # metric = metrics.QuadMeshQualityEvaluator(points, quads)
    # angles = metric.Angle()
    # max_angles = np.max(angles, axis=1)
    # ax2, pc = plot_quad_mesh_colored(points, quads, max_angles, ax=axs[1], cmap='viridis', edgecolor='k', linewidth=0.3, alpha=1.0)
    # axs[1].set_title('Max angle per quad (deg)')
    # fig.colorbar(pc, ax=axs[1], orientation='vertical', label='deg')
    # # ax, line_collection= plot_boundary(points, edgecolor='red', linewidth=1.5, alpha=0.8)
    # # ax.set_title('Boundary')
    # plt.show()
    plot_quad_mesh_general(
        Nx=2,
        Ny=2,
        points=points,
        values=None,
        ax=axs[0],
        xlim=(0, 2),
        ylim=(0, 1),
        title="Quad mesh (edges)",
        cmap="viridis",
        norm=None,
        show_node_id=True,
        linewidth=0.6,
        edgecolor="k",
        facecolor="none",
    )
    metric = metrics.QuadMeshQualityEvaluator(points, quads)
    angles = metric.Angle()
    max_angles = np.max(angles, axis=1)
    print("Max angles per quad:", max_angles)
    plot_quad_mesh_general(
        Nx=2,
        Ny=2,
        points=points,
        values=max_angles,
        ax=axs[1],
        xlim=(0, 2),
        ylim=(0, 1),
        title="Max angle per quad (deg)",
        cmap="viridis",
        norm=None,
        show_node_id=True,
        linewidth=0.3,
        edgecolor="k",
        facecolor="none",
    )
    fig.colorbar(axs[1].collections[0], ax=axs[1], orientation="vertical", label="deg")
    plt.show()

