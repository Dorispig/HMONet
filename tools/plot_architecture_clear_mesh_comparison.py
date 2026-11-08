"""Create readable Ex11 mesh and maximum-angle local comparison figures."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.quality_output import quad_quality_fields


ROOT = Path(r"C:\Users\24846\tmp\architecture_clear_meshes")
OUTPUT = PROJECT_ROOT / "reports/architecture_suite/baseline_e1_e3_air_t"
CASE = "air_t_ext_1_500_21_21"
SIZE = 101
METHODS = (
    ("PDE", "pde", "PDE_Winslow/mesh.npy"),
    ("Baseline", "baseline", "HMONet/corner_corrected/raw/mesh.npy"),
    ("e1 ModifiedMLP", "e1_modified_mlp", "HMONet/corner_corrected/raw/mesh.npy"),
    ("e2 Cross-attention", "e2_cross_attention", "HMONet/corner_corrected/raw/mesh.npy"),
    ("e3 Boundary decoder", "e3_boundary_decoder", "HMONet/corner_corrected/raw/mesh.npy"),
)


def load(parameter: str):
    result = []
    for label, directory, suffix in METHODS:
        path = ROOT / directory / "2D" / CASE / "101x101" / parameter / suffix
        points = np.load(path).reshape(SIZE, SIZE, 2)
        fields, mesh = quad_quality_fields(
            points.reshape(-1, 2), SIZE, SIZE, ["max_angle"]
        )
        values = fields["max_angle"]
        # A larger maximum corner angle indicates a more distorted cell.
        worst = int(np.argmax(values))
        j, i = divmod(worst, SIZE - 1)
        result.append(
            {
                "label": label,
                "grid": points,
                "worst": worst,
                "i": i,
                "j": j,
                "maximum_angle": float(values[worst]),
                "maximum_angle_values": values.reshape(SIZE - 1, SIZE - 1),
                "cell": mesh.vertices[mesh.faces[worst]],
            }
        )
    return result


def draw_grid(axis, grid, row_ids, column_ids, *, linewidth=0.65):
    color = "#34495E"
    for row in row_ids:
        axis.plot(
            grid[row, :, 0], grid[row, :, 1], color=color,
            linewidth=linewidth, alpha=0.92, solid_capstyle="round",
        )
    for column in column_ids:
        axis.plot(
            grid[:, column, 0], grid[:, column, 1], color=color,
            linewidth=linewidth, alpha=0.92, solid_capstyle="round",
        )


def style_axis(axis):
    axis.set_aspect("equal", adjustable="box")
    axis.grid(False)
    axis.tick_params(labelsize=9, width=0.7, length=3)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#667085")


def full_figure(items, parameter):
    figure, axes = plt.subplots(1, len(items), figsize=(22, 4.7), constrained_layout=True)
    selected = sorted(set(range(0, SIZE, 5)) | {SIZE - 1})
    all_points = np.concatenate([item["grid"].reshape(-1, 2) for item in items])
    minimum, maximum = all_points.min(0), all_points.max(0)
    padding = 0.045 * max(maximum - minimum)
    for axis, item in zip(axes, items):
        grid = item["grid"]
        draw_grid(axis, grid, selected, selected, linewidth=0.7)
        # Preserve every physical boundary even though interior lines are thinned.
        for line in (grid[0], grid[-1], grid[:, 0], grid[:, -1]):
            axis.plot(line[:, 0], line[:, 1], color="#111827", linewidth=1.35)
        axis.set_xlim(minimum[0] - padding, maximum[0] + padding)
        axis.set_ylim(minimum[1] - padding, maximum[1] + padding)
        axis.set_title(item["label"], fontsize=15, fontweight="bold", pad=10)
        style_axis(axis)
    figure.suptitle(
        f"{CASE} | parameter={parameter} | 101x101\n"
        "Readable overview: every 5th interior grid line; all boundaries retained",
        fontsize=16,
        fontweight="bold",
    )
    target = OUTPUT / f"mesh_comparison_clear_{parameter}.png"
    figure.savefig(target, dpi=260, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def local_figure(items, parameter):
    # Use a compact neighbourhood around the maximum-angle cells. Every cell
    # is drawn without any overlay, so the actual worst cells remain visible.
    i_min = max(0, min(item["i"] for item in items) - 3)
    i_max = min(SIZE - 1, max(item["i"] for item in items) + 4)
    j_min = max(0, min(item["j"] for item in items) - 6)
    j_max = min(SIZE - 1, max(item["j"] for item in items) + 1)
    points = np.concatenate(
        [item["grid"][j_min : j_max + 1, i_min : i_max + 1].reshape(-1, 2) for item in items]
    )
    minimum, maximum = points.min(0), points.max(0)
    padding = 0.045 * max(maximum - minimum)
    figure, axes = plt.subplots(1, len(items), figsize=(23, 6.3), constrained_layout=True)
    for axis, item in zip(axes, items):
        patch = item["grid"][j_min : j_max + 1, i_min : i_max + 1]
        draw_grid(
            axis,
            patch,
            range(patch.shape[0]),
            range(patch.shape[1]),
            linewidth=1.25,
        )
        axis.set_xlim(minimum[0] - padding, maximum[0] + padding)
        axis.set_ylim(minimum[1] - padding, maximum[1] + padding)
        axis.set_title(
            f"{item['label']}\nworst maximum angle = {item['maximum_angle']:.3f} deg",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )
        style_axis(axis)
    figure.suptitle(
        f"{CASE} | parameter={parameter} | maximum-angle worst-cell neighbourhood\n"
        "Every cell is drawn at high magnification; no cell is covered by a marker",
        fontsize=16,
        fontweight="bold",
    )
    target = OUTPUT / f"maximum_angle_detail_clear_{parameter}.png"
    figure.savefig(target, dpi=280, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def maximum_angle_quality_figure(items, parameter):
    """Draw a magnified, cell-wise maximum-angle quality field."""
    i_min = max(0, min(item["i"] for item in items) - 3)
    i_max = min(SIZE - 1, max(item["i"] for item in items) + 4)
    j_min = max(0, min(item["j"] for item in items) - 6)
    j_max = min(SIZE - 1, max(item["j"] for item in items) + 1)

    local_values = np.concatenate(
        [
            item["maximum_angle_values"][j_min:j_max, i_min:i_max].reshape(-1)
            for item in items
        ]
    )
    norm = Normalize(vmin=float(local_values.min()), vmax=float(local_values.max()))
    cmap = plt.get_cmap("turbo")

    all_points = np.concatenate(
        [
            item["grid"][j_min : j_max + 1, i_min : i_max + 1].reshape(-1, 2)
            for item in items
        ]
    )
    minimum, maximum = all_points.min(0), all_points.max(0)
    padding = 0.04 * max(maximum - minimum)

    figure, axes_grid = plt.subplots(2, 3, figsize=(15.5, 10.6), constrained_layout=True)
    axes = axes_grid.reshape(-1)[: len(items)]
    mappable = None
    for axis, item in zip(axes, items):
        grid = item["grid"]
        polygons = []
        values = []
        for j in range(j_min, j_max):
            for i in range(i_min, i_max):
                polygons.append(
                    np.asarray(
                        [
                            grid[j, i],
                            grid[j, i + 1],
                            grid[j + 1, i + 1],
                            grid[j + 1, i],
                        ]
                    )
                )
                values.append(item["maximum_angle_values"][j, i])
        collection = PolyCollection(
            polygons,
            array=np.asarray(values),
            cmap=cmap,
            norm=norm,
            edgecolors="#111827",
            linewidths=0.72,
            antialiased=True,
        )
        axis.add_collection(collection)
        mappable = collection
        axis.set_xlim(minimum[0] - padding, maximum[0] + padding)
        axis.set_ylim(minimum[1] - padding, maximum[1] + padding)
        axis.set_title(
            f"{item['label']}\nmaximum = {item['maximum_angle']:.3f} deg",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )
        style_axis(axis)

    colorbar_panel = axes_grid.reshape(-1)[-1]
    colorbar_panel.axis("off")
    colorbar_axis = colorbar_panel.inset_axes([0.38, 0.08, 0.16, 0.84])
    colorbar = figure.colorbar(mappable, cax=colorbar_axis)
    colorbar.set_label("Cell maximum angle (deg); larger is worse", fontsize=12)
    colorbar.ax.tick_params(labelsize=10)
    figure.suptitle(
        f"{CASE} | parameter={parameter} | local maximum-angle quality field\n"
        "Common colour scale; every coloured polygon is one mesh cell",
        fontsize=16,
        fontweight="bold",
    )
    target = OUTPUT / f"maximum_angle_quality_map_clear_{parameter}.png"
    figure.savefig(target, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for parameter in ("0.62", "0.65"):
        items = load(parameter)
        local_figure(items, parameter)
        maximum_angle_quality_figure(items, parameter)
        print(
            parameter,
            [
                (item["label"], item["i"], item["j"], item["maximum_angle"])
                for item in items
            ],
        )


if __name__ == "__main__":
    main()
