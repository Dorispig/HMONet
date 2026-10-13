"""Save deterministic first/middle/last PDE grid examples without resampling."""

import json
from pathlib import Path
import numpy as np


def save_dataset_examples(directory, records=None, metadata=None, geometry=None):
    """Use the stored PDE coordinates; middle means zero-based index n//2."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.data.dataset import read_dataset_metadata
    from src.data.mesh.topology import QuadMeshXY
    from src.utils.visualize import plot_quad_meshXY

    directory = Path(directory)
    if records is None:
        records = np.load(directory / "dataset.npy", allow_pickle=True)
    if metadata is None:
        metadata = read_dataset_metadata(directory)
    if geometry is None:
        geometry = np.load(directory / "boundary_geometry.npy", allow_pickle=True)
    nx, ny = metadata["grid_shape"]
    output = directory / "example"
    output.mkdir(exist_ok=True)
    for label, index in (
        ("first", 0),
        ("middle", len(records) // 2),
        ("last", len(records) - 1),
    ):
        points = np.asarray(records[index]["mesh_points"])
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_quad_meshXY(
            QuadMeshXY(nx, ny, points), ax=ax, show_node_id=False, linewidth=0.35
        )
        parameters = geometry[index].get("parameters", {})
        ax.set_title(
            f"{metadata.get('dataset', directory.name)}\nsample {index+1}/{len(records)}: {parameters}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(output / f"{label}.png", dpi=150)
        plt.close(fig)
    return output


def save_dataset_examples_3d(
    directory,
    meshes=None,
    metadata=None,
    parameters=None,
):
    """Save perspective PDE wireframes for the first, middle and last samples."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.visualize import plot_hex_mesh_3d

    directory = Path(directory)
    if metadata is None:
        metadata = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )
    if meshes is None:
        meshes = np.load(directory / "pde_mesh.npy", mmap_mode="r")
    if parameters is None:
        geometry = json.loads(
            (directory / "geometry.json").read_text(encoding="utf-8")
        )
        parameters = [sample.get("parameters", {}) for sample in geometry["samples"]]

    nx, ny, nz = map(int, metadata["grid_shape"])
    output = directory / "example"
    output.mkdir(exist_ok=True)
    selections = (
        ("first", 0),
        ("middle", len(meshes) // 2),
        ("last", len(meshes) - 1),
    )
    for label, index in selections:
        figure = plt.figure(figsize=(8, 7))
        axis = figure.add_subplot(111, projection="3d")
        plot_hex_mesh_3d(
            meshes[index], nx, ny, nz, ax=axis, linewidth=0.28, show_axes=True
        )
        axis.set_title(
            f"{metadata.get('dataset', directory.name)}\n"
            f"sample {index + 1}/{len(meshes)}: {parameters[index]}"
        )
        figure.tight_layout()
        figure.savefig(output / f"{label}.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
    return output
