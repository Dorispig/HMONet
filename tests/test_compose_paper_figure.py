"""Smoke tests for the reusable publication-figure composer."""

from pathlib import Path
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.visualize.compose_paper_figure import main


def test_composes_existing_images_with_one_shared_colorbar():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first, second = root / "first.png", root / "second.png"
        plt.imsave(first, np.linspace(0, 1, 100).reshape(10, 10), cmap="viridis")
        plt.imsave(second, np.linspace(1, 0, 100).reshape(10, 10), cmap="viridis")
        outputs = main(
            [
                "--images", str(first), str(second),
                "--rows", "1", "--cols", "2",
                "--mode", "quality",
                "--crop", "0", "0", "1", "1",
                "--vmin", "0", "--vmax", "1",
                "--output", str(root / "quality_panels"),
                "--formats", "png",
            ]
        )
        assert outputs == [root / "quality_panels.png"]
        assert outputs[0].is_file()


def test_rerenders_mesh_quality_with_one_colorbar():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        mesh = np.stack(np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3)), axis=-1)
        mesh_path = root / "mesh.npy"
        np.save(mesh_path, mesh.reshape(-1, 2))
        outputs = main(
            [
                "--mesh-files", str(mesh_path),
                "--shape", "3", "3",
                "--rows", "1", "--cols", "1",
                "--mode", "quality", "--metric", "max_angle",
                "--no-show-boundary",
                "--output", str(root / "mesh_quality"),
            ]
        )
        assert outputs == [root / "mesh_quality.png"]
        assert outputs[0].is_file()


def test_rerenders_a_physical_local_window():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        mesh = np.stack(
            np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5)), axis=-1
        )
        mesh_path = root / "mesh.npy"
        np.save(mesh_path, mesh.reshape(-1, 2))
        outputs = main(
            [
                "--mesh-files", str(mesh_path),
                "--shape", "5", "5", "--rows", "1", "--cols", "1",
                "--mode", "mesh", "--no-show-boundary",
                "--view-window", "0.25", "0.75", "0.25", "0.75",
                "--output", str(root / "mesh_local"),
            ]
        )
        assert outputs == [root / "mesh_local.png"]
        assert outputs[0].is_file()
