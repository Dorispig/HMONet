"""Check plot columns, including zooms loaded from older unordered results."""

from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from scripts.evaluate.plot_result_zoom import load_comparison, save_zoom
from src.data.mesh.sampling import logical_grid
from src.evaluation.analysis import plot_volumes, save_sections
from src.evaluation.configuration import selected_methods
from src.evaluation.quality_3d import plot_surface_slice_comparison
from src.evaluation.quality_output import compute_quad_quality, save_quality_comparison
from src.evaluation.zoom_3d import generate_zoom, load_meshes


class EvaluationPlotOrderTests(unittest.TestCase):
    unordered = ["HMONet", "MeshONet", "PDE_Winslow", "TFI"]
    ordered = ["TFI", "PDE_Winslow", "MeshONet", "HMONet"]
    labels = ["TFI", "PDE", "MeshONet", "HMONet"]

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="hmonet_plot_order_")
        self.root = Path(temporary.name)
        self.addCleanup(temporary.cleanup)
        self.addCleanup(plt.close, "all")
        self.options = dict(
            save_plots=True,
            save_png=True,
            save_pdf=False,
            save_distribution_plots=False,
        )

    def meshes(self, dimension):
        points = logical_grid((3,) * dimension)
        return {name: points.copy() for name in self.unordered}

    def assert_columns(self, figure, *args, **kwargs):
        panels = []
        for axis in figure.axes:
            spec = axis.get_subplotspec()
            if spec is not None and spec.rowspan.start == 0:
                panels.append((spec.colspan.start, axis.get_title().split(" | ")[0]))
        self.assertEqual(sorted(panels), list(enumerate(self.labels)))

    def saved_source(self, dimension):
        folder = self.root / f"{dimension}D"
        folder.mkdir()
        np.savez_compressed(
            folder / "comparison_mesh_data.npz", **self.meshes(dimension)
        )
        metadata = dict(
            method_order=self.unordered, grid_shape=[3] * dimension, nx=3, ny=3
        )
        (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return folder

    def test_selection_and_legacy_result_loaders(self):
        self.assertEqual(selected_methods("hmonet,meshonet,pde,tfi"), self.ordered)
        self.assertEqual(selected_methods("hmonet,pde"), ["PDE_Winslow", "HMONet"])
        meshes, _, _, _ = load_comparison(self.saved_source(2), None)
        self.assertEqual(list(meshes), self.labels)
        meshes, _, _ = load_meshes(self.saved_source(3))
        self.assertEqual(list(meshes), self.ordered)

    def test_2d_mesh_quality_and_zoom_columns(self):
        meshes = self.meshes(2)
        quality = compute_quad_quality(next(iter(meshes.values())), 3, 3)
        quality["fields"] = {"scaled_jacobian": quality["fields"]["scaled_jacobian"]}
        with patch(
            "src.evaluation.quality_output.save_figure_formats",
            side_effect=self.assert_columns,
        ) as saved:
            save_quality_comparison(
                self.root,
                meshes,
                3,
                3,
                {"boundary_projection": {}},
                quality_cache={name: quality for name in meshes},
                save_distribution_plots=False,
            )
            self.assertEqual(saved.call_count, 2)
        metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["method_order"], self.ordered)
        # Simulate old NPZ and JSON insertion order before loading local views.
        folder = self.saved_source(2)
        np.savez_compressed(
            folder / "comparison_quality_data.npz",
            **{f"{name}__scaled_jacobian": np.ones(4) for name in meshes},
        )
        loaded, fields, nx, ny = load_comparison(folder, "scaled_jacobian")
        args = Namespace(
            output_dir=None,
            parameter_dir=None,
            color_scale="global",
            save_png=True,
            save_pdf=False,
        )
        with patch(
            "scripts.evaluate.plot_result_zoom.save_figure_formats",
            side_effect=self.assert_columns,
        ) as saved:
            for kind in ("mesh", "quality"):
                save_zoom(
                    folder,
                    loaded,
                    fields if kind == "quality" else {},
                    nx,
                    ny,
                    kind,
                    "scaled_jacobian" if kind == "quality" else None,
                    (0.2, 0.8),
                    (0.2, 0.8),
                    {},
                    args,
                )
            self.assertEqual(saved.call_count, 4)

    def test_3d_volume_middle_sections_and_zoom_columns(self):
        meshes = self.meshes(3)
        fields = {name: {"minimum_scaled_jacobian": np.ones(8)} for name in meshes}
        with patch(
            "src.evaluation.analysis.save_figure", side_effect=self.assert_columns
        ) as saved:
            plot_volumes(self.root, meshes, fields, (3, 3, 3), self.options)
            self.assertEqual(saved.call_count, 2)
            save_sections(self.root, meshes, (3, 3, 3), self.options)
            self.assertGreater(saved.call_count, 2)
        for section in ("x_mid", "y_mid", "z_mid"):
            metadata = json.loads(
                (self.root / "boundary_faces" / section / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["method_order"], self.ordered)
        folder = self.saved_source(3)
        check = self.assert_columns

        def capture(figure, *args, **kwargs):
            check(figure)

        with patch("matplotlib.figure.Figure.savefig", new=capture):
            for mode, metric in (
                ("slice", "scaled_jacobian"),
                ("volume", "minimum_scaled_jacobian"),
            ):
                generate_zoom(
                    folder,
                    dict(
                        mode=mode,
                        metric=metric,
                        methods=self.unordered,
                        center_on="worst",
                        plot="both",
                        save_plots=True,
                        save_png=True,
                        save_pdf=False,
                    ),
                )

    def test_3d_summary_columns_retain_matching_data(self):
        fields = {
            name: {"minimum_scaled_jacobian": np.full(8, column)}
            for column, name in enumerate(self.ordered)
        }
        fields = {name: fields[name] for name in self.unordered}
        check = self.assert_columns

        def capture(figure, *args, **kwargs):
            check(figure)
            for axis in figure.axes:
                if axis.images:
                    column = axis.get_subplotspec().colspan.start
                    np.testing.assert_array_equal(
                        axis.images[0].get_array(), np.full((2, 2), column)
                    )

        with patch("matplotlib.figure.Figure.savefig", new=capture):
            plot_surface_slice_comparison(self.root, fields, 3, 3, 3)


if __name__ == "__main__":
    unittest.main()
