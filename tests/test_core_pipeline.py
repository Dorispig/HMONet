"""Core data-generation, model, PDE, package-layout and command-line integration tests."""

import importlib
import pkgutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import sys, tempfile, unittest
import torch


ROOT = Path(__file__).resolve().parents[1]


class DataLayoutTests(unittest.TestCase):
    def test_all_data_modules_import(self):
        import src.data

        for module in pkgutil.walk_packages(src.data.__path__, src.data.__name__ + "."):
            with self.subTest(module=module.name):
                importlib.import_module(module.name)

    def test_moved_generator_uses_project_data_directory(self):
        from src.data.generation.make_dataset_parametric_2d import DEFAULT_OUTPUT_ROOT

        self.assertEqual(DEFAULT_OUTPUT_ROOT, ROOT / "data/raw/2D")

    def test_merged_sampling_in_both_dimensions(self):
        from src.data.mesh.sampling import (
            logical_grid,
            boundary_indices,
            sample_boundary,
        )
        from src.data.geometry.boundary_generator_2d import rectangular_boundary
        from src.data.geometry.boundary_generator_3d import box_boundary

        for shape, boundary in [
            ((5, 7), rectangular_boundary()),
            ((5, 7, 3), box_boundary()),
        ]:
            logical = logical_grid(shape)
            sampled = sample_boundary(boundary, shape)
            self.assertEqual(logical.shape, (int(np.prod(shape)), len(shape)))
            self.assertEqual(sampled.shape, (len(boundary_indices(shape)), len(shape)))
            self.assertTrue(np.isfinite(sampled).all())

    def test_evaluation_version_includes_nested_data_modules(self):
        from unittest.mock import patch
        from src.evaluation import generation

        with patch.object(generation, "digest", side_effect=lambda sources: sources):
            sources = generation.implementation_version()
        expected = ROOT / "src/data/geometry/parametric_curve_2d.py"
        self.assertIn(str(expected.relative_to(ROOT)), sources)
        expected = ROOT / "src/data/backend/curve_tensor.py"
        self.assertIn(str(expected.relative_to(ROOT)), sources)

    def test_generator_cli_outside_project(self):
        with tempfile.TemporaryDirectory() as directory:
            for dimension in (2, 3):
                script = (
                    ROOT
                    / f"src/data/generation/make_dataset_parametric_{dimension}d.py"
                )
                result = subprocess.run(
                    [sys.executable, str(script), "--help"],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())


ROOT = Path(__file__).resolve().parents[1]


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class PipelineTests(unittest.TestCase):
    def test_parametric_curve_and_piecewise_sampling(self):
        from src.data.geometry.parametric_curve_2d import LineSegment, PiecewiseCurve

        curve = PiecewiseCurve(
            [LineSegment([0, 0], [1, 0]), LineSegment([1, 0], [1, 1])],
            allocation="length",
        )
        t, pts = curve.sample_uniform(5, include_junctions=True)
        self.assertTrue(np.any(np.isclose(t, curve.junction_parameters()[0])))
        self.assertTrue(np.allclose(pts[0], [0, 0]))
        self.assertTrue(np.allclose(pts[-1], [1, 1]))

    def test_2d_dataset_generation_and_reader(self):
        from src.data.generation.make_dataset_parametric_2d import generate_explicit_dataset
        from src.data.dataset import ParametricDataset2D

        with tempfile.TemporaryDirectory() as td:
            out = generate_explicit_dataset(
                "case",
                "rectangle",
                [{}],
                grid_shape=(4, 3),
                output_root=td,
            )
            ds = ParametricDataset2D(out, load_boundary=True)
            self.assertEqual(len(ds), 1)
            self.assertEqual(ds.boundary_points.shape, (1, 10, 2))
            self.assertEqual(ds.mesh_points.shape, (1, 12, 2))
            self.assertEqual(set(ds[0]), {"boundary_points", "mesh_points", "boundary"})
            self.assertEqual(ds[0]["boundary"].__class__.__name__, "BoundaryCurveSet2D")

    def test_2d_hmonet_geometry_forward_and_backward(self):
        from src.data.geometry.boundary_generator_2d import rectangular_boundary
        from src.models.model import HMONet2D

        geometry = rectangular_boundary()
        model = HMONet2D(
            3,
            3,
            branch_dim=4,
            trunk_dim=2,
            adjust_hidden_dim=4,
            branch_hidden_dim=4,
            trunk_hidden_dim=4,
            adjust_hidden_layer_num=1,
            branch_hidden_layer_num=1,
            trunk_hidden_layer_num=1,
            output_dim=4,
        )
        b = torch.rand(1, 8, 2, requires_grad=True)
        q = torch.rand(1, 6, 2)
        out = model(
            b[..., 0], b[..., 1], q[..., 0], q[..., 1], boundary_geometry=geometry
        )
        loss = sum(x.square().mean() for x in out)
        loss.backward()
        self.assertEqual(out[0].shape, (1, 6, 1))
        self.assertIsNotNone(next(model.parameters()).grad)

    def test_3d_surface_forward(self):
        from src.data.geometry.boundary_generator_3d import box_boundary
        from src.models.model import HMONet3D

        geometry = box_boundary()
        model = HMONet3D(
            3,
            3,
            3,
            branch_dim=4,
            trunk_dim=3,
            adjust_hidden_dim=4,
            branch_hidden_dim=4,
            trunk_hidden_dim=4,
            adjust_hidden_layer_num=1,
            branch_hidden_layer_num=1,
            trunk_hidden_layer_num=1,
            output_dim=4,
        )
        b = torch.rand(1, 26, 3)
        q = torch.rand(1, 6, 3)
        out = model(
            b[..., 0],
            b[..., 1],
            b[..., 2],
            q[..., 0],
            q[..., 1],
            q[..., 2],
            boundary_geometry=geometry,
        )
        self.assertEqual(out[0].shape, (1, 6, 1))

    def test_3d_dataset_generation_and_reader(self):
        from src.data.generation.make_dataset_parametric_3d import generate_dataset
        from src.data.dataset import ParametricDataset3D

        with tempfile.TemporaryDirectory() as td:
            out = generate_dataset(
                "case",
                "box",
                [{}],
                grid_shape=(4, 4, 4),
                sample_count=5,
                output_root=td,
            )
            ds = ParametricDataset3D(out, load_boundary=True)
            self.assertEqual(ds.boundary_points.shape, (1, 56, 3))
            self.assertEqual(ds.mesh_points.shape, (1, 64, 3))
            self.assertEqual(set(ds[0]), {"boundary_points", "mesh_points", "boundary"})
            self.assertEqual(ds[0]["boundary"].__class__.__name__, "BoundarySurfaceSet3D")
            self.assertFalse(hasattr(ds, "physical_samples"))
            self.assertFalse(hasattr(ds, "sample_parameters"))

            from unittest.mock import patch

            original_load = np.load
            opened = []

            def load_numeric_only(path, *args, **kwargs):
                opened.append(Path(path).name)
                self.assertIn(Path(path).name, {"boundary_samples.npy", "pde_mesh.npy"})
                return original_load(path, *args, **kwargs)

            # MeshONet must still work without geometry or unused sample files.
            for filename in ("geometry.json", "physical_samples.npy", "sample_parameters.npy"):
                (out / filename).unlink()
            with patch("src.data.dataset.np.load", side_effect=load_numeric_only):
                mesh_data = ParametricDataset3D(out, load_boundary=False)
            self.assertEqual(opened, ["boundary_samples.npy", "pde_mesh.npy"])
            self.assertEqual(set(mesh_data[0]), {"boundary_points", "mesh_points"})

    def test_2d_pde_supports_nonuniform_parameters(self):
        from src.data.geometry.boundary_generator_2d import rectangular_boundary
        from scripts.pde.pde_2d import solve_winslow_2d

        g = rectangular_boundary()
        params = {
            "bottom": np.array([0.0, 0.1, 0.6, 1.0]),
            "right": np.array([0.0, 0.2, 1.0]),
            "top": np.array([0.0, 0.3, 0.7, 1.0]),
            "left": np.array([0.0, 0.4, 1.0]),
        }
        result = solve_winslow_2d(g, 4, 3, max_iter=1, parameter_grid=params)
        self.assertEqual(result.shape, (12, 2))
        self.assertTrue(np.isfinite(result).all())

    def test_command_line_entrypoints(self):
        import subprocess

        py = sys.executable
        for script in [
            "src/data/generation/make_dataset_parametric_2d.py",
            "scripts/train/train_2d.py",
            "scripts/train/train_3d.py",
            "scripts/evaluate/evaluate_2d.py",
            "scripts/evaluate/evaluate_3d.py",
        ]:
            result = subprocess.run(
                [py, str(ROOT / script), "--help"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()

