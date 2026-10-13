"""Verify interval sampling and real PDE targets for circle/airfoil datasets."""

import contextlib
import io
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tfi.tfi_2d import tfi_from_curves
from src.data.dataset import (
    ParametricDataset2D,
    read_dataset_metadata,
    read_sample_parameters,
)
from src.data.generation.make_dataset_parametric_2d import (
    DEFAULT_OUTPUT_ROOT,
    generate_circle_airfoil_t_dataset,
    generate_explicit_dataset,
    main,
    sample_parameter_ranges,
    sample_thickness_ranges,
)
from src.data.mesh.topology import GenerateQuadMeshBoundaryIndex
from src.data.geometry.parametric_curve_2d import NACA4Curve


class CircleAirfoilDatasetTests(unittest.TestCase):
    def test_shared_parameter_sampling_supports_signed_geometry_parameters(self):
        values, intervals, counts = sample_parameter_ranges([(-0.5, 0.5)], count=5)
        np.testing.assert_allclose(values, [-0.5, -0.25, 0.0, 0.25, 0.5])
        self.assertEqual(intervals, [[-0.5, 0.5]])
        self.assertEqual(counts, [5])
        with self.assertRaises(ValueError):
            sample_thickness_ranges([(-0.5, 0.5)], count=5)

    def test_default_count_and_length_proportional_sampling(self):
        values, intervals, counts = sample_thickness_ranges(
            [(0.08, 0.10), (0.14, 0.18)]
        )
        self.assertEqual(len(values), 1000)
        self.assertEqual(counts, [333, 667])
        self.assertEqual(len(np.unique(values)), 1000)
        self.assertFalse(np.any((values > 0.10) & (values < 0.14)))
        offset = 0
        for (lower, upper), count in zip(intervals, counts):
            part = values[offset : offset + count]
            self.assertEqual(part[0], lower)
            self.assertEqual(part[-1], upper)
            np.testing.assert_allclose(np.diff(part), (upper - lower) / (count - 1))
            offset += count

    def test_overlapping_ranges_do_not_duplicate_samples(self):
        values, intervals, counts = sample_thickness_ranges(
            [(0.12, 0.20), (0.08, 0.14), (0.20, 0.24)], count=11
        )
        self.assertEqual(intervals, [[0.08, 0.24]])
        self.assertEqual(counts, [11])
        np.testing.assert_allclose(values, np.linspace(0.08, 0.24, 11))

    def test_invalid_ranges_and_insufficient_counts(self):
        for ranges, count in [
            ([], 1000),
            ([(0.12, 0.08)], 1000),
            ([(0.12, 0.12)], 1000),
            ([(0.0, 0.12)], 1000),
            ([(0.08, np.nan)], 1000),
            ([(0.08, 0.10), (0.14, 0.18)], 3),
            ([(0.08, 0.12)], 2.5),
        ]:
            with self.subTest(ranges=ranges, count=count):
                with self.assertRaises(ValueError):
                    sample_thickness_ranges(ranges, count)

    def test_airfoil_surfaces_meet_exactly_at_leading_edge(self):
        curve = NACA4Curve(t=0.12, center=(-0.5, 0.0))
        np.testing.assert_array_equal(curve.evaluate(0.0), [-0.5, 0.0])
        np.testing.assert_array_equal(curve.evaluate(0.0), curve.evaluate(1.0))

    def test_two_files_align_with_sampled_boundaries_and_pde_targets(self):
        nx, ny, count = 21, 11, 4
        with tempfile.TemporaryDirectory() as temporary:
            with contextlib.redirect_stdout(io.StringIO()):
                output = generate_circle_airfoil_t_dataset(
                    "air_t_test", [(0.08, 0.10), (0.16, 0.20)],
                    nx=nx, ny=ny, count=count, output_root=temporary,
                )
            self.assertEqual(
                {path.name for path in output.iterdir() if path.is_file()},
                {"dataset.npy", "boundary_geometry.npy"},
            )
            numerical = np.load(output / "dataset.npy", allow_pickle=True)
            geometry_rows = np.load(output / "boundary_geometry.npy", allow_pickle=True)
            self.assertEqual(numerical.shape, (count,))
            self.assertEqual(geometry_rows.shape, (count,))
            dataset = ParametricDataset2D(output)
            self.assertEqual(dataset.grid_shape, (nx, ny))
            self.assertEqual(dataset.boundary_points.shape, (count, 60, 2))
            self.assertEqual(dataset.mesh_points.shape, (count, ny * nx, 2))
            self.assertEqual(set(vars(dataset)), {
                "root", "load_boundary", "grid_shape", "boundary_points", "mesh_points",
                "boundary",
            })
            metadata = read_dataset_metadata(output)
            self.assertTrue(metadata["all_pde_converged"])
            boundary_ids = GenerateQuadMeshBoundaryIndex(ny, nx)
            counts = dict(bottom=nx, right=ny, top=nx, left=ny)
            for index in range(count):
                item = dataset[index]
                self.assertEqual(set(item), {
                    "boundary_points", "mesh_points", "boundary",
                })
                boundary = item["boundary"]
                self.assertTrue(boundary.validate())
                boundary.top.check_junctions(tol=1e-12)
                np.testing.assert_allclose(boundary.top.evaluate(0.5), [-0.5, 0.0])
                np.testing.assert_array_equal(
                    item["boundary_points"],
                    boundary.sample_closed(counts)[:-1].astype(np.float32),
                )
                self.assertEqual(item["mesh_points"].shape, (ny * nx, 2))
                np.testing.assert_array_equal(
                    item["mesh_points"], numerical[index]["mesh_points"]
                )
                self.assertTrue(np.shares_memory(item["mesh_points"], dataset.mesh_points))
                self.assertTrue(np.shares_memory(
                    item["boundary_points"], dataset.boundary_points
                ))
                mesh = item["mesh_points"].reshape(ny, nx, 2)
                self.assertEqual(numerical[index]["mesh_points"].shape, (ny * nx, 2))
                self.assertEqual(numerical[index]["mesh_points"].dtype, np.float32)
                for actual, expected in [
                    (mesh[0], boundary.bottom.sample(nx)),
                    (mesh[-1], boundary.top.sample(nx)),
                    (mesh[:, 0], boundary.left.sample(ny)),
                    (mesh[:, -1], boundary.right.sample(ny)),
                ]:
                    np.testing.assert_allclose(actual, expected, atol=1e-7)
                np.testing.assert_array_equal(
                    mesh.reshape(-1, 2)[boundary_ids], item["boundary_points"]
                )
                self.assertEqual(geometry_rows[index]["sample_id"], index)
                self.assertEqual(numerical[index]["sample_id"], index)
                self.assertEqual(
                    geometry_rows[index]["dataset_id"], numerical[index]["dataset_id"]
                )
                thickness = geometry_rows[index]["parameters"]["t"]
                self.assertEqual(boundary.metadata["t"], thickness)
                self.assertEqual(read_sample_parameters(output, index), {"t": thickness})
                info = metadata["pde_diagnostics"][index]
                self.assertEqual(info["sample_id"], index)
                self.assertEqual(info["parameters"]["t"], thickness)
                self.assertTrue(info["converged"])
                self.assertGreater(info["iterations"], 1)
                self.assertLess(info["relative_difference"], 1e-6)
                initial = tfi_from_curves(boundary, nx=nx, ny=ny).reshape(ny, nx, 2)
                self.assertGreater(np.max(np.abs(mesh - initial)), 1e-3)

    def test_meshonet_reader_never_opens_geometry_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            with contextlib.redirect_stdout(io.StringIO()):
                output = generate_circle_airfoil_t_dataset(
                    "numeric_only", count=4, nx=11, ny=9, output_root=temporary
                )
            original_load = np.load
            opened = []

            def load_numeric_only(path, *args, **kwargs):
                opened.append(Path(path).name)
                if Path(path).name == "boundary_geometry.npy":
                    raise AssertionError("MeshONet must not load geometry records")
                return original_load(path, *args, **kwargs)

            with patch("src.data.dataset.np.load", side_effect=load_numeric_only):
                dataset = ParametricDataset2D(output / "dataset.npy", load_boundary=False)
            self.assertEqual(opened, ["dataset.npy"])
            self.assertEqual(set(vars(dataset)), {
                "root", "load_boundary", "grid_shape", "boundary_points", "mesh_points",
            })
            numerical = original_load(output / "dataset.npy", allow_pickle=True)
            for index, record in enumerate(numerical):
                item = dataset[index]
                self.assertEqual(set(item), {"boundary_points", "mesh_points"})
                for key in ("boundary_points", "mesh_points"):
                    np.testing.assert_array_equal(item[key], record[key])
            from torch.utils.data import DataLoader

            batch = next(iter(DataLoader(dataset, batch_size=2)))
            self.assertEqual(set(batch), {"boundary_points", "mesh_points"})
            self.assertEqual(tuple(batch["mesh_points"].shape), (2, 99, 2))
            (output / "boundary_geometry.npy").unlink()
            self.assertEqual(len(ParametricDataset2D(output, load_boundary=False)), 4)
            with self.assertRaises(FileNotFoundError):
                ParametricDataset2D(output, load_boundary=True)

    def test_mismatched_or_reordered_pairs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with contextlib.redirect_stdout(io.StringIO()):
                first = generate_circle_airfoil_t_dataset(
                    "first", count=4, nx=11, ny=9, output_root=temporary
                )
                second = generate_circle_airfoil_t_dataset(
                    "second", count=4, nx=11, ny=9, output_root=temporary
                )
            original = np.load(first / "boundary_geometry.npy", allow_pickle=True)
            unrelated = np.load(second / "boundary_geometry.npy", allow_pickle=True)
            np.save(first / "boundary_geometry.npy", unrelated, allow_pickle=True)
            with self.assertRaisesRegex(ValueError, "mismatch: dataset_id"):
                ParametricDataset2D(first)
            np.save(first / "boundary_geometry.npy", original[[0, 2, 1, 3]], allow_pickle=True)
            with self.assertRaisesRegex(ValueError, "sample_id mismatch"):
                ParametricDataset2D(first)
            np.save(first / "boundary_geometry.npy", original[:-1], allow_pickle=True)
            with self.assertRaisesRegex(ValueError, "record count"):
                ParametricDataset2D(first)

    def test_nonconverged_pde_does_not_publish_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "PDE did not converge"):
                generate_circle_airfoil_t_dataset(
                    "unfinished",
                    [(0.08, 0.12)],
                    count=2,
                    output_root=temporary,
                    pde_max_iter=1,
                    pde_tolerance=1e-12,
                )
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_existing_dataset_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("existing data", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                generate_circle_airfoil_t_dataset(
                    "existing", [(0.08, 0.12)], count=2, output_root=temporary
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing data")

    def test_geometry_is_python_only_and_output_defaults_to_project(self):
        self.assertEqual(DEFAULT_OUTPUT_ROOT, ROOT / "data" / "raw" / "2D")
        for generator in (generate_explicit_dataset, generate_circle_airfoil_t_dataset):
            default = inspect.signature(generator).parameters["output_root"].default
            self.assertEqual(default, DEFAULT_OUTPUT_ROOT)
        for option in (
            ["--m", "0.03"],
            ["--p", "0.3"],
            ["--c", "0.8"],
            ["--radius", "2"],
            ["--center", "0", "0"],
            ["--t-range", "0.08", "0.12"],
            ["--output-root", "unused"],
            ["--boundary-type", "rectangle"],
            ["--sample-count", "8"],
            ["--seed", "123"],
        ):
            with self.subTest(option=option):
                with contextlib.redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(SystemExit) as raised:
                        main(option)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("unrecognized arguments", stderr.getvalue())

    def test_catalog_records_actual_parameters_and_retains_other_datasets(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixed = dict(m=0.01, p=0.35, c=0.8, radius=1.5, center=[0.1, -0.2])
            with contextlib.redirect_stdout(io.StringIO()):
                output = generate_circle_airfoil_t_dataset(
                    "custom_airfoil",
                    [(0.10, 0.12)],
                    nx=11,
                    ny=9,
                    count=2,
                    output_root=temporary,
                    **fixed,
                )
                rectangle = generate_explicit_dataset(
                    "rectangle_test",
                    "rectangle",
                    [{}],
                    grid_shape=(4, 3),
                    output_root=temporary,
                )
            metadata = read_dataset_metadata(output)
            self.assertEqual(metadata["fixed_parameters"], fixed)
            self.assertEqual(metadata["parameter_ranges"], {"t": [[0.10, 0.12]]})
            other = read_dataset_metadata(rectangle)
            self.assertEqual(
                other["fixed_parameters"],
                dict(x_min=-1, x_max=1, y_min=-1, y_max=1),
            )
            catalog = (Path(temporary) / "README.md").read_text(encoding="utf-8")
            self.assertEqual(catalog.count("### custom_airfoil"), 1)
            self.assertEqual(catalog.count("### rectangle_test"), 1)
            section = catalog.split("### custom_airfoil", 1)[1].split("### rectangle_test", 1)[0]
            fixed_line = next(
                line for line in section.splitlines() if line.startswith("- 固定几何参数：")
            )
            self.assertEqual(json.loads(fixed_line.split("`")[1]), fixed)
            self.assertIn('"t": [[0.1, 0.12]]', catalog)
            self.assertIn("(ny*nx, 2)", catalog)
            self.assertIn("Winslow PDE 迭代结果", catalog)
            self.assertIn("custom_airfoil/boundary_geometry.npy", catalog)
            self.assertNotIn("\ufffd", catalog)
            self.assertEqual(len([p for p in output.iterdir() if p.is_file()]), 2)


if __name__ == "__main__":
    unittest.main()
