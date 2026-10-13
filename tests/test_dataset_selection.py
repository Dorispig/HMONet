"""Check editable split selection, CLI overrides and real paired PDE outputs."""

import contextlib
import copy
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.generation import make_dataset_parametric_2d as generator
from src.data.dataset import ParametricDataset2D, read_dataset_metadata


class DatasetSelectionTests(unittest.TestCase):
    def test_default_training_ranges_and_automatic_names(self):
        family = generator.DATASET_FAMILIES["circle_airfoil_t"]
        self.assertEqual(family["interpolation"], [[0.055, 0.25], [0.35, 0.60]])
        self.assertEqual(family["extrapolation"], [[0.055, 0.60]])
        values, _, counts = generator.sample_thickness_ranges(family["interpolation"])
        self.assertEqual(len(values), 1000)
        self.assertEqual(counts, [438, 562])
        self.assertFalse(np.any((values > 0.25) & (values < 0.35)))
        for split, tag in (("interpolation", "int"), ("extrapolation", "ext")):
            self.assertEqual(
                generator.build_dataset_name(
                    "circle_airfoil_t", split, 0, 1000, 21, 20
                ),
                f"air_t_{tag}_0_1000_21_20",
            )
        for alias, split in (
            ("int", "interpolation"),
            ("INT", "interpolation"),
            ("ext", "extrapolation"),
            ("Extrapolation", "extrapolation"),
        ):
            self.assertEqual(generator.normalize_split(alias), split)
            self.assertEqual(
                generator.build_dataset_name(
                    "circle_airfoil_t", alias, 0, 1000, 21, 20
                ),
                generator.build_dataset_name(
                    "circle_airfoil_t", split, 0, 1000, 21, 20
                ),
            )

    def test_original_cli_options_aliases_and_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(
                    generator, "SELECTED_DATASETS", [("circle_airfoil_t", "EXT")]
                ),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                outputs = generator.main(
                    [
                        "--size",
                        "4",
                        "--grid-shape",
                        "7x5",
                        "--winslow-max-iter",
                        "500",
                        "--winslow-tolerance",
                        "1e-6",
                        "--timing-samples",
                        "2",
                        "--output-dir",
                        temporary,
                    ]
                )
            self.assertEqual(outputs[0].name, "air_t_ext_0_4_7_5")
            metadata = read_dataset_metadata(outputs[0])
            self.assertEqual(metadata["split"], "extrapolation")
            self.assertEqual(metadata["function"], "circle_airfoil")
            self.assertEqual(metadata["parameter"], "t")
            self.assertEqual(metadata["parameter_intervals"], [[0.055, 0.60]])
            self.assertEqual(metadata["winslow_max_iter"], 500)
            self.assertEqual(metadata["timing_samples_requested"], 2)
            self.assertGreater(metadata["sample_generation_time_seconds"]["total"], 0)
            for percent in range(10, 101, 10):
                self.assertIn(f"Progress: {percent}%", stdout.getvalue())
            self.assertIn("sample 2/4:", stdout.getvalue())
            self.assertNotIn("sample 3/4:", stdout.getvalue())

    def test_family_fields_select_the_curve_constructor_and_parameter(self):
        specification = {
            "output_name": "rectangle_width",
            "function": "rectangle",
            "parameter": "x_max",
            "description": "Rectangle with a varying right boundary.",
            "interpolation": [[1.0, 1.2], [1.4, 1.6]],
            "extrapolation": [[1.0, 1.6]],
        }
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.dict(
                    generator.DATASET_FAMILIES, {"rectangle_width": specification}
                ),
                patch.dict(
                    generator.DATASET_GENERATORS,
                    {
                        "rectangle": generator.BOUNDARY_GENERATORS["rectangle"],
                    },
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                output = generator.generate_dataset(
                    "rectangle_width",
                    "int",
                    size=4,
                    nx=4,
                    ny=3,
                    output_dir=temporary,
                )
            self.assertEqual(output.name, "rectangle_width_int_0_4_4_3")
            metadata = read_dataset_metadata(output)
            self.assertEqual(metadata["parameter"], "x_max")
            self.assertEqual(
                metadata["fixed_parameters"],
                {
                    "x_min": -1,
                    "y_min": -1,
                    "y_max": 1,
                },
            )
            dataset = ParametricDataset2D(output)
            np.testing.assert_allclose(
                dataset.mesh_points[..., 0].max(axis=1), [1.0, 1.2, 1.4, 1.6]
            )

    def test_cli_generates_both_selected_pde_datasets(self):
        generate = generator.generate_selected_datasets
        with tempfile.TemporaryDirectory() as temporary:

            def generate_in_temporary(options, output_dir=None):
                with patch.object(
                    generator,
                    "SELECTED_DATASETS",
                    [("circle_airfoil_t", "int"), ("circle_airfoil_t", "ext")],
                ):
                    return generate(options, output_dir=temporary)

            with contextlib.redirect_stdout(io.StringIO()):
                with patch.object(
                    generator,
                    "generate_selected_datasets",
                    side_effect=generate_in_temporary,
                ):
                    outputs = generator.main(
                        [
                            "--nx",
                            "9",
                            "--ny",
                            "7",
                            "--count",
                            "6",
                            "--timing-samples",
                            "0",
                        ]
                    )
            self.assertEqual(
                [path.name for path in outputs],
                [
                    "air_t_int_0_6_9_7",
                    "air_t_ext_0_6_9_7",
                ],
            )
            expected_values = [
                np.concatenate(
                    [np.linspace(0.055, 0.25, 3), np.linspace(0.35, 0.60, 3)]
                ),
                np.linspace(0.055, 0.60, 6),
            ]
            for output, split, expected in zip(
                outputs, ("interpolation", "extrapolation"), expected_values
            ):
                metadata = read_dataset_metadata(output)
                self.assertEqual(metadata["split"], split)
                self.assertEqual(metadata["size"], 6)
                self.assertEqual(metadata["grid_shape"], [9, 7])
                self.assertTrue(metadata["all_pde_converged"])
                rows = np.load(output / "boundary_geometry.npy", allow_pickle=True)
                np.testing.assert_allclose(
                    [row["parameters"]["t"] for row in rows], expected
                )
                data = ParametricDataset2D(output)
                self.assertTrue(np.isfinite(data.mesh_points).all())
                self.assertEqual(
                    set(data[0]), {"boundary_points", "mesh_points", "boundary"}
                )
                self.assertEqual(
                    {path.name for path in output.iterdir() if path.is_file()},
                    {"dataset.npy", "boundary_geometry.npy"},
                )
            catalog = (Path(temporary) / "README.md").read_text(encoding="utf-8")
            for output in outputs:
                self.assertIn(f"### {output.name}", catalog)

    def test_editable_selection_defaults_and_ranges(self):
        settings = dict(generator.DEFAULT_GENERATION_OPTIONS, nx=7, ny=5, size=4)
        families = copy.deepcopy(generator.DATASET_FAMILIES)
        families["circle_airfoil_t"]["extrapolation"] = [[0.10, 0.20]]
        generate = generator.generate_selected_datasets
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(
                    generator,
                    "SELECTED_DATASETS",
                    [("circle_airfoil_t", "extrapolation")],
                ),
                patch.object(generator, "DEFAULT_GENERATION_OPTIONS", settings),
                patch.object(generator, "DATASET_FAMILIES", families),
                patch.object(
                    generator,
                    "generate_selected_datasets",
                    side_effect=lambda options, output_dir=None: generate(
                        options, output_dir=temporary
                    ),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                outputs = generator.main(["--ny", "9", "--count", "3"])
            self.assertEqual(len(outputs), 1)
            self.assertEqual(outputs[0].name, "air_t_ext_0_3_7_9")
            rows = np.load(outputs[0] / "boundary_geometry.npy", allow_pickle=True)
            np.testing.assert_allclose(
                [row["parameters"]["t"] for row in rows], [0.10, 0.15, 0.20]
            )
            self.assertEqual(settings["ny"], 5)
            self.assertEqual(settings["size"], 4)

    def test_invalid_selection_is_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            for selection in (
                [],
                [("unknown", "interpolation")],
                [("circle_airfoil_t", "invalid")],
                [("circle_airfoil_t", "interpolation", 0, "extra")],
                [("circle_airfoil_t", "interpolation")] * 2,
                [("circle_airfoil_t", "interpolation"), ("circle_airfoil_t", "int")],
            ):
                with self.subTest(selection=selection):
                    with patch.object(generator, "SELECTED_DATASETS", selection):
                        with self.assertRaises(ValueError):
                            generator.generate_selected_datasets(output_dir=temporary)
            self.assertEqual(list(Path(temporary).iterdir()), [])
            family, split, version = generator.SELECTED_DATASETS[0]
            spec = generator.DATASET_FAMILIES[family]
            grid = spec.get("grid_shape", [21, 21])
            (
                Path(temporary)
                / generator.build_dataset_name(family, split, version, 1000, *grid)
            ).mkdir()
            existing = list(Path(temporary).iterdir())
            with patch.object(generator, "generate_dataset") as generate:
                with self.assertRaises(FileExistsError):
                    generator.generate_selected_datasets(output_dir=temporary)
                generate.assert_not_called()
            self.assertEqual(list(Path(temporary).iterdir()), existing)

    def test_invalid_grid_options_fail_before_generation(self):
        for arguments in (
            ["--grid-shape", "21x21x21"],
            ["--grid-shape", "2x21"],
            ["--grid-shape", "invalid"],
            ["--grid-shape", "21x21", "--nx", "11"],
        ):
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        generator.main(arguments)
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
