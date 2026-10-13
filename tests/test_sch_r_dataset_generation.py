"""Tests for the reference sch_r parameter ranges and current 3-D layout."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.dataset import ParametricDataset3D, read_sample_parameters
from src.data.generation.make_dataset_parametric_3d import (
    DATASET_FAMILIES,
    build_dataset_name,
    generate_stadium_cylinder_with_circular_hole_dataset,
    sample_parameter_ranges,
)


class SchRDatasetGenerationTest(unittest.TestCase):
    def test_reference_ranges_and_canonical_names(self):
        specification = DATASET_FAMILIES[
            "stadium_cylinder_with_circular_hole"
        ]
        interpolation = sample_parameter_ranges(
            specification["interpolation"], 1000
        )
        extrapolation = sample_parameter_ranges(
            specification["extrapolation"], 1000
        )

        self.assertEqual(len(interpolation), 1000)
        self.assertEqual(interpolation[0], 0.12)
        self.assertEqual(interpolation[499], 0.24)
        self.assertEqual(interpolation[500], 0.36)
        self.assertEqual(interpolation[-1], 0.48)
        self.assertFalse(
            np.any((interpolation > 0.24) & (interpolation < 0.36))
        )
        self.assertEqual((extrapolation[0], extrapolation[-1]), (0.16, 0.44))
        self.assertEqual(
            build_dataset_name(
                "stadium_cylinder_with_circular_hole",
                "interpolation",
                0,
                1000,
                11,
                11,
                11,
            ),
            "sch_r_int_0_1000_11_11_11",
        )

    def test_small_dataset_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = generate_stadium_cylinder_with_circular_hole_dataset(
                "interpolation",
                size=4,
                nx=5,
                ny=5,
                nz=5,
                sample_count=4,
                output_root=temporary,
                timing_samples=0,
            )
            dataset = ParametricDataset3D(output, load_boundary=True)

            self.assertEqual(len(dataset), 4)
            self.assertEqual(dataset.grid_shape, (5, 5, 5))
            self.assertEqual(dataset.boundary_points.shape, (4, 98, 3))
            self.assertEqual(dataset.mesh_points.shape, (4, 125, 3))
            self.assertEqual(
                read_sample_parameters(Path(output), 0)["hole_radius"], 0.12
            )
            self.assertEqual(
                read_sample_parameters(Path(output), 3)["hole_radius"], 0.48
            )
            self.assertTrue(np.isfinite(dataset.mesh_points).all())


if __name__ == "__main__":
    unittest.main()
