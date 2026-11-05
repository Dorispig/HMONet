"""Tests for the reference sch_r parameter ranges and current 3-D layout."""

import tempfile
from pathlib import Path

import numpy as np

from src.data.dataset import ParametricDataset3D, read_sample_parameters
from src.data.generation.make_dataset_parametric_3d import (
    DATASET_FAMILIES,
    build_dataset_name,
    generate_stadium_cylinder_with_circular_hole_dataset,
    sample_parameter_ranges,
)


class TestSchRDatasetGeneration:
    def test_reference_ranges_and_canonical_names(self):
        specification = DATASET_FAMILIES["stadium_cylinder_with_circular_hole"]
        interpolation = sample_parameter_ranges(specification["interpolation"], 1000)
        extrapolation = sample_parameter_ranges(specification["extrapolation"], 1000)

        assert (len(interpolation)) == (1000)
        assert (interpolation[0]) == (0.12)
        assert (interpolation[499]) == (0.24)
        assert (interpolation[500]) == (0.36)
        assert (interpolation[-1]) == (0.48)
        assert not np.any((interpolation > 0.24) & (interpolation < 0.36))
        assert (extrapolation[0], extrapolation[-1]) == (0.16, 0.44)
        assert (
            build_dataset_name(
                "stadium_cylinder_with_circular_hole",
                "interpolation",
                0,
                1000,
                11,
                11,
                11,
            )
        ) == ("sch_r_int_0_1000_11_11_11")

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

            assert (len(dataset)) == (4)
            assert (dataset.grid_shape) == (5, 5, 5)
            assert (dataset.boundary_points.shape) == (4, 98, 3)
            assert (dataset.mesh_points.shape) == (4, 125, 3)
            assert (read_sample_parameters(Path(output), 0)["hole_radius"]) == (0.12)
            assert (read_sample_parameters(Path(output), 3)["hole_radius"]) == (0.48)
            assert np.isfinite(dataset.mesh_points).all()
