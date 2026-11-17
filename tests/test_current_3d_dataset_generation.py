"""Regression tests for the current six 3-D dataset families."""

import tempfile
from pathlib import Path

import numpy as np

from src.data.dataset import ParametricDataset3D, read_sample_parameters
from src.data.generation.make_dataset_parametric_3d import (
    DATASET_FAMILIES,
    SELECTED_DATASETS,
    build_dataset_name,
    generate_dataset,
    generate_family_dataset,
    sample_parameter_ranges,
)


class TestCurrent3DDatasetGeneration:
    def test_default_selection_uses_current_names_and_shapes(self):
        assert [item[0] for item in SELECTED_DATASETS] == list(DATASET_FAMILIES)
        assert [item[1] for item in SELECTED_DATASETS] == ["extrapolation"] * 6
        assert all(item[3:] == (11, 21, 11) for item in SELECTED_DATASETS)
        assert (
            build_dataset_name("rounded_u_groove_extrusion", "extrapolation", 1, 500, 11, 21, 11)
            == "ugc_d_ext_1_500_11_21_11"
        )

    def test_interpolation_ranges_and_small_round_trip(self):
        specification = DATASET_FAMILIES["rounded_u_groove_extrusion"]
        values = sample_parameter_ranges(specification["interpolation"], 4)
        np.testing.assert_allclose(values, [0.30, 0.65, 1.00, 1.30])
        with tempfile.TemporaryDirectory() as temporary:
            output = generate_family_dataset(
                "sidewall_deformed_block",
                "extrapolation",
                size=2,
                nx=5,
                ny=7,
                nz=5,
                sample_count=4,
                output_root=temporary,
                timing_samples=0,
            )
            dataset = ParametricDataset3D(output, load_boundary=True)
            assert len(dataset) == 2
            assert dataset.grid_shape == (5, 7, 5)
            assert np.isfinite(dataset.mesh_points).all()
            assert read_sample_parameters(Path(output), 0)["sidewall_amplitude"] == 0.04
            assert read_sample_parameters(Path(output), 1)["sidewall_amplitude"] == 0.30

    def test_existing_output_is_skipped_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "existing"
            existing.mkdir()
            assert (
                generate_dataset(
                    "existing",
                    "sidewall_deformed_block",
                    [{"sidewall_amplitude": 0.15}],
                    grid_shape=(5, 7, 5),
                    sample_count=1,
                    output_root=temporary,
                    timing_samples=0,
                )
                == existing
            )
