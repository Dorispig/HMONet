"""Regression tests for the current six 3-D dataset families."""

import json
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
        assert [item[0] for item in SELECTED_DATASETS] == [
            "cylindrical_domain_with_airfoil_hole", "sidewall_deformed_block",
            "rounded_u_notch_extrusion", "tapered_elliptic_bore_block",
            "bent_annular_duct", "lobed_annular_duct",
        ]
        assert [item[1] for item in SELECTED_DATASETS] == ["extrapolation"] * 6
        assert [item[3:] for item in SELECTED_DATASETS] == [
            (11, 21, 11), (11, 21, 11), (41, 41, 11),
            (11, 21, 11), (11, 21, 11), (11, 21, 11),
        ]
        assert (
            build_dataset_name("rounded_u_notch_extrusion", "extrapolation", 1, 500, 41, 41, 11)
            == "run_e_ext_1_500_41_41_11"
        )

    def test_interpolation_ranges_and_small_round_trip(self):
        specification = DATASET_FAMILIES["rounded_u_notch_extrusion"]
        values = sample_parameter_ranges(specification["interpolation"], 4)
        np.testing.assert_allclose(values, [0.55, 1.00, 1.30, 1.50])
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
                workers=2,
            )
            dataset = ParametricDataset3D(output, load_boundary=True)
            assert len(dataset) == 2
            assert dataset.grid_shape == (5, 7, 5)
            assert np.isfinite(dataset.mesh_points).all()
            assert read_sample_parameters(Path(output), 0)["sidewall_amplitude"] == 0.04
            assert read_sample_parameters(Path(output), 1)["sidewall_amplitude"] == 0.30
            metadata = json.loads((Path(output) / "metadata.json").read_text(encoding="utf-8"))
            assert metadata["workers"] == 2

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
