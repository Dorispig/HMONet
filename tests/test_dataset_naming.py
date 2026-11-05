"""Tests for the canonical cross-dimensional dataset naming contract."""

from src.data.generation.dataset_naming import (
    build_dataset_name,
    parse_dataset_name,
)
from src.data.generation import make_dataset_parametric_2d as generator_2d
import pytest


class TestDatasetNaming:
    def test_round_trip_for_two_and_three_dimensions(self):
        cases = (
            ("air_t", "int", 0, 1000, (21, 20)),
            ("sch_r", "extrapolation", 2, 500, (21, 21, 21)),
        )
        for boundary, split, version, size, grid in cases:
            name = build_dataset_name(boundary, split, version, size, *grid)
            parsed = parse_dataset_name(name, len(grid))
            assert (parsed["boundary"]) == (boundary)
            assert (parsed["version"]) == (version)
            assert (parsed["size"]) == (size)
            assert (parsed["grid_shape"]) == (grid)

    def test_invalid_components_are_rejected_without_silent_casts(self):
        invalid_calls = (
            ("bad/name", "int", 0, 10, 21, 20),
            ("air_t", "int", 0, 10, 21.5, 20),
            ("air_t", "int", 0, 10, True, 20),
            ("air_t", "int", 0, 10, 21),
            ("air_t", "int", 0, 10, 21, 20, 5, 4),
        )
        for arguments in invalid_calls:
            with pytest.raises(ValueError):
                build_dataset_name(*arguments)
        for dimension in (1, 4, 2.5, True):
            with pytest.raises(ValueError):
                parse_dataset_name("air_t_int_0_10_21_20", dimension)

    def test_airfoil_default_matches_current_training_grid(self):
        specification = generator_2d.DATASET_FAMILIES["circle_airfoil_t"]
        assert (specification["grid_shape"]) == ([21, 20])
        assert (
            generator_2d.build_dataset_name(
                "circle_airfoil_t",
                "interpolation",
                0,
                1000,
                *specification["grid_shape"],
            )
        ) == ("air_t_int_0_1000_21_20")
