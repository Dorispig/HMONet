"""Regression tests for legacy and power-Fourier coordinate lifting."""

import math

import torch

from src.models.model import HMONet2D, MeshONet2D, MeshONet3D
from src.models.modules import paper_coordinate_lift, power_fourier_lift
from src.models.pi_meshonet import PIMeshONet2D, PIMeshONet3D
import pytest


class TestCoordinateLifting:
    def test_paper_lift_matches_published_two_dimensional_formula(self):
        coordinates = torch.tensor([[[0.25, 0.5]]], dtype=torch.float64)
        actual = paper_coordinate_lift(coordinates, q=3)
        expected = torch.tensor(
            [[[
                math.sin(0.25), math.cos(0.25),
                math.sin(0.5), math.cos(0.5),
                0.25, 0.5,
                0.25**2, 0.5**2,
                0.25**3, 0.5**3,
            ]]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(actual, expected)
        assert actual.shape[-1] == 4 + 2 * 3

    def test_paper_lift_three_dimensional_width(self):
        coordinates = torch.tensor([[[0.2, 0.3, 0.4]]])
        actual = paper_coordinate_lift(coordinates, q=6)
        assert actual.shape[-1] == 6 + 3 * 6

    def test_two_dimensional_order_for_power_two(self):
        coordinates = torch.tensor([[[0.25, 0.5]]], dtype=torch.float64)
        actual = power_fourier_lift(coordinates, 2)
        powers = torch.tensor([[[0.25, 0.25**2, 0.5, 0.5**2]]], dtype=torch.float64)
        expected = torch.cat(
            (powers, torch.sin(2 * math.pi * powers), torch.cos(2 * math.pi * powers)),
            dim=-1,
        )
        torch.testing.assert_close(actual, expected)
        assert (actual.shape[-1]) == (12)

    def test_three_dimensional_width_and_axis_order(self):
        coordinates = torch.tensor([[[0.2, 0.3, 0.4]]])
        actual = power_fourier_lift(coordinates, 2)
        expected_powers = torch.tensor([[[0.2, 0.2**2, 0.3, 0.3**2, 0.4, 0.4**2]]])
        torch.testing.assert_close(actual[..., :6], expected_powers)
        assert (actual.shape[-1]) == (18)

    def test_default_and_explicit_legacy_have_identical_state(self):
        constructors = (
            lambda: MeshONet2D(
                3,
                3,
                branch_dim=4,
                trunk_dim=6,
                adjust_hidden_dim=4,
                branch_hidden_dim=4,
                trunk_hidden_dim=4,
                output_dim=4,
                branch_mlp_hidden_dim=4,
                adjust_hidden_layer_num=1,
                branch_hidden_layer_num=1,
                trunk_hidden_layer_num=1,
                branch_mlp_hidden_layer_num=1,
            ),
            lambda: HMONet2D(
                3,
                3,
                branch_dim=4,
                trunk_dim=6,
                adjust_hidden_dim=4,
                branch_hidden_dim=4,
                trunk_hidden_dim=4,
                output_dim=4,
                branch_mlp_hidden_dim=4,
                adjust_hidden_layer_num=1,
                branch_hidden_layer_num=1,
                trunk_hidden_layer_num=1,
                branch_mlp_hidden_layer_num=1,
            ),
            lambda: PIMeshONet2D(
                3,
                3,
                branch_dim=4,
                trunk_dim=6,
                adjust_hidden_dim=4,
                branch_hidden_dim=4,
                trunk_hidden_dim=4,
                output_dim=4,
                adjust_hidden_layer_num=1,
                branch_hidden_layer_num=1,
                trunk_hidden_layer_num=1,
            ),
        )
        explicit = (
            lambda: MeshONet2D(
                3,
                3,
                branch_dim=4,
                trunk_dim=6,
                adjust_hidden_dim=4,
                branch_hidden_dim=4,
                trunk_hidden_dim=4,
                output_dim=4,
                branch_mlp_hidden_dim=4,
                adjust_hidden_layer_num=1,
                branch_hidden_layer_num=1,
                trunk_hidden_layer_num=1,
                branch_mlp_hidden_layer_num=1,
                coordinate_lift_mode="legacy",
                coordinate_lift_max_power=1,
            ),
            lambda: HMONet2D(
                3,
                3,
                branch_dim=4,
                trunk_dim=6,
                adjust_hidden_dim=4,
                branch_hidden_dim=4,
                trunk_hidden_dim=4,
                output_dim=4,
                branch_mlp_hidden_dim=4,
                adjust_hidden_layer_num=1,
                branch_hidden_layer_num=1,
                trunk_hidden_layer_num=1,
                branch_mlp_hidden_layer_num=1,
                coordinate_lift_mode="legacy",
                coordinate_lift_max_power=1,
            ),
            lambda: PIMeshONet2D(
                3,
                3,
                branch_dim=4,
                trunk_dim=6,
                adjust_hidden_dim=4,
                branch_hidden_dim=4,
                trunk_hidden_dim=4,
                output_dim=4,
                adjust_hidden_layer_num=1,
                branch_hidden_layer_num=1,
                trunk_hidden_layer_num=1,
                coordinate_lift_mode="legacy",
                coordinate_lift_max_power=1,
            ),
        )
        for old_factory, explicit_factory in zip(constructors, explicit):
            torch.manual_seed(31)
            old_model = old_factory()
            torch.manual_seed(31)
            explicit_model = explicit_factory()
            assert (old_model.state_dict().keys()) == (
                explicit_model.state_dict().keys()
            )
            for name, value in old_model.state_dict().items():
                torch.testing.assert_close(value, explicit_model.state_dict()[name])

    def test_new_mode_sets_derived_trunk_widths(self):
        model_2d = HMONet2D(
            3,
            3,
            branch_dim=4,
            trunk_dim=16,
            adjust_hidden_dim=4,
            branch_hidden_dim=4,
            trunk_hidden_dim=4,
            output_dim=4,
            branch_mlp_hidden_dim=4,
            adjust_hidden_layer_num=1,
            branch_hidden_layer_num=1,
            trunk_hidden_layer_num=1,
            branch_mlp_hidden_layer_num=1,
            coordinate_lift_mode="power_fourier",
            coordinate_lift_max_power=2,
        )
        model_3d = MeshONet3D(
            3,
            3,
            3,
            branch_dim=4,
            trunk_dim=16,
            adjust_hidden_dim=4,
            branch_hidden_dim=4,
            trunk_hidden_dim=4,
            output_dim=4,
            adjust_hidden_layer_num=1,
            branch_hidden_layer_num=1,
            trunk_hidden_layer_num=1,
            branch_mlp_hidden_dim=4,
            branch_mlp_hidden_layer_num=1,
            coordinate_lift_mode="power_fourier",
            coordinate_lift_max_power=2,
        )
        pi_3d = PIMeshONet3D(
            3,
            3,
            3,
            branch_dim=4,
            trunk_dim=16,
            adjust_hidden_dim=4,
            branch_hidden_dim=4,
            trunk_hidden_dim=4,
            output_dim=4,
            adjust_hidden_layer_num=1,
            branch_hidden_layer_num=1,
            trunk_hidden_layer_num=1,
            coordinate_lift_mode="power_fourier",
            coordinate_lift_max_power=2,
        )
        assert (model_2d.trunk_dim) == (12)
        assert (model_2d.trunk_net.input_dim) == (12)
        assert (model_3d.trunk_dim) == (18)
        assert (model_3d.trunk.input_dim) == (18)
        assert (pi_3d.trunk_dim) == (18)

    def test_invalid_max_power_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            power_fourier_lift(torch.zeros(1, 1, 2), 0)

    def test_invalid_paper_q_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            paper_coordinate_lift(torch.zeros(1, 1, 2), 0)
