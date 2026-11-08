"""Contracts for the restored HMONet hard map and configurable encoders."""

import torch
from src.models.model import HMONet2D, HMONet3D

SMALL = dict(
    branch_dim=4,
    trunk_dim=3,
    adjust_hidden_dim=4,
    branch_hidden_dim=4,
    trunk_hidden_dim=4,
    adjust_hidden_layer_num=1,
    branch_hidden_layer_num=1,
    trunk_hidden_layer_num=1,
    output_dim=4,
    branch_mlp_hidden_dim=4,
    branch_mlp_hidden_layer_num=1,
)


class TestHMONetProjectionConfiguration:
    def test_corner_corrected_mode_preserves_2d_edges_and_3d_faces(self):
        model2 = HMONet2D(
            3, 3, hmonet_mapping_mode="corner_corrected_affine", **{**SMALL, "trunk_dim": 2}
        ).eval()
        boundary2 = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [0.5, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.5],
                    [1.0, 1.0],
                    [0.5, 1.0],
                    [0.0, 1.0],
                    [0.0, 0.5],
                ]
            ]
        )
        query2 = torch.tensor(
            [
                [
                    [0.0, 0.3],
                    [1.0, 0.7],
                    [0.2, 0.0],
                    [0.8, 1.0],
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [0.3, 0.4],
                ]
            ],
            requires_grad=True,
        )
        actual2 = torch.cat(model2(*boundary2.unbind(-1), *query2.unbind(-1)), -1)
        torch.testing.assert_close(actual2[:, :6], query2[:, :6], atol=1e-6, rtol=0)
        actual2.square().mean().backward()
        assert bool(torch.isfinite(query2.grad).all())

        model3 = HMONet3D(
            3, 3, 3, hmonet_mapping_mode="corner_corrected_affine", **SMALL
        ).eval()
        z, y, x = torch.meshgrid(
            *(torch.linspace(0, 1, 3) for _ in range(3)), indexing="ij"
        )
        grid = torch.stack((x, y, z), -1).reshape(1, -1, 3)
        boundary3 = grid[:, model3.boundary_flat_indices]
        query3 = torch.tensor(
            [
                [
                    [0.0, 0.3, 0.4],
                    [1.0, 0.3, 0.4],
                    [0.2, 0.0, 0.4],
                    [0.2, 1.0, 0.4],
                    [0.2, 0.3, 0.0],
                    [0.2, 0.3, 1.0],
                    [0.0, 0.0, 0.37],
                    [1.0, 1.0, 1.0],
                    [0.3, 0.4, 0.6],
                ]
            ],
            requires_grad=True,
        )
        actual3 = torch.cat(model3(*boundary3.unbind(-1), *query3.unbind(-1)), -1)
        torch.testing.assert_close(actual3[:, :8], query3[:, :8], atol=1e-6, rtol=0)
        actual3.square().mean().backward()
        assert bool(torch.isfinite(query3.grad).all())

    def test_sigmoid_coefficients_stay_in_the_open_unit_interval(self):
        joint2 = torch.randn(2, 8)
        trunk2 = torch.randn(2, 5, 4)
        model = HMONet2D(3, 3, **SMALL)
        coefficients = model.projection_weights(joint2, trunk2)
        assert bool((coefficients > 0).all())
        assert bool((coefficients < 1).all())

    def test_trunk_coefficients_are_query_dependent(self):
        model = HMONet2D(
            3,
            3,
            **SMALL,
        )
        joint = torch.randn(2, 8)
        trunk = torch.randn(2, 7, 4)
        coefficients = model.projection_weights(joint, trunk)
        assert (coefficients.shape) == (2, 7, 4)
        assert (float((coefficients[:, 1:] - coefficients[:, :-1]).abs().sum())) > (0.0)

    def test_attention_encoder_is_optional_and_differentiable(self):
        options = {**SMALL, "trunk_dim": 2}
        model = HMONet2D(3, 3, boundary_self_attention_enabled=True, **options)
        boundary = torch.rand(2, 8, 2)
        query = torch.rand(2, 5, 2, requires_grad=True)
        result = torch.cat(model(*boundary.unbind(-1), *query.unbind(-1)), -1)
        result.square().mean().backward()
        assert (result.shape) == (2, 5, 2)
        assert (query.grad) is not None

    def test_current_mapping_is_exact_on_2d_edges_and_3d_faces(self):
        model2 = HMONet2D(
            3,
            3,
            **{**SMALL, "trunk_dim": 2},
        ).eval()
        boundary2 = torch.rand(1, 8, 2)
        query2 = torch.tensor([[[0.0, 0.3], [1.0, 0.7], [0.2, 0.0], [0.8, 1.0]]])
        projected2 = model2.project(*boundary2.unbind(-1), *query2.unbind(-1))
        actual2 = torch.cat(model2(*boundary2.unbind(-1), *query2.unbind(-1)), -1)
        expected2 = torch.stack(
            tuple(projected2[:, index, index] for index in range(4)), dim=1
        )
        torch.testing.assert_close(actual2, expected2)

        model3 = HMONet3D(
            3,
            3,
            3,
            **SMALL,
        ).eval()
        boundary3 = torch.rand(1, 26, 3)
        query3 = torch.tensor(
            [
                [
                    [0.0, 0.3, 0.4],
                    [1.0, 0.3, 0.4],
                    [0.2, 0.0, 0.4],
                    [0.2, 1.0, 0.4],
                    [0.2, 0.3, 0.0],
                    [0.2, 0.3, 1.0],
                ]
            ]
        )
        projected3 = model3.project(*boundary3.unbind(-1), *query3.unbind(-1))
        actual3 = torch.cat(model3(*boundary3.unbind(-1), *query3.unbind(-1)), -1)
        expected3 = torch.stack(
            tuple(projected3[:, index, index] for index in range(6)), dim=1
        )
        torch.testing.assert_close(actual3, expected3)

    def test_legacy_3d_state_loads_with_sigmoid_branch_initialization(self):
        options = {**SMALL, "weight_head_mlp_enabled": False}
        source = HMONet3D(3, 3, 3, **options)
        legacy = source.state_dict()
        legacy.pop("projection_weight_head.weight")
        legacy.pop("projection_weight_head.bias")
        restored = HMONet3D(3, 3, 3, **options)
        restored.load_state_dict(legacy, strict=True)
        coefficients = restored.projection_coefficients(
            torch.randn(2, 12), torch.randn(2, 5, 4)
        )
        assert (coefficients.shape) == (2, 5, 6)
        assert bool((coefficients > 0).all())
        assert bool((coefficients < 1).all())

    def test_3d_shared_edges_and_corners_use_compatible_exact_values(self):
        model = HMONet3D(3, 3, 3, **SMALL).eval()
        boundary = torch.rand(1, 26, 3)
        query = torch.tensor([[[0.0, 0.0, 0.37], [1.0, 1.0, 1.0]]])
        projected = model.project(*boundary.unbind(-1), *query.unbind(-1))
        actual = torch.cat(model(*boundary.unbind(-1), *query.unbind(-1)), -1)
        edge_expected = 0.5 * (projected[:, 0, 0] + projected[:, 0, 2])
        corner_expected = projected[:, 1, 5]
        torch.testing.assert_close(actual[:, 0], edge_expected)
        torch.testing.assert_close(actual[:, 1], corner_expected)
