"""Contracts for the restored HMONet hard map and configurable encoders."""

import unittest
import torch
from src.models.model import HMONet2D, HMONet3D

SMALL = dict(
    branch_dim=4, trunk_dim=3, adjust_hidden_dim=4, branch_hidden_dim=4,
    trunk_hidden_dim=4, adjust_hidden_layer_num=1, branch_hidden_layer_num=1,
    trunk_hidden_layer_num=1, output_dim=4, branch_mlp_hidden_dim=4,
    branch_mlp_hidden_layer_num=1,
)


class HMONetProjectionConfigurationTests(unittest.TestCase):
    def test_sigmoid_and_softplus_coefficients_are_positive(self):
        joint2 = torch.randn(2, 8)
        for softplus in (False, True):
            model = HMONet2D(3, 3, projection_softplus_enabled=softplus, **SMALL)
            coefficients = model.projection_weights(joint2, None)
            self.assertTrue(bool((coefficients > 0).all()))
            if not softplus:
                self.assertTrue(bool((coefficients < 1).all()))

    def test_trunk_coefficients_are_query_dependent(self):
        model = HMONet2D(
            3, 3, trunk_projection_coefficients_enabled=True,
            projection_softplus_enabled=True, **SMALL,
        )
        joint = torch.randn(2, 8)
        trunk = torch.randn(2, 7, 4)
        coefficients = model.projection_weights(joint, trunk)
        self.assertEqual(coefficients.shape, (2, 7, 4))
        self.assertGreater(float((coefficients[:, 1:] - coefficients[:, :-1]).abs().sum()), 0.0)

    def test_attention_encoder_is_optional_and_differentiable(self):
        options = {**SMALL, "trunk_dim": 2}
        model = HMONet2D(3, 3, boundary_self_attention_enabled=True, **options)
        boundary = torch.rand(2, 8, 2)
        query = torch.rand(2, 5, 2, requires_grad=True)
        result = torch.cat(model(*boundary.unbind(-1), *query.unbind(-1)), -1)
        result.square().mean().backward()
        self.assertEqual(result.shape, (2, 5, 2))
        self.assertIsNotNone(query.grad)

    def test_current_mapping_is_exact_on_2d_edges_and_3d_faces(self):
        model2 = HMONet2D(
            3, 3, trunk_projection_coefficients_enabled=True,
            projection_softplus_enabled=True, **{**SMALL, "trunk_dim": 2},
        ).eval()
        boundary2 = torch.rand(1, 8, 2)
        query2 = torch.tensor([[[0.0, 0.3], [1.0, 0.7], [0.2, 0.0], [0.8, 1.0]]])
        projected2 = model2.project(*boundary2.unbind(-1), *query2.unbind(-1))
        actual2 = torch.cat(model2(*boundary2.unbind(-1), *query2.unbind(-1)), -1)
        expected2 = torch.stack(tuple(projected2[:, index, index] for index in range(4)), dim=1)
        torch.testing.assert_close(actual2, expected2)

        model3 = HMONet3D(
            3, 3, 3, trunk_projection_coefficients_enabled=True,
            projection_softplus_enabled=True, **SMALL,
        ).eval()
        boundary3 = torch.rand(1, 26, 3)
        query3 = torch.tensor([[[0.0, .3, .4], [1.0, .3, .4], [.2, 0.0, .4],
                                [.2, 1.0, .4], [.2, .3, 0.0], [.2, .3, 1.0]]])
        projected3 = model3.project(*boundary3.unbind(-1), *query3.unbind(-1))
        actual3 = torch.cat(model3(*boundary3.unbind(-1), *query3.unbind(-1)), -1)
        expected3 = torch.stack(tuple(projected3[:, index, index] for index in range(6)), dim=1)
        torch.testing.assert_close(actual3, expected3)

    def test_legacy_3d_state_loads_with_unit_branch_coefficients(self):
        source = HMONet3D(3, 3, 3, **SMALL)
        legacy = source.state_dict()
        legacy.pop("projection_weight_head.weight")
        legacy.pop("projection_weight_head.bias")
        restored = HMONet3D(3, 3, 3, **SMALL)
        restored.load_state_dict(legacy, strict=True)
        coefficients = restored.projection_coefficients(torch.randn(2, 12))
        torch.testing.assert_close(coefficients, torch.ones_like(coefficients))


if __name__ == "__main__":
    unittest.main()
