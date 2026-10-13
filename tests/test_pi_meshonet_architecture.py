"""Unit tests for the paper-aligned PI-MeshONet architecture."""

import unittest

import torch

from src.models.pi_meshonet import ModifiedMLP, PIMeshONet2D, PIMeshONet3D


SMALL = dict(
    branch_dim=5,
    adjust_hidden_dim=7,
    branch_hidden_dim=8,
    trunk_hidden_dim=8,
    adjust_hidden_layer_num=1,
    branch_hidden_layer_num=2,
    trunk_hidden_layer_num=2,
    output_dim=6,
)


class PIMeshONetArchitectureTests(unittest.TestCase):
    def test_default_2d_parameter_count_matches_paper_scale(self):
        model = PIMeshONet2D(21, 20)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(parameter_count, 24450)

    def test_modified_mlp_uses_paper_gate_equation(self):
        torch.manual_seed(3)
        layer = ModifiedMLP(2, 4, 2, 3).double()
        inputs = torch.randn(5, 2, dtype=torch.float64)
        u = torch.tanh(layer.feature_u(inputs))
        v = torch.tanh(layer.feature_v(inputs))
        hidden = torch.tanh(layer.initial_hidden(inputs))
        for gate_layer in layer.gates:
            gate = torch.tanh(gate_layer(hidden))
            hidden = (1.0 - gate) * u + gate * v
        expected = layer.output(hidden)
        torch.testing.assert_close(layer(inputs), expected)

    def test_2d_and_3d_shapes_and_gradients(self):
        cases = (
            (PIMeshONet2D, (4, 3), 2, 6),
            (PIMeshONet3D, (3, 3, 3), 3, 9),
        )
        for cls, shape, dimension, trunk_dim in cases:
            model = cls(*shape, trunk_dim=trunk_dim, **SMALL)
            boundary = torch.randn(2, model.boundary_point_count, dimension)
            logical = torch.rand(2, 11, dimension)
            outputs = model(*boundary.unbind(-1), *logical.unbind(-1))
            self.assertEqual(len(outputs), dimension)
            self.assertEqual(tuple(outputs[0].shape), (2, 11, 1))
            torch.cat(outputs, -1).square().mean().backward()
            self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_single_geometry_evaluation_shape(self):
        model = PIMeshONet2D(4, 3, trunk_dim=6, **SMALL)
        boundary = torch.randn(model.boundary_point_count, 2)
        logical = torch.rand(13, 2)
        outputs = model(*boundary.unbind(-1), *logical.unbind(-1))
        self.assertEqual(tuple(outputs[0].shape), (13, 1))


if __name__ == "__main__":
    unittest.main()
