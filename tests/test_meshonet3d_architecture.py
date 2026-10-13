"""Tests for the paper-aligned three-dimensional MeshONet."""

import unittest

import torch

from src.models.model import MeshONet3D


class MeshONet3DArchitectureTests(unittest.TestCase):
    def test_default_32_cubed_parameter_count_and_gradients(self):
        model = MeshONet3D(32, 32, 32)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 154163)
        self.assertTrue(hasattr(model, "branch_fusion"))
        self.assertEqual(len(model.coefficient_heads), 3)

        boundary = torch.randn(2, model.boundary_point_count, 3)
        logical = torch.rand(2, 7, 3)
        outputs = model(*boundary.unbind(-1), *logical.unbind(-1))
        self.assertEqual(tuple(torch.cat(outputs, dim=-1).shape), (2, 7, 3))
        torch.cat(outputs, dim=-1).square().mean().backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.parameters())
        )

    def test_legacy_mode_has_original_state_dict_layout(self):
        model = MeshONet3D(
            4,
            3,
            5,
            shared_branch_fusion_enabled=False,
        )
        self.assertFalse(
            any(key.startswith("branch_fusion.") for key in model.state_dict())
        )
        self.assertFalse(
            any(key.startswith("coefficient_heads.") for key in model.state_dict())
        )


if __name__ == "__main__":
    unittest.main()
