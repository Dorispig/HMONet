"""Tests for the paper-aligned two-dimensional MeshONet fusion block."""

import unittest

import torch

from src.models.model import MeshONet2D


class MeshONet2DArchitectureTests(unittest.TestCase):
    def test_default_uses_shared_fusion_and_coordinate_heads(self):
        model = MeshONet2D(5, 4)
        self.assertTrue(hasattr(model, "branch_fusion"))
        self.assertEqual(len(model.coefficient_heads), 2)

        boundary = torch.randn(2, 14, 2)
        logical = torch.rand(2, 7, 2)
        outputs = model(*boundary.unbind(-1), *logical.unbind(-1))
        self.assertEqual(tuple(torch.cat(outputs, dim=-1).shape), (2, 7, 2))
        torch.cat(outputs, dim=-1).square().mean().backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.parameters())
        )

    def test_legacy_mode_preserves_original_state_dict_layout(self):
        model = MeshONet2D(5, 4, shared_branch_fusion_enabled=False)
        keys = model.state_dict()
        self.assertTrue(any(key.startswith("branch_mlp_x.") for key in keys))
        self.assertTrue(any(key.startswith("branch_mlp_y.") for key in keys))
        self.assertFalse(any(key.startswith("branch_fusion.") for key in keys))
        self.assertFalse(any(key.startswith("coefficient_heads.") for key in keys))


if __name__ == "__main__":
    unittest.main()
