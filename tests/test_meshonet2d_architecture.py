"""Tests for the paper-aligned two-dimensional MeshONet fusion block."""

import torch

from src.models.model import MeshONet2D


class TestMeshONet2DArchitecture:
    def test_default_uses_shared_fusion_and_coordinate_heads(self):
        model = MeshONet2D(5, 4)
        assert hasattr(model, "branch_fusion")
        assert (len(model.coefficient_heads)) == (2)

        boundary = torch.randn(2, 14, 2)
        logical = torch.rand(2, 7, 2)
        outputs = model(*boundary.unbind(-1), *logical.unbind(-1))
        assert (tuple(torch.cat(outputs, dim=-1).shape)) == (2, 7, 2)
        torch.cat(outputs, dim=-1).square().mean().backward()
        assert all(parameter.grad is not None for parameter in model.parameters())

    def test_legacy_mode_preserves_original_state_dict_layout(self):
        model = MeshONet2D(5, 4, shared_branch_fusion_enabled=False)
        keys = model.state_dict()
        assert any(key.startswith("branch_mlp_x.") for key in keys)
        assert any(key.startswith("branch_mlp_y.") for key in keys)
        assert not any(key.startswith("branch_fusion.") for key in keys)
        assert not any(key.startswith("coefficient_heads.") for key in keys)
