"""Tests for the paper-aligned three-dimensional MeshONet."""

import torch

from src.models.model import MeshONet3D


class TestMeshONet3DArchitecture:
    def test_default_32_cubed_parameter_count_and_gradients(self):
        model = MeshONet3D(32, 32, 32)
        assert (sum(p.numel() for p in model.parameters())) == (1266787)
        assert hasattr(model, "branch_fusion")
        assert (len(model.coefficient_heads)) == (3)

        boundary = torch.randn(2, model.boundary_point_count, 3)
        logical = torch.rand(2, 7, 3)
        outputs = model(*boundary.unbind(-1), *logical.unbind(-1))
        assert (tuple(torch.cat(outputs, dim=-1).shape)) == (2, 7, 3)
        torch.cat(outputs, dim=-1).square().mean().backward()
        assert all(parameter.grad is not None for parameter in model.parameters())

    def test_legacy_mode_has_original_state_dict_layout(self):
        model = MeshONet3D(
            4,
            3,
            5,
            shared_branch_fusion_enabled=False,
        )
        assert not any(key.startswith("branch_fusion.") for key in model.state_dict())
        assert not any(
            key.startswith("coefficient_heads.") for key in model.state_dict()
        )
