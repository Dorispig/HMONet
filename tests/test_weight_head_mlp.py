"""Optional geometry head preserves legacy Linear models and supports train/eval."""

import torch
from src.models.model import HMONet2D, HMONet3D
from src.models.modules import MLP

SMALL = dict(
    branch_dim=4,
    trunk_dim=2,
    adjust_hidden_dim=4,
    branch_hidden_dim=4,
    trunk_hidden_dim=4,
    output_dim=4,
    adjust_hidden_layer_num=1,
    branch_hidden_layer_num=1,
    trunk_hidden_layer_num=1,
    branch_mlp_hidden_dim=4,
    branch_mlp_hidden_layer_num=1,
    projection_mlp_hidden_dim=7,
)


class TestWeightHead:
    def build(self, dimension, **options):
        args = {**SMALL, "trunk_dim": dimension, **options}
        return (HMONet2D if dimension == 2 else HMONet3D)(*([3] * dimension), **args)

    def test_default_is_mlp_and_legacy_linear_remains_explicit(self):
        for dim in (2, 3):
            torch.manual_seed(19)
            default = self.build(dim)
            torch.manual_seed(19)
            explicit = self.build(dim, weight_head_mlp_enabled=True)
            assert (default.state_dict().keys()) == (explicit.state_dict().keys())
            for key, value in default.state_dict().items():
                assert torch.equal(value, explicit.state_dict()[key]), key
            explicit.load_state_dict(default.state_dict(), strict=True)
            head = default.weight_head if dim == 2 else default.projection_weight_head
            assert isinstance(head, MLP)
            legacy = self.build(dim, weight_head_mlp_enabled=False)
            legacy_head = (
                legacy.weight_head if dim == 2 else legacy.projection_weight_head
            )
            assert isinstance(legacy_head, torch.nn.Linear)
            assert isinstance(default.bias_head, torch.nn.Linear)

    def test_mlp_matches_location_hidden_structure_and_roundtrips(self):
        for dim in (2, 3):
            model = self.build(dim, weight_head_mlp_enabled=True)
            head = model.weight_head if dim == 2 else model.projection_weight_head
            assert isinstance(head, MLP)
            assert (head.hidden_layer_size) == (
                model.trunk_projection_head.hidden_layer_size
            )
            assert (type(head.activation)) == (
                type(model.trunk_projection_head.activation)
            )
            assert (head.output_dim) == (2 * dim)
            restored = self.build(dim, weight_head_mlp_enabled=True)
            restored.load_state_dict(model.state_dict(), strict=True)
            assert isinstance(model.bias_head, torch.nn.Linear)

    def test_forward_backward_and_exact_boundary_both_modes(self):
        for dim in (2, 3):
            for enabled in (False, True):
                model = self.build(dim, weight_head_mlp_enabled=enabled)
                boundary = torch.rand(1, 8 if dim == 2 else 26, dim)
                query = (0.2 + 0.6 * torch.rand(1, 5, dim)).requires_grad_()
                outputs = torch.cat(model(*boundary.unbind(-1), *query.unbind(-1)), -1)
                outputs.square().mean().backward()
                head = model.weight_head if dim == 2 else model.projection_weight_head
                assert all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in head.parameters()
                )
                query = torch.full((1, 2 * dim, dim), 0.4)
                for axis in range(dim):
                    query[0, 2 * axis, axis] = 0
                    query[0, 2 * axis + 1, axis] = 1
                actual = torch.cat(model(*boundary.unbind(-1), *query.unbind(-1)), -1)
                projected = model.project(*boundary.unbind(-1), *query.unbind(-1))
                expected = torch.stack([projected[:, i, i] for i in range(2 * dim)], 1)
                torch.testing.assert_close(actual, expected)
