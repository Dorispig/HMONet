"""Architecture checks for the variable-length self-attentive 2-D HMONet."""

import torch

from src.models.model import AttentionHMONet2D
from src.models.modules import BoundaryTokenBuilder2D


def _square_edges(nx=4, ny=3):
    horizontal = torch.linspace(0.0, 1.0, nx)
    vertical = torch.linspace(0.0, 1.0, ny)
    bottom = torch.stack((horizontal, torch.zeros_like(horizontal)), dim=-1)
    right = torch.stack((torch.ones_like(vertical), vertical), dim=-1)
    top = torch.stack((horizontal, torch.ones_like(horizontal)), dim=-1)
    left = torch.stack((torch.zeros_like(vertical), vertical), dim=-1)
    return tuple(edge.unsqueeze(0) for edge in (bottom, right, top, left))


def _pad_two(first, second):
    width = max(first.shape[1], second.shape[1])
    result = torch.zeros(2, width, 2)
    mask = torch.ones(2, width, dtype=torch.bool)
    result[0, : first.shape[1]] = first[0]
    result[1, : second.shape[1]] = second[0]
    mask[0, : first.shape[1]] = False
    mask[1, : second.shape[1]] = False
    return result, mask


def test_boundary_token_builder_accepts_variable_edge_lengths_and_padding():
    builder = BoundaryTokenBuilder2D(model_dim=8)
    edges_a = _square_edges(4, 3)
    edges_b = _square_edges(6, 5)
    padded = tuple(_pad_two(a, b) for a, b in zip(edges_a, edges_b))
    edges = tuple(item[0] for item in padded)
    masks = tuple(item[1] for item in padded)

    tokens, packed_mask = builder(*edges, boundary_masks=masks)
    assert tokens.shape == (2, 22, 8)
    assert packed_mask.shape == (2, 22)
    assert int((~packed_mask[0]).sum()) == 14
    assert int((~packed_mask[1]).sum()) == 22


def test_attention_hmonet_forward_attention_and_exact_square_boundary():
    torch.manual_seed(7)
    model = AttentionHMONet2D(
        branch_dim=16,
        trunk_dim=6,
        branch_hidden_dim=16,
        trunk_hidden_dim=16,
        branch_hidden_layer_num=2,
        trunk_hidden_layer_num=2,
        output_dim=16,
        branch_mlp_hidden_dim=16,
        branch_mlp_hidden_layer_num=1,
        attention_dim=16,
        attention_heads=4,
        attention_layers=2,
        attention_feedforward_dim=32,
        boundary_alpha_initial_value=1.0,
    )
    edges = _square_edges(4, 3)
    xi = torch.tensor([[0.0, 1.0, 0.25, 0.75, 0.4]], requires_grad=True)
    eta = torch.tensor([[0.25, 0.75, 0.0, 1.0, 0.6]], requires_grad=True)

    x, y = model(*edges, xi, eta)
    result = torch.cat((x, y), dim=-1)
    expected_boundary = torch.tensor(
        [[0.0, 0.25], [1.0, 0.75], [0.25, 0.0], [0.75, 1.0]]
    )
    assert result.shape == (1, 5, 2)
    assert torch.allclose(result[0, :4], expected_boundary, atol=1e-6)
    assert torch.isfinite(result).all()

    attention = model.boundary_attention_weights(*edges)
    assert attention.shape == (1, 4, 2, 14)
    assert torch.allclose(attention.sum(dim=-1), torch.ones_like(attention[..., 0]))

    result.sum().backward()
    assert model.boundary_attention.direction_queries.grad is not None
    assert torch.isfinite(model.boundary_attention.direction_queries.grad).all()


def test_trainable_boundary_alpha_starts_at_1_01_and_receives_gradient():
    model = AttentionHMONet2D(
        branch_dim=8,
        trunk_dim=6,
        branch_hidden_dim=8,
        trunk_hidden_dim=8,
        branch_hidden_layer_num=1,
        trunk_hidden_layer_num=1,
        output_dim=8,
        branch_mlp_hidden_dim=8,
        branch_mlp_hidden_layer_num=1,
        attention_dim=8,
        attention_heads=2,
        attention_layers=1,
        attention_feedforward_dim=16,
    )
    assert torch.allclose(model.boundary_alpha.detach(), torch.tensor(1.01))
    xi = torch.tensor([[0.0, 1.0, 0.3, 0.7]])
    eta = torch.tensor([[0.2, 0.8, 0.0, 1.0]])
    prediction = torch.cat(model(*_square_edges(), xi, eta), -1)
    target = torch.tensor(
        [[[0.0, 0.2], [1.0, 0.8], [0.3, 0.0], [0.7, 1.0]]]
    )
    loss = (prediction - target).square().sum(-1).mean()
    loss = loss + 0.01 * (model.boundary_alpha - 1).square()
    loss.backward()
    assert model.boundary_alpha.grad is not None
    assert torch.isfinite(model.boundary_alpha.grad)
    assert model.boundary_alpha.grad.abs() > 0


def test_pre_alpha_checkpoint_loads_with_original_alpha_one():
    source = AttentionHMONet2D(
        branch_dim=8, trunk_dim=6, branch_hidden_dim=8, trunk_hidden_dim=8,
        branch_hidden_layer_num=1, trunk_hidden_layer_num=1, output_dim=8,
        branch_mlp_hidden_dim=8, branch_mlp_hidden_layer_num=1,
        attention_dim=8, attention_heads=2, attention_layers=1,
        attention_feedforward_dim=16,
    )
    old_state = {
        key: value for key, value in source.state_dict().items()
        if key != "boundary_alpha"
    }
    restored = AttentionHMONet2D(
        branch_dim=8, trunk_dim=6, branch_hidden_dim=8, trunk_hidden_dim=8,
        branch_hidden_layer_num=1, trunk_hidden_layer_num=1, output_dim=8,
        branch_mlp_hidden_dim=8, branch_mlp_hidden_layer_num=1,
        attention_dim=8, attention_heads=2, attention_layers=1,
        attention_feedforward_dim=16,
    )
    restored.load_state_dict(old_state, strict=True)
    assert torch.allclose(restored.boundary_alpha.detach(), torch.tensor(1.0))


def test_attention_hmonet_changes_boundary_length_between_calls():
    model = AttentionHMONet2D(
        branch_dim=8,
        trunk_dim=6,
        branch_hidden_dim=8,
        trunk_hidden_dim=8,
        branch_hidden_layer_num=1,
        trunk_hidden_layer_num=1,
        output_dim=8,
        branch_mlp_hidden_dim=8,
        branch_mlp_hidden_layer_num=1,
        attention_dim=8,
        attention_heads=2,
        attention_layers=1,
        attention_feedforward_dim=16,
    )
    xi = torch.tensor([[0.2, 0.5, 0.8]])
    eta = torch.tensor([[0.3, 0.5, 0.7]])
    coarse = model(*_square_edges(3, 3), xi, eta)
    fine = model(*_square_edges(9, 7), xi, eta)
    assert coarse[0].shape == fine[0].shape == (1, 3, 1)
    assert all(torch.isfinite(value).all() for value in (*coarse, *fine))

