"""QG-HMONet and Ex11 architecture-switch regression tests."""

import torch

from src.models import HMONet2D, QGHMONet2D, QGHMONet3D


SMALL = dict(
    branch_dim=8,
    trunk_dim=12,
    coordinate_lift_mode="power_fourier",
    coordinate_lift_max_power=2,
    adjust_hidden_dim=8,
    branch_hidden_dim=8,
    trunk_hidden_dim=8,
    adjust_hidden_layer_num=1,
    branch_hidden_layer_num=1,
    trunk_hidden_layer_num=1,
    output_dim=16,
    branch_mlp_hidden_dim=8,
    branch_mlp_hidden_layer_num=1,
    hmonet_mapping_mode="corner_corrected",
)


def perimeter(n):
    t = torch.linspace(0, 1, n)
    return torch.cat((
        torch.stack((t, 0*t), -1),
        torch.stack((0*t[1:]+1, t[1:]), -1),
        torch.stack((t[:-1].flip(0), 0*t[:-1]+1), -1),
        torch.stack((0*t[1:-1], t[1:-1].flip(0)), -1),
    ), 0)


def test_ex11_switches_change_graph_without_changing_contract():
    boundary = perimeter(5).unsqueeze(0)
    query = torch.rand(1, 17)
    for option in (
        "modified_mlp_enabled",
        "query_boundary_cross_attention_enabled",
        "boundary_layer_decoder_enabled",
        "decoupled_weight_decoder_enabled",
    ):
        model = HMONet2D(5, 5, **SMALL, **{option: True})
        output = model(*boundary.unbind(-1), query, query)
        assert tuple(value.shape for value in output) == ((1, 17, 1), (1, 17, 1))


def test_qg_hmonet_2d_has_exact_boundary_and_no_explicit_geometry_inputs():
    model = QGHMONet2D(5, 5, **SMALL)
    boundary = perimeter(5).unsqueeze(0)
    xi = torch.tensor([[0.0, 1.0, 0.3, 0.8]])
    eta = torch.tensor([[0.2, 0.7, 0.0, 1.0]])
    output = torch.cat(model(*boundary.unbind(-1), xi, eta), -1)
    exact = torch.stack((xi, eta), dim=-1)
    assert torch.allclose(output, exact, atol=2e-6, rtol=0)


def test_qg_hmonet_3d_forward_contract():
    model = QGHMONet3D(3, 3, 3, **SMALL)
    boundary = torch.rand(1, model.boundary_point_count, 3)
    query = torch.rand(1, 11)
    output = model(*boundary.unbind(-1), query, query, query)
    assert tuple(value.shape for value in output) == (
        (1, 11, 1), (1, 11, 1), (1, 11, 1)
    )
