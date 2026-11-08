"""QG-HMONet and architecture-switch regression tests."""

from pathlib import Path

import torch
import yaml

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
    hmonet_mapping_mode="corner_corrected_affine",
)


def perimeter(n):
    t = torch.linspace(0, 1, n)
    return torch.cat(
        (
            torch.stack((t, 0 * t), -1),
            torch.stack((0 * t[1:] + 1, t[1:]), -1),
            torch.stack((t[:-1].flip(0), 0 * t[:-1] + 1), -1),
            torch.stack((0 * t[1:-1], t[1:-1].flip(0)), -1),
        ),
        0,
    )


def test_architecture_switches_change_graph_without_changing_contract():
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


def test_boundary_layer_decoder_supports_unbatched_evaluation():
    """Regression test for the per-geometry path used by evaluation."""
    model = HMONet2D(5, 5, **SMALL, boundary_layer_decoder_enabled=True).eval()
    boundary = perimeter(5)
    xi = torch.linspace(0.05, 0.95, 17)
    eta = torch.linspace(0.95, 0.05, 17)
    with torch.no_grad():
        single = model(*boundary.unbind(-1), xi, eta)
        batched = model(
            *boundary.unsqueeze(0).unbind(-1),
            xi.unsqueeze(0),
            eta.unsqueeze(0),
        )
    assert tuple(value.shape for value in single) == ((17, 1), (17, 1))
    assert all(
        torch.allclose(single_value, batched_value.squeeze(0), atol=1e-6, rtol=1e-6)
        for single_value, batched_value in zip(single, batched)
    )


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
        (1, 11, 1),
        (1, 11, 1),
        (1, 11, 1),
    )


def test_declared_models_forward_and_backward_with_production_dimensions():
    """Exercise every retained architecture with its actual baseline dimensions."""
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "experiments/ex3/baseline/experiment.yaml").read_text(encoding="utf-8")
    )
    options = dict(config["model"]["common"])
    options.update(config["model"]["hmonet"])
    options.pop("residual_regularization_weight", None)
    options["trunk_projection_coefficients_enabled"] = True
    cases = (
        (HMONet2D, {"modified_mlp_enabled": True}),
        (HMONet2D, {"query_boundary_cross_attention_enabled": True}),
        (HMONet2D, {"boundary_layer_decoder_enabled": True}),
        (HMONet2D, {"decoupled_weight_decoder_enabled": True}),
        (QGHMONet2D, dict(config["model"]["qghmonet"])),
    )
    boundary = perimeter(21).unsqueeze(0)
    xi = torch.tensor([[0.17, 0.41, 0.63, 0.82]])
    eta = torch.tensor([[0.23, 0.74, 0.36, 0.58]])
    for model_class, overrides in cases:
        model_options = dict(options)
        model_options.update(overrides)
        model = model_class(21, 21, **model_options)
        output = model(*boundary.unbind(-1), xi, eta)
        assert tuple(value.shape for value in output) == ((1, 4, 1), (1, 4, 1))
        loss = sum(value.square().mean() for value in output)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
