"""Regression checks for the Attention-PI-HMONet architecture and PDE losses."""

import torch
from pathlib import Path

from scripts.train.attention_pihmonet_common import (
    autograd_elliptic_residual,
    autograd_physics_losses,
    autograd_pde_loss,
    build_boundary_inputs,
    finite_difference_pde_loss,
    finite_difference_physics_losses,
    finite_difference_physics_losses_at_points,
    predict_attention,
    sample_boundary_logical_points,
)
from src.data.geometry.boundary_generator_2d import rectangular_boundary
from src.data.geometry.boundary_generator_3d import box_boundary
from src.models.model import AttentionHMONet2D, AttentionHMONet3D
from src.data.dataset import ParametricDataset2D, ParametricDataset3D


def _model_2d():
    return AttentionHMONet2D(
        5, 4, branch_dim=12, trunk_dim=6, branch_hidden_dim=16,
        trunk_hidden_dim=16, branch_hidden_layer_num=2,
        trunk_hidden_layer_num=2, output_dim=12, branch_mlp_hidden_dim=16,
        branch_mlp_hidden_layer_num=1, attention_dim=12,
        attention_heads=3, attention_layers=1, attention_feedforward_dim=24,
    )


def _model_3d():
    return AttentionHMONet3D(
        3, 3, 3, branch_dim=12, trunk_dim=6, branch_hidden_dim=16,
        trunk_hidden_dim=16, branch_hidden_layer_num=2,
        trunk_hidden_layer_num=2, output_dim=12, attention_dim=12,
        attention_heads=3, attention_layers=1, attention_feedforward_dim=24,
    )


def test_autograd_elliptic_residual_is_zero_for_affine_maps():
    for dimension in (2, 3):
        logical = torch.rand(2, 4, dimension, requires_grad=True)
        matrix = torch.eye(dimension) * 1.7 + 0.2
        prediction = logical @ matrix
        residual = autograd_elliptic_residual(prediction, logical)
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)


def test_2d_autograd_and_finite_difference_losses_backward():
    geometry = [rectangular_boundary()]
    model = _model_2d()
    boundary = build_boundary_inputs(geometry, (5, 4), torch.device("cpu"), torch.float32)
    autograd = autograd_pde_loss(
        model, boundary, geometry, 1, 3, 2, torch.device("cpu"), torch.float32, 0.1
    )
    finite = finite_difference_pde_loss(
        model, boundary, geometry, 1, 3, [0.1, 0.1], torch.device("cpu"), torch.float32
    )
    assert torch.isfinite(autograd) and torch.isfinite(finite)
    logical = (0.1 + 0.8 * torch.rand(1, 3, 2)).requires_grad_(True)
    prediction = torch.cat(model(*boundary, *logical.unbind(-1), boundary_geometry=geometry), -1)
    losses = autograd_physics_losses(prediction, logical)
    assert all(torch.isfinite(value) for value in losses.values())
    fd_losses = finite_difference_physics_losses(
        model, boundary, geometry, 1, 3, [0.1, 0.1],
        torch.device("cpu"), torch.float32,
    )
    assert all(torch.isfinite(value) for value in fd_losses.values())
    (autograd + finite).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_3d_variable_surfaces_and_exact_faces():
    geometry = [box_boundary()]
    model = _model_3d().eval()
    counts = [[3, 4], [4, 3], [3, 5], [5, 3], [4, 5], [5, 4]]
    faces = build_boundary_inputs(
        geometry, (3, 3, 3), torch.device("cpu"), torch.float32, counts
    )
    xi = torch.tensor([[0.0, 1.0, 0.4]])
    eta = torch.tensor([[0.3, 0.7, 0.0]])
    zeta = torch.tensor([[0.6, 0.2, 1.0]])
    result = torch.cat(
        model(*faces, xi, eta, zeta, boundary_geometry=geometry), -1
    )
    projection = model.project_faces(faces, xi, eta, zeta, geometry)
    assert torch.allclose(result[:, 0], projection[:, 0, 0], atol=1e-6)
    assert torch.allclose(result[:, 1], projection[:, 1, 1], atol=1e-6)
    assert torch.allclose(result[:, 2], projection[:, 2, 2], atol=1e-6)


def test_3d_autograd_and_finite_difference_losses_backward():
    geometry = [box_boundary()]
    model = _model_3d()
    faces = build_boundary_inputs(
        geometry, (3, 3, 3), torch.device("cpu"), torch.float32, [[3, 3]] * 6
    )
    autograd = autograd_pde_loss(
        model, faces, geometry, 1, 2, 3, torch.device("cpu"), torch.float32, 0.1
    )
    finite = finite_difference_pde_loss(
        model, faces, geometry, 1, 2, [0.1] * 3, torch.device("cpu"), torch.float32
    )
    assert torch.isfinite(autograd) and torch.isfinite(finite)
    (autograd + finite).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_boundary_layer_pde_and_jacobian_use_autograd_in_2d_and_3d():
    """Boundary physics points retain first- and second-coordinate gradients."""
    cases = (
        (2, _model_2d(), [rectangular_boundary()], (5, 4), None),
        (3, _model_3d(), [box_boundary()], (3, 3, 3), [[3, 3]] * 6),
    )
    for dimension, model, geometry, shape, counts in cases:
        boundary = build_boundary_inputs(
            geometry, shape, torch.device("cpu"), torch.float32, counts
        )
        logical = sample_boundary_logical_points(
            1, 2 * dimension, dimension, torch.device("cpu"), torch.float32,
            offset=0.01, tangential_margin=0.05,
        )
        prediction = predict_attention(model, boundary, logical, geometry)
        losses = autograd_physics_losses(prediction, logical)
        total = losses["pde_normalized"] + 0.01 * losses["jacobian"]
        assert torch.isfinite(total)
        total.backward()
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )


def test_boundary_layer_pde_and_jacobian_use_finite_differences_in_2d_and_3d():
    """Boundary-layer losses share the centered finite-difference operator."""
    cases = (
        (2, _model_2d(), [rectangular_boundary()], (5, 4), None),
        (3, _model_3d(), [box_boundary()], (3, 3, 3), [[3, 3]] * 6),
    )
    for dimension, model, geometry, shape, counts in cases:
        boundary = build_boundary_inputs(
            geometry, shape, torch.device("cpu"), torch.float32, counts
        )
        steps = [0.1] * dimension
        logical = sample_boundary_logical_points(
            1,
            2 * dimension,
            dimension,
            torch.device("cpu"),
            torch.float32,
            offset=0.1,
            tangential_margin=0.1,
        )
        losses = finite_difference_physics_losses_at_points(
            model, boundary, geometry, 1, logical[0].detach(), steps
        )
        total = losses["pde_normalized"] + 0.01 * losses["jacobian"]
        assert torch.isfinite(total)
        total.backward()
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )


def test_all_stored_parametric_datasets_build_attention_inputs():
    """Every generated dataset with the standard layout must enter the encoder."""
    project = Path(__file__).resolve().parents[1]
    tested = []
    for dimension, dataset_class in ((2, ParametricDataset2D), (3, ParametricDataset3D)):
        for path in sorted((project / f"data/raw/{dimension}D").iterdir()):
            if not path.is_dir():
                continue
            dataset = dataset_class(path, load_boundary=True)
            counts = None if dimension == 2 else [[3, 3]] * 6
            inputs = build_boundary_inputs(
                [dataset[0]["boundary"]], dataset.grid_shape,
                torch.device("cpu"), torch.float32, counts,
            )
            assert len(inputs) == (4 if dimension == 2 else 6)
            assert all(torch.isfinite(value).all() for value in inputs)
            tested.append(path.name)
    assert tested
