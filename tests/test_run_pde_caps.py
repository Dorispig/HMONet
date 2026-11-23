"""run_e dense PDE caps: serialization, exact edges, derivatives and CCAP."""
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.data.geometry.boundary_generator_3d import rounded_u_notch_extrusion_boundary
from src.data.geometry.parametric_geometry_3d import BoundarySurfaceSet3D
from src.data.geometry.boundary_generator_2d import rounded_u_notch_boundary
from src.data.backend.parametric_torch import evaluate_boundary_surfaces_torch
from src.data.mesh.sampling import sample_faces, boundary_indices
from src.data.generation.make_dataset_parametric_3d import (
    _validated_boundary_samples, generate_family_dataset,
)
from src.data.dataset import ParametricDataset3D
from src.evaluation.configuration import create_geometry
from src.models.model import HMONet3D, MeshONet3D
from src.models.pi_meshonet import PIMeshONet3D
from tests.test_convex_projection_weights import SMALL


@pytest.fixture(scope="module")
def boundary():
    return rounded_u_notch_extrusion_boundary(depth=1.55)


def test_cap_uses_dense_pde_without_parameter_redistribution(boundary):
    cap = boundary.z_min
    assert cap is not boundary.z_max
    assert cap.solver_info["grid_shape"] == [201, 201]
    assert cap.solver_info["converged"]
    q = np.linspace(0, 1, 83)
    # Sidewall parameter points stay uniformly spaced, not warped.
    np.testing.assert_allclose(boundary.x_min.evaluate(q, q)[:, 1], 2*q, atol=2e-14)
    two = rounded_u_notch_boundary(width=3., height=2., notch_width=.9, depth=1.55, radius=.25)
    np.testing.assert_allclose(boundary.z_min.evaluate(q, np.ones_like(q))[:, :2],
                               two.top.evaluate(q), atol=2e-14)


def test_serialized_caps_have_independent_payloads_and_load_without_solving(boundary):
    data = json.loads(json.dumps(boundary.to_dict()))
    assert "pde_cap_maps" not in data
    with patch("src.data.geometry.pde_surface_3d._build_cached", side_effect=AssertionError):
        restored = BoundarySurfaceSet3D.from_dict(data)
    assert restored.z_min is not restored.z_max
    for name in boundary.names:
        np.testing.assert_array_equal(getattr(boundary, name).sample(13, 17),
                                      getattr(restored, name).sample(13, 17))


@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 3e-13), (torch.float32, 3e-6)])
def test_torch_matches_numpy_and_allows_second_derivatives(boundary, dtype, tolerance):
    rng = np.random.default_rng(23)
    q = torch.tensor(rng.uniform(.02, .98, (2, 43, 3)), dtype=dtype, requires_grad=True)
    result = evaluate_boundary_surfaces_torch([boundary, boundary], *q.unbind(-1))
    a = q.detach().double().numpy()
    expected = np.stack(boundary.evaluate_faces(*np.moveaxis(a, -1, 0)), -2)
    np.testing.assert_allclose(result.detach().double().numpy(), expected, atol=tolerance, rtol=0)
    first = torch.autograd.grad(result.square().sum(), q, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), q)[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()


def test_coarse_invalid_cap_is_rejected_not_silently_repaired(boundary):
    with pytest.raises(ValueError, match="invalid prescribed boundary faces"):
        _validated_boundary_samples(boundary, (11, 21, 11), 0)
    # The continuous representation is separately checked on a dense grid.
    assert boundary.z_min.solver_info["interpolation_checks"]["401x401"]["all_cells_legal"]


@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 3e-12), (torch.float32, 3e-6)])
def test_three_models_use_same_boundary_and_hmonet_exactly_hits_faces(boundary, dtype, tolerance):
    torch.manual_seed(37)
    shape = (11, 21, 11)
    _, grid = sample_faces(boundary, shape)
    points = torch.tensor(grid.reshape(-1, 3)[boundary_indices(shape)][None], dtype=dtype)
    queries = []
    expected = []
    q = np.linspace(.07, .93, 7)
    for axis in range(3):
        for side in (0, 1):
            query = np.stack((q, q[::-1], .2+.5*q), -1)
            query[:, axis] = side
            queries.append(query)
            free = [i for i in range(3) if i != axis]
            expected.append(getattr(boundary, boundary.names[2*axis+side]).evaluate(
                query[:, free[0]], query[:, free[1]]))
    query = torch.tensor(np.concatenate(queries)[None], dtype=dtype)
    for klass in (HMONet3D, MeshONet3D, PIMeshONet3D):
        options = dict(SMALL)
        if klass is PIMeshONet3D:
            options.pop("branch_mlp_hidden_dim")
            options.pop("branch_mlp_hidden_layer_num")
        if klass is HMONet3D:
            options["hmonet_mapping_mode"] = "CCAP"
        model = klass(*shape, **options).to(dtype=dtype)
        extra = {"boundary_geometry": [boundary]} if klass is HMONet3D else {}
        prediction = torch.cat(model(*points.unbind(-1), *query.unbind(-1), **extra), -1)
        assert torch.isfinite(prediction).all()
        if klass is HMONet3D:
            np.testing.assert_allclose(prediction.detach().numpy()[0], np.concatenate(expected), atol=tolerance, rtol=0)
        prediction.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_generated_dataset_restores_cap_and_matches_evaluation(tmp_path):
    output = generate_family_dataset(
        "rounded_u_notch_extrusion", "extrapolation", size=2,
        parameter_ranges=[(.55, .56)], nx=11, ny=21, nz=11,
        output_root=tmp_path, workers=1, timing_samples=0,
    )
    assert output.name == "run_e_ext_1_2_11_21_11"
    with patch("src.data.geometry.pde_surface_3d._build_cached", side_effect=AssertionError):
        dataset = ParametricDataset3D(output, load_boundary=True)
    assert len(dataset) == 2
    metadata = json.loads((output / "metadata.json").read_text())
    records = json.loads((output / "geometry.json").read_text())["samples"]
    for index, record in enumerate(records):
        parameters = record["parameters"]
        geometry, _ = create_geometry(3, {
            "boundary": "rounded_u_notch_extrusion", "parameter": "depth",
            "fixed_parameters": {k: v for k, v in parameters.items() if k != "depth"},
        }, parameters["depth"])
        _, grid = sample_faces(geometry, tuple(metadata["grid_shape"]))
        expected = grid.reshape(-1, 3)[boundary_indices(metadata["grid_shape"])]
        np.testing.assert_allclose(dataset.boundary_points[index], expected, atol=2e-7, rtol=0)
