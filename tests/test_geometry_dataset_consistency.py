"""Exact surfaces must reproduce generated coordinates in both numerical backends."""
import json
import numpy as np
import pytest
import torch
from src.data.geometry.boundary_generator_3d import generate_boundary
from src.data.geometry.parametric_geometry_3d import BoundarySurfaceSet3D
from src.data.mesh.sampling import sample_faces, boundary_indices
from src.data.parametric_dataset import ParametricDataset3D


def test_inference_rejects_old_cap_weights_but_accepts_new_geometry_parameters():
    from types import SimpleNamespace
    from src.evaluation.inference import validate_training_geometry
    from src.data.geometry.pde_surface_3d import CAP_SCHEME, SIDE_SCHEME

    metadata = dict(family="rounded_u_notch_extrusion", geometry_revision=3,
                    cap_parameterization=CAP_SCHEME, side_parameterization=SIDE_SCHEME,
                    depth=1.52)
    boundary = SimpleNamespace(metadata=metadata)
    old = SimpleNamespace(training_geometry_contract=None)
    with pytest.raises(ValueError, match="Checkpoint boundary parameterization"):
        validate_training_geometry(old, boundary)
    contract = {key: metadata[key] for key in
                ("geometry_revision", "cap_parameterization", "side_parameterization")}
    contract["boundary_type"] = metadata["family"]
    model = SimpleNamespace(training_geometry_contract=contract)
    validate_training_geometry(model, [boundary, SimpleNamespace(metadata={**metadata, "depth": 1.55})])
    with pytest.raises(ValueError, match="Checkpoint boundary parameterization"):
        validate_training_geometry(model, SimpleNamespace(metadata={**metadata, "geometry_revision": 4}))
    validate_training_geometry(old, SimpleNamespace(metadata={"family": "box"}))


def test_ugc_restores_physical_profile_and_exact_centre_depth():
    boundary = generate_boundary("rounded_u_groove_extrusion", groove_depth=1.3)
    xi = np.linspace(0, 1, 101)
    expected_profile = .5 * (np.tanh((3*xi-1.05)/.08) - np.tanh((3*xi-1.95)/.08)) / np.tanh(.45/.08)
    expected_y = 2 - 1.3*expected_profile
    values = boundary.y_max.evaluate(xi, np.zeros_like(xi))
    np.testing.assert_allclose(values[:, 1], expected_y, atol=1e-14)
    assert values[50, 1] == pytest.approx(.7)
    tensor_values = boundary.y_max.evaluate_torch(torch.tensor(xi), torch.zeros(101, dtype=torch.float64))
    np.testing.assert_allclose(tensor_values, values, atol=2e-14)
    restored = BoundarySurfaceSet3D.from_dict(boundary.to_dict())
    np.testing.assert_allclose(restored.y_max.evaluate(xi, xi), boundary.y_max.evaluate(xi, xi))


def test_dataset_rejects_mismatched_serialized_geometry(tmp_path):
    boundary = generate_boundary("box")
    shape = (3, 3, 3)
    _, grid = sample_faces(boundary, shape)
    points = grid.reshape(-1, 3)
    # Interior coordinates are irrelevant to the exact boundary consistency check.
    points[np.isnan(points)] = 0
    stored = points[boundary_indices(shape)]
    (tmp_path / "metadata.json").write_text(json.dumps({"dimension": 3, "size": 1, "grid_shape": shape}))
    (tmp_path / "geometry.json").write_text(json.dumps({"samples": [{"sample_id": 0, "boundary": boundary.to_dict()}]}))
    np.save(tmp_path / "boundary_samples.npy", stored[None])
    np.save(tmp_path / "pde_mesh.npy", points[None])
    ParametricDataset3D(tmp_path)
    wrong = generate_boundary("box", xmax=2)
    (tmp_path / "geometry.json").write_text(json.dumps({"samples": [{"sample_id": 0, "boundary": wrong.to_dict()}]}))
    with pytest.raises(ValueError, match="Dataset boundary mismatch"):
        ParametricDataset3D(tmp_path)

    # The label/boundary check also applies to MeshONet and PI-MeshONet, which
    # do not load the parametric surfaces into their training inputs.
    wrong_mesh = points.copy()
    wrong_mesh[0, 0] += .1
    np.save(tmp_path / "pde_mesh.npy", wrong_mesh[None])
    with pytest.raises(ValueError, match="PDE mesh boundaries"):
        ParametricDataset3D(tmp_path, load_boundary=False)
