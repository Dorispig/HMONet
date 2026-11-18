"""Equivalence checks for sparse 3-D exact face evaluation."""

import torch

from src.data.backend.parametric_torch import (
    evaluate_boundary_surfaces_torch,
    evaluate_selected_boundary_surfaces_torch,
)
from src.data.geometry.boundary_generator_3d import box_boundary


def test_selected_faces_match_dense_projection_and_keep_coordinate_gradients():
    boundaries = [box_boundary(), box_boundary(xmax=1.3, ymax=1.2)]
    coordinates = torch.rand(2, 17, 3, requires_grad=True)
    face_ids = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 0, 2, 4, 1, 3])
    dense = evaluate_boundary_surfaces_torch(
        boundaries, *coordinates.unbind(-1)
    )
    expected = dense.gather(
        2, face_ids[None, :, None, None].expand(2, -1, 1, 3)
    ).squeeze(2)
    actual = evaluate_selected_boundary_surfaces_torch(
        boundaries, *coordinates.unbind(-1), face_ids
    )
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert coordinates.grad is not None
    assert torch.isfinite(coordinates.grad).all()
