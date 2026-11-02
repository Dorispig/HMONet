"""Unit tests for downstream Q1 geometry and assembly."""

import numpy as np
import pytest

from src.data.mesh.sampling import logical_grid
from src.numerics.downstream.q1_fem import (
    MeshAdmissibilityError,
    assemble_q1_system,
    shape_functions,
    structured_connectivity,
)


def test_shape_functions_form_partition_of_unity():
    values, derivatives = shape_functions(0.27, -0.31)
    assert np.isclose(values.sum(), 1.0)
    assert np.allclose(derivatives.sum(axis=0), 0.0)


def test_structured_connectivity_is_counter_clockwise():
    points = logical_grid((4, 3))
    connectivity = structured_connectivity((4, 3))
    quads = points[connectivity]
    cross = np.cross(quads[:, 1] - quads[:, 0], quads[:, 3] - quads[:, 0])
    assert np.all(cross > 0)


def test_q1_assembly_is_symmetric_and_rejects_inversion():
    shape = (5, 4)
    points = logical_grid(shape)
    matrix, load, diagnostics = assemble_q1_system(points, shape, source=2.0)
    assert np.allclose((matrix - matrix.T).data, 0.0)
    assert np.isclose(load.sum(), 2.0)
    assert diagnostics.minimum_determinant > 0

    invalid = points.copy()
    invalid[[0, 1]] = invalid[[1, 0]]
    with pytest.raises(MeshAdmissibilityError):
        assemble_q1_system(invalid, shape)
