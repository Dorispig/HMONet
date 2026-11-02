"""Vectorized Q1 quadrilateral finite-element operators.

The structured grid convention is the same as ``logical_grid``: x/xi varies
fastest, so a flattened ``(nx, ny)`` grid reshapes to ``(ny, nx, 2)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


class MeshAdmissibilityError(ValueError):
    """Raised when a physical Q1 mapping is singular or orientation reversing."""


@dataclass(frozen=True)
class AssemblyDiagnostics:
    element_count: int
    minimum_determinant: float
    maximum_determinant: float
    nonpositive_determinant_count: int


def gauss_rule_2x2():
    """Return the tensor-product two-point Gauss rule on ``[-1, 1]^2``."""
    a = 1.0 / np.sqrt(3.0)
    points = np.array([[-a, -a], [a, -a], [a, a], [-a, a]], dtype=float)
    return points, np.ones(4, dtype=float)


def shape_functions(xi, eta):
    """Return Q1 shape values and reference derivatives at one point."""
    xi, eta = float(xi), float(eta)
    values = 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )
    derivatives = 0.25 * np.array(
        [
            [-(1.0 - eta), -(1.0 - xi)],
            [1.0 - eta, -(1.0 + xi)],
            [1.0 + eta, 1.0 + xi],
            [-(1.0 + eta), 1.0 - xi],
        ]
    )
    return values, derivatives


def structured_connectivity(shape):
    """Return counter-clockwise ``[bl, br, tr, tl]`` element nodes."""
    nx, ny = map(int, shape)
    if nx < 2 or ny < 2:
        raise ValueError("A Q1 mesh requires nx, ny >= 2")
    lower_left = np.arange((ny - 1) * nx, dtype=np.int64).reshape(ny - 1, nx)
    lower_left = lower_left[:, :-1].reshape(-1)
    return np.column_stack(
        [lower_left, lower_left + 1, lower_left + nx + 1, lower_left + nx]
    )


def _as_mesh(points, shape):
    points = np.asarray(points, dtype=float)
    expected = (int(np.prod(shape)), 2)
    if points.shape != expected:
        raise ValueError(f"Expected mesh coordinates with shape {expected}, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Mesh coordinates contain non-finite values")
    return points


def _coefficient_values(coefficient, physical_points):
    if callable(coefficient):
        values = np.asarray(coefficient(physical_points), dtype=float)
    else:
        values = np.asarray(coefficient, dtype=float)
    if values.ndim == 0:
        values = np.full(len(physical_points), float(values))
    values = np.broadcast_to(values, (len(physical_points),))
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("The scalar conductivity must be finite and positive")
    return values


def _source_values(source, physical_points):
    if callable(source):
        values = np.asarray(source(physical_points), dtype=float)
    else:
        values = np.asarray(source, dtype=float)
    if values.ndim == 0:
        values = np.full(len(physical_points), float(values))
    values = np.broadcast_to(values, (len(physical_points),))
    if not np.isfinite(values).all():
        raise ValueError("The source term contains non-finite values")
    return values


def element_quadrature_data(points, shape, determinant_tolerance=1.0e-12):
    """Return reusable geometry at all element/Gauss-point pairs."""
    points = _as_mesh(points, shape)
    connectivity = structured_connectivity(shape)
    coordinates = points[connectivity]
    quadrature, weights = gauss_rule_2x2()
    # The Q1 determinant is affine in each reference coordinate. Checking all
    # four reference corners rejects concave, folded and bow-tie cells that a
    # Gauss-only check could otherwise miss near an element corner.
    validation_points = np.array(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, 0.0]]
    )
    validation_determinants = []
    for xi_eta in validation_points:
        _, dn = shape_functions(*xi_eta)
        jacobian = np.einsum("eia,ib->eab", coordinates, dn)
        validation_determinants.append(np.linalg.det(jacobian))
    validation_determinants = np.stack(validation_determinants, axis=1)
    invalid_elements = np.any(
        validation_determinants <= float(determinant_tolerance), axis=1
    )
    if np.any(invalid_elements):
        raise MeshAdmissibilityError(
            f"Q1 mapping has {int(invalid_elements.sum())} concave, folded or "
            "singular elements; minimum corner determinant="
            f"{float(validation_determinants.min()):.6e}"
        )
    values, gradients, determinants, physical = [], [], [], []
    for xi_eta in quadrature:
        n, dn = shape_functions(*xi_eta)
        # J[a,b] = dx_a / d(reference_b).
        jacobian = np.einsum("eia,ib->eab", coordinates, dn)
        determinant = np.linalg.det(jacobian)
        bad = determinant <= float(determinant_tolerance)
        if np.any(bad):
            minimum = float(determinant.min())
            raise MeshAdmissibilityError(
                f"Q1 mapping has {int(bad.sum())} non-positive/singular "
                f"Gauss-point Jacobians; minimum determinant={minimum:.6e}"
            )
        inverse = np.linalg.inv(jacobian)
        values.append(n)
        gradients.append(np.einsum("ib,eba->eia", dn, inverse))
        determinants.append(determinant)
        physical.append(np.einsum("i,eia->ea", n, coordinates))
    determinant_array = np.column_stack(
        [validation_determinants, np.stack(determinants, axis=1)]
    )
    diagnostics = AssemblyDiagnostics(
        element_count=len(connectivity),
        minimum_determinant=float(determinant_array.min()),
        maximum_determinant=float(determinant_array.max()),
        nonpositive_determinant_count=0,
    )
    return {
        "connectivity": connectivity,
        "coordinates": coordinates,
        "quadrature": quadrature,
        "weights": weights,
        "values": values,
        "gradients": gradients,
        "determinants": determinants,
        "physical_points": physical,
        "diagnostics": diagnostics,
    }


def assemble_q1_system(
    points,
    shape,
    *,
    conductivity=1.0,
    source=0.0,
    determinant_tolerance=1.0e-12,
):
    """Assemble a sparse Q1 system for ``-div(k grad(u)) = source``."""
    points = _as_mesh(points, shape)
    data = element_quadrature_data(points, shape, determinant_tolerance)
    connectivity = data["connectivity"]
    element_count = len(connectivity)
    local_stiffness = np.zeros((element_count, 4, 4), dtype=float)
    local_load = np.zeros((element_count, 4), dtype=float)
    for n, gradients, determinant, physical, weight in zip(
        data["values"],
        data["gradients"],
        data["determinants"],
        data["physical_points"],
        data["weights"],
    ):
        k = _coefficient_values(conductivity, physical)
        q = _source_values(source, physical)
        measure = determinant * weight
        local_stiffness += np.einsum(
            "eia,eja,e->eij", gradients, gradients, k * measure
        )
        local_load += np.einsum("i,e,e->ei", n, q, measure)

    rows = np.repeat(connectivity, 4, axis=1).reshape(-1)
    columns = np.tile(connectivity, (1, 4)).reshape(-1)
    matrix = sparse.coo_matrix(
        (local_stiffness.reshape(-1), (rows, columns)),
        shape=(len(points), len(points)),
    ).tocsr()
    load = np.zeros(len(points), dtype=float)
    np.add.at(load, connectivity.reshape(-1), local_load.reshape(-1))
    return matrix, load, data["diagnostics"]


def integrate_nodal_field_squared(points, shape, values):
    """Integrate the square of a scalar Q1 nodal field."""
    values = np.asarray(values, dtype=float)
    if values.shape != (int(np.prod(shape)),):
        raise ValueError("Nodal scalar field shape does not match the mesh")
    data = element_quadrature_data(points, shape)
    local = values[data["connectivity"]]
    integral = 0.0
    for n, determinant, weight in zip(
        data["values"], data["determinants"], data["weights"]
    ):
        field = np.einsum("i,ei->e", n, local)
        integral += float(np.sum(field**2 * determinant * weight))
    return integral


def integrate_gradient_squared(points, shape, values):
    """Integrate ``|grad(values)|^2`` for a scalar Q1 nodal field."""
    values = np.asarray(values, dtype=float)
    if values.shape != (int(np.prod(shape)),):
        raise ValueError("Nodal scalar field shape does not match the mesh")
    data = element_quadrature_data(points, shape)
    local = values[data["connectivity"]]
    integral = 0.0
    for gradients, determinant, weight in zip(
        data["gradients"], data["determinants"], data["weights"]
    ):
        gradient = np.einsum("ei,eia->ea", local, gradients)
        integral += float(np.sum(np.sum(gradient**2, axis=1) * determinant * weight))
    return integral
