"""Validate all paper-oriented 3-D parametric boundary families."""

import numpy as np
import torch

from src.numerics.traditional.tfi_3d import tfi_from_surfaces
from src.data.backend.parametric_torch import evaluate_boundary_surfaces_torch
from src.data.generation.make_dataset_parametric_3d import DATASET_FAMILIES
from src.data.geometry.boundary_generator_3d import BOUNDARY_GENERATORS
from src.data.geometry.parametric_geometry_3d import BoundarySurfaceSet3D
from src.utils.metrics import evaluate_hexahedral_grid_quality

PAPER_FAMILIES = (
    "cylindrical_domain_with_airfoil_hole",
    "sidewall_deformed_block",
    "rounded_u_groove_extrusion",
    "tapered_elliptic_bore_block",
    "bent_annular_duct",
    "lobed_annular_duct",
)


class TestPaperBoundary3D:
    def test_registered_order_and_round_trip(self):
        assert (tuple(DATASET_FAMILIES)) == (PAPER_FAMILIES)
        assert all(name in BOUNDARY_GENERATORS for name in PAPER_FAMILIES)
        q = np.linspace(0.0, 1.0, 13)
        for name in PAPER_FAMILIES:
            boundary = BOUNDARY_GENERATORS[name]()
            restored = BoundarySurfaceSet3D.from_dict(boundary.to_dict())
            for face in boundary.names:
                np.testing.assert_allclose(
                    getattr(boundary, face).sample(9, 11),
                    getattr(restored, face).sample(9, 11),
                    atol=2e-12,
                )
            if "o_grid" in boundary.metadata.get("topology", ""):
                np.testing.assert_allclose(
                    boundary.y_min.evaluate(q, q), boundary.y_max.evaluate(q, q), atol=2e-12
                )

    def test_default_dense_tfi_has_positive_corner_jacobians(self):
        for name in PAPER_FAMILIES:
            grid = tfi_from_surfaces(
                BOUNDARY_GENERATORS[name](), xi=11, eta=21, zeta=11
            )
            quality = evaluate_hexahedral_grid_quality(
                grid, metrics=["minimum_scaled_jacobian", "flipped_cell"]
            )
            assert (float(quality["minimum_scaled_jacobian"].min())) > (0.0)
            assert (int(quality["flipped_cell"].sum())) == (0)

    def test_sidewall_block_contracts_both_opposing_sidewalls(self):
        boundary = BOUNDARY_GENERATORS["sidewall_deformed_block"](
            length_x=3.0, sidewall_amplitude=0.30
        )
        # At every axial section both x-sidewalls move toward x=1.5.
        left = boundary.x_min.evaluate(np.asarray(0.5), np.asarray(0.5))[0]
        right = boundary.x_max.evaluate(np.asarray(0.5), np.asarray(0.5))[0]
        left_front = boundary.x_min.evaluate(np.asarray(0.5), np.asarray(0.0))[0]
        left_back = boundary.x_min.evaluate(np.asarray(0.5), np.asarray(1.0))[0]
        assert float(left) > 0.0
        assert float(right) < 3.0
        np.testing.assert_allclose(left, 0.45, atol=2e-12)
        np.testing.assert_allclose(right, 2.55, atol=2e-12)
        np.testing.assert_allclose(left_front, left, atol=2e-12)
        np.testing.assert_allclose(left_back, left, atol=2e-12)

    def test_new_surface_families_are_torch_differentiable(self):
        names = PAPER_FAMILIES[1:]
        boundaries = [BOUNDARY_GENERATORS[name]() for name in names]
        query = torch.rand(
            len(boundaries), 17, 3, dtype=torch.float64, requires_grad=True
        )
        values = evaluate_boundary_surfaces_torch(boundaries, *query.unbind(-1))
        assert (values.shape) == (len(boundaries), 17, 6, 3)
        values.square().mean().backward()
        assert torch.isfinite(query.grad).all()
