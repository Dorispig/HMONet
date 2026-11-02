"""Validate all paper-oriented 3-D parametric boundary families."""

import unittest

import numpy as np
import torch

from scripts.tfi.tfi_3d import tfi_from_surfaces
from src.data.backend.parametric_torch import evaluate_boundary_surfaces_torch
from src.data.generation.make_dataset_parametric_3d import DATASET_FAMILIES
from src.data.geometry.boundary_generator_3d import BOUNDARY_GENERATORS
from src.data.geometry.parametric_geometry_3d import BoundarySurfaceSet3D
from src.utils.metrics import evaluate_hexahedral_grid_quality


PAPER_FAMILIES = (
    "cylindrical_domain_with_airfoil_hole",
    "stadium_cylinder_with_circular_hole",
    "annular_nozzle_diffuser",
    "twisted_elliptic_annulus",
    "bent_annular_duct",
    "lobed_annular_duct",
)


class PaperBoundary3DTests(unittest.TestCase):
    def test_registered_order_and_round_trip(self):
        self.assertEqual(tuple(DATASET_FAMILIES), PAPER_FAMILIES)
        self.assertEqual(tuple(BOUNDARY_GENERATORS)[:6], PAPER_FAMILIES)
        q = np.linspace(0.0, 1.0, 13)
        for name in PAPER_FAMILIES:
            with self.subTest(name=name):
                boundary = BOUNDARY_GENERATORS[name]()
                restored = BoundarySurfaceSet3D.from_dict(boundary.to_dict())
                for face in boundary.names:
                    np.testing.assert_allclose(
                        getattr(boundary, face).sample(9, 11),
                        getattr(restored, face).sample(9, 11),
                        atol=2e-12,
                    )
                np.testing.assert_allclose(boundary.y_min.evaluate(q, q), boundary.y_max.evaluate(q, q), atol=2e-12)

    def test_default_dense_tfi_has_positive_corner_jacobians(self):
        for name in PAPER_FAMILIES:
            with self.subTest(name=name):
                grid = tfi_from_surfaces(BOUNDARY_GENERATORS[name](), xi=11, eta=31, zeta=11)
                quality = evaluate_hexahedral_grid_quality(grid, metrics=["minimum_scaled_jacobian", "flipped_cell"])
                self.assertGreater(float(quality["minimum_scaled_jacobian"].min()), 0.0)
                self.assertEqual(int(quality["flipped_cell"].sum()), 0)

    def test_new_surface_families_are_torch_differentiable(self):
        names = PAPER_FAMILIES[2:]
        boundaries = [BOUNDARY_GENERATORS[name]() for name in names]
        query = torch.rand(len(boundaries), 17, 3, dtype=torch.float64, requires_grad=True)
        values = evaluate_boundary_surfaces_torch(boundaries, *query.unbind(-1))
        self.assertEqual(values.shape, (len(boundaries), 17, 6, 3))
        values.square().mean().backward()
        self.assertTrue(torch.isfinite(query.grad).all())


if __name__ == "__main__":
    unittest.main()
