"""Validate all paper-oriented 3-D parametric boundary families."""

import numpy as np
import torch

from src.numerics.traditional.tfi_3d import tfi_from_surfaces
from src.data.backend.parametric_torch import evaluate_boundary_surfaces_torch
from src.data.generation.make_dataset_parametric_3d import DATASET_FAMILIES
from src.data.geometry.boundary_generator_3d import BOUNDARY_GENERATORS
from src.data.geometry.boundary_generator_2d import rounded_u_notch_boundary
from src.data.geometry.parametric_geometry_3d import BoundarySurfaceSet3D
from src.utils.metrics import evaluate_hexahedral_grid_quality

PAPER_FAMILIES = (
    "cylindrical_domain_with_airfoil_hole",
    "sidewall_deformed_block",
    "rounded_u_notch_extrusion",
    "tapered_elliptic_bore_block",
    "bent_annular_duct",
    "lobed_annular_duct",
    "square_to_round_transition",
    "variable_area_s_duct",
    "twisted_blade_passage",
)


class TestPaperBoundary3D:
    def test_registered_order_and_round_trip(self):
        assert all(name in DATASET_FAMILIES for name in PAPER_FAMILIES)
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
            # run_e has separately tested dense PDE caps. Coarse sampling can
            # still yield invalid straight-sided cells; it is not a generic TFI case.
            if name == "rounded_u_notch_extrusion":
                continue
            grid = tfi_from_surfaces(
                BOUNDARY_GENERATORS[name](), xi=11, eta=21, zeta=11
            )
            quality = evaluate_hexahedral_grid_quality(
                grid, metrics=["minimum_scaled_jacobian", "flipped_cell"]
            )
            assert (float(quality["minimum_scaled_jacobian"].min())) > (0.0)
            assert (int(quality["flipped_cell"].sum())) == (0)

    def test_new_industrial_families_are_legal_at_extrapolation_ends(self):
        for name in (
            "square_to_round_transition", "variable_area_s_duct",
            "twisted_blade_passage",
        ):
            specification = DATASET_FAMILIES[name]
            values = np.asarray(specification["extrapolation"], dtype=float)
            for value in (float(values[:, 0].min()), float(values[:, 1].max())):
                parameters = dict(specification["fixed_parameters"])
                parameters[specification["parameter"]] = value
                grid = tfi_from_surfaces(
                    BOUNDARY_GENERATORS[name](**parameters), xi=11, eta=21, zeta=11
                )
                quality = evaluate_hexahedral_grid_quality(
                    grid, metrics=["minimum_scaled_jacobian", "flipped_cell"]
                )
                assert float(quality["minimum_scaled_jacobian"].min()) > 0.0
                assert int(quality["flipped_cell"].sum()) == 0

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

    def test_run_e_cross_section_matches_2d_run_exactly(self):
        depth, radius = 1.5, 0.25
        two_d = rounded_u_notch_boundary(depth=depth, radius=radius)
        three_d = BOUNDARY_GENERATORS["rounded_u_notch_extrusion"](
            depth=depth, radius=radius
        )
        q = np.linspace(0.0, 1.0, 401)
        np.testing.assert_allclose(
            two_d.top.evaluate(q), three_d.z_min.evaluate(q, np.ones_like(q))[:, :2],
            atol=2e-12, rtol=0.0,
        )

    def test_teb_outer_rectangle_uses_four_straight_parameter_sides(self):
        boundary = BOUNDARY_GENERATORS["tapered_elliptic_bore_block"](
            outer_half_width=1.5,
            outer_half_height=1.0,
            minor_axis_out=0.32,
        )
        q = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
        z = np.full_like(q, 0.4)
        outer = boundary.x_max.evaluate(q, z)
        np.testing.assert_allclose(
            outer[:, :2],
            [[1.5, -1.0], [1.5, 1.0], [-1.5, 1.0], [-1.5, -1.0], [1.5, -1.0]],
            atol=2e-12,
            rtol=0.0,
        )
        # Each quarter interval is one affine side, so its midpoint must be
        # the corresponding geometric side midpoint rather than a ray hit.
        midpoints = boundary.x_max.evaluate(
            np.asarray([0.125, 0.375, 0.625, 0.875]),
            np.full(4, 0.4),
        )
        np.testing.assert_allclose(
            midpoints[:, :2],
            [[1.5, 0.0], [0.0, 1.0], [-1.5, 0.0], [0.0, -1.0]],
            atol=2e-12,
            rtol=0.0,
        )

    def test_teb_inner_faces_remain_exact_ellipses(self):
        major, inlet_minor, outlet_minor = 0.32, 0.18, 0.32
        boundary = BOUNDARY_GENERATORS["tapered_elliptic_bore_block"](
            inner_major_axis=major,
            inner_minor_axis_in=inlet_minor,
            minor_axis_out=outlet_minor,
        )
        q = np.linspace(0.0, 1.0, 101)
        inlet = boundary.x_min.evaluate(q, np.zeros_like(q))
        outlet = boundary.x_min.evaluate(q, np.ones_like(q))
        np.testing.assert_allclose(
            (inlet[:, 0] / major) ** 2 + (inlet[:, 1] / inlet_minor) ** 2,
            1.0,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            (outlet[:, 0] / major) ** 2 + (outlet[:, 1] / outlet_minor) ** 2,
            1.0,
            atol=2e-12,
        )

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

    def test_localized_dimple_is_bijective_but_edge_only_tfi_is_flat(self):
        depth, height = 0.25, 1.0
        boundary = BOUNDARY_GENERATORS["localized_dimple_block"](
            length_x=2.0, width_y=1.5, height_z=height, dimple_depth=depth
        )
        center = boundary.z_max.evaluate(np.asarray(0.5), np.asarray(0.5))
        np.testing.assert_allclose(center, [1.0, 0.75, height - depth], atol=2e-12)

        # All four top edges remain the rectangle z=H.  A surface reconstructed
        # from those edges alone is therefore the flat bilinear face and misses
        # the analytic dimple by exactly ``depth`` at the centre.
        edge_only_tfi_center = np.asarray([1.0, 0.75, height])
        np.testing.assert_allclose(
            edge_only_tfi_center - center, [0.0, 0.0, depth], atol=2e-12
        )

        q = np.linspace(0.0, 1.0, 31)
        bump = 64.0 * q ** 3 * (1.0 - q) ** 3
        determinant = 2.0 * 1.5 * (height - depth * bump[:, None] * bump[None, :])
        assert float(determinant.min()) > 0.0

        grid = tfi_from_surfaces(boundary, xi=17, eta=17, zeta=17)
        q = np.linspace(0.0, 1.0, 17)
        zeta, eta, xi = np.meshgrid(q, q, q, indexing="ij")
        bump_x = 64.0 * xi ** 3 * (1.0 - xi) ** 3
        bump_y = 64.0 * eta ** 3 * (1.0 - eta) ** 3
        exact = np.stack(
            (2.0 * xi, 1.5 * eta, zeta * (height - depth * bump_x * bump_y)),
            axis=-1,
        )
        np.testing.assert_allclose(grid, exact, atol=2e-12, rtol=0.0)
        quality = evaluate_hexahedral_grid_quality(
            grid, metrics=["minimum_scaled_jacobian", "flipped_cell"]
        )
        assert float(quality["minimum_scaled_jacobian"].min()) > 0.0
        assert int(quality["flipped_cell"].sum()) == 0
