import unittest

import torch

from scripts.experiments.run_ablation_2d import run_target_interpolation_experiment
from scripts.train.training_common import (
    interpolate_pde_mesh,
    third_refined_surface_parameters,
)
from src.data.geometry.boundary_generator_3d import box_boundary


class GlobalBoundaryIDWInterpolationTests(unittest.TestCase):
    @staticmethod
    def _mesh(shape=(4, 4), amplitude=0.2):
        nx, ny = shape
        rows = []
        for j in range(ny):
            eta = j / (ny - 1)
            for i in range(nx):
                xi = i / (nx - 1)
                y = -4.0 * amplitude * xi * (1.0 - xi) if j == 0 else eta
                rows.append((xi, y))
        return torch.tensor([rows], dtype=torch.float64)

    class Geometry:
        def __init__(self, amplitude=0.2):
            self.amplitude = amplitude
            self.last_xi = None
            self.last_eta = None

        def project_torch(self, xi, eta):
            self.last_xi = xi.detach().clone()
            self.last_eta = eta.detach().clone()
            left = torch.stack((torch.zeros_like(eta), eta), dim=-1)
            right = torch.stack((torch.ones_like(eta), eta), dim=-1)
            bottom = torch.stack(
                (xi, -4.0 * self.amplitude * xi * (1.0 - xi)), dim=-1
            )
            top = torch.stack((xi, torch.ones_like(xi)), dim=-1)
            return torch.stack((left, right, bottom, top), dim=-2)

    def test_boundary_support_matches_exact_curve(self):
        mesh = self._mesh()
        # For nx=4, 1/6 is the midpoint of the first original interval.
        query = torch.tensor([[[1.0 / 6.0, 0.0]]], dtype=torch.float64)
        adjusted = interpolate_pde_mesh(
            mesh, query, (4, 4), "curved_boundary_cell", self.Geometry()
        )
        exact = self.Geometry().project_torch(
            query[..., 0], query[..., 1]
        )[..., 2, :]
        torch.testing.assert_close(adjusted, exact, rtol=0.0, atol=1e-12)

    def test_supports_are_quarter_points_without_original_nodes(self):
        mesh = self._mesh()
        geometry = self.Geometry()
        query = torch.tensor([[[0.5, 0.5]]], dtype=torch.float64)
        interpolate_pde_mesh(
            mesh, query, (4, 4), "curved_boundary_cell", geometry
        )

        horizontal_count = 3 * (4 - 1)
        bottom_xi = geometry.last_xi[0, :horizontal_count]
        expected = torch.tensor(
            [
                1 / 12, 1 / 6, 1 / 4,
                5 / 12, 1 / 2, 7 / 12,
                3 / 4, 5 / 6, 11 / 12,
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(bottom_xi, expected, rtol=0.0, atol=1e-12)
        self.assertEqual(geometry.last_xi.shape[1], 4 * horizontal_count)
        original_nodes = torch.tensor(
            [0.0, 1 / 3, 2 / 3, 1.0], dtype=torch.float64
        )
        self.assertFalse(
            torch.isclose(bottom_xi[:, None], original_nodes[None, :]).any().item()
        )

    def test_global_idw_changes_an_interior_point_toward_curved_boundary(self):
        mesh = self._mesh()
        query = torch.tensor([[[1.0 / 6.0, 1.0 / 3.0]]], dtype=torch.float64)
        linear = interpolate_pde_mesh(mesh, query, (4, 4), "bilinear")
        adjusted = interpolate_pde_mesh(
            mesh, query, (4, 4), "curved_boundary_cell", self.Geometry()
        )
        self.assertLess(adjusted[0, 0, 1], linear[0, 0, 1])
        self.assertAlmostEqual(adjusted[0, 0, 0].item(), linear[0, 0, 0].item())

    def test_all_boundary_supports_participate_globally(self):
        mesh = self._mesh()
        query = torch.tensor([[[0.5, 0.8]]], dtype=torch.float64)
        linear = interpolate_pde_mesh(mesh, query, (4, 4), "bilinear")
        adjusted = interpolate_pde_mesh(
            mesh, query, (4, 4), "curved_boundary_cell", self.Geometry()
        )
        self.assertLess(adjusted[0, 0, 1], linear[0, 0, 1])

    def test_ablation_keeps_the_backward_compatible_mode_name(self):
        variants = run_target_interpolation_experiment({})
        self.assertEqual(
            [item.name for item in variants], ["bilinear", "curved_boundary_cell"]
        )
        self.assertEqual(
            variants[1].overrides,
            ("--target-interpolation", "curved_boundary_cell"),
        )


class GlobalBoundarySurfaceIDWInterpolationTests(unittest.TestCase):
    def test_third_refined_surface_excludes_original_nodes(self):
        u, v = third_refined_surface_parameters(3, 2, dtype=torch.float64)
        cached_u, cached_v = third_refined_surface_parameters(3, 2, dtype=torch.float64)
        self.assertEqual(u.data_ptr(), cached_u.data_ptr())
        self.assertEqual(v.data_ptr(), cached_v.data_ptr())
        self.assertEqual(u.numel(), (3 * (3 - 1) + 1) * (3 * (2 - 1) + 1) - 3 * 2)
        original_u = torch.tensor((0.0, 0.5, 1.0), dtype=torch.float64)
        original_v = torch.tensor((0.0, 1.0), dtype=torch.float64)
        is_original_u = torch.isclose(u[:, None], original_u[None, :]).any(-1)
        is_original_v = torch.isclose(v[:, None], original_v[None, :]).any(-1)
        self.assertFalse((is_original_u & is_original_v).any().item())

    def test_exact_box_surfaces_leave_trilinear_targets_unchanged(self):
        nx, ny, nz = 3, 3, 3
        rows = []
        for k in range(nz):
            zeta = k / (nz - 1)
            for j in range(ny):
                eta = j / (ny - 1)
                for i in range(nx):
                    xi = i / (nx - 1)
                    rows.append((2 * xi - 1, 2 * eta - 1, 2 * zeta - 1))
        mesh = torch.tensor([rows], dtype=torch.float64)
        query = torch.tensor(
            [[[0.17, 0.31, 0.43], [0.72, 0.58, 0.26]]], dtype=torch.float64
        )
        trilinear = interpolate_pde_mesh(mesh, query, (nx, ny, nz), "bilinear")
        adjusted = interpolate_pde_mesh(
            mesh,
            query,
            (nx, ny, nz),
            "curved_boundary_cell",
            [box_boundary()],
        )
        torch.testing.assert_close(adjusted, trilinear, rtol=0.0, atol=1e-12)

    def test_local_idw_accepts_32_neighbors(self):
        nx, ny, nz = 3, 3, 3
        rows = [
            (2 * i / (nx - 1) - 1, 2 * j / (ny - 1) - 1, 2 * k / (nz - 1) - 1)
            for k in range(nz)
            for j in range(ny)
            for i in range(nx)
        ]
        mesh = torch.tensor([rows], dtype=torch.float64)
        query = torch.tensor([[[0.21, 0.37, 0.63]]], dtype=torch.float64)
        expected = interpolate_pde_mesh(mesh, query, (nx, ny, nz), "bilinear")
        adjusted = interpolate_pde_mesh(
            mesh,
            query,
            (nx, ny, nz),
            "curved_boundary_cell",
            [box_boundary()],
            idw_neighbors=32,
            idw_support_chunk_size=8,
        )
        torch.testing.assert_close(adjusted, expected, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
