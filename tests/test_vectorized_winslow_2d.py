"""The accelerated dataset solver must reproduce the legacy PDE equation."""

import unittest
import numpy as np
from src.data.generation.make_dataset_parametric_2d import to_computational_boundary
from src.data.geometry.boundary_generator_2d import generate_boundary
from scripts.pde.pde_2d import solve_winslow_2d
from scripts.pde.vectorized_winslow_2d import solve_winslow_2d_vectorized
from scripts.tfi.tfi_2d import tfi_from_curves


class VectorizedWinslowTests(unittest.TestCase):
    def test_same_pde_solution_on_rectangular_and_square_grids(self):
        for name, parameters in [
            ("circle_airfoil", {"t": 0.3}),
            ("rounded_u_notch", {"depth": 1.4}),
            ("variable_width_s_channel", {"bend": 3.5}),
        ]:
            for nx, ny in [(13, 11), (17, 17)]:
                b = to_computational_boundary(generate_boundary(name, **parameters))
                initial = tfi_from_curves(b, nx=nx, ny=ny).reshape(-1, 2)
                reference, ri = solve_winslow_2d(
                    b, nx, ny, initial_grid=initial, tolerance=1e-9, return_info=True
                )
                actual, ai = solve_winslow_2d_vectorized(
                    b, nx, ny, initial_grid=initial, tolerance=1e-9, return_info=True
                )
                self.assertTrue(ri["converged"] and ai["converged"])
                np.testing.assert_allclose(actual, reference, atol=2e-10, rtol=2e-10)


if __name__ == "__main__":
    unittest.main()
