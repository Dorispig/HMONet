"""Documentation string."""

__all__ = [
    "winslowSolver2D",
    "winslowSolver3D",
    "evaluate_hexahedral_grid_quality",
    "NonuniformWinslowSolver2D",
    "NonuniformWinslowSolver3D",
]
# ============================================================
# Implementation comment.
# ============================================================
from pathlib import Path

# ============================================================
# Implementation comment.
# ============================================================
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ============================================================
# Implementation comment.
# ============================================================
import src.data.mesh.topology as quad_mesh
import src.utils.visualize as visualize
import src.utils.metrics as metrics
from src.utils.metrics import evaluate_hexahedral_grid_quality
import src.utils.math as helper
from scripts.tfi import tfi_legacy as TFI

# ============================================================
# Implementation comment.
# ============================================================

# ============================================================
# Implementation comment.
# ============================================================


# ============================================================
# Implementation comment.
# ============================================================
class winslowSolver2D:
    def __init__(
        self,
        Nx: int,
        Ny: int,
        boundary_values: np.ndarray,
        initial_points: np.ndarray = None,
        max_iter=1000,
        tol=1e-6,
    ):
        """Documentation string."""
        self.Nx = Nx
        self.Ny = Ny
        self.boundary_values = boundary_values
        self.boundary_indices = quad_mesh.GenerateQuadMeshBoundaryIndex(
            self.Ny, self.Nx
        )
        self.initial_points = initial_points
        self.max_iter = max_iter
        self.tol = tol
        self.iterations_performed = 0
        self.last_relative_difference = None

    def solve(self):
        """Documentation string."""
        # Implementation comment.
        # Implementation comment.
        dx = 1.0 / (self.Nx - 1)
        dy = 1.0 / (self.Ny - 1)

        DoF = self.Nx * self.Ny  # Degrees of freedom for the solution

        if self.initial_points is not None:
            solution = self.initial_points.copy()
        else:
            # Initialize the solution array with TFI
            bottomPoints = self.boundary_values[: self.Nx]
            rightPoints = self.boundary_values[self.Nx - 1 : self.Nx + self.Ny - 1]
            topPoints = self.boundary_values[
                self.Nx + self.Ny - 2 : self.Nx + self.Ny + self.Nx - 2
            ]
            leftPoints = self.boundary_values[self.Nx + self.Ny + self.Nx - 3 :]
            # Implementation comment.
            if (
                self.boundary_values[0, 0] != self.boundary_values[-1, 0]
                or self.boundary_values[0, 1] != self.boundary_values[-1, 1]
            ):
                leftPoints = np.append(
                    leftPoints, bottomPoints[0:1], axis=0
                )  # Implementation comment.
            topPoints = topPoints[::-1]  # Implementation comment.
            leftPoints = leftPoints[::-1]  # Implementation comment.

            TFISolver = TFI.TFI2D(bottomPoints, rightPoints, topPoints, leftPoints)
            solution = TFISolver().reshape(-1, 2)  # Implementation comment.
            # solution = np.stack([solution[:DoF], solution[DoF:]], axis=1)
            # return solution
        solution = np.concatenate(
            [solution[:, 0:1].flatten(), solution[:, 1:2].flatten()], axis=0
        )

        # Set the boundary values
        for idx, bd_idx in enumerate(self.boundary_indices):
            solution[bd_idx] = self.boundary_values[idx, 0]  # x-component
            solution[bd_idx + DoF] = self.boundary_values[idx, 1]  # y-component

        # Initialize the linear system of winslow equation

        B_row = list()
        B_col = list()
        B_data = list()

        b = np.zeros((DoF * 2,))

        for idx, bd_idx in enumerate(self.boundary_indices):
            B_row.append(bd_idx)
            B_col.append(bd_idx)
            B_data.append(1.0)

            B_row.append(bd_idx + DoF)
            B_col.append(bd_idx + DoF)
            B_data.append(1.0)

            b[bd_idx] = self.boundary_values[idx, 0]
            b[bd_idx + DoF] = self.boundary_values[idx, 1]

        B = sp.coo_matrix((B_data, (B_row, B_col)), shape=(DoF * 2, DoF * 2))

        for it in range(self.max_iter):

            # Initialize the coefficient lists for the linear system
            E_row = list()
            E_col = list()
            E_data = list()

            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    # Compute the coefficients based on the current solution

                    cx00 = helper.coord_to_idx(i - 1, j - 1, self.Nx, self.Ny, 0)
                    cx01 = helper.coord_to_idx(i - 1, j, self.Nx, self.Ny, 0)
                    cx02 = helper.coord_to_idx(i - 1, j + 1, self.Nx, self.Ny, 0)
                    cx10 = helper.coord_to_idx(i, j - 1, self.Nx, self.Ny, 0)
                    cx11 = helper.coord_to_idx(i, j, self.Nx, self.Ny, 0)
                    cx12 = helper.coord_to_idx(i, j + 1, self.Nx, self.Ny, 0)
                    cx20 = helper.coord_to_idx(i + 1, j - 1, self.Nx, self.Ny, 0)
                    cx21 = helper.coord_to_idx(i + 1, j, self.Nx, self.Ny, 0)
                    cx22 = helper.coord_to_idx(i + 1, j + 1, self.Nx, self.Ny, 0)

                    cy00 = helper.coord_to_idx(i - 1, j - 1, self.Nx, self.Ny, 1)
                    cy01 = helper.coord_to_idx(i - 1, j, self.Nx, self.Ny, 1)
                    cy02 = helper.coord_to_idx(i - 1, j + 1, self.Nx, self.Ny, 1)
                    cy10 = helper.coord_to_idx(i, j - 1, self.Nx, self.Ny, 1)
                    cy11 = helper.coord_to_idx(i, j, self.Nx, self.Ny, 1)
                    cy12 = helper.coord_to_idx(i, j + 1, self.Nx, self.Ny, 1)
                    cy20 = helper.coord_to_idx(i + 1, j - 1, self.Nx, self.Ny, 1)
                    cy21 = helper.coord_to_idx(i + 1, j, self.Nx, self.Ny, 1)
                    cy22 = helper.coord_to_idx(i + 1, j + 1, self.Nx, self.Ny, 1)

                    x_eta = (solution[cx12] - solution[cx10]) / dy / 2
                    x_xi = (solution[cx21] - solution[cx01]) / dx / 2
                    y_xi = (solution[cy21] - solution[cy01]) / dx / 2
                    y_eta = (solution[cy12] - solution[cy10]) / dy / 2

                    alph = x_eta**2 + y_eta**2
                    beta = x_eta * x_xi + y_eta * y_xi
                    gamm = x_xi**2 + y_xi**2

                    # Assemble the coefficients for the linear system
                    # For the x-component

                    # X_00
                    E_row.append(cx11)
                    E_col.append(cx11)
                    E_data.append(-2.0 / (dx * dx) * alph)

                    E_row.append(cx11)
                    E_col.append(cx01)
                    E_data.append(1.0 / (dx * dx) * alph)

                    E_row.append(cx11)
                    E_col.append(cx21)
                    E_data.append(1.0 / (dx * dx) * alph)

                    # X_01
                    E_row.append(cx11)
                    E_col.append(cx22)
                    E_data.append(-1.0 / (2 * dx * dy) * beta)

                    E_row.append(cx11)
                    E_col.append(cx20)
                    E_data.append(1.0 / (2 * dx * dy) * beta)

                    E_row.append(cx11)
                    E_col.append(cx02)
                    E_data.append(1.0 / (2 * dx * dy) * beta)

                    E_row.append(cx11)
                    E_col.append(cx00)
                    E_data.append(-1.0 / (2 * dx * dy) * beta)

                    # X_11
                    E_row.append(cx11)
                    E_col.append(cx11)
                    E_data.append(-2.0 / (dy * dy) * gamm)

                    E_row.append(cx11)
                    E_col.append(cx10)
                    E_data.append(1.0 / (dy * dy) * gamm)

                    E_row.append(cx11)
                    E_col.append(cx12)
                    E_data.append(1.0 / (dy * dy) * gamm)

                    # For the y-component

                    # Y_00
                    E_row.append(cy11)
                    E_col.append(cy11)
                    E_data.append(-2.0 / (dx * dx) * alph)

                    E_row.append(cy11)
                    E_col.append(cy01)
                    E_data.append(1.0 / (dx * dx) * alph)

                    E_row.append(cy11)
                    E_col.append(cy21)
                    E_data.append(1.0 / (dx * dx) * alph)

                    # Y_01
                    E_row.append(cy11)
                    E_col.append(cy22)
                    E_data.append(-1.0 / (2 * dx * dy) * beta)

                    E_row.append(cy11)
                    E_col.append(cy20)
                    E_data.append(1.0 / (2 * dx * dy) * beta)

                    E_row.append(cy11)
                    E_col.append(cy02)
                    E_data.append(1.0 / (2 * dx * dy) * beta)

                    E_row.append(cy11)
                    E_col.append(cy00)
                    E_data.append(-1.0 / (2 * dx * dy) * beta)

                    # Y_11
                    E_row.append(cy11)
                    E_col.append(cy11)
                    E_data.append(-2.0 / (dy * dy) * gamm)

                    E_row.append(cy11)
                    E_col.append(cy10)
                    E_data.append(1.0 / (dy * dy) * gamm)

                    E_row.append(cy11)
                    E_col.append(cy12)
                    E_data.append(1.0 / (dy * dy) * gamm)

            # Ensemble the coefficients for the Winslow equation
            E = sp.coo_matrix((E_data, (E_row, E_col)), shape=(DoF * 2, DoF * 2))
            E = E + B

            new_solution = spla.spsolve(E.tocsr(), b)

            rel_diff = np.linalg.norm(new_solution - solution) / np.linalg.norm(
                solution
            )
            solution = new_solution
            self.iterations_performed = it + 1
            self.last_relative_difference = float(rel_diff)

            # Check for convergence
            if rel_diff < self.tol:
                # print(f"Converged after {it + 1} iterations.")
                break
            else:
                # print(f"Iteration {it + 1}, relative difference: {rel_diff}")
                pass

            # Set the boundary values
            for idx, bd_idx in enumerate(self.boundary_indices):
                solution[bd_idx] = self.boundary_values[idx, 0]  # x-component
                solution[bd_idx + DoF] = self.boundary_values[idx, 1]  # y-component
        solution = np.stack([solution[:DoF], solution[DoF:]], axis=1)
        return solution


class winslowSolver3D:
    """Documentation string."""

    _FACE_NAMES = ("left", "right", "front", "back", "bottom", "top")

    def __init__(
        self,
        Nx: int,
        Ny: int,
        Nz: int,
        boundary_values,
        initial_points: np.ndarray = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        relaxation: float = 1.0,
    ):
        for name, value in (("Nx", Nx), ("Ny", Ny), ("Nz", Nz)):
            if not isinstance(value, (int, np.integer)) or value < 3:
                raise ValueError("""Documentation string.""")
        if not isinstance(boundary_values, dict):
            raise TypeError("""Documentation string.""")
        missing_faces = [
            name for name in self._FACE_NAMES if name not in boundary_values
        ]
        if missing_faces:
            raise ValueError("""Documentation string.""")
        if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError("""Documentation string.""")
        if tol <= 0:
            raise ValueError("""Documentation string.""")
        if not 0 < relaxation <= 1:
            raise ValueError("""Documentation string.""")

        self.Nx, self.Ny, self.Nz = int(Nx), int(Ny), int(Nz)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.relaxation = float(relaxation)
        self.initial_points = initial_points

        self.boundary_values = {
            name: np.asarray(boundary_values[name], dtype=float)
            for name in self._FACE_NAMES
        }
        self.tfi_solver = TFI.TFI3D(
            self.boundary_values["left"],
            self.boundary_values["right"],
            self.boundary_values["front"],
            self.boundary_values["back"],
            self.boundary_values["bottom"],
            self.boundary_values["top"],
        )
        if (
            self.tfi_solver.nu != self.Nx
            or self.tfi_solver.nv != self.Ny
            or self.tfi_solver.nw != self.Nz
        ):
            raise ValueError(
                """Documentation string."""
                """Documentation string."""
                """Documentation string."""
            )
        self.last_iterations = 0
        self.last_relative_difference = np.inf

    def _initial_grid(self) -> np.ndarray:
        if self.initial_points is None:
            grid = self.tfi_solver.generate_grid()
        else:
            points = np.asarray(self.initial_points, dtype=float)
            if points.shape == (self.Nz * self.Ny * self.Nx, 3):
                grid = points.reshape(self.Nz, self.Ny, self.Nx, 3).copy()
            elif points.shape == (self.Nz, self.Ny, self.Nx, 3):
                grid = points.copy()
            else:
                raise ValueError(
                    """Documentation string.""" """Documentation string."""
                )
            if not np.isfinite(grid).all():
                raise ValueError("""Documentation string.""")
        self._apply_boundary(grid)
        return grid

    def _apply_boundary(self, grid: np.ndarray) -> None:
        grid[:, :, 0] = self.boundary_values["left"]
        grid[:, :, -1] = self.boundary_values["right"]
        grid[:, 0, :] = self.boundary_values["front"]
        grid[:, -1, :] = self.boundary_values["back"]
        grid[0, :, :] = self.boundary_values["bottom"]
        grid[-1, :, :] = self.boundary_values["top"]

    def _flat_index(self, i: int, j: int, k: int) -> int:
        return k * self.Ny * self.Nx + j * self.Nx + i

    def solve(self) -> np.ndarray:
        """Documentation string."""
        grid = self._initial_grid()
        dof = self.Nx * self.Ny * self.Nz
        du = 1.0 / (self.Nx - 1)
        dv = 1.0 / (self.Ny - 1)
        dw = 1.0 / (self.Nz - 1)

        boundary_mask = np.zeros((self.Nz, self.Ny, self.Nx), dtype=bool)
        boundary_mask[[0, -1], :, :] = True
        boundary_mask[:, [0, -1], :] = True
        boundary_mask[:, :, [0, -1]] = True
        boundary_indices = np.flatnonzero(boundary_mask.reshape(-1))

        for iteration in range(1, self.max_iter + 1):
            rows, cols, data = [], [], []

            for index in boundary_indices:
                rows.append(int(index))
                cols.append(int(index))
                data.append(1.0)

            for k in range(1, self.Nz - 1):
                for j in range(1, self.Ny - 1):
                    for i in range(1, self.Nx - 1):
                        row = self._flat_index(i, j, k)
                        r_u = (grid[k, j, i + 1] - grid[k, j, i - 1]) / (2 * du)
                        r_v = (grid[k, j + 1, i] - grid[k, j - 1, i]) / (2 * dv)
                        r_w = (grid[k + 1, j, i] - grid[k - 1, j, i]) / (2 * dw)

                        g11 = float(np.dot(r_u, r_u))
                        g22 = float(np.dot(r_v, r_v))
                        g33 = float(np.dot(r_w, r_w))
                        g12 = float(np.dot(r_u, r_v))
                        g13 = float(np.dot(r_u, r_w))
                        g23 = float(np.dot(r_v, r_w))

                        a11 = g22 * g33 - g23 * g23
                        a22 = g11 * g33 - g13 * g13
                        a33 = g11 * g22 - g12 * g12
                        a12 = g13 * g23 - g12 * g33
                        a13 = g12 * g23 - g13 * g22
                        a23 = g12 * g13 - g11 * g23
                        scale = max(a11 + a22 + a33, 1e-14)
                        a11, a22, a33 = a11 / scale, a22 / scale, a33 / scale
                        a12, a13, a23 = a12 / scale, a13 / scale, a23 / scale

                        def add(ii, jj, kk, coefficient):
                            rows.append(row)
                            cols.append(self._flat_index(ii, jj, kk))
                            data.append(coefficient)

                        add(
                            i,
                            j,
                            k,
                            -2 * a11 / du**2 - 2 * a22 / dv**2 - 2 * a33 / dw**2,
                        )
                        add(i - 1, j, k, a11 / du**2)
                        add(i + 1, j, k, a11 / du**2)
                        add(i, j - 1, k, a22 / dv**2)
                        add(i, j + 1, k, a22 / dv**2)
                        add(i, j, k - 1, a33 / dw**2)
                        add(i, j, k + 1, a33 / dw**2)

                        cross_uv = a12 / (2 * du * dv)
                        add(i + 1, j + 1, k, cross_uv)
                        add(i + 1, j - 1, k, -cross_uv)
                        add(i - 1, j + 1, k, -cross_uv)
                        add(i - 1, j - 1, k, cross_uv)

                        cross_uw = a13 / (2 * du * dw)
                        add(i + 1, j, k + 1, cross_uw)
                        add(i + 1, j, k - 1, -cross_uw)
                        add(i - 1, j, k + 1, -cross_uw)
                        add(i - 1, j, k - 1, cross_uw)

                        cross_vw = a23 / (2 * dv * dw)
                        add(i, j + 1, k + 1, cross_vw)
                        add(i, j + 1, k - 1, -cross_vw)
                        add(i, j - 1, k + 1, -cross_vw)
                        add(i, j - 1, k - 1, cross_vw)

            matrix = sp.coo_matrix(
                (data, (rows, cols)),
                shape=(dof, dof),
            ).tocsr()
            old_points = grid.reshape(dof, 3)
            new_points = np.empty_like(old_points)
            for component in range(3):
                rhs = np.zeros(dof)
                rhs[boundary_indices] = old_points[boundary_indices, component]
                new_points[:, component] = spla.spsolve(matrix, rhs)

            if not np.isfinite(new_points).all():
                raise FloatingPointError("""Documentation string.""")
            relaxed_points = (
                self.relaxation * new_points + (1.0 - self.relaxation) * old_points
            )
            new_grid = relaxed_points.reshape(self.Nz, self.Ny, self.Nx, 3)
            self._apply_boundary(new_grid)
            denominator = max(np.linalg.norm(old_points), 1e-14)
            relative_difference = (
                np.linalg.norm(new_grid.reshape(dof, 3) - old_points) / denominator
            )
            grid = new_grid
            self.last_iterations = iteration
            self.last_relative_difference = float(relative_difference)
            if relative_difference < self.tol:
                break

        return grid.reshape(dof, 3)


# Implementation comment.
def _legacy_evaluate_hexahedral_grid_quality(grid: np.ndarray) -> dict:
    """Documentation string."""
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 4 or grid.shape[-1] != 3 or min(grid.shape[:3]) < 2:
        raise ValueError("""Documentation string.""")
    if not np.isfinite(grid).all():
        raise ValueError("""Documentation string.""")

    scaled_jacobians = []
    condition_numbers = []
    edge_aspect_ratios = []
    for k in range(grid.shape[0] - 1):
        for j in range(grid.shape[1] - 1):
            for i in range(grid.shape[2] - 1):
                cell = grid[k : k + 2, j : j + 2, i : i + 2]
                corner_scaled_jacobians = []
                for ck in (0, 1):
                    for cj in (0, 1):
                        for ci in (0, 1):
                            edge_u = cell[ck, cj, 1] - cell[ck, cj, 0]
                            edge_v = cell[ck, 1, ci] - cell[ck, 0, ci]
                            edge_w = cell[1, cj, ci] - cell[0, cj, ci]
                            denominator = (
                                np.linalg.norm(edge_u)
                                * np.linalg.norm(edge_v)
                                * np.linalg.norm(edge_w)
                            )
                            if denominator <= 1e-14:
                                corner_scaled_jacobians.append(-np.inf)
                            else:
                                jacobian = np.column_stack((edge_u, edge_v, edge_w))
                                corner_scaled_jacobians.append(
                                    np.linalg.det(jacobian) / denominator
                                )
                scaled_jacobians.append(min(corner_scaled_jacobians))

                center_u = np.mean(cell[:, :, 1] - cell[:, :, 0], axis=(0, 1))
                center_v = np.mean(cell[:, 1, :] - cell[:, 0, :], axis=(0, 1))
                center_w = np.mean(cell[1, :, :] - cell[0, :, :], axis=(0, 1))
                center_jacobian = np.column_stack((center_u, center_v, center_w))
                condition_numbers.append(np.linalg.cond(center_jacobian))

                edge_lengths = []
                for ck in (0, 1):
                    for cj in (0, 1):
                        edge_lengths.append(
                            np.linalg.norm(cell[ck, cj, 1] - cell[ck, cj, 0])
                        )
                for ck in (0, 1):
                    for ci in (0, 1):
                        edge_lengths.append(
                            np.linalg.norm(cell[ck, 1, ci] - cell[ck, 0, ci])
                        )
                for cj in (0, 1):
                    for ci in (0, 1):
                        edge_lengths.append(
                            np.linalg.norm(cell[1, cj, ci] - cell[0, cj, ci])
                        )
                edge_aspect_ratios.append(max(edge_lengths) / min(edge_lengths))

    return {
        "minimum_scaled_jacobian": np.asarray(scaled_jacobians),
        "center_condition_number": np.asarray(condition_numbers),
        "edge_aspect_ratio": np.asarray(edge_aspect_ratios),
    }


# ============================================================
# Implementation comment.
# ============================================================

if __name__ == "__main__":
    # Implementation comment.
    Nx = Ny = Nz = 10
    duct_length = 3.6
    u_values = np.linspace(-1.0, 1.0, Nx)
    v_values = np.linspace(-1.0, 1.0, Ny)
    w_values = np.linspace(0.0, 1.0, Nz)

    def industrial_duct_mapping(u, v, w):
        """Documentation string."""
        center_x = 0.85 * np.sin(np.pi * w)
        center_y = 0.32 * np.sin(2.0 * np.pi * w)
        half_width = 1.0 - 0.38 * w + 0.10 * np.sin(np.pi * w)
        half_height = 0.72 - 0.22 * w + 0.08 * np.sin(np.pi * w)
        twist = np.deg2rad(75.0) * w + 0.12 * np.sin(np.pi * w)
        local_x = half_width * u
        local_y = half_height * v
        x = center_x + local_x * np.cos(twist) - local_y * np.sin(twist)
        y = center_y + local_x * np.sin(twist) + local_y * np.cos(twist)
        z = duct_length * w
        return np.array([x, y, z])

    boundary_faces = {
        "left": np.array(
            [[industrial_duct_mapping(-1.0, v, w) for v in v_values] for w in w_values]
        ),
        "right": np.array(
            [[industrial_duct_mapping(1.0, v, w) for v in v_values] for w in w_values]
        ),
        "front": np.array(
            [[industrial_duct_mapping(u, -1.0, w) for u in u_values] for w in w_values]
        ),
        "back": np.array(
            [[industrial_duct_mapping(u, 1.0, w) for u in u_values] for w in w_values]
        ),
        "bottom": np.array(
            [[industrial_duct_mapping(u, v, 0.0) for u in u_values] for v in v_values]
        ),
        "top": np.array(
            [[industrial_duct_mapping(u, v, 1.0) for u in u_values] for v in v_values]
        ),
    }

    solver = winslowSolver3D(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        boundary_values=boundary_faces,
        initial_points=None,
        max_iter=100,
        tol=1e-8,
        relaxation=0.8,
    )
    tfi_grid = solver.tfi_solver.generate_grid()
    optimized_points = solver.solve()
    optimized_grid = optimized_points.reshape(Nz, Ny, Nx, 3)
    print("Winslow3D point shape:", optimized_points.shape)
    print("Winslow3D grid shape:", optimized_grid.shape)
    print("Winslow3D iterations:", solver.last_iterations)
    print("Winslow3D relative difference:", solver.last_relative_difference)
    assert optimized_points.shape == (10 * 10 * 10, 3)

    # Implementation comment.
    middle_k = Nz // 2
    section_quads = np.asarray(quad_mesh.GenerateQuadMeshIndex(Ny=Ny, Nx=Nx))
    tfi_section_points_2d = tfi_grid[middle_k, :, :, :2].reshape(-1, 2)
    winslow_section_points_2d = optimized_grid[middle_k, :, :, :2].reshape(-1, 2)
    tfi_section_max_angles = metrics.QuadMeshQualityEvaluator(
        tfi_section_points_2d,
        section_quads,
    ).MaxAngle()
    winslow_section_max_angles = metrics.QuadMeshQualityEvaluator(
        winslow_section_points_2d,
        section_quads,
    ).MaxAngle()

    angle_min = min(
        tfi_section_max_angles.min(),
        winslow_section_max_angles.min(),
    )
    angle_max = max(
        tfi_section_max_angles.max(),
        winslow_section_max_angles.max(),
    )
    angle_figure, angle_axes = plt.subplots(1, 2, figsize=(12, 5))
    _, tfi_angle_collection = visualize.plot_quad_mesh_colored(
        tfi_section_points_2d,
        section_quads,
        tfi_section_max_angles,
        ax=angle_axes[0],
        cmap="viridis",
        edgecolor="k",
        linewidth=0.3,
    )
    _, winslow_angle_collection = visualize.plot_quad_mesh_colored(
        winslow_section_points_2d,
        section_quads,
        winslow_section_max_angles,
        ax=angle_axes[1],
        cmap="viridis",
        edgecolor="k",
        linewidth=0.3,
    )
    tfi_angle_collection.set_clim(angle_min, angle_max)
    winslow_angle_collection.set_clim(angle_min, angle_max)
    angle_axes[0].set_title("TFI3D middle-section maximum angle")
    angle_axes[1].set_title("Winslow3D middle-section maximum angle")
    angle_figure.colorbar(
        winslow_angle_collection,
        ax=angle_axes,
        orientation="vertical",
        label="Maximum angle (degree)",
        shrink=0.85,
    )
    angle_output_path = Path(
        "outputs/results/industrial_duct_middle_section_max_angle.png"
    )
    angle_output_path.parent.mkdir(parents=True, exist_ok=True)
    angle_figure.savefig(angle_output_path, dpi=180, bbox_inches="tight")
    plt.close(angle_figure)

    # Implementation comment.
    tfi_quality = evaluate_hexahedral_grid_quality(tfi_grid)
    winslow_quality = evaluate_hexahedral_grid_quality(optimized_grid)
    quality_names = (
        "minimum_scaled_jacobian",
        "center_condition_number",
        "edge_aspect_ratio",
    )
    print(
        "Middle-section max angle TFI3D (min/mean/max):",
        tfi_section_max_angles.min(),
        tfi_section_max_angles.mean(),
        tfi_section_max_angles.max(),
    )
    print(
        "Middle-section max angle Winslow3D (min/mean/max):",
        winslow_section_max_angles.min(),
        winslow_section_max_angles.mean(),
        winslow_section_max_angles.max(),
    )
    for quality_name in quality_names:
        tfi_values = tfi_quality[quality_name]
        winslow_values = winslow_quality[quality_name]
        print(
            f"{quality_name} TFI3D (min/mean/max):",
            tfi_values.min(),
            tfi_values.mean(),
            tfi_values.max(),
        )
        print(
            f"{quality_name} Winslow3D (min/mean/max):",
            winslow_values.min(),
            winslow_values.mean(),
            winslow_values.max(),
        )
    print(
        "Non-positive scaled Jacobian cells TFI3D/Winslow3D:",
        int(np.sum(tfi_quality["minimum_scaled_jacobian"] <= 0)),
        int(np.sum(winslow_quality["minimum_scaled_jacobian"] <= 0)),
    )

    quality_figure, quality_axes = plt.subplots(1, 3, figsize=(15, 4.5))
    quality_titles = (
        "Minimum scaled Jacobian (higher is better)",
        "Center Jacobian condition (lower is better)",
        "Edge aspect ratio (lower is better)",
    )
    for axis, quality_name, title in zip(
        quality_axes,
        quality_names,
        quality_titles,
    ):
        axis.hist(
            tfi_quality[quality_name],
            bins=25,
            alpha=0.55,
            label="TFI3D",
        )
        axis.hist(
            winslow_quality[quality_name],
            bins=25,
            alpha=0.55,
            label="Winslow3D",
        )
        axis.set_title(title)
        axis.set_ylabel("Cell count")
        axis.legend()
        axis.grid(False)
    quality_figure.tight_layout()
    quality_output_path = Path("outputs/results/industrial_duct_hexa_quality.png")
    quality_figure.savefig(quality_output_path, dpi=180, bbox_inches="tight")
    plt.close(quality_figure)
    print(f"Middle-section visualization saved to: {angle_output_path}")
    print(f"Hexahedral-quality visualization saved to: {quality_output_path}")

    fig = plt.figure(figsize=(14, 7))
    axes = (
        fig.add_subplot(121, projection="3d"),
        fig.add_subplot(122, projection="3d"),
    )

    def draw_hexahedral_grid(axis, grid, title):
        points = grid.reshape(-1, 3)
        axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color="tab:blue",
            s=5,
            alpha=0.75,
        )
        for k in range(Nz):
            for j in range(Ny):
                axis.plot(*grid[k, j, :, :].T, color="0.3", linewidth=0.35)
            for i in range(Nx):
                axis.plot(*grid[k, :, i, :].T, color="0.3", linewidth=0.35)
        for j in range(Ny):
            for i in range(Nx):
                axis.plot(*grid[:, j, i, :].T, color="0.3", linewidth=0.35)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.set_title(title)
        axis.set_box_aspect((2.4, 2.0, duct_length))
        axis.grid(False)

    draw_hexahedral_grid(
        axes[0],
        tfi_grid,
        "TFI3D industrial transition duct",
    )
    draw_hexahedral_grid(
        axes[1],
        optimized_grid,
        "Winslow3D industrial transition duct",
    )
    fig.suptitle("""Documentation string.""")
    output_path = Path(
        "outputs/results/industrial_duct_tfi3d_vs_winslow3d_10x10x10.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Visualization saved to: {output_path}")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


class NonuniformWinslowSolver2D:
    """Iterative Winslow/Laplace smoother for nonuniform computational nodes."""

    def __init__(
        self,
        xi,
        eta,
        boundary_values,
        initial_points=None,
        max_iter=1000,
        tol=1e-6,
        relaxation=1.0,
    ):
        self.xi = np.asarray(xi, dtype=float).reshape(-1)
        self.eta = np.asarray(eta, dtype=float).reshape(-1)
        self.boundary_values = np.asarray(boundary_values, dtype=float)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.relaxation = float(relaxation)
        if self.xi.size < 3 or self.eta.size < 3:
            raise ValueError("xi and eta must each contain at least three nodes")
        if np.any(np.diff(self.xi) <= 0) or np.any(np.diff(self.eta) <= 0):
            raise ValueError("xi and eta must be strictly increasing")
        expected = (self.eta.size, self.xi.size)
        if self.boundary_values.shape != expected + (2,):
            raise ValueError(f"boundary_values must have shape {expected + (2,)}")
        if not 0 < self.relaxation <= 1:
            raise ValueError("relaxation must be in (0, 1]")
        self.initial_points = initial_points
        self.iterations_performed = 0
        self.last_relative_difference = np.inf

    def solve(self):
        grid = (
            self.boundary_values.copy()
            if self.initial_points is None
            else np.asarray(self.initial_points, dtype=float).copy()
        )
        if grid.shape != self.boundary_values.shape:
            raise ValueError(
                "initial_points must have the same shape as boundary_values"
            )
        dxm = self.xi[1:-1] - self.xi[:-2]
        dxp = self.xi[2:] - self.xi[1:-1]
        dym = self.eta[1:-1] - self.eta[:-2]
        dyp = self.eta[2:] - self.eta[1:-1]
        axm = 2.0 / (dxm * (dxm + dxp))
        axp = 2.0 / (dxp * (dxm + dxp))
        aym = 2.0 / (dym * (dym + dyp))
        ayp = 2.0 / (dyp * (dym + dyp))
        denominator = aym[:, None] + ayp[:, None] + axm[None, :] + axp[None, :]
        for iteration in range(1, self.max_iter + 1):
            old = grid.copy()
            candidate = (
                aym[:, None, None] * old[:-2, 1:-1]
                + ayp[:, None, None] * old[2:, 1:-1]
                + axm[None, :, None] * old[1:-1, :-2]
                + axp[None, :, None] * old[1:-1, 2:]
            ) / denominator[..., None]
            grid[1:-1, 1:-1] = (
                self.relaxation * candidate + (1.0 - self.relaxation) * old[1:-1, 1:-1]
            )
            grid[0] = self.boundary_values[0]
            grid[-1] = self.boundary_values[-1]
            grid[:, 0] = self.boundary_values[:, 0]
            grid[:, -1] = self.boundary_values[:, -1]
            rel = np.linalg.norm(grid - old) / max(np.linalg.norm(old), 1e-14)
            self.iterations_performed = iteration
            self.last_relative_difference = float(rel)
            if rel < self.tol:
                break
        return grid


class NonuniformWinslowSolver3D:
    """Iterative Winslow/Laplace smoother for nonuniform 3D computational nodes."""

    def __init__(
        self,
        xi,
        eta,
        zeta,
        boundary_values,
        initial_points=None,
        max_iter=1000,
        tol=1e-6,
        relaxation=1.0,
    ):
        self.xi = np.asarray(xi, dtype=float).reshape(-1)
        self.eta = np.asarray(eta, dtype=float).reshape(-1)
        self.zeta = np.asarray(zeta, dtype=float).reshape(-1)
        self.boundary_values = np.asarray(boundary_values, dtype=float)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.relaxation = float(relaxation)
        self.initial_points = initial_points
        if min(self.xi.size, self.eta.size, self.zeta.size) < 3:
            raise ValueError("all coordinate arrays need at least three nodes")
        if any(np.any(np.diff(a) <= 0) for a in (self.xi, self.eta, self.zeta)):
            raise ValueError("coordinate arrays must be strictly increasing")
        expected = (self.zeta.size, self.eta.size, self.xi.size)
        if self.boundary_values.shape != expected + (3,):
            raise ValueError(f"boundary_values must have shape {expected + (3,)}")
        if not 0 < self.relaxation <= 1:
            raise ValueError("relaxation must be in (0, 1]")
        self.iterations_performed = 0
        self.last_relative_difference = np.inf

    def solve(self):
        grid = (
            self.boundary_values.copy()
            if self.initial_points is None
            else np.asarray(self.initial_points, dtype=float).copy()
        )
        if grid.shape != self.boundary_values.shape:
            raise ValueError(
                "initial_points must have the same shape as boundary_values"
            )

        def coeff(nodes):
            left = nodes[1:-1] - nodes[:-2]
            right = nodes[2:] - nodes[1:-1]
            return 2.0 / (left * (left + right)), 2.0 / (right * (left + right))

        axm, axp = coeff(self.xi)
        aym, ayp = coeff(self.eta)
        azm, azp = coeff(self.zeta)
        denominator = (
            azm[:, None, None]
            + azp[:, None, None]
            + aym[None, :, None]
            + ayp[None, :, None]
            + axm[None, None, :]
            + axp[None, None, :]
        )
        for iteration in range(1, self.max_iter + 1):
            old = grid.copy()
            candidate = (
                azm[:, None, None, None] * old[:-2, 1:-1, 1:-1]
                + azp[:, None, None, None] * old[2:, 1:-1, 1:-1]
                + aym[None, :, None, None] * old[1:-1, :-2, 1:-1]
                + ayp[None, :, None, None] * old[1:-1, 2:, 1:-1]
                + axm[None, None, :, None] * old[1:-1, 1:-1, :-2]
                + axp[None, None, :, None] * old[1:-1, 1:-1, 2:]
            ) / denominator[..., None]
            grid[1:-1, 1:-1, 1:-1] = (
                self.relaxation * candidate
                + (1.0 - self.relaxation) * old[1:-1, 1:-1, 1:-1]
            )
            grid[0] = self.boundary_values[0]
            grid[-1] = self.boundary_values[-1]
            grid[:, 0] = self.boundary_values[:, 0]
            grid[:, -1] = self.boundary_values[:, -1]
            grid[:, :, 0] = self.boundary_values[:, :, 0]
            grid[:, :, -1] = self.boundary_values[:, :, -1]
            rel = np.linalg.norm(grid - old) / max(np.linalg.norm(old), 1e-14)
            self.iterations_performed = iteration
            self.last_relative_difference = float(rel)
            if rel < self.tol:
                break
        return grid


