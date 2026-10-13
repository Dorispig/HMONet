"""Documentation string."""

__all__ = [
    "QuadMeshXY",
    "GenerateQuadMeshIndex",
    "GenerateQuadMeshBoundaryIndex",
    "HexMeshXYZ",
    "GenerateHexMeshIndex",
    "GenerateHexMeshBoundaryIndex",
]
# ============================================================
# Implementation comment.
# ============================================================


# ============================================================
# Implementation comment.
# ============================================================
import numpy as np

# ============================================================
# Implementation comment.
# ============================================================
import src.utils.math as helpers

# ============================================================
# Implementation comment.
# ============================================================


# ============================================================
# Implementation comment.
# ============================================================
# Implementation comment.
def GenerateQuadMeshIndex(Ny, Nx):
    """
    8  -  9  -  10 -  11
    |     |     |     |
    4  -  5  -  6  -  7
    |     |     |     |
    0  -  1  -  2  -  3
    """
    Ny = int(Ny)
    Nx = int(Nx)
    if Nx < 2 or Ny < 2:
        return np.empty((0, 4), dtype=int)
    # Row-major cell order, identical to the former nested loops, without the
    # quadratic cost caused by repeatedly growing an array with np.vstack.
    lower_left = (
        np.arange(Ny - 1, dtype=int)[:, None] * Nx
        + np.arange(Nx - 1, dtype=int)[None, :]
    ).ravel()
    return np.column_stack(
        (
            lower_left,
            lower_left + 1,
            lower_left + Nx + 1,
            lower_left + Nx,
        )
    )


# Implementation comment.
def GenerateQuadMeshBoundaryIndex(Ny, Nx):
    """
    8  -  9  -  10 -  11
    |     |     |     |
    4  -  5  -  6  -  7
    |     |     |     |
    0  -  1  -  2  -  3
    -> boundary vertices: [0, 1, 2, 3, 7, 11, 10, 9, 8, 4]
    """
    Ny = int(Ny)
    Nx = int(Nx)
    if Nx < 2 or Ny < 2:
        return np.empty(0, dtype=int)
    # Counter-clockwise order starting at the lower-left corner. Each corner
    # occurs exactly once, matching the previous coord_to_idx implementation.
    bottom = np.arange(0, Nx - 1, dtype=int)
    right = np.arange(Nx - 1, Ny * Nx - 1, Nx, dtype=int)
    top = (Ny - 1) * Nx + np.arange(Nx - 1, 0, -1, dtype=int)
    left = np.arange(Ny - 1, 0, -1, dtype=int) * Nx
    return np.concatenate((bottom, right, top, left))


# ============================================================
# Implementation comment.
# ============================================================


class QuadMeshXY:
    """Documentation string."""

    def __init__(
        self, Nx: int, Ny: int, vertices: np.ndarray = None, faces: np.ndarray = None
    ):
        """Documentation string."""
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.vertices = vertices
        if faces is None:
            if Nx is None or Ny is None:
                raise ValueError("""Documentation string.""")
            else:
                self.faces = GenerateQuadMeshIndex(self.Ny, self.Nx)
        else:
            self.faces = faces
        self.bound_indexs = GenerateQuadMeshBoundaryIndex(self.Ny, self.Nx)

    def __repr__(self):
        return f"QuadMesh(vertices={self.vertices.shape}, faces={self.faces.shape})"

    def faces_indexs(self):
        """Documentation string."""
        return self.faces

    def bound_indexs(self):
        """Documentation string."""
        return self.bound_indexs


def GenerateHexMeshIndex(Nz, Ny, Nx):
    """Documentation string."""
    cells = []
    index = lambda i, j, k: (k * Ny + j) * Nx + i
    for k in range(Nz - 1):
        for j in range(Ny - 1):
            for i in range(Nx - 1):
                cells.append(
                    [
                        index(i, j, k),
                        index(i + 1, j, k),
                        index(i + 1, j + 1, k),
                        index(i, j + 1, k),
                        index(i, j, k + 1),
                        index(i + 1, j, k + 1),
                        index(i + 1, j + 1, k + 1),
                        index(i, j + 1, k + 1),
                    ]
                )
    return np.asarray(cells, dtype=int)


def GenerateHexMeshBoundaryIndex(Nz, Ny, Nx):
    """Documentation string."""
    ids = np.arange(Nz * Ny * Nx, dtype=int).reshape(Nz, Ny, Nx)
    return {
        "left": ids[:, :, 0].copy(),
        "right": ids[:, :, -1].copy(),
        "front": ids[:, 0, :].copy(),
        "back": ids[:, -1, :].copy(),
        "bottom": ids[0, :, :].copy(),
        "top": ids[-1, :, :].copy(),
    }


class HexMeshXYZ:
    """Documentation string."""

    def __init__(self, Nx: int, Ny: int, Nz: int, vertices=None, cells=None):
        self.Nx, self.Ny, self.Nz = int(Nx), int(Ny), int(Nz)
        if min(self.Nx, self.Ny, self.Nz) < 2:
            raise ValueError("Nx, Ny and Nz must be >= 2")
        self.vertices = None if vertices is None else np.asarray(vertices)
        if self.vertices is not None and self.vertices.shape != (
            self.Nx * self.Ny * self.Nz,
            3,
        ):
            raise ValueError("vertices must have shape (Nx*Ny*Nz, 3)")
        self.cells = (
            GenerateHexMeshIndex(self.Nz, self.Ny, self.Nx)
            if cells is None
            else np.asarray(cells, dtype=int)
        )
        self.boundary_faces = GenerateHexMeshBoundaryIndex(self.Nz, self.Ny, self.Nx)


if __name__ == "__main__":
    points = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])
    # faces = np.array([[0, 1, 2, 3]])
    mesh = QuadMeshXY(Nx=3, Ny=2, vertices=points, faces=None)
    print(mesh)
    print(mesh.faces)
    print(mesh.bound_indexs)

