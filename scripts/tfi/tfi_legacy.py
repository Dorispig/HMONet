"""Documentation string."""

__all__ = [
    "TFI2D",
    "TFI3D",
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

# ============================================================
# Implementation comment.
# ============================================================
# ============================================================
# Implementation comment.
# ============================================================

# ============================================================
# Implementation comment.
# ============================================================


# ============================================================
# Implementation comment.
# ============================================================
class TFI2D:
    def __init__(
        self,
        bottomPoints: np.ndarray,
        rightPoints: np.ndarray,
        topPoints: np.ndarray,
        leftPoints: np.ndarray,
    ):
        """Documentation string."""
        self.bottom = bottomPoints
        self.right = rightPoints
        self.top = topPoints
        self.left = leftPoints
        self.nu = bottomPoints.shape[0]
        self.nv = leftPoints.shape[0]

    def __call__(self, *args, **kwds):
        return self.generate_grid().reshape(-1, 2)

    def values(self, u, v):
        Su = (1 - u) * np.array(self.left[int(v * (self.nv - 1))]) + u * np.array(
            self.right[int(v * (self.nv - 1))]
        )
        Sv = (1 - v) * np.array(self.bottom[int(u * (self.nu - 1))]) + v * np.array(
            self.top[int(u * (self.nu - 1))]
        )
        Sbilinear = (
            (1 - u) * (1 - v) * np.array(self.left[0])
            + u * (1 - v) * np.array(self.right[0])
            + (1 - u) * v * np.array(self.left[-1])
            + u * v * np.array(self.right[-1])
        )
        return Su + Sv - Sbilinear

    def generate_grid(self):
        U, V = np.linspace(0, 1, self.nu), np.linspace(0, 1, self.nv)
        grid = np.zeros((self.nv, self.nu, 2))
        for j, v in enumerate(V):
            for i, u in enumerate(U):
                Su = (1 - u) * np.array(self.left[j]) + u * np.array(self.right[j])
                Sv = (1 - v) * np.array(self.bottom[i]) + v * np.array(self.top[i])
                Sbilinear = (
                    (1 - u) * (1 - v) * np.array(self.left[0])
                    + u * (1 - v) * np.array(self.right[0])
                    + (1 - u) * v * np.array(self.left[-1])
                    + u * v * np.array(self.right[-1])
                )
                grid[j, i] = Su + Sv - Sbilinear
        return grid


class TFI3D:
    """Documentation string."""

    def __init__(
        self,
        leftPoints: np.ndarray,
        rightPoints: np.ndarray,
        frontPoints: np.ndarray,
        backPoints: np.ndarray,
        bottomPoints: np.ndarray,
        topPoints: np.ndarray,
        *,
        compatibility_tolerance: float = 1e-8,
    ):
        self.left = self._as_face("leftPoints", leftPoints)
        self.right = self._as_face("rightPoints", rightPoints)
        self.front = self._as_face("frontPoints", frontPoints)
        self.back = self._as_face("backPoints", backPoints)
        self.bottom = self._as_face("bottomPoints", bottomPoints)
        self.top = self._as_face("topPoints", topPoints)

        self.nw, self.nv = self.left.shape[:2]
        right_shape = (self.nw, self.nv, 3)
        if self.right.shape != right_shape:
            raise ValueError(
                """Documentation string."""
            )

        self.nu = self.front.shape[1]
        expected_shapes = {
            "frontPoints": (self.nw, self.nu, 3),
            "backPoints": (self.nw, self.nu, 3),
            "bottomPoints": (self.nv, self.nu, 3),
            "topPoints": (self.nv, self.nu, 3),
        }
        actual_faces = {
            "frontPoints": self.front,
            "backPoints": self.back,
            "bottomPoints": self.bottom,
            "topPoints": self.top,
        }
        for name, expected_shape in expected_shapes.items():
            if actual_faces[name].shape != expected_shape:
                raise ValueError(
                    """Documentation string."""
                    """Documentation string."""
                )

        if min(self.nu, self.nv, self.nw) < 2:
            raise ValueError("""Documentation string.""")
        if compatibility_tolerance < 0:
            raise ValueError("""Documentation string.""")
        self.compatibility_tolerance = float(compatibility_tolerance)
        self._validate_shared_edges()

    @staticmethod
    def _as_face(name: str, points: np.ndarray) -> np.ndarray:
        face = np.asarray(points, dtype=float)
        if face.ndim != 3 or face.shape[-1] != 3:
            raise ValueError("""Documentation string.""")
        if not np.isfinite(face).all():
            raise ValueError("""Documentation string.""")
        return face

    def _validate_shared_edges(self) -> None:
        edge_pairs = {
            "left-front": (self.left[:, 0], self.front[:, 0]),
            "right-front": (self.right[:, 0], self.front[:, -1]),
            "left-back": (self.left[:, -1], self.back[:, 0]),
            "right-back": (self.right[:, -1], self.back[:, -1]),
            "left-bottom": (self.left[0], self.bottom[:, 0]),
            "right-bottom": (self.right[0], self.bottom[:, -1]),
            "left-top": (self.left[-1], self.top[:, 0]),
            "right-top": (self.right[-1], self.top[:, -1]),
            "front-bottom": (self.front[0], self.bottom[0]),
            "back-bottom": (self.back[0], self.bottom[-1]),
            "front-top": (self.front[-1], self.top[0]),
            "back-top": (self.back[-1], self.top[-1]),
        }
        for name, (first, second) in edge_pairs.items():
            if not np.allclose(
                first,
                second,
                atol=self.compatibility_tolerance,
                rtol=0.0,
            ):
                maximum_error = float(np.max(np.abs(first - second)))
                raise ValueError(
                    """Documentation string."""
                    """Documentation string."""
                )

    def __call__(self) -> np.ndarray:
        """Documentation string."""
        return self.generate_grid().reshape(-1, 3)

    def generate_grid(self) -> np.ndarray:
        """Documentation string."""
        grid = np.empty((self.nw, self.nv, self.nu, 3), dtype=float)
        u_values = np.linspace(0.0, 1.0, self.nu)
        v_values = np.linspace(0.0, 1.0, self.nv)
        w_values = np.linspace(0.0, 1.0, self.nw)

        corners = np.array(
            [
                [
                    [self.bottom[0, 0], self.bottom[0, -1]],
                    [self.bottom[-1, 0], self.bottom[-1, -1]],
                ],
                [
                    [self.top[0, 0], self.top[0, -1]],
                    [self.top[-1, 0], self.top[-1, -1]],
                ],
            ]
        )

        for k, w in enumerate(w_values):
            for j, v in enumerate(v_values):
                for i, u in enumerate(u_values):
                    face_blend = (
                        (1.0 - u) * self.left[k, j]
                        + u * self.right[k, j]
                        + (1.0 - v) * self.front[k, i]
                        + v * self.back[k, i]
                        + (1.0 - w) * self.bottom[j, i]
                        + w * self.top[j, i]
                    )

                    # Implementation comment.
                    edge_blend = (
                        (1.0 - u) * (1.0 - v) * self.left[k, 0]
                        + u * (1.0 - v) * self.right[k, 0]
                        + (1.0 - u) * v * self.left[k, -1]
                        + u * v * self.right[k, -1]
                        + (1.0 - u) * (1.0 - w) * self.left[0, j]
                        + u * (1.0 - w) * self.right[0, j]
                        + (1.0 - u) * w * self.left[-1, j]
                        + u * w * self.right[-1, j]
                        + (1.0 - v) * (1.0 - w) * self.front[0, i]
                        + v * (1.0 - w) * self.back[0, i]
                        + (1.0 - v) * w * self.front[-1, i]
                        + v * w * self.back[-1, i]
                    )

                    corner_blend = np.zeros(3, dtype=float)
                    for corner_w, weight_w in ((0, 1.0 - w), (1, w)):
                        for corner_v, weight_v in ((0, 1.0 - v), (1, v)):
                            for corner_u, weight_u in ((0, 1.0 - u), (1, u)):
                                corner_blend += (
                                    weight_u
                                    * weight_v
                                    * weight_w
                                    * corners[corner_w, corner_v, corner_u]
                                )

                    grid[k, j, i] = face_blend - edge_blend + corner_blend
        return grid


# ============================================================
# Implementation comment.
# ============================================================

if __name__ == "__main__":
    # Implementation comment.
    Nu = Nv = Nw = 5
    radius = 1.0
    height = 2.0
    u_values = np.linspace(-1.0, 1.0, Nu)
    v_values = np.linspace(-1.0, 1.0, Nv)
    w_values = np.linspace(0.0, 1.0, Nw)

    def cylinder_mapping(u, v, w):
        """Documentation string."""
        x = radius * u * np.sqrt(1.0 - 0.5 * v**2)
        y = radius * v * np.sqrt(1.0 - 0.5 * u**2)
        z = height * w
        return np.array([x, y, z])

    left = np.array(
        [[cylinder_mapping(-1.0, v, w) for v in v_values] for w in w_values]
    )
    right = np.array(
        [[cylinder_mapping(1.0, v, w) for v in v_values] for w in w_values]
    )
    front = np.array(
        [[cylinder_mapping(u, -1.0, w) for u in u_values] for w in w_values]
    )
    back = np.array([[cylinder_mapping(u, 1.0, w) for u in u_values] for w in w_values])
    bottom = np.array(
        [[cylinder_mapping(u, v, 0.0) for u in u_values] for v in v_values]
    )
    top = np.array([[cylinder_mapping(u, v, 1.0) for u in u_values] for v in v_values])

    tfi = TFI3D(left, right, front, back, bottom, top)
    cylinder_grid = tfi.generate_grid()
    cylinder_points = cylinder_grid.reshape(-1, 3)
    print("TFI3D grid shape:", cylinder_grid.shape)
    print("TFI3D point count:", cylinder_points.shape[0])
    assert cylinder_points.shape == (5 * 5 * 5, 3)

    fig = plt.figure(figsize=(8, 7))
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(
        cylinder_points[:, 0],
        cylinder_points[:, 1],
        cylinder_points[:, 2],
        color="tab:blue",
        s=18,
    )
    # Implementation comment.
    for k in range(Nw):
        for j in range(Nv):
            axis.plot(*cylinder_grid[k, j, :, :].T, color="0.35", linewidth=0.7)
        for i in range(Nu):
            axis.plot(*cylinder_grid[k, :, i, :].T, color="0.35", linewidth=0.7)
    for j in range(Nv):
        for i in range(Nu):
            axis.plot(*cylinder_grid[:, j, i, :].T, color="0.35", linewidth=0.7)

    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title("""Documentation string.""")
    axis.set_box_aspect((2.0, 2.0, height))
    axis.grid(False)
    output_path = Path("outputs/results/tfi3d_cylinder_5x5x5.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Visualization saved to: {output_path}")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)

