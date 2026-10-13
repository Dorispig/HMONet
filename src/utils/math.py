"""Documentation string."""

# ============================================================
# Implementation comment.
# ============================================================

# ============================================================
# Implementation comment.
# ============================================================
import numpy as np
import torch

# ============================================================
# Implementation comment.
# ============================================================

# ============================================================
# Implementation comment.
# ============================================================


__all__ = [
    "coord_to_idx",
    "compute_discrete_boundary_curvature",
]


# ============================================================
# Implementation comment.
# ============================================================
def coord_to_idx(nx: int, ny: int, Nx: int, Ny: int, n: int) -> int:
    """
    Convert 2D grid coordinates to a 1D index.
    For a grid with dimensions (Ny, Nx), the 2D coordinates (nx, ny) can be mapped to a 1D index using the formula:
    index = ny * Nx + nx + n * Nx * Ny
    8  -  9  -  10 -  11
    |     |     |     |
    4  -  5  -  6  -  7
    |     |     |     |
    0  -  1  -  2  -  3

    Parameters:
    nx (int): Grid coordinate in the x-direction.
    ny (int): Grid coordinate in the y-direction.
    Ny (int): Total number of grid points in the y-direction.

    Returns:
    int: The corresponding 1D index for the given 2D coordinates.
    """
    # return nx * Ny + ny + n * Nx * Ny
    return ny * Nx + nx + n * Nx * Ny


def compute_discrete_boundary_curvature(boundary, quantile=0.95, eps=1e-8):
    """Compute normalized discrete curvature for physical boundaries.

    ``boundary`` has shape ``[B, P, 2]`` or ``[P, 2]``. At vertex ``i`` the
    raw curvature is the turning angle divided by mean adjacent edge length::

        kappa_i = 2 * atan2(|cross(e_prev, e_next)|,
                            dot(e_prev, e_next))
                  / (|e_prev| + |e_next|)

    Absolute curvature is normalized independently for every sample by the
    requested quantile and clipped to ``[0, 1]``. The implementation uses
    vectorized PyTorch operations and can execute on either CPU or GPU. The
    input point ordering is treated as one closed structured boundary loop.

    Args:
        boundary: Physical boundary tensor or array.
        quantile: Per-sample normalization quantile in ``(0, 1]``.
        eps: Numerical lower bound for lengths and normalization scales.

    Returns:
        A tensor of normalized curvature with shape ``[B, P]`` or ``[P]``.
    """
    points = torch.as_tensor(boundary)
    squeeze = points.ndim == 2
    if squeeze:
        points = points.unsqueeze(0)
    if points.ndim != 3 or points.shape[-1] != 2 or points.shape[1] < 3:
        raise ValueError("boundary must have shape [B, P, 2] with P >= 3")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    previous = points - torch.roll(points, 1, dims=1)
    following = torch.roll(points, -1, dims=1) - points
    previous_length = torch.linalg.vector_norm(previous, dim=-1).clamp_min(eps)
    following_length = torch.linalg.vector_norm(following, dim=-1).clamp_min(eps)
    cross = previous[..., 0] * following[..., 1] - previous[..., 1] * following[..., 0]
    dot = (previous * following).sum(dim=-1)
    angle = torch.atan2(cross.abs(), dot)
    curvature = 2.0 * angle / (previous_length + following_length)
    scale = torch.quantile(curvature, quantile, dim=1, keepdim=True).clamp_min(eps)
    normalized = (curvature / scale).clamp(0.0, 1.0)
    return normalized.squeeze(0) if squeeze else normalized


if __name__ == "__main__":
    print(
        "This is a helper module. Please import and use the functions in this module as needed."
    )

