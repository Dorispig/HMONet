"""Traditional algebraic and elliptic structured-grid generation methods."""

from .tfi_2d import tfi_from_curves, tfi_from_parameter_grid
from .tfi_3d import tfi_from_surfaces
from .winslow_2d import solve_winslow_2d
from .winslow_3d import solve_winslow_3d
from .winslow_vectorized_2d import solve_winslow_2d_vectorized

__all__ = [
    "tfi_from_curves",
    "tfi_from_parameter_grid",
    "tfi_from_surfaces",
    "solve_winslow_2d",
    "solve_winslow_3d",
    "solve_winslow_2d_vectorized",
]
