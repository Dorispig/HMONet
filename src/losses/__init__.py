"""Reusable training losses with stable, model-independent APIs."""

from .coordinate import (
    boundary_coordinate_mse,
    coordinate_mse,
    position_weighted_coordinate_mse,
)
from .mesh_regularization import normal_alignment_loss
from .physics import (
    elliptic_grid_normalized_pde_loss,
    elliptic_grid_pde_residual,
    elliptic_grid_scale_invariant_pde_loss,
    grid_line_smoothness_loss,
    jacobian_positive_loss,
    normalized_mapping_barriers,
    normalized_mapping_barriers_from_jacobian,
    orthogonality_loss,
    winslow_energy_and_jacobian_barrier,
)

__all__ = [
    "boundary_coordinate_mse",
    "coordinate_mse",
    "position_weighted_coordinate_mse",
    "normal_alignment_loss",
    "elliptic_grid_pde_residual",
    "elliptic_grid_normalized_pde_loss",
    "elliptic_grid_scale_invariant_pde_loss",
    "grid_line_smoothness_loss",
    "jacobian_positive_loss",
    "normalized_mapping_barriers",
    "normalized_mapping_barriers_from_jacobian",
    "orthogonality_loss",
    "winslow_energy_and_jacobian_barrier",
]
