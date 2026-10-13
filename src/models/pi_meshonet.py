"""Paper-aligned PI-MeshONet operator networks.

The implementation follows equations (5)--(13) of Xiao et al.:
one coordinate-specific branch per physical coordinate, a shared lifted
trunk, gated Modified MLPs, and branch--trunk inner products.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .modules import MLP

__all__ = ["ModifiedMLP", "PIMeshONet2D", "PIMeshONet3D"]


class ModifiedMLP(nn.Module):
    """Gated MLP defined by PI-MeshONet equations (5)--(9)."""

    def __init__(self, input_dim, hidden_dim, hidden_layer_num, output_dim):
        super().__init__()
        if min(int(input_dim), int(hidden_dim), int(output_dim)) < 1:
            raise ValueError("ModifiedMLP dimensions must be positive")
        if int(hidden_layer_num) < 1:
            raise ValueError("ModifiedMLP requires at least one gated layer")
        self.feature_u = nn.Linear(input_dim, hidden_dim)
        self.feature_v = nn.Linear(input_dim, hidden_dim)
        self.initial_hidden = nn.Linear(input_dim, hidden_dim)
        self.gates = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(int(hidden_layer_num))
        )
        self.output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, inputs):
        u = self.activation(self.feature_u(inputs))
        v = self.activation(self.feature_v(inputs))
        hidden = self.activation(self.initial_hidden(inputs))
        for layer in self.gates:
            gate = self.activation(layer(hidden))
            hidden = (1.0 - gate) * u + gate * v
        return self.output(hidden)


class _PIMeshONet(nn.Module):
    """Dimension-independent implementation of the paper architecture."""

    architecture_name = "paper_modified_mlp"

    def __init__(
        self,
        shape,
        *,
        branch_dim=64,
        trunk_dim=16,
        adjust_hidden_dim=128,
        branch_hidden_dim=128,
        trunk_hidden_dim=128,
        adjust_hidden_layer_num=4,
        branch_hidden_layer_num=4,
        trunk_hidden_layer_num=4,
        output_dim=128,
        **_,
    ):
        super().__init__()
        self.declared_grid_shape = tuple(map(int, shape))
        self.dimension = len(self.declared_grid_shape)
        if self.dimension not in (2, 3):
            raise ValueError("PI-MeshONet supports only 2-D and 3-D mappings")
        if min(self.declared_grid_shape) < 2:
            raise ValueError("Every grid axis must contain at least two points")
        minimum_lift = 3 * self.dimension
        if int(trunk_dim) != self.dimension and int(trunk_dim) < minimum_lift:
            raise ValueError(
                f"trunk_dim must equal {self.dimension} for raw coordinates or be "
                f"at least {minimum_lift} for the paper Lift Layer"
            )

        self.trunk_dim = int(trunk_dim)
        self.boundary_point_count = self._boundary_point_count()
        adjust_layers = [int(adjust_hidden_dim)] * int(adjust_hidden_layer_num)
        if not adjust_layers:
            raise ValueError("Adjust Layer requires at least one hidden layer")
        self.adjust_layers = nn.ModuleList(
            MLP(self.boundary_point_count, adjust_layers, int(branch_dim))
            for _ in range(self.dimension)
        )
        self.branch_layers = nn.ModuleList(
            ModifiedMLP(
                int(branch_dim),
                int(branch_hidden_dim),
                int(branch_hidden_layer_num),
                int(output_dim),
            )
            for _ in range(self.dimension)
        )
        self.trunk = ModifiedMLP(
            self.trunk_dim,
            int(trunk_hidden_dim),
            int(trunk_hidden_layer_num),
            int(output_dim),
        )
        self.bias = nn.Parameter(torch.zeros(self.dimension))

    def _boundary_point_count(self):
        if self.dimension == 2:
            nx, ny = self.declared_grid_shape
            return 2 * (nx + ny) - 4
        nx, ny, nz = self.declared_grid_shape
        return (
            2 * (nx * ny + nx * nz + ny * nz)
            - 4 * (nx + ny + nz)
            + 8
        )

    def lift(self, coordinates):
        """Apply the trigonometric/polynomial Lift Layer in equation (10)."""
        if self.trunk_dim == self.dimension:
            return coordinates
        features = []
        for axis in coordinates.unbind(dim=-1):
            features.extend((torch.sin(axis), torch.cos(axis)))
        features.extend(coordinates.unbind(dim=-1))
        power = 2
        while len(features) < self.trunk_dim:
            features.extend(coordinates.pow(power).unbind(dim=-1))
            power += 1
        return torch.stack(features[: self.trunk_dim], dim=-1)

    @staticmethod
    def _flatten_boundary(coordinate):
        if coordinate.dim() == 3:
            if coordinate.shape[1] != 1:
                raise ValueError("Boundary coordinate must have shape [B, 1, Nb]")
            coordinate = coordinate[:, 0, :]
        if coordinate.dim() not in (1, 2):
            raise ValueError("Boundary coordinate must have shape [Nb] or [B, Nb]")
        return coordinate

    def _forward(self, boundary_coordinates, logical_coordinates):
        if len(boundary_coordinates) != self.dimension:
            raise ValueError("One boundary coordinate field is required per axis")
        logical = torch.stack(logical_coordinates, dim=-1)
        trunk_features = self.trunk(self.lift(logical))
        outputs = []
        for axis, values in enumerate(boundary_coordinates):
            values = self._flatten_boundary(values)
            if values.shape[-1] != self.boundary_point_count:
                raise ValueError(
                    "Boundary point count does not match the declared training grid"
                )
            branch = self.branch_layers[axis](self.adjust_layers[axis](values))
            while branch.dim() < trunk_features.dim():
                branch = branch.unsqueeze(-2)
            outputs.append(
                torch.sum(branch * trunk_features, dim=-1, keepdim=True)
                + self.bias[axis]
            )
        return tuple(outputs)


class PIMeshONet2D(_PIMeshONet):
    """Two-dimensional PI-MeshONet from the paper architecture."""

    def __init__(self, Nx, Ny, **kwargs):
        # This balanced configuration contains 24,450 trainable parameters for
        # the default 21 x 20 training grid, matching the paper's reported
        # 24,452 parameters to within two parameters. Explicit caller options
        # continue to take precedence over these two-dimensional defaults.
        defaults = {
            "branch_dim": 20,
            "trunk_dim": 16,
            "adjust_hidden_dim": 20,
            "adjust_hidden_layer_num": 1,
            "branch_hidden_dim": 32,
            "branch_hidden_layer_num": 4,
            "trunk_hidden_dim": 32,
            "trunk_hidden_layer_num": 3,
            "output_dim": 32,
        }
        for name, value in defaults.items():
            kwargs.setdefault(name, value)
        super().__init__((Nx, Ny), **kwargs)
        self.Nx, self.Ny = self.declared_grid_shape

    def forward(self, p_x, p_y, xi, eta):
        return self._forward((p_x, p_y), (xi, eta))


class PIMeshONet3D(_PIMeshONet):
    """Three-dimensional extension with three branches and one shared trunk."""

    def __init__(self, Nx, Ny, Nz, **kwargs):
        kwargs.setdefault("trunk_dim", 24)
        super().__init__((Nx, Ny, Nz), **kwargs)
        self.Nx, self.Ny, self.Nz = self.declared_grid_shape

    def forward(self, p_x, p_y, p_z, xi, eta, zeta):
        return self._forward((p_x, p_y, p_z), (xi, eta, zeta))
