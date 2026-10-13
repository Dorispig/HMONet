#!/usr/bin/env MGNet
# -*- coding: utf-8 -*-
"Neural operator models for structured-grid geometry mapping."

from __future__ import annotations

__all__ = [
    "MeshONet2D",
    "PIMeshONet2D",
    "HMONet2D",
    "AttentionHMONet2D",
    "AttentionHMONet3D",
    "MeshONet3D",
    "PIMeshONet3D",
    "HMONet3D",
]

# ============================================================
# Module imports and model definitions.
# ============================================================

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ============================================================
# Third-party tensor and numerical library imports.
# ============================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pi_meshonet import PIMeshONet2D, PIMeshONet3D


from src.data.backend.parametric_torch import (
    evaluate_boundary_curves_torch,
    evaluate_boundary_surfaces_torch,
)

# ============================================================
# Local neural-network module imports.
# ============================================================
# from .utils import helper_function
import src.models.modules as modules
from src.models.inference_optimizations import blend_current_2d, fuse_coordinates
from src.data.backend.inference_options import ROUTES

# ============================================================
# Neural-network dependencies and local model modules.
# ============================================================

# Module-level compatibility defaults.
DEFAULT_THRESHOLD: float = 0.5
MAX_RETRIES: int = 3


def _bounded_projection_coefficients(logits, minimum):
    """Map unconstrained logits to ``[minimum, 1]``."""
    return minimum + (1.0 - minimum) * torch.sigmoid(logits)


def _convex_projection_weights(boundary_weights):
    """Build a smooth convex partition while retaining an interior network term.

    Each rational boundary influence is in ``[0, 1]``. Their complementary
    product is the raw-network weight, and the remaining mass is distributed
    in proportion to the boundary influences. Thus the center weight stays
    positive at every strict interior point instead of being clipped to zero
    whenever the raw boundary weights happen to sum above one.
    """
    influence = boundary_weights.clamp(0.0, 1.0)
    center = torch.prod(1.0 - influence, dim=-1, keepdim=True)
    total = influence.sum(dim=-1, keepdim=True)
    boundary = torch.where(
        total > 0.0,
        (1.0 - center) * influence / total.clamp_min(torch.finfo(total.dtype).tiny),
        torch.zeros_like(influence),
    )
    return center, boundary


def _smooth_normal_coordinate(parameter, smoothing):
    """Blend the identity coordinate with smoothstep without flattening it fully."""
    return (1.0 - smoothing) * parameter + smoothing * parameter.square() * (
        3.0 - 2.0 * parameter
    )


def _lifted_tangential_bubble(parameter, beta):
    """Lift the interior of ``4t(1-t)`` while preserving zero endpoints."""
    bubble = 4.0 * parameter * (1.0 - parameter)
    return bubble / (beta + (1.0 - beta) * bubble)


def _rational_projection_weights_2d(
    xi,
    eta,
    coefficients,
    transition_width=0.2,
    normal_smoothing=0.5,
    tangential_lift_beta=0.6,
):
    """Return width-controlled rational weights for four 2-D boundaries."""
    u = _smooth_normal_coordinate(xi, normal_smoothing)
    v = _smooth_normal_coordinate(eta, normal_smoothing)
    qu = _lifted_tangential_bubble(xi, tangential_lift_beta)
    qv = _lifted_tangential_bubble(eta, tangential_lift_beta)
    tangential = torch.stack((qv, qv, qu, qu), dim=-1)
    normal = torch.stack((u, u, v, v), dim=-1)
    face_factor = torch.stack((1.0 - u, u, 1.0 - v, v), dim=-1)
    influence = coefficients * transition_width * tangential
    denominator = normal * (1.0 - normal) + influence
    denominator = denominator.masked_fill(denominator == 0, 1.0)
    raw = influence * face_factor / denominator
    return _convex_projection_weights(raw)


def _rational_projection_weights_3d(
    xi,
    eta,
    zeta,
    coefficients,
    transition_width=0.2,
    normal_smoothing=0.5,
    tangential_lift_beta=0.6,
):
    """Extend the 2-D rational hard-map weights to six 3-D faces.

    A face uses the bubble in its normal coordinate in the denominator and the
    product of the other two coordinate bubbles as its tangential influence.
    The six resulting nonnegative weights are finally converted to a convex
    partition together with the raw-network center weight.
    """
    u = _smooth_normal_coordinate(xi, normal_smoothing)
    v = _smooth_normal_coordinate(eta, normal_smoothing)
    w = _smooth_normal_coordinate(zeta, normal_smoothing)
    qu = _lifted_tangential_bubble(xi, tangential_lift_beta)
    qv = _lifted_tangential_bubble(eta, tangential_lift_beta)
    qw = _lifted_tangential_bubble(zeta, tangential_lift_beta)
    tangential = torch.stack(
        (qv * qw, qv * qw, qu * qw, qu * qw, qu * qv, qu * qv), dim=-1
    )
    normal = torch.stack((u, u, v, v, w, w), dim=-1)
    face_factor = torch.stack((1.0 - u, u, 1.0 - v, v, 1.0 - w, w), dim=-1)
    influence = coefficients * transition_width * tangential
    denominator = normal * (1.0 - normal) + influence
    denominator = denominator.masked_fill(denominator == 0, 1.0)
    raw = influence * face_factor / denominator
    return _convex_projection_weights(raw)


# ============================================================
# Core model classes.
# ============================================================


class MeshONet2D(nn.Module):
    "MeshONet operator network for mapping a 2D computational grid to physical coordinates."

    def __init__(
        self,
        Nx,
        Ny,
        branch_dim=24,
        trunk_dim=24,
        adjust_hidden_dim=8,
        branch_hidden_dim=24,
        trunk_hidden_dim=32,
        adjust_hidden_layer_num=3,
        branch_hidden_layer_num=4,
        trunk_hidden_layer_num=4,
        output_dim=32,
        branch_mlp_hidden_dim=24,
        branch_mlp_hidden_layer_num=3,
    ):
        super().__init__()
        if branch_mlp_hidden_dim <= 0:
            raise ValueError("branch_mlp_hidden_dim must be positive")
        if branch_mlp_hidden_layer_num <= 0:
            raise ValueError("branch_mlp_hidden_layer_num must be positive")

        # Store grid dimensions and derive the flattened boundary feature size.
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        if self.Nx < 2 or self.Ny < 2:
            raise ValueError("Nx and Ny must both be at least 2")
        self.adjust_dim = 2 * (self.Nx + self.Ny) - 4
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.enhance_dim = trunk_dim
        self.adjust_hidden_dim = adjust_hidden_dim
        self.branch_hidden_dim = branch_hidden_dim
        self.trunk_hidden_dim = trunk_hidden_dim
        self.adjust_hidden_layer_num = adjust_hidden_layer_num
        self.branch_hidden_layer_num = branch_hidden_layer_num
        self.trunk_hidden_layer_num = trunk_hidden_layer_num
        self.output_dim = output_dim
        self.branch_mlp_hidden_dim = branch_mlp_hidden_dim
        self.branch_mlp_hidden_layer_num = branch_mlp_hidden_layer_num
        # Learnable scalar biases for the two output coordinates.
        self.b0_x = nn.Parameter(torch.zeros(1))
        self.b0_y = nn.Parameter(torch.zeros(1))

        self.adjust_net_x = (
            nn.Sequential(
                modules.MLP(
                    input_dim=self.adjust_dim,
                    hidden_layer_size=[adjust_hidden_dim] * adjust_hidden_layer_num,
                    output_dim=branch_dim,
                ),
                nn.Tanh(),
            )
            if adjust_hidden_dim > 0 and adjust_hidden_layer_num > 0
            else nn.Identity()
        )
        self.adjust_net_y = (
            nn.Sequential(
                modules.MLP(
                    input_dim=self.adjust_dim,
                    hidden_layer_size=[adjust_hidden_dim] * adjust_hidden_layer_num,
                    output_dim=branch_dim,
                ),
                nn.Tanh(),
            )
            if adjust_hidden_dim > 0 and adjust_hidden_layer_num > 0
            else nn.Identity()
        )
        # Axis-specific branch networks map boundary features to latent vectors.
        self.branch_net_x = modules.MLP(
            input_dim=branch_dim,
            hidden_layer_size=[branch_hidden_dim] * branch_hidden_layer_num,
            output_dim=output_dim,
        )
        self.branch_net_y = modules.MLP(
            input_dim=branch_dim,
            hidden_layer_size=[branch_hidden_dim] * branch_hidden_layer_num,
            output_dim=output_dim,
        )
        branch_mlp_layers = [branch_mlp_hidden_dim] * branch_mlp_hidden_layer_num
        self.branch_mlp_x = modules.MLP(
            input_dim=2 * output_dim,
            hidden_layer_size=branch_mlp_layers,
            output_dim=output_dim,
        )
        self.branch_mlp_y = modules.MLP(
            input_dim=2 * output_dim,
            hidden_layer_size=branch_mlp_layers,
            output_dim=output_dim,
        )
        # The trunk network maps computational coordinates to latent vectors.
        self.trunk_net = modules.MLP(
            input_dim=trunk_dim,
            hidden_layer_size=[trunk_hidden_dim] * trunk_hidden_layer_num,
            output_dim=output_dim,
        )

    def data_enhance(self, xi_eta):
        # Add sinusoidal, coordinate, and higher-order polynomial features.
        enhance_data = torch.cat(
            [torch.sin(xi_eta), torch.cos(xi_eta), xi_eta], dim=-1
        )  # [B, N, 6]
        if self.enhance_dim > 6:
            for i in range((self.enhance_dim - 6) // 2):
                enhance_data = torch.cat(
                    [enhance_data, torch.pow(xi_eta, i + 2)], dim=-1
                )
        return enhance_data  # [B, N, enhance_dim]

    def forward(self, p_x, p_y, xi, eta):
        "Compute physical coordinates from boundary data and computational coordinates."
        if p_x.dim() == 3:
            p_x = p_x[:, 0, :]
            p_y = p_y[:, 0, :]
        xi_eta = torch.stack((xi, eta), dim=-1)  # [B, N, 2]
        h_x = self.branch_net_x(self.adjust_net_x(p_x))
        h_y = self.branch_net_y(self.adjust_net_y(p_y))
        h = torch.cat((h_x, h_y), dim=-1)
        b_i_x = self.branch_mlp_x(h).unsqueeze(-2)
        b_i_y = self.branch_mlp_y(h).unsqueeze(-2)
        if self.trunk_dim == 2:
            trunk_out = self.trunk_net(xi_eta)  # [B, N, output_dim]
        else:
            trunk_out = self.trunk_net(self.data_enhance(xi_eta))  # [B, N, output_dim]

        # Learnable scalar biases for the two output coordinates.
        x = torch.sum(b_i_x * trunk_out, dim=-1, keepdim=True) + self.b0_x  # [B, N, 1]
        y = torch.sum(b_i_y * trunk_out, dim=-1, keepdim=True) + self.b0_y  # [B, N, 1]
        # x = torch.einsum('bi,bj->bij', branch_out_x.squeeze(-1), trunk_out.squeeze(-1))  # [B, N]
        # y = torch.einsum('bi,bj->bij', branch_out_y.squeeze(-1), trunk_out.squeeze(-1))  # [B, N]
        # result = torch.cat((x, y), dim=-1)  # [B, N, 2]
        return x, y


class HMONet2D(nn.Module):
    "HMONet 2D model with hard boundary projection and optional Coons residual mapping."

    def __init__(
        self,
        Nx,
        Ny,
        branch_dim=64,
        trunk_dim=18,
        adjust_hidden_dim=128,
        branch_hidden_dim=128,
        trunk_hidden_dim=128,
        adjust_hidden_layer_num=4,
        branch_hidden_layer_num=4,
        trunk_hidden_layer_num=4,
        output_dim=64,
        branch_mlp_hidden_dim=128,
        branch_mlp_hidden_layer_num=2,
        hmonet_mapping_mode="current",
        trunk_projection_coefficients_enabled=False,
        projection_softplus_enabled=False,
        boundary_self_attention_enabled=False,
        projection_mlp_hidden_dim=64,
        attention_dim=64,
        attention_heads=4,
        attention_layers=2,
        attention_feedforward_dim=128,
        attention_dropout=0.0,
        # Deprecated compatibility options. The restored original hard map ignores them.
        projection_coefficient_min=0.1,
        projection_transition_width=0.2,
        projection_normal_smoothing=0.5,
        projection_tangential_lift_beta=0.6,
    ):
        super().__init__()
        if branch_mlp_hidden_dim <= 0:
            raise ValueError("branch_mlp_hidden_dim must be positive")
        if branch_mlp_hidden_layer_num <= 0:
            raise ValueError("branch_mlp_hidden_layer_num must be positive")
        hmonet_mapping_mode = str(hmonet_mapping_mode).strip().lower().replace("-", "_")
        if hmonet_mapping_mode not in {"current", "coons_residual"}:
            raise ValueError("hmonet_mapping_mode must be current or coons_residual")
        self.trunk_projection_coefficients_enabled = bool(
            trunk_projection_coefficients_enabled
        )
        self.projection_softplus_enabled = bool(projection_softplus_enabled)
        self.boundary_self_attention_enabled = bool(boundary_self_attention_enabled)
        if int(projection_mlp_hidden_dim) <= 0:
            raise ValueError("projection_mlp_hidden_dim must be positive")
        if int(attention_dim) % int(attention_heads) != 0:
            raise ValueError("attention_dim must be divisible by attention_heads")


        # Store grid dimensions and configure the HMONet mapping.
        self.Nx = Nx
        self.Ny = Ny
        self.adjust_dim = 2 * (Nx + Ny) - 4  # Flattened boundary feature dimension.
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.enhance_dim = trunk_dim
        self.adjust_hidden_dim = adjust_hidden_dim
        self.branch_hidden_dim = branch_hidden_dim
        self.trunk_hidden_dim = trunk_hidden_dim
        self.adjust_hidden_layer_num = adjust_hidden_layer_num
        self.branch_hidden_layer_num = branch_hidden_layer_num
        self.trunk_hidden_layer_num = trunk_hidden_layer_num
        self.output_dim = output_dim
        self.branch_mlp_hidden_dim = branch_mlp_hidden_dim
        self.branch_mlp_hidden_layer_num = branch_mlp_hidden_layer_num
        self.hmonet_mapping_mode = hmonet_mapping_mode
        # Keep legacy attributes only for loading saved configurations.
        self.projection_coefficient_min = float(projection_coefficient_min)
        self.projection_transition_width = float(projection_transition_width)
        self.projection_normal_smoothing = float(projection_normal_smoothing)
        self.projection_tangential_lift_beta = float(projection_tangential_lift_beta)
        self.adjust_net_x = (
            nn.Sequential(
                modules.MLP(
                    input_dim=self.adjust_dim,
                    hidden_layer_size=[adjust_hidden_dim] * adjust_hidden_layer_num,
                    output_dim=branch_dim,
                ),
                nn.Tanh(),
            )
            if adjust_hidden_dim > 0 and adjust_hidden_layer_num > 0
            else nn.Identity()
        )
        self.adjust_net_y = (
            nn.Sequential(
                modules.MLP(
                    input_dim=self.adjust_dim,
                    hidden_layer_size=[adjust_hidden_dim] * adjust_hidden_layer_num,
                    output_dim=branch_dim,
                ),
                nn.Tanh(),
            )
            if adjust_hidden_dim > 0 and adjust_hidden_layer_num > 0
            else nn.Identity()
        )
        # Axis-specific branch networks map boundary features to latent vectors.
        self.branch_net_x = modules.MLP(
            input_dim=branch_dim,
            hidden_layer_size=[branch_hidden_dim] * branch_hidden_layer_num,
            output_dim=output_dim,
        )
        self.branch_net_y = modules.MLP(
            input_dim=branch_dim,
            hidden_layer_size=[branch_hidden_dim] * branch_hidden_layer_num,
            output_dim=output_dim,
        )
        branch_mlp_layers = [branch_mlp_hidden_dim] * branch_mlp_hidden_layer_num
        self.branch_mlp_x = modules.MLP(
            input_dim=2 * output_dim,
            hidden_layer_size=branch_mlp_layers,
            output_dim=output_dim,
        )
        self.branch_mlp_y = modules.MLP(
            input_dim=2 * output_dim,
            hidden_layer_size=branch_mlp_layers,
            output_dim=output_dim,
        )
        # Reuse the joint boundary feature instead of evaluating four extra
        # deep MLPs on the raw boundary coordinates.
        self.bias_head = nn.Linear(2 * output_dim, 2)
        self.weight_head = nn.Linear(2 * output_dim, 4)
        self.trunk_projection_head = (
            modules.MLP(
                input_dim=output_dim,
                hidden_layer_size=[int(projection_mlp_hidden_dim)],
                output_dim=4,
            )
            if self.trunk_projection_coefficients_enabled
            else None
        )
        self.boundary_attention = (
            modules.BoundarySelfAttentionEncoder2D(
                model_dim=int(attention_dim),
                num_heads=int(attention_heads),
                num_layers=int(attention_layers),
                feedforward_dim=int(attention_feedforward_dim),
                dropout=float(attention_dropout),
                output_dim=branch_dim,
            )
            if self.boundary_self_attention_enabled
            else None
        )
        # The trunk network maps computational coordinates to latent vectors.
        self.trunk_net = modules.MLP(
            input_dim=trunk_dim,
            hidden_layer_size=[trunk_hidden_dim] * trunk_hidden_layer_num,
            output_dim=output_dim,
        )

    def data_enhance(self, xi_eta):
        enhance_data = torch.cat(
            [torch.sin(xi_eta), torch.cos(xi_eta), xi_eta], dim=-1
        )  # [B, N, 6]
        if self.enhance_dim > 6:
            for i in range((self.enhance_dim - 6) // 2):
                enhance_data = torch.cat(
                    [enhance_data, torch.pow(xi_eta, i + 2)], dim=-1
                )
        return enhance_data  # [B, N, enhance_dim]

    def _directed_boundary_edges(self, p_x, p_y):
        """Reconstruct four parameter-directed edges from the perimeter vector."""
        boundary = torch.stack((p_x, p_y), dim=-1)
        bottom = boundary[:, : self.Nx]
        right = boundary[:, self.Nx - 1 : self.Nx + self.Ny - 1]
        top_start = self.Nx + self.Ny - 2
        top = boundary[:, top_start : top_start + self.Nx].flip(1)
        left_start = 2 * self.Nx + self.Ny - 3
        left = torch.cat((boundary[:, left_start:], boundary[:, :1]), dim=1).flip(1)
        return bottom, right, top, left

    def encode_boundary(self, p_x, p_y):
        """Encode the boundary with either MLPs or joint boundary self-attention."""
        if self.boundary_attention is None:
            descriptor_x = self.adjust_net_x(p_x)
            descriptor_y = self.adjust_net_y(p_y)
        else:
            descriptor_x, descriptor_y = self.boundary_attention(
                *self._directed_boundary_edges(p_x, p_y)
            )
        h_x = self.branch_net_x(descriptor_x)
        h_y = self.branch_net_y(descriptor_y)
        return h_x, h_y, torch.cat((h_x, h_y), dim=-1)

    def _activate_projection_coefficients(self, logits):
        """Apply the selected positive geometry-coefficient activation."""
        return F.softplus(logits) if self.projection_softplus_enabled else torch.sigmoid(logits)

    def projection_weights(self, joint_geometry, trunk_features):
        """Return branch coefficients, optionally modulated by trunk coefficients."""
        branch = self._activate_projection_coefficients(self.weight_head(joint_geometry))
        if self.trunk_projection_head is None:
            return branch.unsqueeze(-2)
        location = self._activate_projection_coefficients(
            self.trunk_projection_head(trunk_features)
        )
        return branch.unsqueeze(-2) * location

    def project_linear(self, p_x, p_y, xi, eta, boundary_geometry=None):
        """Project onto exact curves or linearly interpolated sampled edges."""
        unbatched = p_x.dim() == 1
        if unbatched:
            p_x, p_y, xi, eta = p_x[None], p_y[None], xi[None], eta[None]
        if p_x.dim() == 3:
            p_x, p_y = p_x[:, 0, :], p_y[:, 0, :]
        if boundary_geometry is not None:
            result = (
                boundary_geometry.project_torch(xi, eta)
                if hasattr(boundary_geometry, "project_torch")
                else evaluate_boundary_curves_torch(boundary_geometry, xi, eta)
            )
            return result.squeeze(0) if unbatched else result
        boundary = torch.stack((p_x, p_y), dim=-1)
        size = boundary.shape[-2]
        x, y = xi.clamp(0, 1) * (self.Nx - 1), eta.clamp(0, 1) * (self.Ny - 1)
        ix, iy = x.floor().long().clamp(0, self.Nx - 2), y.floor().long().clamp(
            0, self.Ny - 2
        )
        ax, ay = x - ix, y - iy
        left_i, right_i = (size - iy) % size, self.Nx - 1 + iy
        top_i = 2 * self.Nx + self.Ny - 4 - ix

        def interp(a, b, w):
            return (1 - w).unsqueeze(-1) * a + w.unsqueeze(-1) * b

        def gather(indices):
            return boundary.gather(1, indices[..., None].expand(-1, -1, 2))

        result = torch.stack(
            (
                interp(gather(left_i), gather((left_i - 1) % size), ay),
                interp(gather(right_i), gather(right_i + 1), ay),
                interp(gather(ix), gather(ix + 1), ax),
                interp(gather(top_i + 1), gather(top_i), ax),
            ),
            dim=-2,
        )
        return result.squeeze(0) if unbatched else result

    def project(self, p_x, p_y, xi, eta, boundary_geometry=None):
        """Project using parameter curves or sampled boundary points."""
        return self.project_linear(
            p_x, p_y, xi, eta, boundary_geometry=boundary_geometry
        )

    def evaluate_coons_points(self, p_x, p_y, xi, eta, boundary_geometry=None):
        """Evaluate the Coons/TFI map only at the supplied query points.

        No full grid is constructed.  ``xi`` and ``eta`` may contain arbitrary
        fixed, random, or batched query coordinates with shape ``[B, N]``.
        """
        if p_x.dim() == 3:
            p_x = p_x[:, 0, :]
            p_y = p_y[:, 0, :]
        projected = self.project(p_x, p_y, xi, eta, boundary_geometry=boundary_geometry)
        left, right, bottom, top = projected.unbind(dim=-2)

        boundary = torch.stack((p_x, p_y), dim=-1)
        p00 = boundary[..., 0, :].unsqueeze(-2)
        p10 = boundary[..., self.Nx - 1, :].unsqueeze(-2)
        p11 = boundary[..., self.Nx + self.Ny - 2, :].unsqueeze(-2)
        p01 = boundary[..., 2 * self.Nx + self.Ny - 3, :].unsqueeze(-2)

        xi_vector = xi.unsqueeze(-1)
        eta_vector = eta.unsqueeze(-1)
        side_blend = (
            (1.0 - xi_vector) * left
            + xi_vector * right
            + (1.0 - eta_vector) * bottom
            + eta_vector * top
        )
        corner_blend = (
            (1.0 - xi_vector) * (1.0 - eta_vector) * p00
            + xi_vector * (1.0 - eta_vector) * p10
            + (1.0 - xi_vector) * eta_vector * p01
            + xi_vector * eta_vector * p11
        )
        return side_blend - corner_blend

    @staticmethod
    def coons_residual_envelope(xi, eta):
        """Boundary-vanishing bubble, normalized to one at the square centre."""
        return 16.0 * xi * (1.0 - xi) * eta * (1.0 - eta)

    def forward(
        self,
        p_x,
        p_y,
        xi,
        eta,
        return_displacement=False,
        boundary_geometry=None,
    ):
        """Map queries with exact parameter curves or linear boundary interpolation."""
        if (
            self.training
            and hasattr(boundary_geometry, "project_torch")
            and not getattr(boundary_geometry, "supports_coordinate_gradients", False)
        ):
            raise ValueError("Prepared geometry adapters are for evaluation only")
        if p_x.dim() == 3:
            p_x = p_x[:, 0, :]
            p_y = p_y[:, 0, :]
        xi_eta = torch.stack((xi, eta), dim=-1)  # [B, N, 2]
        h_x, h_y, h = self.encode_boundary(p_x, p_y)
        b_i_x = self.branch_mlp_x(h).unsqueeze(-2)
        b_i_y = self.branch_mlp_y(h).unsqueeze(-2)
        if self.trunk_dim == 2:
            trunk_out = self.trunk_net(xi_eta)  # [B, N, output_dim]
        else:
            trunk_out = self.trunk_net(self.data_enhance(xi_eta))  # [B, N, output_dim]

        bias = self.bias_head(h).unsqueeze(-2)  # [B, 1, 2]
        x_bias = bias[..., 0:1]
        y_bias = bias[..., 1:2]
        # Fuse branch and trunk features to predict physical coordinates.
        if (
            ROUTES.get()[3]
            and not torch.is_grad_enabled()
            and trunk_out.shape[-2] >= 1024
        ):
            raw_prediction = fuse_coordinates(trunk_out, b_i_x, b_i_y, bias)
        else:
            x = torch.sum(b_i_x * trunk_out, dim=-1, keepdim=True) + x_bias
            y = torch.sum(b_i_y * trunk_out, dim=-1, keepdim=True) + y_bias
            raw_prediction = torch.cat((x, y), dim=-1)
        mapping_mode = getattr(self, "hmonet_mapping_mode", "current")
        if mapping_mode == "coons_residual":
            coons = self.evaluate_coons_points(
                p_x,
                p_y,
                xi,
                eta,
                boundary_geometry=boundary_geometry,
            )
            envelope = self.coons_residual_envelope(xi, eta).unsqueeze(-1)
            displacement = envelope * raw_prediction
            result = coons + displacement
            if return_displacement:
                return result[..., 0:1], result[..., 1:2], displacement
            return result[..., 0:1], result[..., 1:2]
        if mapping_mode != "current":
            raise ValueError("hmonet_mapping_mode must be current or coons_residual")

        weight = self.projection_weights(h, trunk_out)
        projected = self.project(
            p_x, p_y, xi, eta, boundary_geometry=boundary_geometry
        )
        w1, w2, w3, w4 = weight.unbind(dim=-1)
        xi_bubble = xi * (1.0 - xi)
        eta_bubble = eta * (1.0 - eta)
        tiny = torch.finfo(raw_prediction.dtype).tiny

        denominator_1 = xi_bubble + w1 * eta_bubble
        denominator_2 = xi_bubble + w2 * eta_bubble
        denominator_3 = eta_bubble + w3 * xi_bubble
        denominator_4 = eta_bubble + w4 * xi_bubble
        safe = lambda value: torch.where(value.abs() > tiny, value, torch.ones_like(value))
        wp1 = w1 * (1.0 - xi) * eta_bubble / safe(denominator_1)
        wp2 = w2 * xi * eta_bubble / safe(denominator_2)
        wp3 = w3 * xi_bubble * (1.0 - eta) / safe(denominator_3)
        wp4 = w4 * xi_bubble * eta / safe(denominator_4)
        center_weight = 1.0 - wp1 - wp2 - wp3 - wp4
        result = (
            center_weight[..., None] * raw_prediction
            + wp1[..., None] * projected[..., 0, :]
            + wp2[..., None] * projected[..., 1, :]
            + wp3[..., None] * projected[..., 2, :]
            + wp4[..., None] * projected[..., 3, :]
        )
        corner_mask = ((xi == 0) | (xi == 1)) & ((eta == 0) | (eta == 1))
        corner_points = torch.where(
            (xi == 0)[..., None], projected[..., 0, :], projected[..., 1, :]
        )
        result = torch.where(corner_mask[..., None], corner_points, result)
        if return_displacement:
            # The optional regularizer applies only to coons_residual models.
            displacement = torch.zeros_like(result)
            return result[..., 0:1], result[..., 1:2], displacement
        return result[..., 0:1], result[..., 1:2]


class AttentionHMONet2D(HMONet2D):
    """HMONet with joint boundary attention and trainable relaxed projection.

    The boundary encoder builds directed tokens for the bottom, right, top and
    left curves, applies shared multi-head self-attention, and uses separate
    learned pooling queries for the x and y branches.  In ``current`` mapping
    mode, four positive geometry coefficients from the branch are multiplied
    elementwise by four positive coordinate coefficients from the trunk.  The
    resulting local coefficients modulate the same exact parametric-boundary
    projection used by :class:`HMONet2D`. ``boundary_alpha`` replaces the
    partition-of-unity constant during two-dimensional physics training;
    alpha equal to one recovers the original exact-boundary mapping.

    This class only defines the network architecture. It does not add a loss,
    optimizer, scheduler, data loader or training entry point.
    """

    def __init__(
        self,
        Nx=None,
        Ny=None,
        branch_dim=64,
        trunk_dim=18,
        adjust_hidden_dim=128,
        branch_hidden_dim=128,
        trunk_hidden_dim=128,
        adjust_hidden_layer_num=4,
        branch_hidden_layer_num=4,
        trunk_hidden_layer_num=4,
        output_dim=64,
        branch_mlp_hidden_dim=128,
        branch_mlp_hidden_layer_num=2,
        hmonet_mapping_mode="current",
        attention_dim=64,
        attention_heads=4,
        attention_layers=2,
        attention_feedforward_dim=128,
        attention_dropout=0.0,
        boundary_alpha_initial_value=1.01,
    ):
        # The parent supplies the operator and hard-mapping layers. Its fixed
        # boundary adjust layers are immediately removed, so a minimal dummy
        # grid avoids tying this architecture to a boundary sampling length.
        super().__init__(
            Nx=2,
            Ny=2,
            branch_dim=branch_dim,
            trunk_dim=trunk_dim,
            adjust_hidden_dim=adjust_hidden_dim,
            branch_hidden_dim=branch_hidden_dim,
            trunk_hidden_dim=trunk_hidden_dim,
            adjust_hidden_layer_num=adjust_hidden_layer_num,
            branch_hidden_layer_num=branch_hidden_layer_num,
            trunk_hidden_layer_num=trunk_hidden_layer_num,
            output_dim=output_dim,
            branch_mlp_hidden_dim=branch_mlp_hidden_dim,
            branch_mlp_hidden_layer_num=branch_mlp_hidden_layer_num,
            hmonet_mapping_mode=hmonet_mapping_mode,
        )
        self.boundary_attention = modules.BoundarySelfAttentionEncoder2D(
            model_dim=attention_dim,
            num_heads=attention_heads,
            num_layers=attention_layers,
            feedforward_dim=attention_feedforward_dim,
            dropout=attention_dropout,
            output_dim=branch_dim,
        )

        # The attention encoder replaces the flattened-boundary adjust layers.
        del self.adjust_net_x
        del self.adjust_net_y
        del self.weight_head
        self.Nx = None
        self.Ny = None
        self.declared_grid_shape = (
            None if Nx is None else int(Nx),
            None if Ny is None else int(Ny),
        )
        self.branch_projection_head = nn.Linear(2 * output_dim, 4)
        self.trunk_projection_head = nn.Linear(output_dim, 4)
        alpha_initial = float(boundary_alpha_initial_value)
        if not np.isfinite(alpha_initial):
            raise ValueError("boundary_alpha_initial_value must be finite")
        self.boundary_alpha = nn.Parameter(torch.tensor(alpha_initial))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs,
    ):
        """Load pre-alpha checkpoints as the original exact-boundary model."""
        alpha_key = prefix + "boundary_alpha"
        if alpha_key not in state_dict:
            state_dict[alpha_key] = torch.ones_like(self.boundary_alpha)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _edge_masks(edges, boundary_masks):
        return modules.BoundaryTokenBuilder2D._normalize_masks(edges, boundary_masks)

    @staticmethod
    def _interpolate_edge(edge, mask, parameter):
        """Piecewise-linearly evaluate padded edge samples at normalized parameters."""
        lengths = (~mask).sum(dim=1)
        scaled = parameter.clamp(0.0, 1.0) * (lengths[:, None] - 1).to(parameter.dtype)
        lower = scaled.floor().long()
        upper = torch.minimum(lower + 1, lengths[:, None] - 1)
        fraction = (scaled - lower.to(scaled.dtype))[..., None]
        gather_shape = (*lower.shape, 2)
        p0 = edge.gather(1, lower[..., None].expand(gather_shape))
        p1 = edge.gather(1, upper[..., None].expand(gather_shape))
        return p0 + fraction * (p1 - p0)

    def encode_boundary(self, bottom, right, top, left, boundary_masks=None):
        """Encode all boundary edges jointly, then form axis-specific branches."""
        descriptor_x, descriptor_y = self.boundary_attention(
            bottom,
            right,
            top,
            left,
            boundary_masks=boundary_masks,
        )
        h_x = self.branch_net_x(descriptor_x)
        h_y = self.branch_net_y(descriptor_y)
        return h_x, h_y, torch.cat((h_x, h_y), dim=-1)

    def projection_weights(self, joint_geometry, trunk_features):
        """Factorize every projection coefficient into geometry and location terms."""
        geometry = torch.sigmoid(self.branch_projection_head(joint_geometry))
        location = torch.sigmoid(self.trunk_projection_head(trunk_features))
        return self.projection_coefficient_min + (
            1.0 - self.projection_coefficient_min
        ) * geometry.unsqueeze(-2) * location

    def project_edges(
        self,
        bottom,
        right,
        top,
        left,
        xi,
        eta,
        boundary_geometry=None,
        boundary_masks=None,
    ):
        """Evaluate four exact curves or interpolate the four sampled edges."""
        if boundary_geometry is not None:
            return (
                boundary_geometry.project_torch(xi, eta)
                if hasattr(boundary_geometry, "project_torch")
                else evaluate_boundary_curves_torch(boundary_geometry, xi, eta)
            )
        edges = (bottom, right, top, left)
        masks = self._edge_masks(edges, boundary_masks)
        modules.BoundaryTokenBuilder2D._validate(edges, masks)
        return torch.stack(
            (
                self._interpolate_edge(left, masks[3], eta),
                self._interpolate_edge(right, masks[1], eta),
                self._interpolate_edge(bottom, masks[0], xi),
                self._interpolate_edge(top, masks[2], xi),
            ),
            dim=-2,
        )

    def evaluate_coons_points_from_edges(
        self,
        bottom,
        right,
        top,
        left,
        xi,
        eta,
        boundary_geometry=None,
        boundary_masks=None,
    ):
        """Evaluate a Coons map from variable-length directed boundaries."""
        edges = (bottom, right, top, left)
        masks = self._edge_masks(edges, boundary_masks)
        projected = self.project_edges(
            *edges,
            xi,
            eta,
            boundary_geometry=boundary_geometry,
            boundary_masks=masks,
        )
        left_p, right_p, bottom_p, top_p = projected.unbind(dim=-2)
        lengths = tuple((~mask).sum(dim=1) for mask in masks)

        def endpoint(edge, length, last):
            index = length - 1 if last else torch.zeros_like(length)
            return edge.gather(1, index[:, None, None].expand(-1, 1, 2))

        p00 = endpoint(bottom, lengths[0], False)
        p10 = endpoint(bottom, lengths[0], True)
        p01 = endpoint(top, lengths[2], False)
        p11 = endpoint(top, lengths[2], True)
        u, v = xi[..., None], eta[..., None]
        side = (1 - u) * left_p + u * right_p + (1 - v) * bottom_p + v * top_p
        corner = (
            (1 - u) * (1 - v) * p00
            + u * (1 - v) * p10
            + (1 - u) * v * p01
            + u * v * p11
        )
        return side - corner

    def _apply_current_mapping(self, xi, eta, weight, raw_prediction, projected):
        fused = blend_current_2d(xi, eta, weight, raw_prediction, projected)
        if fused is not None:
            corner_mask = ((xi == 0) | (xi == 1)) & ((eta == 0) | (eta == 1))
            correction = (self.boundary_alpha - 1) * raw_prediction
            return fused + torch.where(
                corner_mask[..., None], torch.zeros_like(correction), correction
            )
        _, boundary_weights = _rational_projection_weights_2d(
            xi,
            eta,
            weight,
            self.projection_transition_width,
            self.projection_normal_smoothing,
            self.projection_tangential_lift_beta,
        )
        wp1, wp2, wp3, wp4 = boundary_weights.unbind(-1)
        residual_weight = self.boundary_alpha - wp1 - wp2 - wp3 - wp4
        result = (
            residual_weight[..., None] * raw_prediction
            + wp1[..., None] * projected[..., 0, :]
            + wp2[..., None] * projected[..., 1, :]
            + wp3[..., None] * projected[..., 2, :]
            + wp4[..., None] * projected[..., 3, :]
        )
        corner_mask = ((xi == 0) | (xi == 1)) & ((eta == 0) | (eta == 1))
        corner_points = torch.where(
            (xi == 0)[..., None], projected[..., 0, :], projected[..., 1, :]
        )
        return torch.where(corner_mask[..., None], corner_points, result)

    def forward(
        self,
        bottom,
        right,
        top,
        left,
        xi,
        eta,
        return_displacement=False,
        boundary_geometry=None,
        boundary_masks=None,
    ):
        """Map queries from four variable-length, directed boundary sequences."""
        if (
            self.training
            and hasattr(boundary_geometry, "project_torch")
            and not getattr(boundary_geometry, "supports_coordinate_gradients", False)
        ):
            raise ValueError("Prepared geometry adapters are for evaluation only")
        if xi.dim() != 2 or eta.shape != xi.shape:
            raise ValueError("xi and eta must have matching [batch, query] shapes")
        edges = (bottom, right, top, left)
        masks = self._edge_masks(edges, boundary_masks)
        h_x, h_y, h = self.encode_boundary(*edges, boundary_masks=masks)
        b_i_x = self.branch_mlp_x(h).unsqueeze(-2)
        b_i_y = self.branch_mlp_y(h).unsqueeze(-2)
        xi_eta = torch.stack((xi, eta), dim=-1)
        trunk_out = (
            self.trunk_net(xi_eta)
            if self.trunk_dim == 2
            else self.trunk_net(self.data_enhance(xi_eta))
        )
        bias = self.bias_head(h).unsqueeze(-2)
        x = torch.sum(b_i_x * trunk_out, dim=-1, keepdim=True) + bias[..., 0:1]
        y = torch.sum(b_i_y * trunk_out, dim=-1, keepdim=True) + bias[..., 1:2]
        raw_prediction = torch.cat((x, y), dim=-1)

        if self.hmonet_mapping_mode == "coons_residual":
            coons = self.evaluate_coons_points_from_edges(
                *edges,
                xi,
                eta,
                boundary_geometry=boundary_geometry,
                boundary_masks=masks,
            )
            displacement = self.coons_residual_envelope(xi, eta)[..., None] * raw_prediction
            result = coons + displacement
        else:
            projected = self.project_edges(
                *edges,
                xi,
                eta,
                boundary_geometry=boundary_geometry,
                boundary_masks=masks,
            )
            weight = self.projection_weights(h, trunk_out)
            result = self._apply_current_mapping(
                xi,
                eta,
                weight,
                raw_prediction,
                projected,
            )
            displacement = torch.zeros_like(result)
        outputs = (result[..., 0:1], result[..., 1:2])
        return (*outputs, displacement) if return_displacement else outputs

    def boundary_attention_weights(
        self,
        bottom,
        right,
        top,
        left,
        boundary_masks=None,
    ):
        """Expose directional pooling weights for analysis and visualization."""
        _, _, attention = self.boundary_attention(
            bottom,
            right,
            top,
            left,
            boundary_masks=boundary_masks,
            return_attention=True,
        )
        return attention


# ============================================================
# Shared 3D model components.


class _OperatorNet3D(nn.Module):
    "Shared 3D operator-network backbone used by MeshONet3D and HMONet3D."

    def __init__(
        self,
        Nx,
        Ny,
        Nz,
        branch_dim=64,
        trunk_dim=24,
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
        self.Nx, self.Ny, self.Nz = map(int, (Nx, Ny, Nz))
        if min(self.Nx, self.Ny, self.Nz) < 2:
            raise ValueError("Nx, Ny and Nz must be >= 2")
        self.boundary_point_count = (
            2 * (self.Nx * self.Ny + self.Nx * self.Nz + self.Ny * self.Nz)
            - 4 * (self.Nx + self.Ny + self.Nz)
            + 8
        )
        ids = torch.arange(self.Nx * self.Ny * self.Nz).reshape(
            self.Nz, self.Ny, self.Nx
        )
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[[0, -1], :, :] = True
        mask[:, [0, -1], :] = True
        mask[:, :, [0, -1]] = True
        boundary_flat_indices = ids[mask]
        full_to_boundary = torch.full_like(ids, -1).reshape(-1)
        full_to_boundary[boundary_flat_indices] = torch.arange(
            boundary_flat_indices.numel(), dtype=torch.long
        )
        face_full = (
            ids[:, :, 0].reshape(-1),
            ids[:, :, -1].reshape(-1),
            ids[:, 0, :].reshape(-1),
            ids[:, -1, :].reshape(-1),
            ids[0, :, :].reshape(-1),
            ids[-1, :, :].reshape(-1),
        )
        for name, face in zip(
            ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"), face_full
        ):
            self.register_buffer(
                f"boundary_face_{name}_indices",
                full_to_boundary[face].clone(),
                persistent=False,
            )
        self._boundary_face_index_names = tuple(
            f"boundary_face_{name}_indices"
            for name in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
        )
        self.trunk_dim = int(trunk_dim)
        self.output_dim = int(output_dim)
        adjust_layers = [adjust_hidden_dim] * adjust_hidden_layer_num
        branch_layers = [branch_hidden_dim] * branch_hidden_layer_num
        trunk_layers = [trunk_hidden_dim] * trunk_hidden_layer_num
        self.adjust = nn.ModuleList(
            [
                nn.Sequential(
                    modules.MLP(self.boundary_point_count, adjust_layers, branch_dim),
                    nn.Tanh(),
                )
                for _ in range(3)
            ]
        )
        self.branches = nn.ModuleList(
            [modules.MLP(branch_dim, branch_layers, output_dim) for _ in range(3)]
        )
        self.trunk = modules.MLP(self.trunk_dim, trunk_layers, output_dim)
        self.bias = nn.Parameter(torch.zeros(3))

    def data_enhance(self, coordinates):
        base = torch.cat(
            (torch.sin(coordinates), torch.cos(coordinates), coordinates), dim=-1
        )
        power = 2
        while base.shape[-1] < self.trunk_dim:
            base = torch.cat((base, coordinates.pow(power)), dim=-1)
            power += 1
        return base[..., : self.trunk_dim]

    def _forward_with_geometry(self, p_x, p_y, p_z, xi, eta, zeta):
        """Return coordinate predictions and the joint boundary descriptor."""
        coordinates = torch.stack((xi, eta, zeta), dim=-1)
        trunk = self.trunk(
            coordinates if self.trunk_dim == 3 else self.data_enhance(coordinates)
        )
        outputs = []
        geometry_features = []
        for axis, boundary_coordinate in enumerate((p_x, p_y, p_z)):
            if boundary_coordinate.dim() == 3:
                boundary_coordinate = boundary_coordinate[:, 0, :]
            branch = self.branches[axis](self.adjust[axis](boundary_coordinate))
            geometry_features.append(branch)
            expanded = branch
            while expanded.dim() < trunk.dim():
                expanded = expanded.unsqueeze(-2)
            outputs.append(
                torch.sum(expanded * trunk, dim=-1, keepdim=True) + self.bias[axis]
            )
        return tuple(outputs), torch.cat(geometry_features, dim=-1)

    def forward(self, p_x, p_y, p_z, xi, eta, zeta):
        outputs, _ = self._forward_with_geometry(p_x, p_y, p_z, xi, eta, zeta)
        return outputs


class MeshONet3D(_OperatorNet3D):
    """Paper-aligned 3-D MeshONet with shared branch-feature fusion."""

    def __init__(
        self,
        Nx,
        Ny,
        Nz,
        branch_mlp_hidden_dim=24,
        branch_mlp_hidden_layer_num=3,
        shared_branch_fusion_enabled=True,
        **kwargs,
    ):
        # Together with the shared fusion block below, these defaults contain
        # exactly 154,163 trainable parameters at a 32 x 32 x 32 resolution.
        defaults = {
            "branch_dim": 24,
            "trunk_dim": 24,
            "adjust_hidden_dim": 8,
            "adjust_hidden_layer_num": 4,
            "branch_hidden_dim": 20,
            "branch_hidden_layer_num": 4,
            "trunk_hidden_dim": 20,
            "trunk_hidden_layer_num": 4,
            "output_dim": 24,
        }
        for name, value in defaults.items():
            kwargs.setdefault(name, value)
        super().__init__(Nx, Ny, Nz, **kwargs)
        self.shared_branch_fusion_enabled = bool(shared_branch_fusion_enabled)
        self.branch_mlp_hidden_dim = int(branch_mlp_hidden_dim)
        self.branch_mlp_hidden_layer_num = int(branch_mlp_hidden_layer_num)
        if self.shared_branch_fusion_enabled:
            if self.branch_mlp_hidden_dim < 1:
                raise ValueError("branch_mlp_hidden_dim must be positive")
            if self.branch_mlp_hidden_layer_num < 1:
                raise ValueError("branch_mlp_hidden_layer_num must be positive")
            fusion_layers = (
                [self.branch_mlp_hidden_dim] * self.branch_mlp_hidden_layer_num
            )
            self.branch_fusion = modules.MLP(
                input_dim=3 * self.output_dim,
                hidden_layer_size=fusion_layers,
                output_dim=self.branch_mlp_hidden_dim,
            )
            self.coefficient_heads = nn.ModuleList(
                nn.Linear(self.branch_mlp_hidden_dim, self.output_dim)
                for _ in range(3)
            )

    def forward(self, p_x, p_y, p_z, xi, eta, zeta):
        if not self.shared_branch_fusion_enabled:
            return super().forward(p_x, p_y, p_z, xi, eta, zeta)

        coordinates = torch.stack((xi, eta, zeta), dim=-1)
        trunk = self.trunk(
            coordinates if self.trunk_dim == 3 else self.data_enhance(coordinates)
        )
        branch_features = []
        for axis, boundary_coordinate in enumerate((p_x, p_y, p_z)):
            if boundary_coordinate.dim() == 3:
                boundary_coordinate = boundary_coordinate[:, 0, :]
            branch_features.append(
                self.branches[axis](self.adjust[axis](boundary_coordinate))
            )
        fused = self.branch_fusion(torch.cat(branch_features, dim=-1))
        outputs = []
        for axis, head in enumerate(self.coefficient_heads):
            coefficients = head(fused)
            while coefficients.dim() < trunk.dim():
                coefficients = coefficients.unsqueeze(-2)
            outputs.append(
                torch.sum(coefficients * trunk, dim=-1, keepdim=True)
                + self.bias[axis]
            )
        return tuple(outputs)


class HMONet3D(_OperatorNet3D):
    """3-D HMONet with selectable six-face or TFI-residual mapping."""

    def __init__(
        self,
        *args,
        hmonet_mapping_mode="current",
        trunk_projection_coefficients_enabled=False,
        projection_softplus_enabled=False,
        boundary_self_attention_enabled=False,
        projection_mlp_hidden_dim=64,
        attention_dim=64,
        attention_heads=4,
        attention_layers=2,
        attention_feedforward_dim=128,
        attention_dropout=0.0,
        # Deprecated compatibility options. The restored original hard map ignores them.
        projection_coefficient_min=0.1,
        projection_transition_width=0.2,
        projection_normal_smoothing=0.5,
        projection_tangential_lift_beta=0.6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hmonet_mapping_mode = str(hmonet_mapping_mode).lower().replace("-", "_")
        if self.hmonet_mapping_mode not in {"current", "coons_residual"}:
            raise ValueError("hmonet_mapping_mode must be current or coons_residual")
        self.trunk_projection_coefficients_enabled = bool(
            trunk_projection_coefficients_enabled
        )
        self.projection_softplus_enabled = bool(projection_softplus_enabled)
        self.boundary_self_attention_enabled = bool(boundary_self_attention_enabled)
        self.projection_coefficient_min = float(projection_coefficient_min)
        self.projection_transition_width = float(projection_transition_width)
        self.projection_normal_smoothing = float(projection_normal_smoothing)
        self.projection_tangential_lift_beta = float(projection_tangential_lift_beta)
        self.projection_weight_head = nn.Linear(3 * self.output_dim, 6)
        self.trunk_projection_head = (
            modules.MLP(
                input_dim=self.output_dim,
                hidden_layer_size=[int(projection_mlp_hidden_dim)],
                output_dim=6,
            )
            if self.trunk_projection_coefficients_enabled
            else None
        )
        self.boundary_attention = (
            modules.BoundarySelfAttentionEncoder3D(
                model_dim=int(attention_dim),
                num_heads=int(attention_heads),
                num_layers=int(attention_layers),
                feedforward_dim=int(attention_feedforward_dim),
                dropout=float(attention_dropout),
                output_dim=self.adjust[0][0].output_dim,
            )
            if self.boundary_self_attention_enabled
            else None
        )
        ids = torch.arange(self.Nx * self.Ny * self.Nz).reshape(
            self.Nz, self.Ny, self.Nx
        )
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[[0, -1], :, :] = True
        mask[:, [0, -1], :] = True
        mask[:, :, [0, -1]] = True
        self.register_buffer(
            "boundary_flat_indices", ids[mask].reshape(-1), persistent=False
        )
        self.register_buffer("_face_priority", torch.arange(1, 7), persistent=False)

    def _activate_projection_coefficients(self, logits):
        return F.softplus(logits) if self.projection_softplus_enabled else torch.sigmoid(logits)

    def projection_coefficients(self, joint_geometry, trunk_features=None):
        """Return branch coefficients, optionally modulated by trunk coefficients."""
        branch = self._activate_projection_coefficients(
            self.projection_weight_head(joint_geometry)
        )
        if self.trunk_projection_head is None:
            return branch.unsqueeze(-2)
        location = self._activate_projection_coefficients(
            self.trunk_projection_head(trunk_features)
        )
        return branch.unsqueeze(-2) * location

    def _forward_hmonet_geometry(self, p_x, p_y, p_z, xi, eta, zeta):
        coordinates = torch.stack((xi, eta, zeta), dim=-1)
        trunk = self.trunk(
            coordinates if self.trunk_dim == 3 else self.data_enhance(coordinates)
        )
        boundary_coordinates = (p_x, p_y, p_z)
        if self.boundary_attention is None:
            descriptors = []
            for axis, coordinate in enumerate(boundary_coordinates):
                if coordinate.dim() == 3:
                    coordinate = coordinate[:, 0, :]
                descriptors.append(self.adjust[axis](coordinate))
        else:
            faces = self._boundary_faces(p_x, p_y, p_z)
            shaped_faces = (
                faces[0].reshape(-1, self.Nz, self.Ny, 3),
                faces[1].reshape(-1, self.Nz, self.Ny, 3),
                faces[2].reshape(-1, self.Nz, self.Nx, 3),
                faces[3].reshape(-1, self.Nz, self.Nx, 3),
                faces[4].reshape(-1, self.Ny, self.Nx, 3),
                faces[5].reshape(-1, self.Ny, self.Nx, 3),
            )
            descriptors = self.boundary_attention(*shaped_faces)
        geometry = [branch(descriptor) for branch, descriptor in zip(self.branches, descriptors)]
        outputs = []
        for axis, feature in enumerate(geometry):
            outputs.append(
                torch.sum(feature.unsqueeze(-2) * trunk, dim=-1, keepdim=True)
                + self.bias[axis]
            )
        return tuple(outputs), torch.cat(geometry, dim=-1), trunk

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Load pre-coefficient 3-D checkpoints with their former unit weights."""
        weight_key = prefix + "projection_weight_head.weight"
        bias_key = prefix + "projection_weight_head.bias"
        if weight_key not in state_dict:
            state_dict[weight_key] = torch.zeros_like(self.projection_weight_head.weight)
        if bias_key not in state_dict:
            state_dict[bias_key] = torch.full_like(
                self.projection_weight_head.bias, 20.0
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _interp2(face, a, b):
        h, w = face.shape[1:3]
        x = a.clamp(0, 1) * (w - 1)
        y = b.clamp(0, 1) * (h - 1)
        i = x.floor().long().clamp(0, w - 2)
        j = y.floor().long().clamp(0, h - 2)
        fx = x - i
        fy = y - j
        batch = torch.arange(face.shape[0], device=face.device)[:, None].expand_as(i)
        return (
            ((1 - fx) * (1 - fy))[..., None] * face[batch, j, i]
            + (fx * (1 - fy))[..., None] * face[batch, j, i + 1]
            + ((1 - fx) * fy)[..., None] * face[batch, j + 1, i]
            + (fx * fy)[..., None] * face[batch, j + 1, i + 1]
        )

    @staticmethod
    def _interp1(edge, t):
        n = edge.shape[1]
        x = t.clamp(0, 1) * (n - 1)
        i = x.floor().long().clamp(0, n - 2)
        f = x - i
        batch = torch.arange(edge.shape[0], device=edge.device)[:, None].expand_as(i)
        return (1 - f)[..., None] * edge[batch, i] + f[..., None] * edge[batch, i + 1]

    def _boundary_grid(self, p_x, p_y, p_z):
        if p_x.dim() == 1:
            p_x, p_y, p_z = p_x.unsqueeze(0), p_y.unsqueeze(0), p_z.unsqueeze(0)
        values = torch.stack((p_x, p_y, p_z), -1)
        if values.dim() == 4:
            values = values[:, 0]
        grid = values.new_zeros((values.shape[0], self.Nx * self.Ny * self.Nz, 3))
        grid[:, self.boundary_flat_indices] = values
        return grid.reshape(values.shape[0], self.Nz, self.Ny, self.Nx, 3)

    def _boundary_faces(self, p_x, p_y, p_z):
        if p_x.dim() == 1:
            p_x, p_y, p_z = p_x.unsqueeze(0), p_y.unsqueeze(0), p_z.unsqueeze(0)
        values = torch.stack((p_x, p_y, p_z), -1)
        if values.dim() == 4:
            values = values[:, 0]
        return tuple(
            values[:, getattr(self, name), :]
            for name in self._boundary_face_index_names
        )

    @staticmethod
    def _interp2_fast(face, a, b):
        batch, height, width, channels = face.shape
        scaled_a = a.clamp(0, 1) * (width - 1)
        scaled_b = b.clamp(0, 1) * (height - 1)
        ia = scaled_a.floor().long().clamp(0, width - 2)
        ib = scaled_b.floor().long().clamp(0, height - 2)
        wa, wb = scaled_a - ia, scaled_b - ib
        indices = torch.stack(
            (
                ib * width + ia,
                ib * width + ia + 1,
                (ib + 1) * width + ia + 1,
                (ib + 1) * width + ia,
            ),
            -1,
        )
        weights = torch.stack(
            (
                (1 - wa) * (1 - wb),
                wa * (1 - wb),
                wa * wb,
                (1 - wa) * wb,
            ),
            -1,
        )
        flat_face = face.reshape(batch, height * width, channels)
        offsets = torch.arange(batch, device=face.device)[:, None, None] * (
            height * width
        )
        corners = flat_face.reshape(batch * height * width, channels)[
            (indices + offsets).reshape(batch, -1)
        ].reshape(batch, indices.shape[-2], 4, channels)
        return (corners * weights[..., None]).sum(-2)

    def project(self, p_x, p_y, p_z, xi, eta, zeta, boundary_geometry=None):
        """Project queries onto six exact parameter surfaces or sampled faces."""
        unbatched = p_x.dim() == 1
        if boundary_geometry is not None:
            if unbatched:
                xi, eta, zeta = xi[None], eta[None], zeta[None]
            result = evaluate_boundary_surfaces_torch(boundary_geometry, xi, eta, zeta)
            return result.squeeze(0) if unbatched else result
        if unbatched:
            xi, eta, zeta = xi[None], eta[None], zeta[None]
        faces = self._boundary_faces(p_x, p_y, p_z)
        result = torch.stack(
            (
                self._interp2_fast(
                    faces[0].reshape(-1, self.Nz, self.Ny, 3), eta, zeta
                ),
                self._interp2_fast(
                    faces[1].reshape(-1, self.Nz, self.Ny, 3), eta, zeta
                ),
                self._interp2_fast(faces[2].reshape(-1, self.Nz, self.Nx, 3), xi, zeta),
                self._interp2_fast(faces[3].reshape(-1, self.Nz, self.Nx, 3), xi, zeta),
                self._interp2_fast(faces[4].reshape(-1, self.Ny, self.Nx, 3), xi, eta),
                self._interp2_fast(faces[5].reshape(-1, self.Ny, self.Nx, 3), xi, eta),
            ),
            dim=-2,
        )
        return result.squeeze(0) if unbatched else result

    def evaluate_coons_points(
        self,
        p_x,
        p_y,
        p_z,
        xi,
        eta,
        zeta,
        boundary_geometry=None,
    ):
        """Evaluate exact Gordon--Hall TFI: face blend - edge blend + corners."""
        unbatched = p_x.dim() == 1
        g = self._boundary_grid(p_x, p_y, p_z)
        if unbatched:
            xi, eta, zeta = xi[None], eta[None], zeta[None]
        f = self.project(
            p_x,
            p_y,
            p_z,
            xi.squeeze(0) if unbatched else xi,
            eta.squeeze(0) if unbatched else eta,
            zeta.squeeze(0) if unbatched else zeta,
            boundary_geometry=boundary_geometry,
        )
        if unbatched:
            f = f[None]
        u, v, w = xi[..., None], eta[..., None], zeta[..., None]
        surface = (
            (1 - u) * f[..., 0, :]
            + u * f[..., 1, :]
            + (1 - v) * f[..., 2, :]
            + v * f[..., 3, :]
            + (1 - w) * f[..., 4, :]
            + w * f[..., 5, :]
        )
        if boundary_geometry is not None:
            # Gordon--Hall cancellation requires exact shared edges as well as
            # exact faces. Interpolating sampled edges breaks curved-face BCs.
            zero, one = torch.zeros_like(xi), torch.ones_like(xi)
            edge_queries = (
                (zero, zero, zeta),
                (one, zero, zeta),
                (zero, one, zeta),
                (one, one, zeta),
                (zero, eta, zero),
                (one, eta, zero),
                (zero, eta, one),
                (one, eta, one),
                (xi, zero, zero),
                (xi, one, zero),
                (xi, zero, one),
                (xi, one, one),
            )
            corners = (
                xi.new_tensor(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [1, 1, 0],
                        [0, 0, 1],
                        [1, 0, 1],
                        [0, 1, 1],
                        [1, 1, 1],
                    ]
                )
                .unsqueeze(0)
                .expand(xi.shape[0], -1, -1)
            )
            queries = torch.cat(
                [
                    torch.cat([torch.stack(q, -1) for q in edge_queries], dim=1),
                    corners,
                ],
                dim=1,
            )
            faces = evaluate_boundary_surfaces_torch(
                boundary_geometry, *queries.unbind(-1)
            )
            face_ids = torch.tensor(
                [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3], device=xi.device
            ).repeat_interleave(xi.shape[1])
            face_ids = torch.cat(
                [face_ids, torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=xi.device)]
            )
            values = faces.gather(
                2, face_ids[None, :, None, None].expand(xi.shape[0], -1, 1, 3)
            ).squeeze(2)
            e = values[:, :-8].reshape(xi.shape[0], 12, xi.shape[1], 3).unbind(1)
            c = values[:, -8:].unbind(1)
            edge = (
                (1 - u) * (1 - v) * e[0]
                + u * (1 - v) * e[1]
                + (1 - u) * v * e[2]
                + u * v * e[3]
                + (1 - u) * (1 - w) * e[4]
                + u * (1 - w) * e[5]
                + (1 - u) * w * e[6]
                + u * w * e[7]
                + (1 - v) * (1 - w) * e[8]
                + v * (1 - w) * e[9]
                + (1 - v) * w * e[10]
                + v * w * e[11]
            )
            corner = (
                (1 - u) * (1 - v) * (1 - w) * c[0][:, None]
                + u * (1 - v) * (1 - w) * c[1][:, None]
                + (1 - u) * v * (1 - w) * c[2][:, None]
                + u * v * (1 - w) * c[3][:, None]
                + (1 - u) * (1 - v) * w * c[4][:, None]
                + u * (1 - v) * w * c[5][:, None]
                + (1 - u) * v * w * c[6][:, None]
                + u * v * w * c[7][:, None]
            )
        else:
            iw = lambda edge, t: self._interp1(edge, t)
            edge = (
                (1 - u) * (1 - v) * iw(g[:, :, 0, 0], zeta)
                + u * (1 - v) * iw(g[:, :, 0, -1], zeta)
                + (1 - u) * v * iw(g[:, :, -1, 0], zeta)
                + u * v * iw(g[:, :, -1, -1], zeta)
                + (1 - u) * (1 - w) * iw(g[:, 0, :, 0], eta)
                + u * (1 - w) * iw(g[:, 0, :, -1], eta)
                + (1 - u) * w * iw(g[:, -1, :, 0], eta)
                + u * w * iw(g[:, -1, :, -1], eta)
                + (1 - v) * (1 - w) * iw(g[:, 0, 0, :], xi)
                + v * (1 - w) * iw(g[:, 0, -1, :], xi)
                + (1 - v) * w * iw(g[:, -1, 0, :], xi)
                + v * w * iw(g[:, -1, -1, :], xi)
            )
            corner = (
                (1 - u) * (1 - v) * (1 - w) * g[:, 0, 0, 0, None, :]
                + u * (1 - v) * (1 - w) * g[:, 0, 0, -1, None, :]
                + (1 - u) * v * (1 - w) * g[:, 0, -1, 0, None, :]
                + u * v * (1 - w) * g[:, 0, -1, -1, None, :]
                + (1 - u) * (1 - v) * w * g[:, -1, 0, 0, None, :]
                + u * (1 - v) * w * g[:, -1, 0, -1, None, :]
                + (1 - u) * v * w * g[:, -1, -1, 0, None, :]
                + u * v * w * g[:, -1, -1, -1, None, :]
            )
        result = surface - edge + corner
        return result.squeeze(0) if unbatched else result

    def forward(
        self,
        p_x,
        p_y,
        p_z,
        xi,
        eta,
        zeta,
        return_displacement=False,
        boundary_geometry=None,
    ):
        """Map queries with six exact surfaces or interpolated sampled faces."""
        if (
            self.training
            and hasattr(boundary_geometry, "project_torch")
            and not getattr(boundary_geometry, "supports_coordinate_gradients", False)
        ):
            raise ValueError("Prepared geometry adapters are for evaluation only")
        raw_parts, joint_geometry, trunk_features = self._forward_hmonet_geometry(
            p_x, p_y, p_z, xi, eta, zeta
        )
        raw = torch.cat(raw_parts, -1)
        if self.hmonet_mapping_mode == "coons_residual":
            envelope = (64 * xi * (1 - xi) * eta * (1 - eta) * zeta * (1 - zeta))[
                ..., None
            ]
            displacement = envelope * raw
            result = (
                self.evaluate_coons_points(
                    p_x,
                    p_y,
                    p_z,
                    xi,
                    eta,
                    zeta,
                    boundary_geometry=boundary_geometry,
                )
                + displacement
            )
        else:
            # The Coons branch evaluates its own faces; only current needs these.
            f = self.project(
                p_x, p_y, p_z, xi, eta, zeta, boundary_geometry=boundary_geometry
            )
            distances = torch.stack((xi, 1 - xi, eta, 1 - eta, zeta, 1 - zeta), -1)
            coefficients = self.projection_coefficients(joint_geometry, trunk_features)

            # Six-face extension of the 2D rational hard-constraint map.  Each
            # face uses its normal coordinate and the product of the two
            # tangential bubbles; no convex normalization is applied.
            q_x = xi * (1.0 - xi)
            q_y = eta * (1.0 - eta)
            q_z = zeta * (1.0 - zeta)
            tangent_x = q_y * q_z
            tangent_y = q_x * q_z
            tangent_z = q_x * q_y
            g_x_min, g_x_max, g_y_min, g_y_max, g_z_min, g_z_max = (
                coefficients.unbind(dim=-1)
            )

            tiny = torch.finfo(raw.dtype).tiny

            def safe_denominator(value: torch.Tensor) -> torch.Tensor:
                return torch.where(value.abs() > tiny, value, torch.ones_like(value))

            denominator_x_min = safe_denominator(q_x + g_x_min * tangent_x)
            denominator_x_max = safe_denominator(q_x + g_x_max * tangent_x)
            denominator_y_min = safe_denominator(q_y + g_y_min * tangent_y)
            denominator_y_max = safe_denominator(q_y + g_y_max * tangent_y)
            denominator_z_min = safe_denominator(q_z + g_z_min * tangent_z)
            denominator_z_max = safe_denominator(q_z + g_z_max * tangent_z)

            w_x_min = g_x_min * (1.0 - xi) * tangent_x / denominator_x_min
            w_x_max = g_x_max * xi * tangent_x / denominator_x_max
            w_y_min = g_y_min * (1.0 - eta) * tangent_y / denominator_y_min
            w_y_max = g_y_max * eta * tangent_y / denominator_y_max
            w_z_min = g_z_min * (1.0 - zeta) * tangent_z / denominator_z_min
            w_z_max = g_z_max * zeta * tangent_z / denominator_z_max
            face_weights = torch.stack(
                (w_x_min, w_x_max, w_y_min, w_y_max, w_z_min, w_z_max),
                dim=-1,
            )
            center_weight = 1.0 - face_weights.sum(dim=-1)
            result = center_weight[..., None] * raw + (
                face_weights[..., None] * f
            ).sum(dim=-2)
            displacement = torch.zeros_like(result)
            # At edges/corners, preserve the original last-face-wins order.
            priority = ((distances.abs() < 1e-7) * self._face_priority).amax(-1)
            indices = (
                (priority - 1)
                .clamp_min(0)[..., None, None]
                .expand(*priority.shape, 1, 3)
            )
            boundary = f.gather(-2, indices).squeeze(-2)
            result = torch.where((priority > 0)[..., None], boundary, result)
        xyz = (result[..., 0:1], result[..., 1:2], result[..., 2:3])
        return (*xyz, displacement) if return_displacement else xyz


class AttentionHMONet3D(HMONet3D):
    """Six-surface self-attention HMONet with factorized hard-map weights."""

    face_names = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")

    def __init__(self, Nx=None, Ny=None, Nz=None, branch_dim=64, trunk_dim=24,
                 branch_hidden_dim=128, trunk_hidden_dim=128,
                 branch_hidden_layer_num=4, trunk_hidden_layer_num=4,
                 output_dim=128, hmonet_mapping_mode="current",
                 attention_dim=64, attention_heads=4, attention_layers=2,
                 attention_feedforward_dim=128, attention_dropout=0.0, **kwargs):
        super().__init__(
            2, 2, 2, branch_dim=branch_dim, trunk_dim=trunk_dim,
            branch_hidden_dim=branch_hidden_dim, trunk_hidden_dim=trunk_hidden_dim,
            branch_hidden_layer_num=branch_hidden_layer_num,
            trunk_hidden_layer_num=trunk_hidden_layer_num, output_dim=output_dim,
            hmonet_mapping_mode=hmonet_mapping_mode, **kwargs,
        )
        self.boundary_attention = modules.BoundarySelfAttentionEncoder3D(
            model_dim=attention_dim, num_heads=attention_heads,
            num_layers=attention_layers, feedforward_dim=attention_feedforward_dim,
            dropout=attention_dropout, output_dim=branch_dim,
        )
        del self.adjust
        self.Nx = self.Ny = self.Nz = None
        self.declared_grid_shape = tuple(
            None if value is None else int(value) for value in (Nx, Ny, Nz)
        )
        self.joint_branches = nn.ModuleList([
            modules.MLP(3 * output_dim, [branch_hidden_dim] * 2, output_dim)
            for _ in range(3)
        ])
        self.bias_head = nn.Linear(3 * output_dim, 3)
        self.branch_projection_head = nn.Linear(3 * output_dim, 6)
        self.trunk_projection_head = nn.Linear(output_dim, 6)

    @staticmethod
    def _face_masks(faces, masks):
        return modules.BoundarySurfaceTokenBuilder3D._normalize_masks(faces, masks)

    @staticmethod
    def _interpolate_face(face, mask, u, v):
        valid = ~mask
        nu = valid.any(2).sum(1)
        nv = valid.any(1).sum(1)
        su = u.clamp(0, 1) * (nu[:, None] - 1).to(u.dtype)
        sv = v.clamp(0, 1) * (nv[:, None] - 1).to(v.dtype)
        iu, iv = su.floor().long(), sv.floor().long()
        ju = torch.minimum(iu + 1, nu[:, None] - 1)
        jv = torch.minimum(iv + 1, nv[:, None] - 1)
        fu, fv = (su - iu.to(su.dtype))[..., None], (sv - iv.to(sv.dtype))[..., None]
        batch = torch.arange(face.shape[0], device=face.device)[:, None].expand_as(iu)
        p00, p10 = face[batch, iu, iv], face[batch, ju, iv]
        p01, p11 = face[batch, iu, jv], face[batch, ju, jv]
        return (1-fu)*(1-fv)*p00 + fu*(1-fv)*p10 + (1-fu)*fv*p01 + fu*fv*p11

    def encode_boundary(self, faces, masks):
        descriptors = self.boundary_attention(*faces, surface_masks=masks)
        axes = tuple(branch(descriptor) for branch, descriptor in zip(self.branches, descriptors))
        joint = torch.cat(axes, -1)
        return tuple(head(joint) for head in self.joint_branches), joint

    def project_faces(self, faces, xi, eta, zeta, boundary_geometry=None,
                      surface_masks=None):
        if boundary_geometry is not None:
            return evaluate_boundary_surfaces_torch(boundary_geometry, xi, eta, zeta)
        masks = self._face_masks(faces, surface_masks)
        modules.BoundarySurfaceTokenBuilder3D._validate(faces, masks)
        return torch.stack((
            self._interpolate_face(faces[0], masks[0], eta, zeta),
            self._interpolate_face(faces[1], masks[1], eta, zeta),
            self._interpolate_face(faces[2], masks[2], xi, zeta),
            self._interpolate_face(faces[3], masks[3], xi, zeta),
            self._interpolate_face(faces[4], masks[4], xi, eta),
            self._interpolate_face(faces[5], masks[5], xi, eta),
        ), -2)

    def projection_weights(self, joint, trunk):
        learned = torch.sigmoid(self.branch_projection_head(joint)).unsqueeze(-2) * \
            torch.sigmoid(self.trunk_projection_head(trunk))
        return self.projection_coefficient_min + (
            1.0 - self.projection_coefficient_min
        ) * learned

    def forward(self, x_min, x_max, y_min, y_max, z_min, z_max,
                xi, eta, zeta, return_displacement=False,
                boundary_geometry=None, surface_masks=None):
        if xi.dim() != 2 or eta.shape != xi.shape or zeta.shape != xi.shape:
            raise ValueError("xi, eta and zeta must have matching [batch, query] shapes")
        faces = (x_min, x_max, y_min, y_max, z_min, z_max)
        masks = self._face_masks(faces, surface_masks)
        branch_axes, joint = self.encode_boundary(faces, masks)
        coordinates = torch.stack((xi, eta, zeta), -1)
        trunk = self.trunk(coordinates if self.trunk_dim == 3 else self.data_enhance(coordinates))
        bias = self.bias_head(joint).unsqueeze(-2)
        raw = torch.cat([
            (axis.unsqueeze(-2) * trunk).sum(-1, keepdim=True) + bias[..., i:i+1]
            for i, axis in enumerate(branch_axes)
        ], -1)
        projected = self.project_faces(
            faces, xi, eta, zeta, boundary_geometry=boundary_geometry,
            surface_masks=masks,
        )
        if self.hmonet_mapping_mode == "coons_residual":
            raise NotImplementedError(
                "AttentionHMONet3D currently supports the requested current mapping only"
            )
        distances = torch.stack((xi, 1-xi, eta, 1-eta, zeta, 1-zeta), -1)
        coefficients = self.projection_weights(joint, trunk)
        center_weight, face_weights = _rational_projection_weights_3d(
            xi,
            eta,
            zeta,
            coefficients,
            self.projection_transition_width,
            self.projection_normal_smoothing,
            self.projection_tangential_lift_beta,
        )
        result = center_weight * raw + (
            face_weights[..., None] * projected
        ).sum(-2)
        priority = ((distances.abs() < 1e-7) * self._face_priority).amax(-1)
        indices = (priority - 1).clamp_min(0)[..., None, None].expand(*priority.shape, 1, 3)
        exact = projected.gather(-2, indices).squeeze(-2)
        result = torch.where((priority > 0)[..., None], exact, result)
        xyz = tuple(result[..., i:i+1] for i in range(3))
        return (*xyz, torch.zeros_like(result)) if return_displacement else xyz


if __name__ == "__main__":
    model = AttentionHMONet2D(Nx=21, Ny=21)
    print(model)

