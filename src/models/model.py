#!/usr/bin/env MGNet
# -*- coding: utf-8 -*-
"Neural operator models for structured-grid geometry mapping."

from __future__ import annotations

__all__ = [
    "MeshONet2D",
    "PIMeshONet2D",
    "HMONet2D",
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
from src.models.inference_optimizations import fuse_coordinates
from src.data.backend.inference_options import ROUTES

# ============================================================
# Neural-network dependencies and local model modules.
# ============================================================

# Module-level compatibility defaults.
DEFAULT_THRESHOLD: float = 0.5
MAX_RETRIES: int = 3


def _safe_rational_denominator(value: torch.Tensor) -> torch.Tensor:
    """Avoid only the removable zero-over-zero values at logical corners."""
    tiny = torch.finfo(value.dtype).tiny
    return torch.where(value.abs() > tiny, value, torch.ones_like(value))


def _rational_affine_weights_2d(xi, eta, coefficients):
    """Return the original four-boundary rational affine weights."""
    q_x = xi * (1.0 - xi)
    q_y = eta * (1.0 - eta)
    g_l, g_r, g_b, g_t = coefficients.unbind(dim=-1)
    w_l = g_l * (1.0 - xi) * q_y / _safe_rational_denominator(q_x + g_l * q_y)
    w_r = g_r * xi * q_y / _safe_rational_denominator(q_x + g_r * q_y)
    w_b = g_b * q_x * (1.0 - eta) / _safe_rational_denominator(q_y + g_b * q_x)
    w_t = g_t * q_x * eta / _safe_rational_denominator(q_y + g_t * q_x)
    boundary = torch.stack((w_l, w_r, w_b, w_t), dim=-1)
    return 1.0 - boundary.sum(dim=-1), boundary


def _corner_corrected_weights_2d(xi, eta, coefficients):
    """Smooth, affine four-side/four-corner weights (scheme B).

    Corner weights can be negative; the map is not a convex combination.
    The fixed raw-field gain 1/2 leaves a substantial learned interior term
    while making that term vanish on every exact boundary.
    """
    sx, sy = xi * (1.0 - xi), eta * (1.0 - eta)
    rho_l, rho_r, rho_b, rho_t = coefficients.unbind(-1)
    lam_l, lam_r = 1.0 - 4.0 * sx * rho_l, 1.0 - 4.0 * sx * rho_r
    lam_b, lam_t = 1.0 - 4.0 * sy * rho_b, 1.0 - 4.0 * sy * rho_t
    beta = 8.0 * sx * sy  # 16*sx*sy*a0, a0 = 1/2.
    scale = 1.0 - beta
    side = torch.stack((
        scale * (1.0 - xi) * lam_l,
        scale * xi * lam_r,
        scale * (1.0 - eta) * lam_b,
        scale * eta * lam_t,
    ), dim=-1)
    corners = torch.stack((
        scale * (1.0 - xi) * (1.0 - eta) * (1.0 - lam_l - lam_b),
        scale * xi * (1.0 - eta) * (1.0 - lam_r - lam_b),
        scale * (1.0 - xi) * eta * (1.0 - lam_l - lam_t),
        scale * xi * eta * (1.0 - lam_r - lam_t),
    ), dim=-1)
    return beta, side, corners


def _rational_affine_weights_3d(xi, eta, zeta, coefficients):
    """Return six-face rational weights with a compatible shared-edge limit.

    Open face interiors use the direct rational formula.  At a shared logical
    edge the two incident exact surface evaluations are averaged.  Compatible
    parameter surfaces agree there, so this is the common exact edge value and
    avoids the removable zero-over-zero form of the face formula.  The eight
    logical corners are handled by explicit exact assignment in the caller.
    """
    q_x = xi * (1.0 - xi)
    q_y = eta * (1.0 - eta)
    q_z = zeta * (1.0 - zeta)
    tangent_x, tangent_y, tangent_z = q_y * q_z, q_x * q_z, q_x * q_y
    g_x0, g_x1, g_y0, g_y1, g_z0, g_z1 = coefficients.unbind(dim=-1)
    boundary = torch.stack(
        (
            g_x0 * (1.0 - xi) * tangent_x
            / _safe_rational_denominator(q_x + g_x0 * tangent_x),
            g_x1 * xi * tangent_x
            / _safe_rational_denominator(q_x + g_x1 * tangent_x),
            g_y0 * (1.0 - eta) * tangent_y
            / _safe_rational_denominator(q_y + g_y0 * tangent_y),
            g_y1 * eta * tangent_y
            / _safe_rational_denominator(q_y + g_y1 * tangent_y),
            g_z0 * (1.0 - zeta) * tangent_z
            / _safe_rational_denominator(q_z + g_z0 * tangent_z),
            g_z1 * zeta * tangent_z
            / _safe_rational_denominator(q_z + g_z1 * tangent_z),
        ),
        dim=-1,
    )
    distances = torch.stack((xi, 1.0 - xi, eta, 1.0 - eta, zeta, 1.0 - zeta), -1)
    incidence = distances.abs() < 1.0e-7
    incidence_count = incidence.sum(dim=-1)
    edge_weights = incidence.to(boundary.dtype) / incidence_count.clamp_min(1)[..., None]
    boundary = torch.where((incidence_count == 2)[..., None], edge_weights, boundary)
    return 1.0 - boundary.sum(dim=-1), boundary, incidence_count


# ============================================================
# Core model classes.
# ============================================================


class MeshONet2D(nn.Module):
    "MeshONet operator network for mapping a 2D computational grid to physical coordinates."

    def __init__(
        self,
        Nx,
        Ny,
        branch_dim=32,
        trunk_dim=16,
        adjust_hidden_dim=64,
        branch_hidden_dim=64,
        trunk_hidden_dim=64,
        adjust_hidden_layer_num=4,
        branch_hidden_layer_num=4,
        trunk_hidden_layer_num=4,
        output_dim=64,
        branch_mlp_hidden_dim=64,
        branch_mlp_hidden_layer_num=4,
        shared_branch_fusion_enabled=True,
        coordinate_lift_mode="legacy",
        coordinate_lift_max_power=1,
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
        self.configured_trunk_dim = int(trunk_dim)
        self.coordinate_lift_mode = modules.normalize_coordinate_lift_mode(
            coordinate_lift_mode
        )
        self.coordinate_lift_max_power = int(coordinate_lift_max_power)
        self.trunk_dim = modules.coordinate_lift_input_dim(
            2,
            self.configured_trunk_dim,
            self.coordinate_lift_mode,
            self.coordinate_lift_max_power,
        )
        self.enhance_dim = self.trunk_dim
        self.adjust_hidden_dim = adjust_hidden_dim
        self.branch_hidden_dim = branch_hidden_dim
        self.trunk_hidden_dim = trunk_hidden_dim
        self.adjust_hidden_layer_num = adjust_hidden_layer_num
        self.branch_hidden_layer_num = branch_hidden_layer_num
        self.trunk_hidden_layer_num = trunk_hidden_layer_num
        self.output_dim = output_dim
        self.branch_mlp_hidden_dim = branch_mlp_hidden_dim
        self.branch_mlp_hidden_layer_num = branch_mlp_hidden_layer_num
        self.shared_branch_fusion_enabled = bool(shared_branch_fusion_enabled)
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
        if self.shared_branch_fusion_enabled:
            self.branch_fusion = modules.MLP(
                input_dim=2 * output_dim,
                hidden_layer_size=branch_mlp_layers,
                output_dim=branch_mlp_hidden_dim,
            )
            self.coefficient_heads = nn.ModuleList(
                nn.Linear(branch_mlp_hidden_dim, output_dim) for _ in range(2)
            )
        else:
            # Preserve the original graph for loading pre-update checkpoints.
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
            input_dim=self.trunk_dim,
            hidden_layer_size=[trunk_hidden_dim] * trunk_hidden_layer_num,
            output_dim=output_dim,
        )

    def data_enhance(self, xi_eta):
        if self.coordinate_lift_mode == "power_fourier":
            return modules.power_fourier_lift(
                xi_eta, self.coordinate_lift_max_power
            )
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
        if self.shared_branch_fusion_enabled:
            fused = self.branch_fusion(h)
            b_i_x, b_i_y = (
                head(fused).unsqueeze(-2) for head in self.coefficient_heads
            )
        else:
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
        trunk_projection_coefficients_enabled=True,
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
        weight_head_mlp_enabled=True,
        coordinate_lift_mode="legacy",
        coordinate_lift_max_power=1,
        modified_mlp_enabled=False,
        query_boundary_cross_attention_enabled=False,
        boundary_layer_decoder_enabled=False,
        decoupled_weight_decoder_enabled=False,
        multiscale_boundary_encoder_enabled=False,
        cross_attention_heads=4,
        boundary_layer_width=0.08,
        local_boundary_dim=32,
        boundary_decoder_hidden_dim=32,
        weight_adapter_hidden_dim=32,
    ):
        super().__init__()
        if branch_mlp_hidden_dim <= 0:
            raise ValueError("branch_mlp_hidden_dim must be positive")
        if branch_mlp_hidden_layer_num <= 0:
            raise ValueError("branch_mlp_hidden_layer_num must be positive")
        hmonet_mapping_mode = str(hmonet_mapping_mode).strip().lower().replace("-", "_")
        if hmonet_mapping_mode not in {"current", "coons_residual", "corner_corrected"}:
            raise ValueError("hmonet_mapping_mode must be current, coons_residual, or corner_corrected")
        # The trunk flag is retained only to load archived branch-only
        # checkpoints. All current training entry points force it on.
        self.trunk_projection_coefficients_enabled = bool(
            trunk_projection_coefficients_enabled
        )
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
        self.configured_trunk_dim = int(trunk_dim)
        self.coordinate_lift_mode = modules.normalize_coordinate_lift_mode(
            coordinate_lift_mode
        )
        self.coordinate_lift_max_power = int(coordinate_lift_max_power)
        self.trunk_dim = modules.coordinate_lift_input_dim(
            2,
            self.configured_trunk_dim,
            self.coordinate_lift_mode,
            self.coordinate_lift_max_power,
        )
        self.enhance_dim = self.trunk_dim
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
        self.modified_mlp_enabled = bool(modified_mlp_enabled)
        self.query_boundary_cross_attention_enabled = bool(
            query_boundary_cross_attention_enabled
        )
        self.boundary_layer_decoder_enabled = bool(boundary_layer_decoder_enabled)
        self.decoupled_weight_decoder_enabled = bool(
            decoupled_weight_decoder_enabled
        )
        self.multiscale_boundary_encoder_enabled = bool(
            multiscale_boundary_encoder_enabled
        )
        self.boundary_layer_width = float(boundary_layer_width)
        if self.boundary_layer_width <= 0:
            raise ValueError("boundary_layer_width must be positive")
        # ``projection_coefficient_min`` remains active in the attention
        # variant. The three following arguments are accepted only so archived
        # configuration files still load; their former convex/smoothed-bubble
        # mapping has been removed.
        self.projection_coefficient_min = float(projection_coefficient_min)
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
        self.weight_head_mlp_enabled = bool(weight_head_mlp_enabled)
        self.weight_head = (
            modules.MLP(
                input_dim=2 * output_dim,
                hidden_layer_size=[int(projection_mlp_hidden_dim)],
                output_dim=4,
            )
            if self.weight_head_mlp_enabled
            else nn.Linear(2 * output_dim, 4)
        )
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
            input_dim=self.trunk_dim,
            hidden_layer_size=[trunk_hidden_dim] * trunk_hidden_layer_num,
            output_dim=output_dim,
        )
        if self.modified_mlp_enabled:
            self.branch_net_x = modules.ModifiedGatedMLP(
                branch_dim, branch_hidden_dim, branch_hidden_layer_num, output_dim
            )
            self.branch_net_y = modules.ModifiedGatedMLP(
                branch_dim, branch_hidden_dim, branch_hidden_layer_num, output_dim
            )
            self.trunk_net = modules.ModifiedGatedMLP(
                self.trunk_dim, trunk_hidden_dim, trunk_hidden_layer_num, output_dim
            )
        self.boundary_token_projection = (
            nn.Sequential(nn.Linear(2, output_dim), nn.Tanh())
            if self.query_boundary_cross_attention_enabled
            and not self.multiscale_boundary_encoder_enabled
            else None
        )
        self.multiscale_boundary_encoder = (
            modules.MultiScaleBoundaryEncoder(
                2, 4, model_dim=int(local_boundary_dim), output_dim=output_dim,
                num_heads=int(cross_attention_heads), dropout=float(attention_dropout),
            )
            if self.multiscale_boundary_encoder_enabled else None
        )
        self.query_boundary_attention = (
            modules.QueryBoundaryCrossAttention(
                output_dim, output_dim, int(cross_attention_heads), float(attention_dropout)
            )
            if self.query_boundary_cross_attention_enabled else None
        )
        self.boundary_layer_decoder = (
            modules.MLP(3 * output_dim, [int(boundary_decoder_hidden_dim)] * 2, 2)
            if self.boundary_layer_decoder_enabled else None
        )
        self.weight_geometry_adapter = (
            modules.MLP(2 * output_dim, [int(weight_adapter_hidden_dim)], 2 * output_dim)
            if self.decoupled_weight_decoder_enabled else nn.Identity()
        )
        self.weight_location_adapter = (
            modules.MLP(output_dim, [int(weight_adapter_hidden_dim)], output_dim)
            if self.decoupled_weight_decoder_enabled else nn.Identity()
        )

    def data_enhance(self, xi_eta):
        if self.coordinate_lift_mode == "power_fourier":
            return modules.power_fourier_lift(
                xi_eta, self.coordinate_lift_max_power
            )
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
        """Map all geometry and location coefficients to the open unit interval."""
        return torch.sigmoid(logits)

    def projection_weights(self, joint_geometry, trunk_features):
        """Return branch coefficients modulated by location-dependent trunk coefficients."""
        branch_geometry = self.weight_geometry_adapter(joint_geometry)
        location_features = self.weight_location_adapter(trunk_features)
        branch = self._activate_projection_coefficients(self.weight_head(branch_geometry))
        if self.trunk_projection_head is None:  # archived checkpoint compatibility
            return branch.unsqueeze(-2)
        location = self._activate_projection_coefficients(
            self.trunk_projection_head(location_features)
        )
        return branch.unsqueeze(-2) * location

    def _enhance_query_features(self, p_x, p_y, trunk_features):
        if self.query_boundary_attention is None:
            return trunk_features
        if self.multiscale_boundary_encoder is not None:
            tokens = self.multiscale_boundary_encoder(
                self._directed_boundary_edges(p_x, p_y)
            )
        else:
            tokens = self.boundary_token_projection(torch.stack((p_x, p_y), dim=-1))
        return self.query_boundary_attention(trunk_features, tokens)

    def _add_boundary_layer_residual(self, raw, joint_geometry, trunk_features, xi_eta):
        if self.boundary_layer_decoder is None:
            return raw
        # ``joint_geometry`` is [D] for single-geometry inference and [B, D]
        # during batched training.  Broadcast it over the query axis without
        # assuming that a batch dimension is present.
        expected_prefix = trunk_features.shape[:-2]
        if joint_geometry.shape[:-1] != expected_prefix:
            raise ValueError(
                "Boundary descriptor and trunk feature batch shapes do not match: "
                f"{tuple(joint_geometry.shape)} versus {tuple(trunk_features.shape)}"
            )
        geometry = joint_geometry.unsqueeze(-2).expand(
            *expected_prefix,
            trunk_features.shape[-2],
            joint_geometry.shape[-1],
        )
        residual = self.boundary_layer_decoder(
            torch.cat((geometry, trunk_features), dim=-1)
        )
        distance = torch.minimum(xi_eta, 1.0 - xi_eta).amin(dim=-1, keepdim=True)
        gate = torch.exp(-distance / self.boundary_layer_width)
        return raw + gate * residual

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
        trunk_out = self._enhance_query_features(p_x, p_y, trunk_out)
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
        raw_prediction = self._add_boundary_layer_residual(
            raw_prediction, h, trunk_out, xi_eta
        )
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
        if mapping_mode not in {"current", "corner_corrected"}:
            raise ValueError("hmonet_mapping_mode must be current, coons_residual, or corner_corrected")

        weight = self.projection_weights(h, trunk_out)
        projected = self.project(
            p_x, p_y, xi, eta, boundary_geometry=boundary_geometry
        )
        if mapping_mode == "corner_corrected":
            beta, boundary_weights, corner_weights = _corner_corrected_weights_2d(
                xi, eta, weight
            )
            # The sampled perimeter includes the four exact common endpoints.
            # Reuse them instead of evaluating every parameter curve again at
            # four corners for every query chunk.
            boundary = torch.stack((p_x, p_y), dim=-1)
            corners = torch.stack(
                (
                    boundary[..., 0, :],
                    boundary[..., self.Nx - 1, :],
                    boundary[..., 2 * self.Nx + self.Ny - 3, :],
                    boundary[..., self.Nx + self.Ny - 2, :],
                ),
                dim=-2,
            )
            result = beta[..., None] * raw_prediction + (
                boundary_weights[..., None] * projected
            ).sum(dim=-2) + (
                corner_weights[..., None] * corners.unsqueeze(-3)
            ).sum(dim=-2)
            if return_displacement:
                return result[..., 0:1], result[..., 1:2], torch.zeros_like(result)
            return result[..., 0:1], result[..., 1:2]
        center_weight, boundary_weights = _rational_affine_weights_2d(
            xi, eta, weight
        )
        result = center_weight[..., None] * raw_prediction + (
            boundary_weights[..., None] * projected
        ).sum(dim=-2)
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
        coordinate_lift_mode="legacy",
        coordinate_lift_max_power=1,
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
        self.configured_trunk_dim = int(trunk_dim)
        self.coordinate_lift_mode = modules.normalize_coordinate_lift_mode(
            coordinate_lift_mode
        )
        self.coordinate_lift_max_power = int(coordinate_lift_max_power)
        self.trunk_dim = modules.coordinate_lift_input_dim(
            3,
            self.configured_trunk_dim,
            self.coordinate_lift_mode,
            self.coordinate_lift_max_power,
        )
        self.output_dim = int(output_dim)
        self.branch_hidden_dim = int(branch_hidden_dim)
        self.branch_hidden_layer_num = int(branch_hidden_layer_num)
        self.trunk_hidden_dim = int(trunk_hidden_dim)
        self.trunk_hidden_layer_num = int(trunk_hidden_layer_num)
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
        if self.coordinate_lift_mode == "power_fourier":
            return modules.power_fourier_lift(
                coordinates, self.coordinate_lift_max_power
            )
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
        branch_mlp_hidden_dim=64,
        branch_mlp_hidden_layer_num=4,
        shared_branch_fusion_enabled=True,
        **kwargs,
    ):
        # Keep the default widths and depths aligned with the compact ex4 HMONet.
        defaults = {
            "branch_dim": 32,
            "trunk_dim": 16,
            "adjust_hidden_dim": 64,
            "adjust_hidden_layer_num": 4,
            "branch_hidden_dim": 64,
            "branch_hidden_layer_num": 4,
            "trunk_hidden_dim": 64,
            "trunk_hidden_layer_num": 4,
            "output_dim": 64,
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
        trunk_projection_coefficients_enabled=True,
        boundary_self_attention_enabled=False,
        projection_mlp_hidden_dim=64,
        attention_dim=64,
        attention_heads=4,
        attention_layers=2,
        attention_feedforward_dim=128,
        attention_dropout=0.0,
        branch_mlp_hidden_dim=128,
        branch_mlp_hidden_layer_num=2,
        # Deprecated compatibility options. The restored original hard map ignores them.
        projection_coefficient_min=0.1,
        projection_transition_width=0.2,
        projection_normal_smoothing=0.5,
        projection_tangential_lift_beta=0.6,
        weight_head_mlp_enabled=True,
        modified_mlp_enabled=False,
        query_boundary_cross_attention_enabled=False,
        boundary_layer_decoder_enabled=False,
        decoupled_weight_decoder_enabled=False,
        multiscale_boundary_encoder_enabled=False,
        cross_attention_heads=4,
        boundary_layer_width=0.08,
        local_boundary_dim=32,
        boundary_decoder_hidden_dim=32,
        weight_adapter_hidden_dim=32,
        **kwargs,
    ):
        if int(branch_mlp_hidden_dim) <= 0:
            raise ValueError("branch_mlp_hidden_dim must be positive")
        if int(branch_mlp_hidden_layer_num) <= 0:
            raise ValueError("branch_mlp_hidden_layer_num must be positive")
        super().__init__(*args, **kwargs)
        self.hmonet_mapping_mode = str(hmonet_mapping_mode).lower().replace("-", "_")
        if self.hmonet_mapping_mode not in {"current", "coons_residual", "corner_corrected"}:
            raise ValueError("hmonet_mapping_mode must be current, coons_residual, or corner_corrected")
        # The trunk flag is retained only to load archived branch-only
        # checkpoints. All current training entry points force it on.
        self.trunk_projection_coefficients_enabled = bool(
            trunk_projection_coefficients_enabled
        )
        self.boundary_self_attention_enabled = bool(boundary_self_attention_enabled)
        self.projection_coefficient_min = float(projection_coefficient_min)
        self.branch_mlp_hidden_dim = int(branch_mlp_hidden_dim)
        self.branch_mlp_hidden_layer_num = int(branch_mlp_hidden_layer_num)
        self.modified_mlp_enabled = bool(modified_mlp_enabled)
        self.query_boundary_cross_attention_enabled = bool(
            query_boundary_cross_attention_enabled
        )
        self.boundary_layer_decoder_enabled = bool(boundary_layer_decoder_enabled)
        self.decoupled_weight_decoder_enabled = bool(decoupled_weight_decoder_enabled)
        self.multiscale_boundary_encoder_enabled = bool(
            multiscale_boundary_encoder_enabled
        )
        self.boundary_layer_width = float(boundary_layer_width)
        if self.boundary_layer_width <= 0:
            raise ValueError("boundary_layer_width must be positive")
        joint_geometry_dim = 3 * self.output_dim
        branch_mlp_layers = [
            self.branch_mlp_hidden_dim
        ] * self.branch_mlp_hidden_layer_num
        self.branch_mlp_x = modules.MLP(
            input_dim=joint_geometry_dim,
            hidden_layer_size=branch_mlp_layers,
            output_dim=self.output_dim,
        )
        self.branch_mlp_y = modules.MLP(
            input_dim=joint_geometry_dim,
            hidden_layer_size=branch_mlp_layers,
            output_dim=self.output_dim,
        )
        self.branch_mlp_z = modules.MLP(
            input_dim=joint_geometry_dim,
            hidden_layer_size=branch_mlp_layers,
            output_dim=self.output_dim,
        )
        self.bias_head = nn.Linear(joint_geometry_dim, 3)
        del self.bias
        if int(projection_mlp_hidden_dim) <= 0:
            raise ValueError("projection_mlp_hidden_dim must be positive")
        self.weight_head_mlp_enabled = bool(weight_head_mlp_enabled)
        self.projection_weight_head = (
            modules.MLP(
                input_dim=joint_geometry_dim,
                hidden_layer_size=[int(projection_mlp_hidden_dim)],
                output_dim=6,
            )
            if self.weight_head_mlp_enabled
            else nn.Linear(joint_geometry_dim, 6)
        )
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
        corner_ids = torch.stack(
            (
                ids[0, 0, 0], ids[0, 0, -1],
                ids[0, -1, 0], ids[0, -1, -1],
                ids[-1, 0, 0], ids[-1, 0, -1],
                ids[-1, -1, 0], ids[-1, -1, -1],
            )
        )
        self.register_buffer(
            "_corner_boundary_positions",
            torch.searchsorted(self.boundary_flat_indices, corner_ids),
            persistent=False,
        )
        self.register_buffer("_face_priority", torch.arange(1, 7), persistent=False)
        if self.modified_mlp_enabled:
            branch_dim = int(self.adjust[0][0].output_dim)
            self.branches = nn.ModuleList(
                modules.ModifiedGatedMLP(
                    branch_dim, self.branch_hidden_dim, self.branch_hidden_layer_num,
                    self.output_dim,
                )
                for _ in range(3)
            )
            self.trunk = modules.ModifiedGatedMLP(
                self.trunk_dim, self.trunk_hidden_dim, self.trunk_hidden_layer_num,
                self.output_dim,
            )
        self.boundary_token_projection = (
            nn.Sequential(nn.Linear(3, self.output_dim), nn.Tanh())
            if self.query_boundary_cross_attention_enabled
            and not self.multiscale_boundary_encoder_enabled else None
        )
        self.multiscale_boundary_encoder = (
            modules.MultiScaleBoundaryEncoder(
                3, 6, model_dim=int(local_boundary_dim), output_dim=self.output_dim,
                num_heads=int(cross_attention_heads), dropout=float(attention_dropout),
            )
            if self.multiscale_boundary_encoder_enabled else None
        )
        self.query_boundary_attention = (
            modules.QueryBoundaryCrossAttention(
                self.output_dim, self.output_dim, int(cross_attention_heads),
                float(attention_dropout),
            )
            if self.query_boundary_cross_attention_enabled else None
        )
        self.boundary_layer_decoder = (
            modules.MLP(
                4 * self.output_dim,
                [int(boundary_decoder_hidden_dim)] * 2,
                3,
            )
            if self.boundary_layer_decoder_enabled else None
        )
        self.weight_geometry_adapter = (
            modules.MLP(
                3 * self.output_dim, [int(weight_adapter_hidden_dim)], 3 * self.output_dim
            )
            if self.decoupled_weight_decoder_enabled else nn.Identity()
        )
        self.weight_location_adapter = (
            modules.MLP(
                self.output_dim, [int(weight_adapter_hidden_dim)], self.output_dim
            )
            if self.decoupled_weight_decoder_enabled else nn.Identity()
        )

    def _activate_projection_coefficients(self, logits):
        return torch.sigmoid(logits)

    def projection_coefficients(self, joint_geometry, trunk_features=None):
        """Return branch coefficients modulated by location-dependent trunk coefficients."""
        branch = self._activate_projection_coefficients(
            self.projection_weight_head(self.weight_geometry_adapter(joint_geometry))
        )
        if self.trunk_projection_head is None:  # archived checkpoint compatibility
            return branch.unsqueeze(-2)
        location = self._activate_projection_coefficients(
            self.trunk_projection_head(self.weight_location_adapter(trunk_features))
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
        joint_geometry = torch.cat(geometry, dim=-1)
        if self.query_boundary_attention is not None:
            if self.multiscale_boundary_encoder is not None:
                tokens = self.multiscale_boundary_encoder(self._boundary_faces(p_x, p_y, p_z))
            else:
                physical_boundary = torch.stack((p_x, p_y, p_z), dim=-1)
                if physical_boundary.dim() == 4:
                    physical_boundary = physical_boundary[:, 0]
                tokens = self.boundary_token_projection(physical_boundary)
            trunk = self.query_boundary_attention(trunk, tokens)
        coordinate_coefficients = (
            self.branch_mlp_x(joint_geometry),
            self.branch_mlp_y(joint_geometry),
            self.branch_mlp_z(joint_geometry),
        )
        bias = self.bias_head(joint_geometry).unsqueeze(-2)
        outputs = tuple(
            torch.sum(coefficients.unsqueeze(-2) * trunk, dim=-1, keepdim=True)
            + bias[..., axis : axis + 1]
            for axis, coefficients in enumerate(coordinate_coefficients)
        )
        if self.boundary_layer_decoder is not None:
            geometry_queries = joint_geometry.unsqueeze(-2).expand(-1, trunk.shape[-2], -1)
            residual = self.boundary_layer_decoder(
                torch.cat((geometry_queries, trunk), dim=-1)
            )
            distance = torch.minimum(coordinates, 1.0 - coordinates).amin(
                dim=-1, keepdim=True
            )
            gate = torch.exp(-distance / self.boundary_layer_width)
            raw = torch.cat(outputs, dim=-1) + gate * residual
            outputs = tuple(raw[..., axis : axis + 1] for axis in range(3))
        return outputs, joint_geometry, trunk

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
        if isinstance(self.projection_weight_head, nn.Linear) and weight_key not in state_dict:
            state_dict[weight_key] = torch.zeros_like(self.projection_weight_head.weight)
        if isinstance(self.projection_weight_head, nn.Linear) and bias_key not in state_dict:
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

    def evaluate_corner_corrected_points(
        self, p_x, p_y, p_z, xi, eta, zeta, raw, coefficients,
        boundary_geometry=None,
    ):
        """Six-face, twelve-edge, eight-corner extension of scheme B."""
        unbatched = xi.dim() == 1
        if unbatched:
            p_x, p_y, p_z = p_x[None], p_y[None], p_z[None]
            xi, eta, zeta = xi[None], eta[None], zeta[None]
            raw, coefficients = raw[None], coefficients[None]
        zero, one = torch.zeros_like(xi), torch.ones_like(xi)
        qx, qy, qz = xi * (1 - xi), eta * (1 - eta), zeta * (1 - zeta)
        rho = coefficients.unbind(-1)
        lam = (
            1 - 4 * qx * rho[0], 1 - 4 * qx * rho[1],
            1 - 4 * qy * rho[2], 1 - 4 * qy * rho[3],
            1 - 4 * qz * rho[4], 1 - 4 * qz * rho[5],
        )
        ax = ((1 - xi) * lam[0], xi * lam[1])
        ay = ((1 - eta) * lam[2], eta * lam[3])
        az = ((1 - zeta) * lam[4], zeta * lam[5])
        dx, dy, dz = (1 - xi, xi), (1 - eta, eta), (1 - zeta, zeta)
        beta = 32 * qx * qy * qz  # 64*qx*qy*qz*a0, a0 = 1/2.

        faces = self.project(
            p_x, p_y, p_z, xi, eta, zeta,
            boundary_geometry=boundary_geometry,
        )
        if xi.device.type == "cpu":
            # Scalar accumulation avoids large temporary weight tensors on CPU.
            face_part = sum(
                a[..., None] * faces[..., i, :]
                for i, a in enumerate((*ax, *ay, *az))
            )
        else:
            ax_values = torch.stack(ax, -1)
            ay_values = torch.stack(ay, -1)
            az_values = torch.stack(az, -1)
            face_weights = torch.cat((ax_values, ay_values, az_values), -1)
            face_part = (face_weights[..., None] * faces).sum(-2)

        # Evaluate each shared edge and corner from one incident exact face.
        # Compatible parameter surfaces must agree on those intersections.
        edge_queries = (
            (zero, zero, zeta), (one, zero, zeta),
            (zero, one, zeta), (one, one, zeta),
            (zero, eta, zero), (one, eta, zero),
            (zero, eta, one), (one, eta, one),
            (xi, zero, zero), (xi, one, zero),
            (xi, zero, one), (xi, one, one),
        )
        queries = torch.cat([torch.stack(q, -1) for q in edge_queries], dim=1)
        values = self.project(
            p_x, p_y, p_z, *queries.unbind(-1),
            boundary_geometry=boundary_geometry,
        )
        face_ids = torch.tensor(
            [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3], device=xi.device
        ).repeat_interleave(xi.shape[1])
        values = values.gather(
            2, face_ids[None, :, None, None].expand(xi.shape[0], -1, 1, 3)
        ).squeeze(2)
        edges = values.reshape(xi.shape[0], 12, xi.shape[1], 3).transpose(1, 2)
        if xi.device.type == "cpu":
            edge_part = sum(
                (ax[i] * ay[j])[..., None] * edges[..., j * 2 + i, :]
                for j in range(2) for i in range(2)
            ) + sum(
                (ax[i] * az[k])[..., None] * edges[..., 4 + k * 2 + i, :]
                for k in range(2) for i in range(2)
            ) + sum(
                (ay[j] * az[k])[..., None] * edges[..., 8 + k * 2 + j, :]
                for k in range(2) for j in range(2)
            )
        else:
            edge_weights = torch.cat(
                (
                    (ay_values[..., :, None] * ax_values[..., None, :]).flatten(-2),
                    (az_values[..., :, None] * ax_values[..., None, :]).flatten(-2),
                    (az_values[..., :, None] * ay_values[..., None, :]).flatten(-2),
                ),
                -1,
            )
            edge_part = (edge_weights[..., None] * edges).sum(-2)
        boundary = torch.stack((p_x, p_y, p_z), -1)
        if boundary.dim() == 4:
            boundary = boundary[:, 0]
        corners = boundary.index_select(1, self._corner_boundary_positions)
        remainder = (1 - ax[0] - ax[1]) * (1 - ay[0] - ay[1]) * (
            1 - az[0] - az[1]
        )
        if xi.device.type == "cpu":
            corner_part = sum(
                (
                    ax[i] * ay[j] * az[k]
                    + remainder * dx[i] * dy[j] * dz[k]
                )[..., None] * corners[:, k * 4 + j * 2 + i, None, :]
                for k in range(2) for j in range(2) for i in range(2)
            )
        else:
            corner_weights = (
                az_values[..., :, None, None]
                * ay_values[..., None, :, None]
                * ax_values[..., None, None, :]
                + remainder[..., None, None, None]
                * torch.stack(dz, -1)[..., :, None, None]
                * torch.stack(dy, -1)[..., None, :, None]
                * torch.stack(dx, -1)[..., None, None, :]
            ).flatten(-3)
            corner_part = (corner_weights[..., None] * corners[:, None]).sum(-2)
        result = beta[..., None] * raw + (1 - beta)[..., None] * (
            face_part - edge_part + corner_part
        )
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
        elif self.hmonet_mapping_mode == "corner_corrected":
            coefficients = self.projection_coefficients(joint_geometry, trunk_features)
            result = self.evaluate_corner_corrected_points(
                p_x, p_y, p_z, xi, eta, zeta, raw, coefficients,
                boundary_geometry=boundary_geometry,
            )
            displacement = torch.zeros_like(result)
        else:
            # The Coons branch evaluates its own faces; only current needs these.
            f = self.project(
                p_x, p_y, p_z, xi, eta, zeta, boundary_geometry=boundary_geometry
            )
            coefficients = self.projection_coefficients(joint_geometry, trunk_features)
            center_weight, face_weights, incidence_count = (
                _rational_affine_weights_3d(xi, eta, zeta, coefficients)
            )
            result = center_weight[..., None] * raw + (
                face_weights[..., None] * f
            ).sum(dim=-2)
            displacement = torch.zeros_like(result)
            # Shared edges use the compatible two-face limit above. Only the
            # eight logical corners need direct exact assignment.
            distances = torch.stack((xi, 1 - xi, eta, 1 - eta, zeta, 1 - zeta), -1)
            priority = ((distances.abs() < 1e-7) * self._face_priority).amax(-1)
            indices = (
                (priority - 1)
                .clamp_min(0)[..., None, None]
                .expand(*priority.shape, 1, 3)
            )
            boundary = f.gather(-2, indices).squeeze(-2)
            result = torch.where((incidence_count >= 3)[..., None], boundary, result)
        xyz = (result[..., 0:1], result[..., 1:2], result[..., 2:3])
        return (*xyz, displacement) if return_displacement else xyz


