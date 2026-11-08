"""Query--Geometry Gated HMONet models.

QG-HMONet is the **Query--Geometry Gated Hard-Boundary Mesh Operator
Network**.  It augments HMONet with learned multiscale boundary tokens,
query-to-boundary cross-attention, a gated residual query fusion, a
boundary-layer decoder, and factorized coordinate/weight decoders.  It does
not compute curvature, normals, signed distances, or other explicit local
geometric differential features.
"""

from __future__ import annotations

import torch

from .mapping_modes import CORNER_CORRECTED_AFFINE
from .model import HMONet2D, HMONet3D, _corner_corrected_affine_weights_2d

__all__ = ["QGHMONet2D", "QGHMONet3D"]


class QGHMONet2D(HMONet2D):
    """Two-dimensional Query--Geometry Gated HMONet.

    The implementation is intentionally written as six named stages so the
    QG-HMONet architecture can be read directly from this file:

    1. ``encode_global_geometry``: two HMONet boundary branches -> ``h``;
    2. ``encode_local_boundary``: shared Conv1D 3/5/9 and self-attention -> ``E``;
    3. ``encode_queries``: coordinate lift, trunk and query-boundary attention;
    4. ``decode_coordinates`` / ``decode_mapping_weights``: separate outputs;
    5. ``project_exact_boundary``: exact parametric boundary projections;
    6. ``hard_boundary_fusion``: corner-corrected exact-boundary mapping.

    Layer implementations are shared with HMONet through ``modules.py``; only
    their architecture-level composition is defined here.  This avoids two
    independent copies of the same hard-boundary formulas.
    """

    architecture_name = "query_geometry_gated_hard_boundary_operator"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("query_boundary_cross_attention_enabled", True)
        kwargs.setdefault("boundary_layer_decoder_enabled", True)
        kwargs.setdefault("decoupled_weight_decoder_enabled", True)
        kwargs.setdefault("multiscale_boundary_encoder_enabled", True)
        kwargs.setdefault("local_boundary_dim", 16)
        super().__init__(*args, **kwargs)

    def encode_global_geometry(self, p_x, p_y):
        """Stage 1: obtain global x/y branch features and their concatenation."""
        h_x, h_y, h = self.encode_boundary(p_x, p_y)
        return {"h_x": h_x, "h_y": h_y, "h": h}

    def encode_local_boundary(self, p_x, p_y):
        """Stage 2: encode four directed edges into local boundary tokens E."""
        edges = self._directed_boundary_edges(p_x, p_y)
        return self.multiscale_boundary_encoder(edges)

    def encode_queries(self, xi, eta, boundary_tokens):
        """Stage 3: coordinate lift -> trunk -> query-to-boundary attention."""
        coordinates = torch.stack((xi, eta), dim=-1)
        lifted = (
            coordinates
            if self.trunk_dim == 2
            else self.data_enhance(coordinates)
        )
        trunk = self.trunk_net(lifted)
        enhanced = self.query_boundary_attention(trunk, boundary_tokens)
        return {
            "coordinates": coordinates,
            "lifted": lifted,
            "trunk": trunk,
            "enhanced_trunk": enhanced,
        }

    def decode_coordinates(self, global_features, query_features):
        """Stage 4a: branch--trunk products, bias, and boundary-layer decoder."""
        h = global_features["h"]
        trunk = query_features["enhanced_trunk"]
        branch_x = self.branch_mlp_x(h).unsqueeze(-2)
        branch_y = self.branch_mlp_y(h).unsqueeze(-2)
        bias = self.bias_head(h).unsqueeze(-2)
        raw = torch.cat(
            (
                (branch_x * trunk).sum(-1, keepdim=True) + bias[..., 0:1],
                (branch_y * trunk).sum(-1, keepdim=True) + bias[..., 1:2],
            ),
            dim=-1,
        )
        return self._add_boundary_layer_residual(
            raw, h, trunk, query_features["coordinates"]
        )

    def decode_mapping_weights(self, global_features, query_features):
        """Stage 4b: independently adapted geometry/location mapping weights."""
        return self.projection_weights(
            global_features["h"], query_features["enhanced_trunk"]
        )

    def project_exact_boundary(self, p_x, p_y, xi, eta, boundary_geometry=None):
        """Stage 5: evaluate left/right/bottom/top projection points."""
        return self.project(
            p_x, p_y, xi, eta, boundary_geometry=boundary_geometry
        )

    def hard_boundary_fusion(
        self, p_x, p_y, xi, eta, raw_coordinates, mapping_weights, projections
    ):
        """Stage 6: apply the corner-corrected exact-boundary formula."""
        beta, boundary_weights, corner_weights = _corner_corrected_affine_weights_2d(
            xi, eta, mapping_weights
        )
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
        return (
            beta[..., None] * raw_coordinates
            + (boundary_weights[..., None] * projections).sum(dim=-2)
            + (
                corner_weights[..., None] * corners.unsqueeze(-3)
            ).sum(dim=-2)
        )

    def forward(
        self,
        p_x,
        p_y,
        xi,
        eta,
        return_displacement=False,
        boundary_geometry=None,
    ):
        """Run the six explicit QG-HMONet stages and return physical coordinates."""
        if self.hmonet_mapping_mode != CORNER_CORRECTED_AFFINE:
            return super().forward(
                p_x, p_y, xi, eta,
                return_displacement=return_displacement,
                boundary_geometry=boundary_geometry,
            )
        if (
            self.training
            and hasattr(boundary_geometry, "project_torch")
            and not getattr(boundary_geometry, "supports_coordinate_gradients", False)
        ):
            raise ValueError("Prepared geometry adapters are for evaluation only")
        if p_x.dim() == 3:
            p_x, p_y = p_x[:, 0, :], p_y[:, 0, :]

        global_features = self.encode_global_geometry(p_x, p_y)
        boundary_tokens = self.encode_local_boundary(p_x, p_y)
        query_features = self.encode_queries(xi, eta, boundary_tokens)
        raw_coordinates = self.decode_coordinates(global_features, query_features)
        mapping_weights = self.decode_mapping_weights(
            global_features, query_features
        )
        projections = self.project_exact_boundary(
            p_x, p_y, xi, eta, boundary_geometry=boundary_geometry
        )
        result = self.hard_boundary_fusion(
            p_x, p_y, xi, eta,
            raw_coordinates, mapping_weights, projections,
        )
        outputs = (result[..., 0:1], result[..., 1:2])
        if return_displacement:
            return (*outputs, torch.zeros_like(result))
        return outputs


class QGHMONet3D(HMONet3D):
    """Three-dimensional QG-HMONet with the same six-stage composition.

    The global encoder has x/y/z branches, the local encoder consumes six
    directed boundary faces, the trunk queries are three-dimensional, and the
    exact projector returns six face projections.  HMONet3D supplies the 3-D
    corner-corrected hard-boundary formula while the QG-specific modules are
    enabled explicitly below.
    """

    architecture_name = "query_geometry_gated_hard_boundary_operator"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("query_boundary_cross_attention_enabled", True)
        kwargs.setdefault("boundary_layer_decoder_enabled", True)
        kwargs.setdefault("decoupled_weight_decoder_enabled", True)
        kwargs.setdefault("multiscale_boundary_encoder_enabled", True)
        kwargs.setdefault("local_boundary_dim", 16)
        super().__init__(*args, **kwargs)
