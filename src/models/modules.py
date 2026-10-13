"""Neural network building blocks used by the 2D and 3D operators."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "MGNet2DBlock",
    "MLP",
    "BoundaryTokenBuilder2D",
    "BoundarySelfAttentionEncoder2D",
    "BoundarySurfaceTokenBuilder3D",
    "BoundarySelfAttentionEncoder3D",
]


class MGNet2DBlock(nn.Module):
    """Coordinate feature block with sinusoidal encoding and gated hidden layers.

    The block receives two computational coordinates, augments them with
    sine, cosine, and identity features, and returns the requested output
    dimension.
    """

    CLASS_VERSION: str = "1.0"

    def __init__(self, hidden_dim=100, hidden_layer_num=4, output_dim=1):
        """Create the coordinate feature block."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layer_num = hidden_layer_num
        self.output_dim = output_dim
        self.activation = nn.Tanh()

        # Six features are sin(xi), sin(eta), cos(xi), cos(eta), xi, and eta.
        self.expanding_layer = nn.Linear(6, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layer_num)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.excitation_block = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, xi, eta):
        """Map computational coordinates to the learned output features."""
        xy = torch.cat((xi, eta), dim=1)
        # Encode coordinates before applying the learned nonlinear mapping.
        enhance_data = torch.cat([torch.sin(xy), torch.cos(xy), xy], dim=1)
        m = self.activation(self.expanding_layer(enhance_data))
        for layer in self.hidden_layers:
            m = self.activation(layer(m))
            # The excitation layer provides an additional feature transformation.
            m = self.excitation_block(m)
        return self.output_layer(m)


class MLP(nn.Module):
    """Configurable fully connected multilayer perceptron."""

    def __init__(
        self, input_dim=2, hidden_layer_size=None, output_dim=1, activation=None
    ):
        """Create an MLP with the requested layer widths and activation."""
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layer_size = (
            hidden_layer_size if hidden_layer_size is not None else [128] * 4
        )
        self.output_dim = output_dim
        self.activation = activation if activation is not None else nn.Tanh()
        self.input_layer = nn.Linear(input_dim, self.hidden_layer_size[0])
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_layer_size[i], self.hidden_layer_size[i + 1])
                for i in range(len(self.hidden_layer_size) - 1)
            ]
        )
        self.output_layer = nn.Linear(self.hidden_layer_size[-1], output_dim)

    def forward(self, x):
        """Apply the input layer, hidden layers, activations, and output layer."""
        m = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            m = self.activation(layer(m))
        return self.output_layer(m)


class BoundaryTokenBuilder2D(nn.Module):
    """Build tokens from four directed, optionally padded boundary sequences."""

    edge_names = ("bottom", "right", "top", "left")

    def __init__(self, model_dim):
        super().__init__()
        self.coordinate_projection = nn.Linear(4, model_dim)
        self.edge_embedding = nn.Embedding(4, model_dim)
        self.normalization = nn.LayerNorm(model_dim)

    @staticmethod
    def _normalize_masks(edges, masks):
        if masks is None:
            return tuple(
                torch.zeros(edge.shape[:2], dtype=torch.bool, device=edge.device)
                for edge in edges
            )
        if isinstance(masks, dict):
            masks = tuple(masks[name] for name in BoundaryTokenBuilder2D.edge_names)
        if len(masks) != 4:
            raise ValueError("boundary_masks must contain bottom/right/top/left masks")
        return tuple(mask.to(device=edge.device, dtype=torch.bool) for edge, mask in zip(edges, masks))

    @staticmethod
    def _validate(edges, masks):
        batch = edges[0].shape[0]
        device, dtype = edges[0].device, edges[0].dtype
        for name, edge, mask in zip(BoundaryTokenBuilder2D.edge_names, edges, masks):
            if edge.dim() != 3 or edge.shape[-1] != 2:
                raise ValueError(f"{name} must have shape [batch, points, 2]")
            if edge.shape[0] != batch or edge.device != device or edge.dtype != dtype:
                raise ValueError("all four boundary tensors must share batch, device and dtype")
            if mask.shape != edge.shape[:2]:
                raise ValueError(f"{name} mask must have shape [batch, points]")
            valid = ~mask
            if torch.any(valid.sum(dim=1) < 2):
                raise ValueError(f"every {name} boundary requires at least two valid points")
            if torch.any(mask[:, :-1] & valid[:, 1:]):
                raise ValueError("valid boundary points must precede padding points")

    @staticmethod
    def _edge_parameters(edge, mask):
        lengths = (~mask).sum(dim=1)
        positions = torch.arange(edge.shape[1], device=edge.device)[None, :]
        t = positions.to(edge.dtype) / (lengths[:, None] - 1).to(edge.dtype)
        return t.clamp(0.0, 1.0).masked_fill(mask, 0.0)

    def forward(self, bottom, right, top, left, boundary_masks=None):
        """Return packed tokens and a ``True=padding`` attention mask."""
        edges = (bottom, right, top, left)
        masks = self._normalize_masks(edges, boundary_masks)
        self._validate(edges, masks)
        token_parts = []
        for edge_id, (edge, mask) in enumerate(zip(edges, masks)):
            t = self._edge_parameters(edge, mask)
            if edge_id == 0:
                computational = torch.stack((t, torch.zeros_like(t)), dim=-1)
            elif edge_id == 1:
                computational = torch.stack((torch.ones_like(t), t), dim=-1)
            elif edge_id == 2:
                computational = torch.stack((t, torch.ones_like(t)), dim=-1)
            else:
                computational = torch.stack((torch.zeros_like(t), t), dim=-1)
            physical = edge.masked_fill(mask[..., None], 0.0)
            base = self.coordinate_projection(torch.cat((physical, computational), dim=-1))
            identity = self.edge_embedding(
                torch.full((edge.shape[1],), edge_id, device=edge.device, dtype=torch.long)
            )[None, :, :]
            token_parts.append(self.normalization(base + identity))
        return torch.cat(token_parts, dim=1), torch.cat(masks, dim=1)


class BoundarySelfAttentionEncoder2D(nn.Module):
    """Encode all four directed boundaries with shared self-attention.

    A Transformer encoder models interactions both along an edge and across
    different edges. Two learned pooling queries then extract axis-specific
    geometry descriptors for the x and y operator branches.
    """

    def __init__(
        self,
        model_dim=64,
        num_heads=4,
        num_layers=2,
        feedforward_dim=128,
        dropout=0.0,
        output_dim=64,
    ):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.token_builder = BoundaryTokenBuilder2D(model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(model_dim),
        )
        self.direction_queries = nn.Parameter(torch.empty(2, model_dim))
        nn.init.normal_(self.direction_queries, mean=0.0, std=model_dim ** -0.5)
        self.direction_pool = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, output_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        bottom,
        right,
        top,
        left,
        boundary_masks=None,
        return_attention=False,
    ):
        """Return x/y geometry descriptors and optional pooling attention maps."""
        tokens, padding_mask = self.token_builder(
            bottom,
            right,
            top,
            left,
            boundary_masks=boundary_masks,
        )
        tokens = self.encoder(tokens, src_key_padding_mask=padding_mask)
        queries = self.direction_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        pooled, attention = self.direction_pool(
            queries,
            tokens,
            tokens,
            key_padding_mask=padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        descriptors = self.output_projection(pooled)
        h_x, h_y = descriptors.unbind(dim=1)
        if return_attention:
            return h_x, h_y, attention
        return h_x, h_y


class BoundarySurfaceTokenBuilder3D(nn.Module):
    """Build tokens from six directed, optionally padded parameter surfaces.

    Every face is supplied as ``[batch, nu, nv, 3]``.  The two logical
    parameters are inferred from the valid rectangular prefix of each padded
    face, so one model can consume different surface resolutions.
    """

    face_names = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")

    def __init__(self, model_dim):
        super().__init__()
        self.coordinate_projection = nn.Linear(6, model_dim)
        self.face_embedding = nn.Embedding(6, model_dim)
        self.normalization = nn.LayerNorm(model_dim)

    @staticmethod
    def _normalize_masks(faces, masks):
        if masks is None:
            return tuple(
                torch.zeros(face.shape[:3], dtype=torch.bool, device=face.device)
                for face in faces
            )
        if isinstance(masks, dict):
            masks = tuple(masks[name] for name in BoundarySurfaceTokenBuilder3D.face_names)
        if len(masks) != 6:
            raise ValueError("surface_masks must contain all six face masks")
        return tuple(
            mask.to(device=face.device, dtype=torch.bool)
            for face, mask in zip(faces, masks)
        )

    @staticmethod
    def _validate(faces, masks):
        batch = faces[0].shape[0]
        device, dtype = faces[0].device, faces[0].dtype
        for name, face, mask in zip(BoundarySurfaceTokenBuilder3D.face_names, faces, masks):
            if face.dim() != 4 or face.shape[-1] != 3:
                raise ValueError(f"{name} must have shape [batch, nu, nv, 3]")
            if face.shape[0] != batch or face.device != device or face.dtype != dtype:
                raise ValueError("all six faces must share batch, device and dtype")
            if mask.shape != face.shape[:3]:
                raise ValueError(f"{name} mask must have shape [batch, nu, nv]")
            valid = ~mask
            rows = valid.any(dim=2).sum(dim=1)
            columns = valid.any(dim=1).sum(dim=1)
            if torch.any(rows < 2) or torch.any(columns < 2):
                raise ValueError(f"every {name} face needs at least a 2 x 2 valid grid")
            expected = (
                torch.arange(face.shape[1], device=face.device)[None, :, None] < rows[:, None, None]
            ) & (
                torch.arange(face.shape[2], device=face.device)[None, None, :] < columns[:, None, None]
            )
            if not torch.equal(valid, expected):
                raise ValueError("valid face samples must form a rectangular prefix")

    @staticmethod
    def _surface_parameters(face, mask):
        valid = ~mask
        nu = valid.any(dim=2).sum(dim=1)
        nv = valid.any(dim=1).sum(dim=1)
        u = torch.arange(face.shape[1], device=face.device, dtype=face.dtype)[None, :, None]
        v = torch.arange(face.shape[2], device=face.device, dtype=face.dtype)[None, None, :]
        u = u / (nu[:, None, None] - 1).to(face.dtype)
        v = v / (nv[:, None, None] - 1).to(face.dtype)
        return u.expand_as(mask).masked_fill(mask, 0.0), v.expand_as(mask).masked_fill(mask, 0.0)

    def forward(self, *faces, surface_masks=None):
        masks = self._normalize_masks(faces, surface_masks)
        self._validate(faces, masks)
        tokens, padding = [], []
        for face_id, (face, mask) in enumerate(zip(faces, masks)):
            u, v = self._surface_parameters(face, mask)
            if face_id == 0:
                logical = torch.stack((torch.zeros_like(u), u, v), -1)
            elif face_id == 1:
                logical = torch.stack((torch.ones_like(u), u, v), -1)
            elif face_id == 2:
                logical = torch.stack((u, torch.zeros_like(u), v), -1)
            elif face_id == 3:
                logical = torch.stack((u, torch.ones_like(u), v), -1)
            elif face_id == 4:
                logical = torch.stack((u, v, torch.zeros_like(u)), -1)
            else:
                logical = torch.stack((u, v, torch.ones_like(u)), -1)
            shape = face.shape
            physical = face.masked_fill(mask[..., None], 0.0)
            base = self.coordinate_projection(torch.cat((physical, logical), -1))
            identity = self.face_embedding.weight[face_id].view(1, 1, 1, -1)
            tokens.append(self.normalization(base + identity).reshape(shape[0], -1, base.shape[-1]))
            padding.append(mask.reshape(shape[0], -1))
        return torch.cat(tokens, 1), torch.cat(padding, 1)


class BoundarySelfAttentionEncoder3D(nn.Module):
    """Jointly encode six surfaces and pool x/y/z geometry descriptors."""

    def __init__(self, model_dim=64, num_heads=4, num_layers=2,
                 feedforward_dim=128, dropout=0.0, output_dim=64):
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.token_builder = BoundarySurfaceTokenBuilder3D(model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=feedforward_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(model_dim))
        self.direction_queries = nn.Parameter(torch.empty(3, model_dim))
        nn.init.normal_(self.direction_queries, std=model_dim ** -0.5)
        self.direction_pool = nn.MultiheadAttention(model_dim, num_heads,
                                                    dropout=dropout, batch_first=True)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Linear(model_dim, output_dim), nn.Tanh()
        )

    def forward(self, *faces, surface_masks=None, return_attention=False):
        tokens, padding = self.token_builder(*faces, surface_masks=surface_masks)
        tokens = self.encoder(tokens, src_key_padding_mask=padding)
        queries = self.direction_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        pooled, attention = self.direction_pool(
            queries, tokens, tokens, key_padding_mask=padding,
            need_weights=return_attention, average_attn_weights=False,
        )
        descriptors = self.output_projection(pooled).unbind(dim=1)
        return (*descriptors, attention) if return_attention else descriptors


if __name__ == "__main__":
    block = MGNet2DBlock(hidden_dim=100, hidden_layer_num=4, output_dim=1)
    print(block)

