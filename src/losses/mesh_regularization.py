"""Geometry and mesh-shape regularization losses."""

import torch


def normal_alignment_loss(direction, normal, epsilon=1e-12):
    """Penalize the component of ``direction`` tangent to a boundary."""
    unit_normal = normal / torch.linalg.vector_norm(
        normal, dim=-1, keepdim=True
    ).clamp_min(float(epsilon))
    normal_component = (direction * unit_normal).sum(dim=-1, keepdim=True)
    tangent = direction - normal_component * unit_normal
    return tangent.square().sum(dim=-1).mean()
