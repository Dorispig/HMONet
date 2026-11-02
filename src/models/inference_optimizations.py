"""Stateless forward optimizations; model parameters and checkpoints are unchanged."""

import torch


def fuse_coordinates(trunk, branch_x, branch_y, bias):
    """Contract feature dimensions directly instead of materializing two products."""
    branch = torch.cat((branch_x, branch_y), dim=-2)
    return torch.matmul(trunk, branch.transpose(-1, -2)) + bias
