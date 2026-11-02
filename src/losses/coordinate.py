"""Coordinate-supervision losses shared by all operator models."""


def coordinate_mse(prediction, target, *, coordinate_sum=False):
    """Return coordinate MSE, optionally summing coordinate channels first."""
    squared = (prediction - target).square()
    return squared.sum(dim=-1).mean() if coordinate_sum else squared.mean()


def position_weighted_coordinate_mse(prediction, target, logical, weight_fn):
    """Return coordinate-summed MSE multiplied by a logical-position weight."""
    weight = weight_fn(logical)
    while weight.ndim < prediction.ndim:
        weight = weight.unsqueeze(-1)
    return (weight * (prediction - target)).square().sum(dim=-1).mean()


def boundary_coordinate_mse(prediction, target, *, factor=1.0):
    """Return coordinate-summed boundary MSE with an explicit scale factor."""
    return float(factor) * (prediction - target).square().sum(dim=-1).mean()
