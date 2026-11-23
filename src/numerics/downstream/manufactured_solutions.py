"""Manufactured scalar-diffusion solutions for downstream mesh tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ManufacturedSolution2D:
    """Exact field, gradient, and forcing for ``-div(k grad(u)) = q``."""

    name: str
    conductivity: float = 1.0

    def value(self, points):
        points = np.asarray(points, dtype=float)
        x, y = points[..., 0], points[..., 1]
        if self.name == "sine_product":
            return np.sin(np.pi * x) * np.sin(np.pi * y)
        if self.name == "exponential_sine":
            return np.exp(0.5 * x) * np.sin(2.0 * np.pi * y)
        raise RuntimeError(f"Unsupported manufactured solution {self.name!r}")

    def gradient(self, points):
        points = np.asarray(points, dtype=float)
        x, y = points[..., 0], points[..., 1]
        if self.name == "sine_product":
            return np.stack(
                [
                    np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
                    np.pi * np.sin(np.pi * x) * np.cos(np.pi * y),
                ],
                axis=-1,
            )
        if self.name == "exponential_sine":
            exponential = np.exp(0.5 * x)
            return np.stack(
                [
                    0.5 * exponential * np.sin(2.0 * np.pi * y),
                    2.0 * np.pi * exponential * np.cos(2.0 * np.pi * y),
                ],
                axis=-1,
            )
        raise RuntimeError(f"Unsupported manufactured solution {self.name!r}")

    def source(self, points):
        if self.name == "sine_product":
            # -k Delta u = 2 k pi^2 u.
            factor = 2.0 * np.pi**2
        elif self.name == "exponential_sine":
            # Delta u = (1/4 - 4 pi^2) u.
            factor = 4.0 * np.pi**2 - 0.25
        else:
            raise RuntimeError(f"Unsupported manufactured solution {self.name!r}")
        return float(self.conductivity) * factor * self.value(points)


def create_manufactured_solution(name="sine_product", *, conductivity=1.0):
    """Construct a registered two-dimensional manufactured solution."""
    key = str(name).strip().lower().replace("-", "_")
    aliases = {
        "sine_product": "sine_product",
        "sin_product": "sine_product",
        "exponential_sine": "exponential_sine",
        "exp_sine": "exponential_sine",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown manufactured solution {name!r}; expected one of "
            "'sine_product' or 'exponential_sine'"
        )
    conductivity = float(conductivity)
    if not np.isfinite(conductivity) or conductivity <= 0:
        raise ValueError("Manufactured-solution conductivity must be positive")
    return ManufacturedSolution2D(aliases[key], conductivity)
