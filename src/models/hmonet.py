"""Public HMONet model API.

The class implementations remain checkpoint-compatible with the historical
``src.models.model`` module; this dedicated module is the canonical import
location for new code.
"""

from .model import HMONet2D, HMONet3D

__all__ = ["HMONet2D", "HMONet3D"]
