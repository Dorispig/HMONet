"""Backward-compatible import path for parameterized-boundary datasets.

New code should import :mod:`src.data.parametric_dataset` directly.
"""

from src.data.parametric_dataset import *  # noqa: F401,F403
