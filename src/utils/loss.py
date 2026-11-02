"""Compatibility imports for the former loss module.

Use :mod:`src.losses` in new code. The old ``ComputeWinslowLoss`` name is
intentionally removed: it returned elliptic PDE residual components rather
than a Winslow energy and was not used by the active training pipeline.
"""

from src.losses import *  # noqa: F401,F403
