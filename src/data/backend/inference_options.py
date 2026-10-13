"""Internal switches for reproducible inference ablations; all routes default on."""

from contextlib import contextmanager
from contextvars import ContextVar

ROUTES = ContextVar("hmonet_inference_routes", default=(True, True, True, True))


@contextmanager
def inference_routes(blend=True, direct=True, scalar=True, contraction=True):
    """Temporarily disable individual routes without modifying model state."""
    token = ROUTES.set((blend, direct, scalar, contraction))
    try:
        yield
    finally:
        ROUTES.reset(token)
