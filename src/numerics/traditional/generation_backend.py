"""Process-local backend selection shared by end caps and volume generation."""
from contextlib import contextmanager
from contextvars import ContextVar

_OPTIONS = ContextVar('winslow_generation_options', default=None)


@contextmanager
def generation_solver_context(options):
    token = _OPTIONS.set(options)
    try:
        yield
    finally:
        _OPTIONS.reset(token)


def current_solver_options():
    return _OPTIONS.get() or {}
