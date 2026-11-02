"""Compatibility imports for the relocated physics-loss implementation.

New code must import from :mod:`src.losses.physics`. This module remains so
existing commands and third-party imports keep working during migration.
"""

from src.losses.physics import *  # noqa: F401,F403
