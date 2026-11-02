"""Module alias for the relocated mesh-quality implementation."""

import sys
from src.metrics import quality as _implementation

sys.modules[__name__] = _implementation
