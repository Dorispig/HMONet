"""Module alias for the relocated training-state engine.

Using a true module alias (rather than ``import *``) keeps legacy monkeypatches
and archived scripts attached to the same globals as :mod:`src.engine.state`.
"""

import sys
from src.engine import state as _implementation

sys.modules[__name__] = _implementation
