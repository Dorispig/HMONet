# Traditional structured-grid methods

This package contains non-neural numerical baselines:

- `tfi_2d.py`, `tfi_3d.py`: transfinite interpolation (algebraic TFI);
- `winslow_2d.py`, `winslow_3d.py`: Winslow elliptic grid solvers;
- `winslow_vectorized_2d.py`: vectorized two-dimensional Winslow solver;
- `tfi_legacy.py`, `winslow_legacy.py`: compatibility implementations used by
  archived datasets and tests.

Command-line wrappers under `scripts/tfi/` and `scripts/pde/` remain available
for historical commands, but new Python code should import this package.
