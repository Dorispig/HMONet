# Code structure

The reusable implementation lives below `src/`; files below `scripts/` are
command-line adapters only.

```text
src/
├── data/
│   ├── parametric_dataset.py   # current 2-D/3-D parameterized-boundary schema
│   ├── backend/                # batched curve/surface evaluation
│   └── generation/             # parameterized dataset generators
├── models/
│   ├── hmonet.py               # canonical HMONet API
│   ├── meshonet.py             # canonical MeshONet API
│   ├── pi_meshonet.py          # PI-MeshONet implementation
│   ├── modules.py              # shared neural layers
│   └── model.py                # checkpoint-compatible legacy implementation
├── losses/
│   ├── coordinate.py           # supervised coordinate losses
│   ├── physics.py              # named inverse-elliptic-grid PDE losses
│   └── mesh_regularization.py  # geometric regularizers
├── numerics/
│   ├── differential.py         # finite-difference/autograd derivatives
│   └── traditional/            # algebraic TFI and Winslow solvers
├── metrics/                    # geometry, quad/hex and quality metrics
├── evaluation/                 # evaluation orchestration
└── engine/                     # reusable training state and loops
```

Compatibility wrappers retain old import and command paths so archived
checkpoints, manifests, and user scripts remain usable. New code must import
the canonical modules above. `src/data/dataset.py`, `src/utils/loss.py`, and
the old TFI/PDE script modules contain no independent implementation.

The project no longer contains a reader or generator for the oldest
point-only dataset schema. The apparent `dataset.npy` file in a current 2-D
dataset is the mesh-coordinate half of the parameterized pair and must not be
deleted; its matching `boundary_geometry.npy` and metadata establish schema
`parametric_2d_pair_v1`.
