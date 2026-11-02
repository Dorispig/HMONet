# Contributing to HMONet

HMONet is research software. Contributions must preserve numerical correctness, experimental traceability, and compatibility with archived results. Before investing in a large change, open an issue describing the scientific motivation and expected validation.

## 1. Development setup

```bash
git clone https://github.com/Dorispig/HMONet.git
cd HMONet
conda create -n HMONet python=3.10 -y
conda activate HMONet
pip install -r requirements-dev.txt
```

See [environment documentation](docs/environment.md) for CUDA-specific installation and platform notes.

## 2. Branches and pull requests

1. Create a focused branch from the current default branch.
2. Keep unrelated formatting, generated data, checkpoints, and experiment outputs out of the change.
3. Add or update tests and documentation with the implementation.
4. Run the checks below before opening a pull request.
5. Complete the scientific-reproducibility section of the pull request template.

Small commits with descriptive messages are preferred. Do not rewrite published experiment history or force-push over shared result branches.

## 3. Required checks

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python -m pytest tests -q
python scripts/experiments/run.py --help
python scripts/train/train_2d.py --help
python scripts/train/train_3d.py --help
python scripts/evaluate/evaluate_2d.py --help
python scripts/evaluate/evaluate_3d.py --help
```

Tests that require a GPU, a large external dataset, or a trained checkpoint must be marked and documented rather than silently skipped. A pull request should not claim full validation when only a smoke test was run.

## 4. Scientific changes

Changes to a model, loss, sampler, numerical derivative, PDE solver, quality metric, target interpolation, or stopping rule must include:

- the mathematical definition and units/scaling;
- the affected two- and three-dimensional paths;
- a deterministic test on a known geometry;
- gradient/finite-difference validation where applicable;
- behavior at boundaries, corners, shared edges, and degenerate cases;
- changes to training and evaluation fingerprints;
- at least one before/after result or a reason why numerical output is unchanged.

Do not change a frozen experiment YAML in place after results have been reported. Create a new experiment or version, and record the change in `CHANGELOG.md`.

## 5. Code organization

Reusable implementation belongs below `src/`; `scripts/` contains command-line adapters only:

```text
src/
├── data/          # parameterized dataset schema, geometry backends, generators
├── models/        # model-specific APIs and shared neural modules
├── losses/        # coordinate, named PDE and mesh-regularization losses
├── numerics/      # derivatives, downstream solvers, TFI/Winslow methods
├── metrics/       # geometry and quad/hex quality metrics
├── evaluation/    # mesh providers, retention, quality and plotting
└── engine/        # reusable training state and loops

scripts/
├── train/
├── evaluate/
├── experiments/
└── downstream/
```

Reusable numerical code must not be duplicated inside an experiment runner. New losses belong in `src/losses/`, metrics in `src/metrics/`, derivatives in `src/numerics/`, and model code in the model-specific module. Keep archived import paths as thin compatibility wrappers when checkpoints or manifests depend on them; wrappers must contain no independent implementation.

The oldest point-only dataset reader/generator is not supported. Current `dataset.npy` is the mesh-coordinate half of the parameterized pair and requires matching `boundary_geometry.npy` plus schema metadata; do not treat it as a standalone legacy dataset.

## 6. Data and artifacts

Do not commit private data, credentials, large datasets, checkpoints, TensorBoard logs, or generated evaluation trees. For a reproducible external artifact, provide:

- generation or download instructions;
- relative path and expected size;
- SHA-256 checksum;
- license and provenance;
- corresponding Git commit and configuration.

Generated data that contains nonconverged PDE samples, nonpositive Jacobians, or boundary inconsistency must be labeled explicitly and must not be presented as valid supervision without justification.

## 7. Documentation and citations

Use concise technical language and define every symbol. Clearly distinguish node counts from cell counts, raw predictions from boundary-postprocessed results, and historical configurations from current defaults. New scientific claims require an appropriate primary reference; do not copy wording, figures, or tables from a publication.

## 8. Reporting problems

Use the GitHub issue templates. Include the smallest reproducing command, configuration, traceback, environment, dataset identity, checkpoint identity, and whether local modifications were present. Remove credentials and personal paths before posting.

By contributing, you agree that your contribution is distributed under the repository's [BSD 3-Clause License](LICENSE).

