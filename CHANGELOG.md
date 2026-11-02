# Changelog

All notable changes to HMONet are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases should use [Semantic Versioning](https://semver.org/) when the public API becomes stable.

## [Unreleased]

### Added

- Initial research-preview implementation of two- and three-dimensional HMONet, MeshONet, PI-MeshONet, PI-HMONet, and Attention-PI-HMONet workflows.
- Parameterized-boundary dataset generation and paired dataset schema.
- TFI and Winslow reference mesh generation.
- Formal training, evaluation, compact result retention, mesh-quality analysis, and dynamic multi-GPU task queues.
- Numbered two-dimensional ablation suites and downstream two-dimensional Poisson benchmark scaffolding.
- Complete command, environment, dataset-generation, reproducibility, multi-GPU, output-retention, checkpoint-compatibility, boundary-catalog, and known-limitations documentation.
- Three-dimensional training and evaluation command references.
- Development dependencies and GitHub continuous-integration templates.

### Changed

- Unified current ablation examples around `scripts/experiments/run.py`.
- Marked obsolete Softplus results and removed evaluation options as historical rather than current behavior.
- Synchronized the HMONet architecture document with the compact 64-wide formal configuration and Branch×Trunk projection coefficients.
- Consolidated user documentation into eight workflow-oriented files and moved generated ablation reports to `reports/ablations/`.

### Notes

- The current repository is a development research preview rather than a tagged stable release or stable public API.
- Large datasets, checkpoints, and generated results are not distributed through Git.
- The preferred paper citation will be added after the associated manuscript receives final author metadata and a persistent identifier.

[Unreleased]: https://github.com/Dorispig/HMONet/commits/main

