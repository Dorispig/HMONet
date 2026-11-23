# Test suite

The project test suite uses `pytest`. Run every command below from the project
root after activating the project environment.

Dataset/3-D evaluation regression checks (no trained checkpoints required):

```bash
python -m pytest tests/test_dataset_batch_command.py tests/test_geometry_dataset_consistency.py tests/test_volume_evaluation_scope.py
```

These cover batch selection/revisions, the UGC exact profile, rejection of
serialized-boundary mismatches, and skipping volume calculations while retaining
only requested middle sections. Passing these checks does not certify convergence
of a full training run or global non-overlap of arbitrary volume meshes.

run_e dense PDE cap regression (serialization, NumPy/Torch agreement, exact
CCAP boundaries, model backward passes, dataset/evaluation consistency):

```bash
python -m pytest tests/test_run_pde_caps.py tests/test_run_cap_batch_interpolation.py
python scripts/data/check_run_e_grids.py --grids 21x21x11 61x21x11 51x51x11 --values 0.55 1.50 1.52 1.55
```

The second command solves real volume PDEs and records local hex validity and
prescribed-face failures. It does not create or overwrite a training dataset.

All nine current 3-D families now use independent planar PDE caps and the
original side-surface geometry (TFI in each original parameter chart).
Run the shared surface/dataset regression tests with:

```bash
python -m pytest tests/test_planar_pde_surfaces.py tests/test_current_3d_dataset_generation.py tests/test_dataset_batch_command.py tests/test_geometry_dataset_consistency.py
```

These check exact shared edges, unchanged side geometry, independent cap
serialization, loading without solving a PDE, NumPy/Torch values and coordinate
derivatives, and rejection of old-format or mismatched numeric data. Dataset
name versions default to 1; the saved geometry revision distinguishes the new
parameterization. The run_e default is 61x21x11; the other eight are 11x21x11.

Compare pre-optimization per-geometry interpolation with the batched version
using identical surfaces, weights and queries (including batch=64). Run on an
idle GPU; this checks full outputs/loss/parameter gradients before timing:

```bash
python scripts/benchmark/benchmark_run_cap_batch.py --device cuda:0 --repeats 20 --train-repeats 10 --warmup 3
```

Timing excludes cap construction, optimizer updates, data loading and periodic
full-grid MSE. The JSON report is saved under `reports/3D/run_e_pde_caps/`.

## Install test dependencies

```bash
python -m pip install -r requirements-dev.txt
```

## Common commands

Run the complete suite:

```bash
python -m pytest
```

Run the fast suite and skip long integration tests:

```bash
python -m pytest -m "not slow"
```

Run one file, one test, or tests matching a name:

```bash
python -m pytest tests/test_progress_output.py
python -m pytest tests/test_progress_output.py::test_training_progress_keeps_all_losses_with_short_names
python -m pytest -k "projection and boundary"
```

Useful diagnostic modes:

```bash
python -m pytest -x -vv
python -m pytest --lf
python -m pytest --durations=20
python -m pytest --collect-only -q
```

## Test conventions

- Test files are named `test_*.py`; test functions are named `test_*`.
- Test classes are optional. When used, their names start with `Test` and they
  do not define `__init__`.
- Use plain `assert`, `pytest.raises`, `pytest.warns`, fixtures, and
  `pytest.mark.parametrize` for new tests.
- Use `tmp_path` or `tempfile.TemporaryDirectory` for generated files. Tests
  must not write into formal datasets, checkpoints, experiments, or results.
- Mark end-to-end tests with `@pytest.mark.slow`. The complete suite remains
  the required regression check before publishing or merging major changes.
- CUDA-specific assertions must check device availability and retain a CPU
  path whenever the tested operation supports CPU execution.

GPU数据生成回归：`python -m pytest tests/test_gpu_dataset_generation.py -q`。包含CPU参考数值对照、真实CUDA数值对照和双设备队列模拟；没有CUDA时跳过真实GPU测试。双设备模拟不等于真实多卡硬件验证。
