# Test suite

The project test suite uses `pytest`. Run every command below from the project
root after activating the project environment.

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
