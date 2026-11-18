"""Train selected datasets, then evaluate the exact checkpoints from this run."""

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.model_selection import parse_model_selection
from src.utils.gpu_queue import (
    GpuQueueError,
    GpuQueueInterrupted,
    GpuTask,
    parse_gpu_ids,
    run_gpu_queue,
)
from src.utils.run_status import log_error, print_completion_summary


# Keep formal queue-log paths within the same portable budget as ablations.
FORMAL_RUN_TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
FORMAL_PATH_LIMIT = 234
FORMAL_SYSTEM_PATH_LIMIT = 260 if os.name == "nt" else 4096


def validate_formal_run_tag(value: str) -> str:
    """Validate the user-visible tag used in formal multi-GPU output paths."""
    value = str(value).strip()
    if not FORMAL_RUN_TAG_PATTERN.fullmatch(value):
        raise ValueError(
            "--run-tag must use letters, digits, '.', '_' or '-', and begin with a letter or digit"
        )
    return value


def _formal_shape_names(value, fallback):
    """Parse optional comma-separated formal evaluation grids for dry runs."""
    if value is None:
        return ["x".join(map(str, shape)) for shape in fallback]
    names = []
    for item in str(value).split(","):
        pieces = item.strip().lower().split("x")
        if len(pieces) < 2 or not all(piece.isdigit() for piece in pieces):
            return ["x".join(map(str, shape)) for shape in fallback]
        names.append("x".join(pieces))
    return names or ["x".join(map(str, shape)) for shape in fallback]


def _formal_values(value, fallback):
    if value is None:
        return list(fallback)
    try:
        return [float(item.strip()) for item in str(value).split(",") if item.strip()]
    except ValueError:
        return list(fallback)


def _formal_grid_names(tokens, fallback, dimension):
    """Return printable shapes from compact or legacy evaluation overrides."""
    compact = _command_values(tokens, "--grids")
    if compact:
        try:
            from src.evaluation.configuration import compact_grid_shapes

            return [
                "x".join(map(str, shape))
                for shape in compact_grid_shapes(compact, dimension)
            ]
        except (TypeError, ValueError):
            return ["x".join(map(str, shape)) for shape in fallback]
    return _formal_shape_names(_command_option(tokens, "--grid-shapes"), fallback)


def _formal_value_list(tokens, fallback):
    """Support both ``--values a b`` and the legacy comma spelling."""
    values = _command_values(tokens, "--values")
    return _formal_values(",".join(values) if values else None, fallback)


def validate_formal_evaluation_request(dimension, tokens, cases, case_names):
    """Validate the final evaluation overrides before any training starts.

    The combined formal entry point deliberately owns ``--case`` and model
    checkpoint selection, but it forwards grid and value overrides to the
    shared evaluator.  Validate those forwarded overrides here as well, so a
    malformed request cannot waste a training run before failing in the
    evaluation phase.
    """
    from src.evaluation.configuration import (
        compact_grid_shapes,
        create_geometry,
        csv_values,
    )

    compact = _command_values(tokens, "--grids")
    legacy = _command_option(tokens, "--grid-shapes")
    if compact and legacy is not None:
        raise ValueError("--grids and --grid-shapes are mutually exclusive")

    requested_shapes = (
        compact_grid_shapes(compact, dimension)
        if compact
        else compact_grid_shapes(legacy, dimension)
        if legacy is not None
        else None
    )
    requested_values = _command_values(tokens, "--values")
    values_override = csv_values(requested_values) if requested_values else None
    if values_override is not None and (
        not values_override or not all(math.isfinite(value) for value in values_override)
    ):
        raise ValueError("--values must contain one or more finite numbers")

    checked = []
    for name in case_names:
        case = cases[name]
        shapes = requested_shapes
        if shapes is None:
            raw_shapes = case.get(
                "grid_shapes", [case.get("grid_shape", case.get("train_grid"))]
            )
            shapes = [tuple(map(int, shape)) for shape in raw_shapes]
        values = (
            values_override
            if values_override is not None
            else case.get("values", [None])
        )
        if not isinstance(values, (list, tuple)):
            values = [values]
        for shape in shapes:
            if len(shape) != dimension or min(shape) < 2:
                raise ValueError(
                    f"Invalid {dimension}-D evaluation grid for {name}: {shape}"
                )
        for value in values:
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"Invalid evaluation value for {name}: {value!r}")
            # Constructing the exact parametric boundary validates that the
            # requested physical parameter is usable without allocating an
            # output mesh or executing a network/PDE solve.
            create_geometry(dimension, case, value)
        checked.append(
            {
                "name": name,
                "grids": ["x".join(map(str, shape)) for shape in shapes],
                "values": [
                    "default" if value is None else f"{float(value):g}"
                    for value in values
                ],
            }
        )
    return checked


def report_formal_dry_run_paths(args, train_extra, evaluate_extra, paths, cases, models):
    """Preview the concrete formal roots without inventing deep suffixes."""
    evaluation_root = Path(
        _command_option(evaluate_extra, "--output-root", "outputs/results")
    )
    if not evaluation_root.is_absolute():
        evaluation_root = ROOT / evaluation_root
    if evaluation_root.name != f"{args.dimension}D":
        evaluation_root /= f"{args.dimension}D"
    train_root = Path(
        _command_option(train_extra, "--output-root", f"outputs/training/{args.dimension}D")
    )
    if not train_root.is_absolute():
        train_root = ROOT / train_root
    configured = yaml.safe_load(
        Path(args.train_config or f"configs/{args.dimension}D/train.yaml").read_text(
            encoding="utf-8"
        )
    ) or {}
    seed = int(_command_option(train_extra, "--seed", configured.get("training", {}).get("seed", 0)))
    mappings = configured.get("model", {}).get("hmonet", {}).get(
        "hmonet_mapping_mode", "corner_corrected_affine"
    )
    model_names = {
        "hmonet": "HMONet", "qghmonet": "QG-HMONet",
        "meshonet": "MeshONet", "pimeshonet": "PI-MeshONet",
    }
    candidates = []
    for path in paths:
        dataset_name = path.parent.name if path.is_file() else path.name
        case = cases[dataset_name]
        for grid in _formal_grid_names(
            evaluate_extra, case.get("grid_shapes", []), args.dimension
        ):
            for value in _formal_value_list(
                evaluate_extra, case.get("values", [])
            ):
                candidates.append((
                    "formal evaluation root",
                    evaluation_root / dataset_name / grid / f"{value:g}",
                ))
        for model in models:
            display = model_names.get(model, model)
            mapping = mappings if model in {"hmonet", "qghmonet"} else None
            training = train_root / dataset_name / display
            if mapping:
                training /= str(mapping)
            candidates.append((
                "formal training",
                training,
            ))
        if args.gpus:
            candidates.append((
                "formal queue log",
                ROOT / "outputs/gpu_queue/formal" / dataset_name / "_".join(models)
                / f"seed_{seed}" / args.run_tag / "train_evaluate.log",
            ))
    warnings = [
        (
            category,
            path,
            len(str(path.resolve())),
            len(str(path.resolve())) >= FORMAL_SYSTEM_PATH_LIMIT
            or any(len(part) > 255 for part in path.parts),
        )
        for category, path in candidates
        if len(str(path.resolve())) > FORMAL_PATH_LIMIT
        or len(str(path.resolve())) >= FORMAL_SYSTEM_PATH_LIMIT
        or any(len(part) > 255 for part in path.parts)
    ]
    print(
        f"[dry-run paths] checked={len(candidates)} | warning-threshold={FORMAL_PATH_LIMIT} | "
        f"system-limit={FORMAL_SYSTEM_PATH_LIMIT} | Linux-component=255",
        flush=True,
    )
    if not warnings:
        print("[dry-run paths] all selected formal output paths are below the warning threshold.", flush=True)
    else:
        for category, path, length, would_fail in warnings:
            print(
                f"[dry-run paths] {'WOULD-FAIL' if would_fail else 'WARNING'} | "
                f"{category} | length={length}, warning-threshold={FORMAL_PATH_LIMIT}, "
                f"system={FORMAL_SYSTEM_PATH_LIMIT} | {path}",
                flush=True,
            )
        print(
            "[dry-run paths] warnings do not stop this dry run. "
            "WOULD-FAIL entries would exceed the real system limit during a run.",
            flush=True,
        )
    return warnings


def normalize_evaluate_methods(value):
    """Normalize an optional subset while preserving the evaluator's method order."""
    if value is None:
        return None
    from src.evaluation.configuration import selected_methods

    return [
        "pde" if name == "PDE_Winslow" else name.lower().replace("-", "")
        for name in selected_methods(value)
    ]


def extract_evaluation_model(evaluate_extra, evaluate_methods=None):
    """Consume a model selector while forwarding remaining evaluation overrides."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--model",
        help="Model alias or comma-separated model list",
    )
    selected, remaining = parser.parse_known_args(evaluate_extra)
    if selected.model is None:
        return normalize_evaluate_methods(evaluate_methods), remaining
    if evaluate_methods is not None:
        raise ValueError(
            "Use either --evaluate-methods or --evaluate-args '--model ...', not both"
        )
    models = parse_model_selection(selected.model)
    return normalize_evaluate_methods(["tfi", "pde"] + models), remaining


def evaluation_commands(
    summaries, config_path, cases, checkpoint, evaluate_methods=None
):
    """Build per-dataset evaluations and pin checkpoints trained in this run.

    Explicitly requested models that were not trained in this invocation are
    left without a checkpoint argument, so the evaluator resolves their latest
    compatible existing checkpoint.
    """
    selected = normalize_evaluate_methods(evaluate_methods)
    grouped = {}
    for summary in summaries:
        name = summary["dataset_name"]
        if name not in cases:
            raise ValueError(f"Missing evaluation case: {name}")
        grouped.setdefault(name, {})[summary["model"].lower()] = summary
    commands = []
    for name, models in grouped.items():
        methods = selected if selected is not None else ["tfi", "pde"] + list(models)
        command = [
            "--config",
            str(config_path),
            "--case",
            name,
            "--methods",
            ",".join(methods),
        ]
        for model, summary in models.items():
            if model not in methods:
                continue
            path = Path(summary[f"{checkpoint}_checkpoint"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            command += [f"--{model}-checkpoint", str(path)]
            if model == "hmonet":
                command += [f"--{model}-mapping-mode", summary["hmonet_mapping_mode"]]
        commands.append(command)
    return commands


_EVALUATION_METHOD_NAMES = {
    "hmonet": "HMONet",
    "qghmonet": "QG-HMONet",
    "meshonet": "MeshONet",
    "pimeshonet": "PI-MeshONet",
}


def _find_existing_checkpoint(dimension, dataset_name, method, mapping, checkpoint):
    """Resolve one compatible saved best/final checkpoint for evaluate-only."""
    from src.models.mapping_modes import normalize_mapping_mode

    root = ROOT / f"outputs/checkpoints/{dimension}D" / dataset_name
    filename = f"{checkpoint}.pt"
    candidates = []
    for path in root.rglob(filename):
        parts = path.relative_to(root).parts
        if method not in parts:
            continue
        if method in {"HMONet", "QG-HMONet"}:
            wanted = normalize_mapping_mode(mapping)
            compatible = []
            for part in parts:
                try:
                    compatible.append(normalize_mapping_mode(part))
                except ValueError:
                    pass
            if wanted not in compatible:
                continue
        candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.parent.name, path.stat().st_mtime_ns))


def existing_evaluation_commands(
    dimension,
    dataset_names,
    config_path,
    evaluation_config,
    models,
    checkpoint,
    evaluate_methods=None,
):
    """Build pinned evaluations from existing formal checkpoints.

    Unlike training-backed commands, this path has no new summary file from
    which to obtain checkpoint locations.  Resolve exactly one compatible
    ``best.pt`` or ``final.pt`` per requested network and pass it explicitly,
    so evaluate-only never silently switches to another checkpoint type.
    """
    selected = normalize_evaluate_methods(evaluate_methods)
    mapping = evaluation_config.get("evaluation", {}).get(
        "hmonet_mapping_mode", "corner_corrected_affine"
    )
    commands = []
    for name in dataset_names:
        methods = selected if selected is not None else ["tfi", "pde", *models]
        command = ["--config", str(config_path), "--case", name, "--methods", ",".join(methods)]
        for alias in methods:
            method = _EVALUATION_METHOD_NAMES.get(alias)
            if method is None:
                continue
            path = _find_existing_checkpoint(
                dimension, name, method, mapping, checkpoint
            )
            if path is None:
                raise FileNotFoundError(
                    f"No compatible {checkpoint}.pt for {method}/{mapping}: "
                    f"outputs/checkpoints/{dimension}D/{name}"
                )
            command += [f"--{alias}-checkpoint", str(path.resolve())]
            if alias == "hmonet":
                command += ["--hmonet-mapping-mode", str(mapping)]
        commands.append(command)
    return commands


def _without_value_options(tokens, names):
    """Remove selected one-value options from a forwarded CLI token list."""
    names = set(names)
    result = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        key = token.split("=", 1)[0]
        if key in names:
            if "=" not in token:
                if index + 1 >= len(tokens):
                    raise ValueError(f"Missing value for {token}")
                index += 2
            else:
                index += 1
            continue
        result.append(token)
        index += 1
    return result


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _formal_queue_fingerprint(
    *,
    dimension,
    dataset,
    models,
    train_config,
    evaluate_config,
    train_arguments,
    evaluate_arguments,
    evaluate_methods,
    checkpoint,
):
    """Create a stable identity for one formal train-and-evaluate queue task."""
    dataset = Path(dataset).resolve()

    def config_digest(value):
        path = Path(value)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        return hashlib.sha256(path.read_bytes()).hexdigest()

    if dataset.is_dir():
        data_files = sorted(dataset.glob("*.npy"))
    else:
        data_files = [dataset]
    dataset_signature = [
        {
            "name": item.name,
            "size": item.stat().st_size,
            "modified_ns": item.stat().st_mtime_ns,
        }
        for item in data_files
    ]
    payload = {
        "dimension": int(dimension),
        "dataset": str(dataset),
        "dataset_signature": dataset_signature,
        "models": list(models),
        "train_config_sha256": config_digest(train_config),
        "evaluate_config_sha256": config_digest(evaluate_config),
        "train_arguments": list(train_arguments),
        "evaluate_arguments": list(evaluate_arguments),
        "evaluate_methods": list(evaluate_methods or []),
        "checkpoint": str(checkpoint),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _command_option(tokens, name, default=None):
    """Read one ``--name value`` or ``--name=value`` option."""
    for index, token in enumerate(tokens):
        if token == name:
            if index + 1 >= len(tokens):
                raise ValueError(f"Missing value for {name}")
            return tokens[index + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return default


def _command_values(tokens, name):
    """Read all values following one option until the next CLI switch."""
    for index, token in enumerate(tokens):
        if token.startswith(name + "="):
            return [token.split("=", 1)[1]]
        if token == name:
            values = []
            for candidate in tokens[index + 1 :]:
                if candidate.startswith("--"):
                    break
                values.append(candidate)
            return values
    return []


def _option_present(tokens, name):
    return any(token == name or token.startswith(name + "=") for token in tokens)


def _append_simple_option(parser, tokens, name, values, *, source):
    """Append a friendly top-level option unless advanced args already set it."""
    if values is None:
        return
    if _option_present(tokens, name):
        parser.error(
            f"Use either {source} or {name} inside the advanced argument string, not both"
        )
    if not isinstance(values, (list, tuple)):
        values = [values]
    tokens.extend([name, *map(str, values)])


def _infer_dataset_mode(value):
    """Infer a split from the stable ``*_int_*``/``*_ext_*`` dataset name."""
    path = Path(value)
    name = path.parent.name if path.name == "dataset.npy" else path.name
    if "_int_" in name:
        return "interpolation"
    if "_ext_" in name:
        return "extrapolation"
    return None


def _queue_status_path(result_path: Path) -> Path:
    """Store worker phase state beside, but separately from, final summaries."""
    return Path(result_path).with_name("queue_status.json")


def _write_queue_worker_status(result_path: Path, **payload) -> None:
    """Persist a small, readable train/evaluate status for the queue parent."""
    path = _queue_status_path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _formal_queue_process_rows(records):
    """Render per-model formal queue tasks with separate phase outcomes."""
    rows = []
    for record in records:
        metadata = dict(record.get("metadata") or {})
        status_path = metadata.get("queue_status_path")
        try:
            state = json.loads(Path(status_path).read_text(encoding="utf-8"))
            state = state if isinstance(state, dict) else {}
        except (OSError, TypeError, ValueError):
            state = {}
        status = str(record.get("status", "unknown"))
        training = state.get("training")
        evaluation = state.get("evaluation")
        if status == "dry_run":
            training = evaluation = "dry_run"
        elif status == "completed":
            training = training or "completed"
            evaluation = evaluation or "completed"
        elif status in {"failed", "interrupted", "cancelled"}:
            training = training or "not-started"
            evaluation = evaluation or "not-started"
        else:
            training = training or "unknown"
            evaluation = evaluation or "unknown"
        error = (
            state.get("error")
            or record.get("error")
            or (log_error(record.get("log")) if status != "completed" else None)
        )
        rows.append(
            {
                "task": record.get("task", "formal-task"),
                "status": status,
                "dataset": metadata.get("dataset"),
                "variant": metadata.get("experiment"),
                "seed": metadata.get("seed"),
                "training": training,
                "evaluation": evaluation,
                "error": error,
                "log": record.get("log"),
            }
        )
    return rows


def _formal_local_process_rows(summaries, *, training, evaluation, error=None):
    """Describe serial formal-model runs with the same columns as GPU jobs."""
    rows = []
    for summary in summaries:
        rows.append(
            {
                "task": f"train_evaluate__{summary.get('dataset_name', 'dataset')}__{summary.get('model', 'model')}",
                "status": "failed" if error else "completed",
                "dataset": summary.get("dataset_name"),
                "variant": summary.get("model"),
                "seed": summary.get("seed"),
                "training": training,
                "evaluation": evaluation,
                "error": error,
            }
        )
    if not rows and error:
        rows.append(
            {
                "task": "train_evaluate",
                "status": "failed",
                "training": training,
                "evaluation": evaluation,
                "error": repr(error),
            }
        )
    return rows


def _write_formal_completion_summary(queue_root: Path, payload: dict) -> None:
    """Keep the latest queue-level completion view beside queue_summary.json."""
    Path(queue_root).mkdir(parents=True, exist_ok=True)
    (Path(queue_root) / "completion_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _formal_evaluation_root(dimension: int, evaluate_extra) -> Path:
    """Resolve the canonical root used by the shared evaluator."""
    root = Path(_command_option(evaluate_extra, "--output-root", "outputs/results"))
    if not root.is_absolute():
        root = ROOT / root
    if root.name != f"{dimension}D":
        root /= f"{dimension}D"
    return root


def _keep_uncompacted_evaluation_outputs(tokens):
    """Preserve generated meshes and method analyses for queue consolidation."""
    if _option_present(tokens, "--compact-outputs") or _option_present(
        tokens, "--no-compact-outputs"
    ):
        return list(tokens)
    return [*tokens, "--no-compact-outputs"]


def _cache_signature(directory: Path):
    """Read a method-generation signature without treating absent metadata as valid."""
    metadata_file = Path(directory) / "generation_metadata.json"
    if not metadata_file.is_file():
        return None
    try:
        return json.loads(metadata_file.read_text(encoding="utf-8")).get("signature")
    except (OSError, json.JSONDecodeError):
        return None


def _analysis_signature(directory: Path):
    """Read a method-analysis signature without accepting incomplete metadata."""
    metadata_file = Path(directory) / "analysis_metadata.json"
    if not metadata_file.is_file():
        return None
    try:
        return json.loads(metadata_file.read_text(encoding="utf-8")).get("signature")
    except (OSError, json.JSONDecodeError):
        return None


def promote_worker_evaluation_caches(dimension: int, staging_roots, canonical_root: Path):
    """Publish matching per-method worker artifacts for the final comparison.

    Each GPU worker writes a complete single-model evaluation under a private
    root.  The final canonical evaluation has the same ``case/grid/value``
    layout, so both mesh-generation caches and single-method quality analyses
    can be populated without sharing writable directories between workers.
    Only valid signed artifacts are copied; a conflicting canonical artifact
    is left untouched and will be regenerated normally by the evaluator.
    """
    canonical_root = Path(canonical_root)
    copied = reused = conflicts = missing = 0
    analysis_copied = analysis_reused = analysis_conflicts = 0
    seen_caches = set()
    seen_analyses = set()
    for staging_root in map(Path, staging_roots):
        worker_root = staging_root / f"{dimension}D"
        if not worker_root.is_dir():
            missing += 1
            continue
        for result_file in worker_root.rglob("results.json"):
            source_case_root = result_file.parent
            source_cache_root = source_case_root / ".cache"
            if not source_cache_root.is_dir():
                missing += 1
                continue
            relative_case_root = source_case_root.relative_to(worker_root)
            target_case_root = canonical_root / relative_case_root
            target_cache_root = canonical_root / relative_case_root / ".cache"
            for source_metadata in source_cache_root.rglob("generation_metadata.json"):
                source_cache = source_metadata.parent
                relative_cache = source_cache.relative_to(source_cache_root)
                target_cache = target_cache_root / relative_cache
                key = (str(target_cache.resolve()).lower(), _cache_signature(source_cache))
                if key in seen_caches:
                    reused += 1
                    continue
                seen_caches.add(key)
                source_signature = _cache_signature(source_cache)
                if source_signature is None or not (source_cache / "generation.npz").is_file():
                    missing += 1
                    continue
                target_signature = _cache_signature(target_cache)
                if target_cache.exists() and target_signature not in (None, source_signature):
                    conflicts += 1
                    continue
                if target_signature == source_signature and (target_cache / "generation.npz").is_file():
                    reused += 1
                    continue
                target_cache.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source_cache, target_cache, dirs_exist_ok=True)
                copied += 1
            for source_metadata in source_case_root.rglob("analysis_metadata.json"):
                source_analysis = source_metadata.parent
                relative_analysis = source_analysis.relative_to(source_case_root)
                # Only method-level artifacts are reusable.  A worker's
                # comparison group contains a different method set and is
                # deliberately rebuilt once by the canonical joint stage.
                if relative_analysis.parts[0] in {".cache", "comparisons"}:
                    continue
                target_analysis = target_case_root / relative_analysis
                source_signature = _analysis_signature(source_analysis)
                key = (str(target_analysis.resolve()).lower(), source_signature)
                if key in seen_analyses:
                    analysis_reused += 1
                    continue
                seen_analyses.add(key)
                if source_signature is None:
                    missing += 1
                    continue
                target_signature = _analysis_signature(target_analysis)
                if target_analysis.exists() and target_signature not in (None, source_signature):
                    analysis_conflicts += 1
                    continue
                if target_signature == source_signature:
                    analysis_reused += 1
                    continue
                target_analysis.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source_analysis, target_analysis, dirs_exist_ok=True)
                analysis_copied += 1
    details = {
        "generation_copied": copied,
        "generation_already_available": reused,
        "generation_conflicts": conflicts,
        "analysis_copied": analysis_copied,
        "analysis_already_available": analysis_reused,
        "analysis_conflicts": analysis_conflicts,
        "missing": missing,
    }
    print(
        "[formal evaluation cache] "
        f"generation_copied={copied}, generation_reused={reused}, "
        f"analysis_copied={analysis_copied}, analysis_reused={analysis_reused}, "
        f"conflicts={conflicts + analysis_conflicts}, missing={missing}",
        flush=True,
    )
    return details


def _queued_formal_run(
    args,
    train_extra,
    evaluate_extra,
    models,
    paths,
    evaluate_config,
    cases,
    selected_methods,
    resolved_dataset_mode,
):
    """Queue independent ``dataset × model`` training-and-evaluation tasks.

    A single formal dataset normally compares several models.  Splitting that
    comparison into model-specific jobs makes all requested GPUs useful while
    retaining the required train-then-evaluate lifecycle for every model.
    Worker evaluations write into private staging roots, so simultaneous jobs
    never contend for the canonical comparison directory.  After the queue
    finishes, one consolidated evaluation writes the usual multi-model result.
    """
    import shlex

    gpu_ids = parse_gpu_ids(args.gpus)
    queue_root = Path(args.queue_log_root or ROOT / "outputs/gpu_queue/formal")
    stripped = _without_value_options(
        train_extra, {"--model", "--dataset", "--dataset-mode", "--device"}
    )
    requested_mode = next(
        (
            train_extra[index + 1]
            for index, token in enumerate(train_extra[:-1])
            if token == "--dataset-mode"
        ),
        resolved_dataset_mode,
    )
    tasks = []
    result_files = []
    script = ROOT / "scripts/train/train_evaluate.py"
    # A worker uses a private evaluation root.  The canonical root is written
    # only by the post-queue consolidation below, after all workers finish.
    # Keep private artifacts intact so their generated meshes can be promoted
    # into the canonical cache without rerunning inference or PDE generation.
    worker_evaluation_extra = _without_value_options(
        evaluate_extra, {"--device", "--output-root"}
    )
    worker_evaluation_extra = _keep_uncompacted_evaluation_outputs(
        worker_evaluation_extra
    )
    for path in paths:
        dataset_name = path.parent.name if path.is_file() else path.name
        task_mode = requested_mode
        if task_mode in (None, "both"):
            task_mode = (
                "interpolation"
                if "_int_" in dataset_name
                else "extrapolation" if "_ext_" in dataset_name else None
            )
        if task_mode not in ("interpolation", "extrapolation"):
            raise ValueError(
                f"Cannot infer interpolation/extrapolation mode for {dataset_name!r}; "
                "pass --train-args '--dataset-mode interpolation' (or extrapolation)."
            )
        train_config = args.train_config or f"configs/{args.dimension}D/train.yaml"
        evaluation_config = (
            args.evaluate_config or f"configs/{args.dimension}D/evaluate.yaml"
        )
        configured = yaml.safe_load(
            (ROOT / train_config).read_text(encoding="utf-8-sig")
            if not Path(train_config).is_absolute()
            else Path(train_config).read_text(encoding="utf-8-sig")
        ) or {}
        for model in models:
            worker_extra = [
                *stripped,
                "--model",
                model,
                "--dataset",
                str(path),
                "--dataset-mode",
                task_mode,
                "--device",
                "cuda:0",
            ]
            seed = int(
                _command_option(
                    worker_extra,
                    "--seed",
                    configured.get("training", {}).get("seed", 0),
                )
            )
            task_directory = (
                Path(dataset_name) / model / f"seed_{seed}" / args.run_tag
            )
            result_file = queue_root / task_directory / "queue_result.json"
            status_file = _queue_status_path(result_file)
            staging_root = queue_root / ".evaluation_work" / task_directory
            worker_evaluate_arguments = [
                *worker_evaluation_extra,
                "--device",
                "cuda:0",
                "--output-root",
                str(staging_root),
            ]
            fingerprint = _formal_queue_fingerprint(
                dimension=args.dimension,
                dataset=path,
                models=[model],
                train_config=train_config,
                evaluate_config=evaluation_config,
                train_arguments=worker_extra,
                evaluate_arguments=worker_evaluate_arguments,
                # Worker evaluation always includes its own model plus TFI/PDE.
                evaluate_methods=None,
                checkpoint=args.checkpoint,
            )
            if not args.dry_run:
                result_file.parent.mkdir(parents=True, exist_ok=True)
            result_files.append(result_file)
            command = [
                sys.executable,
                str(script),
                "--dimension",
                str(args.dimension),
                "--train-config",
                str(train_config),
                "--evaluate-config",
                str(evaluation_config),
                "--checkpoint",
                args.checkpoint,
                "--train-args",
                shlex.join(worker_extra),
                "--evaluate-args",
                shlex.join(worker_evaluate_arguments),
                "--queue-worker",
                "--queue-result",
                str(result_file),
            ]
            tasks.append(
                GpuTask(
                    name=f"train_evaluate__{dataset_name}__{model}",
                    command=command,
                    log_relative_path=task_directory / "train_evaluate.log",
                    clear_log_directory=True,
                    metadata={
                        "dataset": dataset_name,
                        "experiment": model,
                        "seed": seed,
                        "run_tag": args.run_tag,
                        "task_fingerprint": fingerprint,
                        "queue_result_path": str(result_file),
                        "queue_status_path": str(status_file),
                        "staging_root": str(staging_root),
                    },
                )
            )
    try:
        task_records = run_gpu_queue(
            tasks,
            gpu_ids,
            cwd=ROOT,
            log_root=queue_root,
            continue_on_error=args.continue_on_error,
            dry_run=args.dry_run,
            terminal_interval=args.queue_terminal_interval,
        )
    except (GpuQueueError, GpuQueueInterrupted) as error:
        task_records = error.records
        completion = {
            "status": "interrupted" if isinstance(error, GpuQueueInterrupted) else "failed",
            "gpu_ids": gpu_ids,
            "tasks": task_records,
            "processes": _formal_queue_process_rows(task_records),
            "error": str(error),
        }
        print_completion_summary(
            completion["processes"], title="Formal GPU queue completion summary"
        )
        _write_formal_completion_summary(queue_root, completion)
        raise
    if args.dry_run:
        print_completion_summary(
            _formal_queue_process_rows(task_records),
            title="Formal GPU queue dry-run summary",
        )
        return []
    summaries = []
    for result_file in result_files:
        if result_file.is_file():
            summaries.extend(json.loads(result_file.read_text(encoding="utf-8")))
    failed = [record for record in task_records if record["status"] == "failed"]
    consolidation = "not-run"
    cache_promotion = None
    if summaries:
        # Publish the immutable generation artifacts from private worker roots
        # before one canonical multi-model comparison.  The evaluator then
        # cache-hits HMONet/MeshONet/PI-MeshONet/TFI/PDE meshes and performs
        # only the cross-method summaries and comparison plots.
        cache_promotion = promote_worker_evaluation_caches(
            args.dimension,
            [record["metadata"]["staging_root"] for record in task_records if record["status"] == "completed"],
            _formal_evaluation_root(args.dimension, evaluate_extra),
        )
        evaluate = importlib.import_module(f"scripts.evaluate.evaluate_{args.dimension}d")
        consolidated_evaluate_extra = _without_value_options(
            evaluate_extra, {"--device"}
        )
        consolidated_evaluate_extra = _keep_uncompacted_evaluation_outputs(
            consolidated_evaluate_extra
        )
        consolidated_evaluate_extra += ["--device", f"cuda:{gpu_ids[0]}"]
        try:
            for command in evaluation_commands(
                summaries, evaluate_config, cases, args.checkpoint, selected_methods
            ):
                evaluate.main(command + consolidated_evaluate_extra)
            consolidation = "completed"
        except Exception as error:
            consolidation = "failed"
            completion = {
                "status": "failed",
                "gpu_ids": gpu_ids,
                "tasks": task_records,
                "processes": _formal_queue_process_rows(task_records),
                "consolidation": consolidation,
                "evaluation_cache_promotion": cache_promotion,
                "error": repr(error),
            }
            print_completion_summary(
                completion["processes"], title="Formal GPU queue completion summary"
            )
            _write_formal_completion_summary(queue_root, completion)
            raise
    completion = {
        "status": "completed_with_failures" if failed else "completed",
        "gpu_ids": gpu_ids,
        "tasks": task_records,
        "processes": _formal_queue_process_rows(task_records),
        "consolidation": consolidation,
        "evaluation_cache_promotion": cache_promotion,
    }
    print_completion_summary(
        completion["processes"], title="Formal GPU queue completion summary"
    )
    _write_formal_completion_summary(queue_root, completion)
    if failed and not summaries:
        return []
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=[2, 3], default=2)
    parser.add_argument("--train-config")
    parser.add_argument("--evaluate-config")
    parser.add_argument(
        "--dataset",
        help=(
            "Dataset directory name (for example swb_a_ext_1_500_11_21_11), "
            "a project-relative path, or an absolute path."
        ),
    )
    parser.add_argument(
        "--dataset-mode",
        choices=["interpolation", "extrapolation", "both"],
        help="Training split when --dataset is omitted; inferred from a named dataset otherwise.",
    )
    parser.add_argument(
        "--models",
        "--model",
        dest="models",
        nargs="+",
        help="Models to train, for example hmonet meshonet pimeshonet; aliases both, pi and all are supported.",
    )
    parser.add_argument("--seed", type=int, help="Override the training seed.")
    parser.add_argument("--device", help="Training/evaluation device, for example cuda:0 or cpu.")
    parser.add_argument("--epochs", type=int, help="Override the maximum training epochs.")
    parser.add_argument("--batch-size", type=int, help="Override the training batch size.")
    parser.add_argument("--lr", type=float, help="Override each selected model's initial learning rate.")
    grid_group = parser.add_mutually_exclusive_group()
    grid_group.add_argument(
        "--grids",
        nargs="+",
        metavar="N_OR_SHAPE",
        help=(
            "Evaluation sizes or explicit shapes, for example 101 201 401 or "
            "11x31x11 21x61x21; scalar N expands to NxN or NxNxN."
        ),
    )
    grid_group.add_argument(
        "--grid-shapes",
        help=(
            "Legacy explicit evaluation grids, for example 101x101,201x201 "
            "or 11x31x11,21x61x21."
        ),
    )
    parser.add_argument(
        "--values",
        nargs="+",
        help="Evaluation parameter values, for example --values 0.62 0.65.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate selected datasets, cases, grids and values without training.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help=(
            "Skip training and evaluate existing compatible checkpoints. "
            "--checkpoint selects best.pt or final.pt."
        ),
    )
    parser.add_argument(
        "--evaluate-methods",
        help="Comma-separated evaluation subset, e.g. tfi,pde,meshonet,hmonet; defaults to selected models plus TFI/PDE",
    )
    parser.add_argument("--checkpoint", choices=["best", "final"], default="best")
    parser.add_argument(
        "--run-tag",
        "--tag",
        dest="run_tag",
        default="default",
        help="Readable queue-log run tag (default: default).",
    )
    parser.add_argument(
        "--train-args", default="", help="Quoted training CLI overrides"
    )
    parser.add_argument(
        "--evaluate-args",
        default="",
        help="Quoted evaluation CLI overrides, including --model; cases and checkpoints remain automatic",
    )
    parser.add_argument(
        "--quality-plots",
        nargs="+",
        metavar="METRIC",
        help=(
            "Optional per-metric quality figures for evaluation: 'all' or "
            "named metrics. Default: numeric summaries and mesh figures only."
        ),
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        help="Dynamic GPU queue, e.g. --gpus 0 1 2 3 or --gpus 0,1,2,3.",
    )
    parser.add_argument("--queue-log-root")
    parser.add_argument(
        "--queue-terminal-interval",
        type=int,
        default=500,
        help="Print one complete queued epoch summary every N epochs.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration and reanalysis during the evaluation stage.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--queue-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--queue-result", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.run_tag = validate_formal_run_tag(args.run_tag)
    import shlex

    train_extra = shlex.split(args.train_args, posix=True)
    evaluate_extra = shlex.split(args.evaluate_args, posix=True)
    _append_simple_option(
        parser, train_extra, "--dataset", args.dataset, source="--dataset"
    )
    requested_mode = args.dataset_mode
    if requested_mode is None and args.dataset is not None:
        requested_mode = _infer_dataset_mode(args.dataset)
    _append_simple_option(
        parser,
        train_extra,
        "--dataset-mode",
        requested_mode,
        source="--dataset-mode",
    )
    _append_simple_option(
        parser,
        train_extra,
        "--model",
        ",".join(args.models) if args.models else None,
        source="--models/--model",
    )
    for option, value in (
        ("--seed", args.seed),
        ("--device", args.device),
        ("--epochs", args.epochs),
        ("--batch-size", args.batch_size),
        ("--lr", args.lr),
    ):
        _append_simple_option(parser, train_extra, option, value, source=option)
    _append_simple_option(
        parser, evaluate_extra, "--grids", args.grids, source="--grids"
    )
    _append_simple_option(
        parser,
        evaluate_extra,
        "--grid-shapes",
        args.grid_shapes,
        source="--grid-shapes",
    )
    _append_simple_option(
        parser, evaluate_extra, "--values", args.values, source="--values"
    )
    if _option_present(evaluate_extra, "--grids") and _option_present(
        evaluate_extra, "--grid-shapes"
    ):
        parser.error("--grids and --grid-shapes are mutually exclusive")
    if args.force:
        if _option_present(evaluate_extra, "--force"):
            parser.error("Use --force either directly or inside --evaluate-args, not both")
        evaluate_extra.append("--force")
    if args.quality_plots:
        if any(token.split("=", 1)[0] == "--quality-plots" for token in evaluate_extra):
            parser.error(
                "Use --quality-plots either directly or inside --evaluate-args, not both"
            )
        evaluate_extra += ["--quality-plots", *args.quality_plots]
    try:
        selected, evaluate_extra = extract_evaluation_model(
            evaluate_extra, args.evaluate_methods
        )
    except ValueError as error:
        parser.error(str(error))
    if any(token.split("=", 1)[0] == "--config" for token in train_extra):
        parser.error("Use --train-config instead of --config inside --train-args")
    reserved = {
        "--config",
        "--case",
        "--all-cases",
        "--dataset",
        "--methods",
        "--model",
        "--checkpoint",
        "--hmonet-checkpoint",
        "--qghmonet-checkpoint",
        "--meshonet-checkpoint",
        "--pimeshonet-checkpoint",
        "--hmonet-mapping-mode",
    }
    if any(token.split("=", 1)[0] in reserved for token in evaluate_extra):
        parser.error(
            "Evaluation cases, methods and checkpoints are supplied by this pipeline"
        )
    os.chdir(ROOT)
    train_config = Path(args.train_config or f"configs/{args.dimension}D/train.yaml")
    evaluate_config = Path(
        args.evaluate_config or f"configs/{args.dimension}D/evaluate.yaml"
    )
    config = yaml.safe_load(train_config.read_text(encoding="utf-8"))
    evaluation_config = yaml.safe_load(evaluate_config.read_text(encoding="utf-8"))
    from src.utils.config_validation import validate_evaluate_config

    validate_evaluate_config(evaluation_config, dimension=args.dimension)
    cases = evaluation_config["cases"]
    # Parse dataset/grid overrides before expensive training begins.
    preflight = argparse.ArgumentParser(add_help=False)
    preflight.add_argument(
        "--model",
        help="Model alias or comma-separated model list",
    )
    preflight.add_argument("--dataset")
    preflight.add_argument("--dataset-mode")
    for axis in ("nx", "ny", "nz"):
        preflight.add_argument(f"--{axis}", type=int)
    overrides, _ = preflight.parse_known_args(train_extra)
    choice = overrides.model or config.get("model", {}).get("name", "hmonet")
    trained = set(parse_model_selection(choice))
    # Explicit evaluation methods may include an already trained model. Models
    # trained in this invocation are pinned to their new checkpoints; other
    # requested models are resolved by the evaluator from existing checkpoints.
    from src.engine.training import resolve_dataset_paths, resolve_explicit_dataset_path
    from src.data.parametric_dataset import ParametricDataset2D, ParametricDataset3D

    mode = overrides.dataset_mode or config.get("training", {}).get(
        "dataset", "interpolation"
    )
    paths = (
        [
            resolve_explicit_dataset_path(
                overrides.dataset, dimension=args.dimension, project_root=ROOT
            )
        ]
        if overrides.dataset
        else resolve_dataset_paths(config, mode, ROOT)
    )
    if not paths:
        raise ValueError("No training datasets selected")
    dataset_class = ParametricDataset2D if args.dimension == 2 else ParametricDataset3D
    for path in paths:
        data = dataset_class(path, load_boundary=False)
        for axis, actual in zip(
            ("nx", "ny", "nz")[: args.dimension], data.grid_shape
        ):
            requested = getattr(overrides, axis)
            if requested is not None and int(requested) != int(actual):
                raise ValueError(
                    f"Dataset {path}: explicit --{axis}={requested} differs from "
                    f"saved grid {data.grid_shape}"
                )
        name = Path(path).parent.name if Path(path).is_file() else Path(path).name
        if name not in cases:
            raise ValueError(
                f"Add an evaluation case for {name} before running the pipeline"
            )
        print(
            f"Validated: {name}, {len(data)} samples, grid={data.grid_shape}",
            flush=True,
        )
        del data
    try:
        validation = validate_formal_evaluation_request(
            args.dimension,
            evaluate_extra,
            cases,
            [
                Path(path).parent.name if Path(path).is_file() else Path(path).name
                for path in paths
            ],
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    for record in validation:
        print(
            "Validated evaluation: "
            f"{record['name']}, grids={','.join(record['grids'])}, "
            f"values={','.join(record['values'])}",
            flush=True,
        )
    if args.dry_run:
        report_formal_dry_run_paths(
            args, train_extra, evaluate_extra, paths, cases, sorted(trained)
        )
        return []
    if args.check_only:
        return []
    if args.gpus and args.queue_worker:
        parser.error("--gpus cannot be combined with the internal queue worker")
    if args.evaluate_only:
        if args.queue_worker:
            parser.error("--evaluate-only cannot be combined with the internal queue worker")
        if args.gpus:
            parser.error(
                "--evaluate-only currently runs one canonical evaluation; use --device instead of --gpus"
            )
        evaluate = importlib.import_module(
            f"scripts.evaluate.evaluate_{args.dimension}d"
        )
        dataset_names = [
            Path(path).parent.name if Path(path).is_file() else Path(path).name
            for path in paths
        ]
        commands = existing_evaluation_commands(
            args.dimension,
            dataset_names,
            evaluate_config,
            evaluation_config,
            sorted(trained),
            args.checkpoint,
            selected,
        )
        evaluate_only_extra = _keep_uncompacted_evaluation_outputs(evaluate_extra)
        print(
            "Evaluate-only: reusing existing checkpoints; "
            "complete method artifacts will be retained.",
            flush=True,
        )
        try:
            records = [evaluate.main(command + evaluate_only_extra) for command in commands]
        except Exception as error:
            print_completion_summary(
                _formal_local_process_rows(
                    [
                        {"dataset_name": name, "model": model, "seed": args.seed}
                        for name in dataset_names
                        for model in sorted(trained)
                    ],
                    training="not-run",
                    evaluation="failed",
                    error=repr(error),
                ),
                title="Formal evaluate-only completion summary",
            )
            raise
        print_completion_summary(
            _formal_local_process_rows(
                [
                    {"dataset_name": name, "model": model, "seed": args.seed}
                    for name in dataset_names
                    for model in sorted(trained)
                ],
                training="not-run",
                evaluation="completed",
            ),
            title="Formal evaluate-only completion summary",
        )
        return records
    if args.gpus:
        return _queued_formal_run(
            args,
            train_extra,
            evaluate_extra,
            sorted(trained),
            paths,
            evaluate_config,
            cases,
            selected,
            mode,
        )
    train = importlib.import_module(f"scripts.train.train_{args.dimension}d")
    evaluate = importlib.import_module(f"scripts.evaluate.evaluate_{args.dimension}d")
    if args.queue_worker:
        if not args.queue_result:
            parser.error("Internal queue workers require --queue-result")
        result_path = Path(args.queue_result)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.unlink(missing_ok=True)
        _queue_status_path(result_path).unlink(missing_ok=True)
        _write_queue_worker_status(
            result_path, training="running", evaluation="not-started"
        )
    else:
        result_path = None
    summaries = []
    try:
        summaries = train.main(["--config", str(train_config)] + train_extra)
        if result_path is not None:
            _write_queue_worker_status(
                result_path,
                training="completed",
                evaluation="running",
                summaries=summaries,
            )
        for command in evaluation_commands(
            summaries, evaluate_config, cases, args.checkpoint, selected
        ):
            evaluate.main(command + evaluate_extra)
    except Exception as error:
        if result_path is not None:
            _write_queue_worker_status(
                result_path,
                training="completed" if summaries else "failed",
                evaluation="failed" if summaries else "not-started",
                summaries=summaries,
                error=repr(error),
                traceback=traceback.format_exc(),
            )
        elif not args.queue_worker:
            print_completion_summary(
                _formal_local_process_rows(
                    summaries,
                    training="completed" if summaries else "failed",
                    evaluation="failed" if summaries else "not-started",
                    error=repr(error),
                ),
                title="Formal serial completion summary",
            )
        raise
    if result_path is not None:
        result_path.write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        _write_queue_worker_status(
            result_path,
            training="completed",
            evaluation="completed",
            summaries=summaries,
        )
    elif not args.queue_worker:
        print_completion_summary(
            _formal_local_process_rows(
                summaries, training="completed", evaluation="completed"
            ),
            title="Formal serial completion summary",
        )
    return summaries


if __name__ == "__main__":
    main()
