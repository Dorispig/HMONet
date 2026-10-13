"""Load matching checkpoints and time network forward passes independently."""

import inspect
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .artifacts import file_digest, read_json
from .configuration import ROOT, resolve_path
from src.data.mesh.sampling import sample_boundary
from src.data.backend.curve_kernels import curve_kernel_mode, kernel_status
from src.models.model import (
    HMONet2D, MeshONet2D, HMONet3D, MeshONet3D,
    AttentionHMONet2D, AttentionHMONet3D,
    PIMeshONet2D, PIMeshONet3D,
)
from scripts.train.attention_pihmonet_common import build_boundary_inputs


def resolve_device(value):
    value = str(value)
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def find_checkpoint(dimension, dataset, method, mapping, options, split=None):
    """Filter model and mapping before choosing the most recent checkpoint."""
    explicit = options.get(
        f"{method.lower().replace('-', '')}_checkpoint"
    ) or options.get("checkpoint")
    if isinstance(explicit, dict):
        explicit = explicit.get(mapping)
    if explicit:
        path = resolve_path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Explicit {method} checkpoint not found: {path}")
        return path
    root = ROOT / f"outputs/checkpoints/{dimension}D" / dataset
    candidates = []
    for path in root.rglob("best.pt"):
        parts = path.relative_to(root).parts
        if method not in parts:
            continue
        if method in ("HMONet", "PI-HMONet", "Attention-PI-HMONet") and mapping not in parts:
            continue
        candidates.append(path)
    return (
        max(candidates, key=lambda p: (p.parent.name, p.stat().st_mtime_ns))
        if candidates
        else None
    )


def load_network(path, method, dimension, mapping, device, train_config=None):
    """Support full .pt checkpoints and .pth weights with adjacent parameters."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config") if isinstance(payload, dict) else None
    if not config:
        config = read_json(path.parent / "config.json")
    if not config and (path.parent / "config.yaml").exists():
        config = yaml.safe_load(
            (path.parent / "config.yaml").read_text(encoding="utf-8-sig")
        )
    if not config and train_config:
        config = yaml.safe_load(
            resolve_path(train_config).read_text(encoding="utf-8-sig")
        )
    if not config:
        raise ValueError(f"Missing model/grid configuration for {path}")
    stated = (
        payload.get("model_name", method)
        if "model_state_dict" in payload
        else config.get("model", {}).get("name", method)
    )
    normalize = lambda value: "".join(ch for ch in str(value).lower() if ch.isalnum())
    if normalize(stated) != normalize(method):
        raise ValueError(f"Checkpoint is {stated}, requested {method}: {path}")
    grid = payload.get("grid", config.get("grid"))
    shape = (
        tuple(int(grid[k]) for k in ("nx", "ny", "nz")[:dimension])
        if isinstance(grid, dict)
        else tuple(grid)
    )
    if len(shape) != dimension:
        raise ValueError("Checkpoint dimension does not match evaluation")
    model_config = config.get("model", {})
    if method == "Attention-PI-HMONet" and model_config.get("name") == "attention_pihmonet":
        kwargs = {key: value for key, value in model_config.items() if key != "name"}
    else:
        kwargs = {
            **model_config.get("common", {}),
            **model_config.get(method.lower().replace("-", ""), {}),
        }
    if method in ("HMONet", "PI-HMONet", "Attention-PI-HMONet"):
        stored_mapping = kwargs.get("hmonet_mapping_mode", "current")
        if stored_mapping != mapping:
            raise ValueError(
                f"Checkpoint mapping is {stored_mapping}, requested {mapping}: {path}"
            )
    state_dict = payload.get("model_state_dict", payload)
    paper_pi_architecture = method == "PI-MeshONet" and any(
        key.startswith("branch_layers.") for key in state_dict
    )
    cls = {
        (2, "HMONet"): HMONet2D,
        (2, "MeshONet"): MeshONet2D,
        (2, "PI-MeshONet"): PIMeshONet2D if paper_pi_architecture else MeshONet2D,
        (2, "PI-HMONet"): HMONet2D,
        (2, "Attention-PI-HMONet"): AttentionHMONet2D,
        (3, "HMONet"): HMONet3D,
        (3, "MeshONet"): MeshONet3D,
        (3, "PI-MeshONet"): PIMeshONet3D if paper_pi_architecture else MeshONet3D,
        (3, "PI-HMONet"): HMONet3D,
        (3, "Attention-PI-HMONet"): AttentionHMONet3D,
    }[(dimension, method)]
    if cls is MeshONet3D and not any(
        key.startswith("branch_fusion.") for key in state_dict
    ):
        # Checkpoints created before the paper-aligned 3-D fusion update have
        # no shared fusion block. Preserve their original inference graph.
        kwargs["shared_branch_fusion_enabled"] = False
    kwargs.pop("residual_regularization_weight", None)
    kwargs.pop("architecture", None)
    if dimension == 2 and not paper_pi_architecture:
        allowed = inspect.signature(cls.__init__).parameters
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    model = cls(*shape, **kwargs)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    model.boundary_sample_counts = (
        config.get("attentionpihmonet_training", {}).get("boundary_sample_counts")
        if method == "Attention-PI-HMONet"
        else None
    ) or config.get("boundary_encoder", {}).get("sample_counts")
    summary = read_json(path.parent / "training_summary.json", {})
    return (
        model,
        shape,
        {
            "checkpoint": str(path),
            "checkpoint_sha256": file_digest(path),
            "training_grid": shape,
            "epoch": payload.get("epoch"),
            "training_seconds": summary.get(
                "training_time_seconds",
                summary.get("elapsed_seconds", summary.get("elapsed_time_seconds")),
            ),
        },
    )


def summarize_inference_samples(samples):
    """Discard exactly one minimum and maximum, then average the remaining runs.

    One or two explicitly requested repeats use the ordinary mean because
    trimming both ends would leave no measured runs.
    """
    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Timing samples must be a nonempty finite sequence")
    used = np.sort(values)[1:-1] if values.size >= 3 else values
    return float(used.mean()), {
        "inference_timing_statistic": "trimmed_mean" if values.size >= 3 else "mean",
        "inference_discarded_samples": 2 if values.size >= 3 else 0,
        "inference_used_samples": int(used.size),
    }


@torch.inference_mode()
def predict(
    model, method, geometry, input_shape, queries, device, options, runtime=None
):
    with curve_kernel_mode(options.get("curve_kernel_backend", "auto") != "torch"):
        return _predict(
            model,
            method,
            geometry,
            input_shape,
            queries,
            device,
            options,
            False,
            runtime,
        )


@torch.inference_mode()
def predict_batch(
    model, method, geometries, input_shape, queries, device, options, runtime=None
):
    """Predict geometries sharing one query grid; report batch and per-item times."""
    if not geometries:
        raise ValueError("At least one geometry is required")
    with curve_kernel_mode(options.get("curve_kernel_backend", "auto") != "torch"):
        return _predict(
            model,
            method,
            list(geometries),
            input_shape,
            queries,
            device,
            options,
            True,
            runtime,
        )


def _predict(
    model, method, geometry, input_shape, queries, device, options, batched, runtime
):
    """Time the full raw forward, including HMONet's internal hard constraint.

    Both networks use the same warmup and measurement protocol. Projection
    points are evaluated inside each HMONet forward, with no external cache.
    """
    batch_size = len(geometry) if batched else 1
    kernel_before = kernel_status()
    boundary_values = (
        np.stack([sample_boundary(g, input_shape) for g in geometry])
        if batched
        else sample_boundary(geometry, input_shape)
    )
    boundary = torch.as_tensor(boundary_values, dtype=torch.float32, device=device)
    query = torch.as_tensor(queries, dtype=torch.float32, device=device)
    explicit_single_batch = bool(
        not batched
        and (
            method == "Attention-PI-HMONet"
            or getattr(model, "boundary_self_attention_enabled", False)
        )
    )
    if batched:
        query = query[None].expand(batch_size, -1, -1).contiguous()
    elif explicit_single_batch:
        # Attention encoders always consume [batch, points, coordinates], even
        # when evaluation contains only one geometry.
        query = query[None]
        if method != "Attention-PI-HMONet":
            boundary = boundary[None]
    b = boundary.unbind(-1)
    attention_boundary = None
    if method == "Attention-PI-HMONet":
        geometry_list = list(geometry) if batched else [geometry]
        attention_boundary = build_boundary_inputs(
            geometry_list, input_shape, device, torch.float32,
            getattr(model, "boundary_sample_counts", None),
        )
    chunk_size = int(options["query_batch_size"])
    chunks = tuple(query.split(chunk_size, dim=-2))
    geometries = (geometry,) * len(chunks)

    def forward(index):
        args = (
            (*attention_boundary, *chunks[index].unbind(-1))
            if method == "Attention-PI-HMONet"
            else (*b, *chunks[index].unbind(-1))
        )
        return (
            model(*args, boundary_geometry=geometries[index])
            if method in ("HMONet", "PI-HMONet", "Attention-PI-HMONet")
            else model(*args)
        )

    warmup_runs = int(options["inference_warmup_runs"])
    for _ in range(warmup_runs):
        for index in range(len(chunks)):
            forward(index)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    final = None
    for repetition in range(int(options["inference_timing_repeats"])):
        elapsed = 0.0
        pieces = []
        for index in range(len(chunks)):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = forward(index)
                end.record()
                end.synchronize()
                elapsed += start.elapsed_time(end) / 1000.0
            else:
                started = time.perf_counter()
                output = forward(index)
                elapsed += time.perf_counter() - started
            if repetition == int(options["inference_timing_repeats"]) - 1:
                pieces.append(torch.cat(output, dim=-1))
        samples.append(elapsed)
        if pieces:
            # Only the last measured output is needed. Transfer it once instead
            # of synchronizing/copying every chunk in every timing repetition.
            final = torch.cat(pieces, dim=-2).cpu().numpy()
    if explicit_single_batch:
        final = final[0]
    if not np.isfinite(final).all():
        raise FloatingPointError(f"{method} returned non-finite coordinates")
    batch_seconds, timing_summary = summarize_inference_samples(samples)
    forward_seconds = batch_seconds / batch_size
    return final, {
        **timing_summary,
        "network_inference_seconds": forward_seconds,
        "inference_samples_seconds": [value / batch_size for value in samples],
        "inference_batch_size": batch_size,
        "batch_inference_seconds": batch_seconds,
        "inference_time_basis": "per_item_batch_average" if batched else "single_grid",
        "curve_kernel_backend": options.get("curve_kernel_backend", "auto"),
        "curve_kernel_calls": kernel_status()["calls"] - kernel_before["calls"],
        "curve_kernel_fallbacks": (
            kernel_status()["failures"] if method in ("HMONet", "PI-HMONet", "Attention-PI-HMONet") else {}
        ),
        "effective_inference_warmup_runs": warmup_runs,
        "inference_warmup_runs": options["inference_warmup_runs"],
        "inference_timing_repeats": options["inference_timing_repeats"],
        "query_batch_size": options["query_batch_size"],
        "device": str(device),
    }


def postprocess(raw, exact_boundary, ids, mode, power=2.0, chunk_size=4096):
    """Keep raw immutable; optionally spread boundary displacement by IDW."""
    result = np.asarray(raw, dtype=float).copy()
    if mode == "none":
        return result
    if mode not in ("direct", "inverse_distance"):
        raise ValueError(f"Unknown boundary projection: {mode}")
    displacement = exact_boundary - result[ids]
    if mode == "inverse_distance":
        interior = np.setdiff1d(np.arange(len(result)), ids)
        for start in range(0, len(interior), chunk_size):
            chunk = interior[start : start + chunk_size]
            distance = np.linalg.norm(
                raw[chunk, None, :] - raw[ids][None, :, :], axis=-1
            )
            weights = np.maximum(distance, 1e-10) ** (-float(power))
            weights /= weights.sum(axis=1, keepdims=True)
            result[chunk] += weights @ displacement
    result[ids] = exact_boundary
    return result
