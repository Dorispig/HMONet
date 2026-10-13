"""Generate selected meshes with exact cache identities and no substitute methods."""

import time
from pathlib import Path

import numpy as np
import torch

from .artifacts import digest, file_digest, array_digest, read_json, write_json
from .configuration import ROOT, resolve_path
from .inference import load_network, predict, resolve_device
from src.data.mesh.sampling import boundary_indices, logical_grid, sample_boundary


def implementation_version():
    """Invalidate numerical/analysis caches after relevant source changes."""
    folders = (
        "src/models",
        "src/data",
        "src/evaluation",
        "src/utils",
        "scripts/tfi",
        "scripts/pde",
        "scripts/evaluate",
    )
    return digest(
        {
            str(p.relative_to(ROOT)): file_digest(p)
            for folder in folders
            for p in sorted((ROOT / folder).rglob("*.py"))
        }
    )


def tfi(geometry, shape, queries=None):
    queries = logical_grid(shape) if queries is None else queries
    if len(shape) == 2:
        from scripts.tfi.tfi_2d import tfi_from_curves

        return tfi_from_curves(geometry, parameter_grid=queries).reshape(-1, 2)
    from scripts.tfi.tfi_3d import tfi_from_surfaces

    return tfi_from_surfaces(geometry, parameter_grid=queries).reshape(-1, 3)


def solve_pde(geometry, shape, options, initial=None, iterations=None):
    """Return the actual iterative solution and its convergence diagnostics."""
    if min(shape) < 3:
        raise ValueError("PDE generation requires at least three nodes per axis")
    if len(shape) == 2:
        from scripts.pde.pde_2d import solve_winslow_2d as solver
    else:
        from scripts.pde.pde_3d import solve_winslow_3d as solver
    if initial is None:
        initial = tfi(geometry, shape)
    points, info = solver(
        geometry,
        *shape,
        initial_grid=initial,
        max_iter=int(iterations or options["winslow_max_iter"]),
        tolerance=float(options["winslow_tolerance"]),
        return_info=True,
    )
    points = np.asarray(points, dtype=float).reshape(-1, len(shape))
    if not np.isfinite(points).all():
        raise FloatingPointError("PDE returned non-finite mesh coordinates")
    if not info["converged"]:
        message = f"PDE has not converged: {info}"
        if options["require_pde_convergence"] and iterations is None:
            raise RuntimeError(message)
        print("WARNING: " + message, flush=True)
    exact = sample_boundary(geometry, shape)
    maximum_error = float(np.max(np.abs(points[boundary_indices(shape)] - exact)))
    if maximum_error > 1e-7:
        raise RuntimeError(f"PDE changed prescribed boundary nodes: {maximum_error:g}")
    return points, dict(info, boundary_max_coordinate_error=maximum_error)


def generate(
    directory,
    method,
    geometry,
    shape,
    mapping,
    options,
    checkpoint,
    train_config,
    version,
    models,
    defer=False,
):
    """Reuse only the matching method's mesh; an unavailable method is never TFI."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    network = method in ("HMONet", "MeshONet", "PI-MeshONet", "PI-HMONet", "Attention-PI-HMONet")
    signature_data = {
        "implementation": version,
        "method": method,
        "shape": shape,
        "geometry": geometry.to_dict(),
        "mapping": mapping if method in ("HMONet", "PI-HMONet", "Attention-PI-HMONet") else None,
    }
    if network:
        signature_data.update(
            checkpoint=str(checkpoint),
            checkpoint_sha256=file_digest(checkpoint),
            config=read_json(checkpoint.parent / "config.json"),
            train_config_sha256=file_digest(resolve_path(train_config)),
            device=str(resolve_device(options["device"])),
            torch_version=torch.__version__,
            timing={
                k: options[k]
                for k in (
                    "inference_warmup_runs",
                    "inference_timing_repeats",
                    "query_batch_size",
                    "inference_batch_size",
                )
            },
        )
        if method in ("HMONet", "PI-HMONet", "Attention-PI-HMONet"):
            signature_data["inference_path"] = "direct-no-setup-v1"
            signature_data["curve_kernel_backend"] = options.get(
                "curve_kernel_backend", "auto"
            )
    elif method == "PDE_Winslow":
        signature_data.update(
            {
                k: options[k]
                for k in (
                    "winslow_max_iter",
                    "winslow_tolerance",
                    "require_pde_convergence",
                )
            }
        )
    signature = digest(signature_data)
    cache_path = directory / "generation.npz"
    pending_key = (str(cache_path.resolve()), signature)
    generated_in_batch = pending_key in models.get("batched_generation", set())
    metadata = read_json(directory / "generation_metadata.json", {})
    reusable = (
        (
            (options["reuse_generated_results"] and not options["force_regenerate"])
            or generated_in_batch
        )
        and metadata.get("signature") == signature
        and cache_path.is_file()
        and metadata.get("file_sha256") == file_digest(cache_path)
    )
    if reusable:
        if defer:
            return None
        with np.load(cache_path) as archive:
            points = archive["points"].copy()
        print(
            f"  {'use batch-generated' if generated_in_batch else 'reuse generated'} {method}",
            flush=True,
        )
        if generated_in_batch:
            models["batched_generation"].discard(pending_key)
        return (
            points,
            dict(metadata["timing"], cache_hit=not generated_in_batch),
            digest({"input": signature, "mesh": array_digest(points)}),
        )
    if options["analysis_only"]:
        raise RuntimeError(
            f"analysis-only requires a matching generation cache for {method}: {directory}"
        )
    print(f"  generate {method}", flush=True)
    started = time.perf_counter()
    if network:
        device = resolve_device(options["device"])
        key = (
            str(checkpoint),
            str(device),
            mapping,
            signature_data["checkpoint_sha256"],
        )
        if key not in models:
            models[key] = load_network(
                checkpoint, method, len(shape), mapping, device, train_config
            )
        model, input_shape, details = models[key]
        if defer:
            return dict(
                model=model,
                input_shape=input_shape,
                details=details,
                method=method,
                geometry=geometry,
                shape=shape,
                device=device,
                options=options,
                signature=signature,
                signature_data=signature_data,
                cache_path=cache_path,
                pending_key=pending_key,
            )
        points, timing = predict(
            model,
            method,
            geometry,
            input_shape,
            logical_grid(shape),
            device,
            options,
            runtime=models.setdefault("inference_runtime", {}),
        )
        timing.update(details)
    elif method == "TFI":
        points = tfi(geometry, shape)
        timing = {"total_seconds": time.perf_counter() - started}
    elif method == "PDE_Winslow":
        points, info = solve_pde(geometry, shape, options)
        timing = {"total_seconds": time.perf_counter() - started, "pde": info}
    else:
        raise ValueError(f"Unknown evaluation method {method}")
    if (
        points.shape != (int(np.prod(shape)), len(shape))
        or not np.isfinite(points).all()
    ):
        raise ValueError(f"Invalid {method} mesh shape/coordinates: {points.shape}")
    np.savez_compressed(cache_path, points=points)
    timing["cache_hit"] = False
    write_json(
        directory / "generation_metadata.json",
        {
            "signature": signature,
            "inputs": signature_data,
            "timing": timing,
            "file_sha256": file_digest(cache_path),
        },
    )
    return points, timing, digest({"input": signature, "mesh": array_digest(points)})


def execute_prediction_batches(requests, models):
    """Populate normal generation caches using compatible geometry batches."""
    from .inference import predict_batch

    groups = {}
    for request in requests:
        options = request["options"]
        key = (
            id(request["model"]),
            tuple(request["shape"]),
            options.get("curve_kernel_backend", "auto"),
            *[
                options[name]
                for name in (
                    "query_batch_size",
                    "inference_warmup_runs",
                    "inference_timing_repeats",
                    "inference_batch_size",
                )
            ],
        )
        groups.setdefault(key, []).append(request)
    for group in groups.values():
        size = int(group[0]["options"]["inference_batch_size"])
        for start in range(0, len(group), size):
            batch = group[start : start + size]
            item = batch[0]
            arguments = (
                item["model"],
                item["method"],
                [r["geometry"] for r in batch] if len(batch) > 1 else item["geometry"],
                item["input_shape"],
                logical_grid(item["shape"]),
                item["device"],
                item["options"],
            )
            points, timing = (predict_batch if len(batch) > 1 else predict)(
                *arguments, runtime=models.setdefault("inference_runtime", {})
            )
            if len(batch) == 1:
                points = points[None]
            print(
                f"  batch generate {item['method']}: count={len(batch)}, "
                f"batch_forward={timing['batch_inference_seconds']:.6f}s",
                flush=True,
            )
            for mesh, request in zip(points, batch):
                if (
                    mesh.shape
                    != (int(np.prod(request["shape"])), len(request["shape"]))
                    or not np.isfinite(mesh).all()
                ):
                    raise ValueError("Invalid mesh produced by batched inference")
                cache_path = request["cache_path"]
                np.savez_compressed(cache_path, points=mesh)
                write_json(
                    cache_path.parent / "generation_metadata.json",
                    {
                        "signature": request["signature"],
                        "inputs": request["signature_data"],
                        "timing": dict(timing, **request["details"], cache_hit=False),
                        "file_sha256": file_digest(cache_path),
                    },
                )
                models.setdefault("batched_generation", set()).add(
                    request["pending_key"]
                )
