"""Stable access to mesh archives produced by the regular evaluator."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.mesh.sampling import boundary_indices, sample_boundary
from src.evaluation.artifacts import read_json
from src.utils.paths import resolve_project_path


METHOD_ALIASES = {
    "tfi": "TFI",
    "pde": "PDE_Winslow",
    "pde_winslow": "PDE_Winslow",
    "meshonet": "MeshONet",
    "pimeshonet": "PI-MeshONet",
    "pi-meshonet": "PI-MeshONet",
    "hmonet": "HMONet",
}


def canonical_method(name):
    key = str(name).strip().lower()
    return METHOD_ALIASES.get(key, str(name))


def mesh_archive_path(evaluation_root, dataset, shape, value):
    evaluation_root = resolve_project_path(evaluation_root)
    if evaluation_root.name != "2D":
        evaluation_root /= "2D"
    tag = f"{float(value):g}" if isinstance(value, (int, float)) else str(value)
    return evaluation_root / dataset / "x".join(map(str, shape)) / tag / "generated_meshes.npz"


def load_evaluated_meshes(evaluation_root, dataset, shape, value, methods=None):
    """Load standard evaluator output and its recorded timing metadata."""
    path = mesh_archive_path(evaluation_root, dataset, shape, value)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing evaluated mesh archive: {path}. Run the normal 2-D "
            "evaluation with matching dataset/value/grid first; compact-only "
            "results must be regenerated with --generation-only or --no-compact-outputs."
        )
    requested = None if methods is None else [canonical_method(name) for name in methods]
    with np.load(path) as archive:
        available = list(archive.files)
        selected = available if requested is None else requested
        missing = [method for method in selected if method not in archive.files]
        if missing:
            raise KeyError(f"{path} does not contain {missing}; available={available}")
        meshes = {method: archive[method].copy() for method in selected}
    expected = (int(np.prod(shape)), 2)
    for method, points in meshes.items():
        if points.shape != expected or not np.isfinite(points).all():
            raise ValueError(f"Invalid {method} mesh in {path}: {points.shape}")
    result = read_json(path.parent / "results.json", {}) or {}
    return meshes, result.get("timings", {}), path


def replace_with_exact_boundary(points, geometry, shape):
    """Return a copy with only boundary nodes replaced by exact curve samples."""
    result = np.asarray(points, dtype=float).copy()
    ids = boundary_indices(shape)
    exact = sample_boundary(geometry, shape)
    result[ids] = exact
    return result


def boundary_geometry_error(points, geometry, shape):
    ids = boundary_indices(shape)
    exact = sample_boundary(geometry, shape)
    distance = np.linalg.norm(np.asarray(points)[ids] - exact, axis=1)
    return {
        "boundary_mean_error": float(distance.mean()),
        "boundary_max_error": float(distance.max()),
        "boundary_rms_error": float(np.sqrt(np.mean(distance**2))),
    }


def generation_seconds(timings, method):
    timing = timings.get(method, {})
    for key in ("total_model_seconds", "network_inference_seconds", "total_seconds"):
        value = timing.get(key)
        if value is not None:
            return float(value)
    return float("nan")
