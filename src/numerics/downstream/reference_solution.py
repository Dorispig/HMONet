"""Small reference-solution cache for repeated downstream comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.evaluation.artifacts import array_digest, digest, read_json, write_json


def reference_identity(mesh, shape, problem):
    return digest(
        {
            "mesh_sha256": array_digest(np.asarray(mesh)),
            "shape": list(shape),
            "problem": problem,
            "implementation": "q1-poisson-v1",
        }
    )


def load_reference(path, identity):
    path = Path(path)
    metadata = read_json(path.with_suffix(".json"), {})
    if metadata.get("identity") != identity or not path.is_file():
        return None
    with np.load(path) as archive:
        return archive["solution"].copy()


def save_reference(path, identity, solution, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, solution=np.asarray(solution, dtype=float))
    write_json(path.with_suffix(".json"), {"identity": identity, **(metadata or {})})
