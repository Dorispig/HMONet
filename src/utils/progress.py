"""Progress display policy shared by project training entry points."""

from __future__ import annotations

import os

from tqdm import tqdm as _tqdm


def in_gpu_queue() -> bool:
    """Return whether this process is a compact multi-GPU queue worker."""
    return os.environ.get("HMONET_GPU_QUEUE", "").strip() == "1"


def training_progress(iterable=None, *args, **kwargs):
    """Use normal tqdm interactively and suppress batch bars in queue workers."""
    if in_gpu_queue():
        kwargs["disable"] = True
    return _tqdm(iterable, *args, **kwargs)
