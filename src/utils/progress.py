"""Terminal-width-aware progress and logging shared by all training routes."""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from numbers import Real

from tqdm import tqdm as _tqdm


_SHORT_METRIC_NAMES = {
    "training_loss": "loss",
    "coordinate_loss": "coord",
    "selected_internal_loss": "sel_mse",
    "random_internal_loss": "rand_mse",
    "loss_interior": "interior",
    "loss_boundary": "bnd",
    "boundary_loss": "bnd",
    "boundary_normal_loss": "normal",
    "boundary_normal": "normal",
    "pde_loss": "pde",
    "loss_pde": "pde",
    "normalized_pde": "pde",
    "loss_pde_normalized": "pde",
    "jacobian_loss": "jac",
    "jacobian_barrier": "jac",
    "normalized_jacobian_loss": "jac",
    "normalized_jacobian": "jac",
    "jacobian": "jac",
    "singular_value_loss": "sv",
    "singular_value": "sv",
    "winslow_loss": "win",
    "winslow": "win",
    "grid_smoothness": "smooth",
    "smoothness_loss": "smooth",
    "displacement_regularization": "reg",
    "teacher_anchor": "anchor",
    "alpha_penalty": "alpha_pen",
    "learning_rate": "lr",
    "dataset_mse": "mse",
}


def short_metric_name(name: str) -> str:
    """Return the concise terminal label without changing persisted keys."""
    return _SHORT_METRIC_NAMES.get(str(name), str(name))


def terminal_width(default: int = 120) -> int:
    """Return a practical terminal width for consoles and redirected logs."""
    configured = os.environ.get("HMONET_TERMINAL_WIDTH", "").strip()
    if configured:
        try:
            return max(48, int(configured))
        except ValueError:
            pass
    return max(48, shutil.get_terminal_size((default, 24)).columns)


def _format_value(value, precision: int = 3) -> str:
    if isinstance(value, Real) and not isinstance(value, bool):
        return f"{float(value):.{precision}e}"
    return str(value)


def wrapped_text(text: str, *, width: int | None = None, indent: str = "  ") -> str:
    """Wrap a log message without breaking metric names, paths or numbers."""
    return textwrap.fill(
        str(text),
        width=width or terminal_width(),
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def wrapped_print(
    text: str,
    *,
    width: int | None = None,
    indent: str = "  ",
    file=None,
    flush: bool = True,
) -> None:
    """Print a message with automatic wrapping at the active terminal width."""
    print(
        wrapped_text(text, width=width, indent=indent),
        file=file or sys.stdout,
        flush=flush,
    )


def print_key_values(
    prefix: str,
    values,
    *,
    precision: int = 6,
    width: int | None = None,
    file=None,
) -> None:
    """Print a comma-separated metric record and wrap only between fields."""
    if hasattr(values, "items"):
        values = values.items()
    tokens = [f"{key}={_format_value(value, precision)}" for key, value in values]
    wrapped_print(
        str(prefix) + (", ".join(tokens) if tokens else ""),
        width=width,
        indent="  ",
        file=file,
    )


class _TrainingTqdm(_tqdm):
    """Use the established tqdm layout with concise loss labels."""

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        fields = {}
        if ordered_dict is not None:
            fields.update(ordered_dict)
        fields.update(kwargs)
        concise = {}
        for key, value in fields.items():
            concise[short_metric_name(key)] = value
        return super().set_postfix(concise, refresh=refresh)


def in_gpu_queue() -> bool:
    """Return whether this process is a compact multi-GPU queue worker."""
    return os.environ.get("HMONET_GPU_QUEUE", "").strip() == "1"


def training_progress(iterable=None, *args, **kwargs):
    """Use normal tqdm interactively and suppress batch bars in queue workers."""
    if in_gpu_queue():
        kwargs["disable"] = True
    return _TrainingTqdm(iterable, *args, **kwargs)
