"""Canonical model selection shared by training and train/evaluate entry points."""

from __future__ import annotations


MODEL_NAMES = (
    "hmonet",
    "meshonet",
    "pimeshonet",
    "pihmonet",
    "attentionpihmonet",
)

MODEL_ALIASES = {
    "both": ("hmonet", "meshonet"),
    "pi": ("pimeshonet", "pihmonet", "attentionpihmonet"),
    "all": MODEL_NAMES,
}


def parse_model_selection(value):
    """Expand aliases and comma-separated model names while preserving order."""
    if isinstance(value, str):
        tokens = value.split(",")
    elif isinstance(value, (list, tuple)):
        tokens = []
        for item in value:
            tokens.extend(str(item).split(","))
    else:
        raise ValueError("Model selection must be a string or a sequence of names")

    selected = []
    for raw_name in tokens:
        name = raw_name.strip().lower().replace("-", "")
        if not name:
            continue
        expanded = MODEL_ALIASES.get(name, (name,))
        for model_name in expanded:
            if model_name not in MODEL_NAMES:
                supported = ", ".join(MODEL_NAMES + tuple(MODEL_ALIASES))
                raise ValueError(
                    f"Unknown model '{raw_name.strip()}'. Supported values: {supported}; "
                    "multiple values may be separated by commas"
                )
            if model_name not in selected:
                selected.append(model_name)

    if not selected:
        raise ValueError("At least one model must be selected")
    return selected
