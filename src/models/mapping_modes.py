"""Canonical names and compatibility aliases for HMONet hard-boundary maps."""

from __future__ import annotations

RATIONAL_AFFINE = "rational_affine"
COONS_RESIDUAL = "coons_residual"
CORNER_CORRECTED_AFFINE = "corner_corrected_affine"
BOTH = "both"

CANONICAL_MAPPING_MODES = (
    RATIONAL_AFFINE,
    COONS_RESIDUAL,
    CORNER_CORRECTED_AFFINE,
)

# Archived configurations and checkpoints remain readable after the rename.
MAPPING_MODE_ALIASES = {
    "current": RATIONAL_AFFINE,
    "rational": RATIONAL_AFFINE,
    "rational_affine_projection": RATIONAL_AFFINE,
    "coons": COONS_RESIDUAL,
    "coons_residual_projection": COONS_RESIDUAL,
    "corner_corrected": CORNER_CORRECTED_AFFINE,
    "corner_corrected_projection": CORNER_CORRECTED_AFFINE,
}

# A pure naming change must not invalidate completed experiment fingerprints.
FINGERPRINT_MAPPING_NAMES = {
    RATIONAL_AFFINE: "current",
    COONS_RESIDUAL: COONS_RESIDUAL,
    CORNER_CORRECTED_AFFINE: "corner_corrected",
}

# Reader-facing output directories must stay well below Windows MAX_PATH while
# configuration files and checkpoint identities retain their explicit names.
MAPPING_MODE_PATH_NAMES = {
    RATIONAL_AFFINE: "RAP",
    COONS_RESIDUAL: "CR",
    CORNER_CORRECTED_AFFINE: "CCAP",
}


def normalize_mapping_mode(value, *, allow_both: bool = False) -> str:
    """Return one canonical mapping name, accepting archived aliases."""
    mode = str(value).strip().lower().replace("-", "_")
    mode = MAPPING_MODE_ALIASES.get(mode, mode)
    allowed = set(CANONICAL_MAPPING_MODES)
    if allow_both:
        allowed.add(BOTH)
    if mode not in allowed:
        choices = ", ".join(
            (*CANONICAL_MAPPING_MODES, *((BOTH,) if allow_both else ()))
        )
        raise ValueError(f"hmonet_mapping_mode must be one of: {choices}")
    return mode


def mapping_mode_choices(*, allow_both: bool = False, include_aliases: bool = True):
    """Choices for command-line parsers; canonical names are listed first."""
    choices = list(CANONICAL_MAPPING_MODES)
    if allow_both:
        choices.append(BOTH)
    if include_aliases:
        choices.extend(alias for alias in MAPPING_MODE_ALIASES if alias not in choices)
    return choices


def fingerprint_mapping_mode(value) -> str:
    """Return the historical spelling used by completed-run fingerprints."""
    return FINGERPRINT_MAPPING_NAMES[normalize_mapping_mode(value)]


def mapping_mode_path_name(value) -> str:
    """Return the stable short folder name for a hard-boundary map."""
    return MAPPING_MODE_PATH_NAMES[normalize_mapping_mode(value)]
