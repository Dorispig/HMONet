import pytest

from src.models.mapping_modes import (
    COONS_RESIDUAL,
    CORNER_CORRECTED_AFFINE,
    RATIONAL_AFFINE,
    fingerprint_mapping_mode,
    normalize_mapping_mode,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("rational_affine", RATIONAL_AFFINE),
        ("current", RATIONAL_AFFINE),
        ("coons_residual", COONS_RESIDUAL),
        ("coons", COONS_RESIDUAL),
        ("corner_corrected_affine", CORNER_CORRECTED_AFFINE),
        ("corner_corrected", CORNER_CORRECTED_AFFINE),
    ],
)
def test_mapping_names_and_archived_aliases(value, expected):
    assert normalize_mapping_mode(value) == expected


def test_mapping_rename_preserves_historical_fingerprint_names():
    assert fingerprint_mapping_mode(RATIONAL_AFFINE) == "current"
    assert fingerprint_mapping_mode(CORNER_CORRECTED_AFFINE) == "corner_corrected"


def test_invalid_mapping_name_is_rejected():
    with pytest.raises(ValueError, match="hmonet_mapping_mode"):
        normalize_mapping_mode("unknown")
