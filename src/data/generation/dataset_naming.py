"""Canonical dataset names shared by two- and three-dimensional generators."""

from __future__ import annotations

import re


_OUTPUT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*$")


def validate_version(version):
    """Return a non-negative, user-controlled parameter-range version."""
    if isinstance(version, bool) or int(version) != version or int(version) < 0:
        raise ValueError("dataset version must be a non-negative integer")
    return int(version)


def build_dataset_name(output_name, split, version, size, *grid_shape):
    """Build ``boundary_int/ext_version_size_grid``."""
    output_name = str(output_name)
    if _OUTPUT_NAME_PATTERN.fullmatch(output_name) is None:
        raise ValueError(
            "output_name must contain only letters, digits, and single underscores"
        )
    normalized = str(split).lower()
    aliases = {
        "int": "int",
        "interpolation": "int",
        "ext": "ext",
        "extrapolation": "ext",
    }
    if normalized not in aliases:
        raise ValueError("split must be interpolation/int or extrapolation/ext")
    version = validate_version(version)
    if isinstance(size, bool) or int(size) != size or int(size) < 1:
        raise ValueError("dataset size must be a positive integer")
    if len(grid_shape) not in (2, 3):
        raise ValueError("grid shape must contain two or three dimensions >= 3")
    if any(
        isinstance(value, bool) or int(value) != value or int(value) < 3
        for value in grid_shape
    ):
        raise ValueError("grid dimensions must be integers >= 3")
    dimensions = tuple(int(value) for value in grid_shape)
    suffix = "_".join(map(str, dimensions))
    return f"{output_name}_{aliases[normalized]}_{version}_{int(size)}_{suffix}"


def default_version_for_split(split):
    """Return the default parameter-range version for either split."""
    return 0


def parse_dataset_name(name, dimension):
    """Parse a canonical name and return its boundary, version, size and grid."""
    if isinstance(dimension, bool) or int(dimension) != dimension:
        raise ValueError("dimension must be 2 or 3")
    count = int(dimension)
    if count not in (2, 3):
        raise ValueError("dimension must be 2 or 3")
    pattern = rf"^(?P<boundary>.+)_(?P<split>int|ext)_(?P<version>\d+)_(?P<size>\d+)_(?P<grid>(?:\d+_){{{count - 1}}}\d+)$"
    match = re.match(pattern, str(name))
    if match is None:
        return None
    return {
        "boundary": match.group("boundary"),
        "split": "interpolation" if match.group("split") == "int" else "extrapolation",
        "version": int(match.group("version")),
        "size": int(match.group("size")),
        "grid_shape": tuple(int(value) for value in match.group("grid").split("_")),
    }
