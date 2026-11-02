"""Retention helpers for compact, reusable evaluation outputs."""

from __future__ import annotations

from pathlib import Path


IMAGE_SUFFIXES = {".jpeg", ".jpg", ".pdf", ".png", ".svg", ".webp"}


def compact_evaluation_directory(directory: Path) -> Path:
    """Keep only the canonical summary and generated images below one case."""
    directory = Path(directory)
    result_file = directory / "results.json"
    if not result_file.is_file():
        raise FileNotFoundError(
            f"Cannot compact evaluation without its summary: {result_file}"
        )
    for child in list(directory.rglob("*")):
        if not child.is_file():
            continue
        if child == result_file or child.suffix.lower() in IMAGE_SUFFIXES:
            continue
        child.unlink()
    for child in sorted(
        (item for item in directory.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        # Keep the named comparison location valid even when plotting is off.
        # Empty directories contain no retained payload and cost negligible
        # space, while results.json may legitimately reference this path.
        if "comparisons" in child.parts:
            continue
        try:
            child.rmdir()
        except OSError:
            pass
    return result_file
