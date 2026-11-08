"""Retention helpers for compact, reusable evaluation outputs."""

from __future__ import annotations

from pathlib import Path
import re
import shutil


IMAGE_SUFFIXES = {".jpeg", ".jpg", ".pdf", ".png", ".svg", ".webp"}


def _method_group(parts) -> str:
    aliases = {
        "tfi": "TFI",
        "pde": "PDE",
        "meshonet": "MeshONet",
        "pimeshonet": "PI-MeshONet",
        "hmonet": "HMONet",
        "qghmonet": "QG-HMONet",
        "pihmonet": "PI-HMONet",
    }
    ignored = {
        "grid", "quality", "none", "raw", "direct", "idw", "rational",
        "affine", "corner", "corrected", "coons", "residual",
    }
    for part in reversed(tuple(parts)):
        methods = []
        for token in re.split(r"[_+]+", part.lower()):
            if not token or token.isdigit() or token in ignored:
                continue
            method = aliases.get(token)
            if method and method not in methods:
                methods.append(method)
        if len(methods) >= 2:
            return "_".join(methods)
    return "methods"


def compact_image_path(root: Path, relative: Path) -> Path:
    """Map evaluator-internal image paths to the public compact plot tree."""
    parts = list(relative.parts)
    filename = parts[-1]
    lowered = [part.lower() for part in parts]
    if len(parts) == 1:
        return root / "plots" / filename
    if parts[0].lower() == "grid_quality":
        return root / "plots" / "quality" / filename
    if parts[0].lower() == "comparisons":
        category = ["comparison", _method_group(parts[1:-1])]
        if "grid_quality" in lowered:
            category.append("quality")
        return root / "plots" / Path(*category) / filename
    aliases = {
        "pde_winslow": "PDE",
        "tfi": "TFI",
        "hmonet": "HMONet",
        "qg-hmonet": "QG-HMONet",
        "meshonet": "MeshONet",
        "pi-meshonet": "PI-MeshONet",
    }
    method = aliases.get(parts[0].lower(), re.sub(r"[^A-Za-z0-9._+-]+", "_", parts[0]))
    category = [method]
    if "raw" in lowered:
        category.append("raw")
    elif method in {"HMONet", "QG-HMONet", "MeshONet", "PI-MeshONet"}:
        category.append("processed")
    if "grid_quality" in lowered:
        category.append("quality")
    return root / "plots" / Path(*category) / filename


def reorganize_evaluation_images(directory: Path) -> int:
    """Move all evaluation images into the compact public plot hierarchy."""
    directory = Path(directory)
    images = [
        child
        for child in directory.rglob("*")
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
    ]
    moved = 0
    for child in images:
        relative = child.relative_to(directory)
        if relative.parts and relative.parts[0] == "plots":
            continue
        destination = compact_image_path(directory, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            stem, suffix = destination.stem, destination.suffix
            index = 2
            while destination.exists():
                destination = destination.with_name(f"{stem}_{index}{suffix}")
                index += 1
        shutil.move(str(child), str(destination))
        moved += 1

    for child in sorted(
        (item for item in directory.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        try:
            child.rmdir()
        except OSError:
            pass
    return moved


def compact_evaluation_directory(directory: Path) -> Path:
    """Keep only the canonical summary and generated images below one case."""
    directory = Path(directory)
    result_file = directory / "results.json"
    if not result_file.is_file():
        raise FileNotFoundError(
            f"Cannot compact evaluation without its summary: {result_file}"
        )
    reorganize_evaluation_images(directory)

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
        try:
            child.rmdir()
        except OSError:
            pass
    return result_file
