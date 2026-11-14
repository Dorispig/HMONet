"""Keep evaluation summaries while publishing plots under readable paths.

The evaluator may create deeply nested implementation folders. This module
retains results.json plus plot files, while deliberately using full,
reader-facing method names in the public plot hierarchy.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil


IMAGE_SUFFIXES = {".jpeg", ".jpg", ".pdf", ".png", ".svg", ".webp"}

_METHOD_NAMES = {
    "tfi": "TFI",
    "pde": "PDE_Winslow",
    "pde_winslow": "PDE_Winslow",
    "meshonet": "MeshONet",
    "pimeshonet": "PI-MeshONet",
    "pi-meshonet": "PI-MeshONet",
    "hmonet": "HMONet",
    "qghmonet": "QG-HMONet",
    "qg-hmonet": "QG-HMONet",
    "pihmonet": "PI-HMONet",
    "attentionpihmonet": "Attention-PI-HMONet",
}
_LEGACY_METHOD_NAMES = {
    "tfi": "TFI",
    "t": "TFI",
    "pde": "PDE_Winslow",
    "p": "PDE_Winslow",
    "hm": "HMONet",
    "h": "HMONet",
    "qg": "QG-HMONet",
    "mo": "MeshONet",
    "m": "MeshONet",
    "pi": "PI-MeshONet",
    "pih": "PI-HMONet",
}


def _safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(value)).strip("._")
    return value or "misc"


def _comparison_group(parts) -> str:
    """Derive a reader-facing method group from an evaluator folder name."""
    for part in reversed(tuple(parts)):
        names = []
        for token in re.split(r"[_+]+", str(part).lower()):
            if not token or token.isdigit():
                continue
            name = _METHOD_NAMES.get(token) or _LEGACY_METHOD_NAMES.get(token)
            if name and name not in names:
                names.append(name)
        if len(names) >= 2:
            return "_".join(names)
    return "methods"


def _legacy_public_path(root: Path, parts: list[str]) -> Path:
    """Translate the former compact hierarchy without losing images."""
    if len(parts) == 1:
        return root / "plots" / parts[0]
    if parts[1] == "c":
        group = _comparison_group(parts[2:3]) if len(parts) > 2 else "methods"
        tail = parts[3:]
        destination = root / "plots" / "comparison" / group
        if tail and tail[0] == "q":
            destination /= "quality"
            tail = tail[1:]
        return destination.joinpath(*tail)
    if parts[1] == "q":
        return root / "plots" / "quality" / Path(*parts[2:])
    method = _LEGACY_METHOD_NAMES.get(parts[1], _safe_component(parts[1]))
    tail = parts[2:]
    destination = root / "plots" / method
    if tail and tail[0] == "r":
        destination /= "raw"
        tail = tail[1:]
    elif tail and tail[0] == "p":
        destination /= "processed"
        tail = tail[1:]
    if tail and tail[0] == "q":
        destination /= "quality"
        tail = tail[1:]
    return destination.joinpath(*tail)


def readable_image_path(root: Path, relative: Path) -> Path:
    """Map raw evaluator or former compact paths to public plot paths."""
    root = Path(root)
    parts = list(Path(relative).parts)
    if not parts:
        return root / "plots" / "misc"
    if parts[0] == "p":
        return _legacy_public_path(root, parts)
    if parts[0] == "plots":
        parts = parts[1:]
        if parts and parts[0] == "comparison":
            # Canonicalize older readable groups such as ``TFI_PDE_HMONet``
            # to the full method spelling used by newly published results.
            group = _comparison_group(parts[1:2]) if len(parts) > 1 else "methods"
            return root / "plots" / "comparison" / group / Path(*parts[2:])
        # A few older migrations produced ``plots/p/...`` instead of moving
        # the former compact root itself.  Interpret it as legacy as well.
        if parts and parts[0] == "p":
            return _legacy_public_path(root, parts)
    lowered = [part.lower() for part in parts]
    filename = parts[-1]
    if len(parts) == 1:
        return root / "plots" / filename
    if parts[0].lower() == "grid_quality":
        return root / "plots" / "quality" / filename
    if parts[0].lower() in {"comparisons", "comparison"}:
        destination = root / "plots" / "comparison" / _comparison_group(parts[1:-1])
        if "grid_quality" in lowered or "quality" in lowered:
            destination /= "quality"
        return destination / filename
    method = _METHOD_NAMES.get(parts[0].lower(), _safe_component(parts[0]))
    destination = root / "plots" / method
    if "raw" in lowered:
        destination /= "raw"
    elif method in {
        "HMONet", "QG-HMONet", "MeshONet", "PI-MeshONet",
        "PI-HMONet", "Attention-PI-HMONet",
    }:
        destination /= "processed"
    if "grid_quality" in lowered or "quality" in lowered:
        destination /= "quality"
    return destination / filename


def reorganize_evaluation_images(directory: Path) -> int:
    """Move every retained image to the readable public hierarchy."""
    directory = Path(directory)
    images = [
        child for child in directory.rglob("*")
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
    ]
    moved = 0
    for child in images:
        destination = readable_image_path(directory, child.relative_to(directory))
        if destination == child:
            continue
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
    """Keep canonical summary and readable plots below one evaluation case."""
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
        # Local-view metadata is small, is referenced by downstream zoom
        # commands, and is not a per-cell quality artifact.
        if child.suffix.lower() == ".json" and "zoom_views" in child.relative_to(directory).parts:
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
