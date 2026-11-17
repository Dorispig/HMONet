"""Cross-platform serialization and resolution of project-owned paths."""
from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_NAME = PROJECT_ROOT.name.lower()
LEGACY_PROJECT_NAME = "hmonet_p"
PROJECT_TOP_LEVEL_NAMES = {
    ".git",
    "configs",
    "data",
    "docs",
    "experiments",
    "outputs",
    "paper",
    "scripts",
    "src",
    "tests",
}


# Keep the guard aligned with the classic Windows API used by the local
# project tooling.  The warning threshold is advisory and is applied only to
# the concrete path that is about to be created or written.
WINDOWS_CLASSIC_PATH_LIMIT = 260
LINUX_PATH_LIMIT = 4096
OUTPUT_PATH_WARNING_LIMIT = int(WINDOWS_CLASSIC_PATH_LIMIT * 0.9)


def system_path_limit() -> int:
    """Return the active platform's total-path limit used by project outputs."""
    return WINDOWS_CLASSIC_PATH_LIMIT if os.name == "nt" else LINUX_PATH_LIMIT


def output_path_length(path: str | Path) -> int:
    """Return the absolute spelling length the operating system receives."""
    return len(str(Path(path).resolve()))


def validate_output_path(
    path: str | Path,
    *,
    context: str = "Output path",
    warn: bool = True,
) -> int:
    """Check one concrete output path before creating or writing it.

    A warning is emitted only when this exact path is long; no artificial
    suffix or speculative reserve is added.  Reaching the real system limit
    raises before the filesystem operation is attempted.
    """
    path = Path(path)
    length = output_path_length(path)
    limit = system_path_limit()
    oversized = [part for part in path.parts if len(part) > 255]
    if length >= limit or oversized:
        detail = (
            f"components above 255 characters: {oversized}"
            if oversized
            else f"length={length}, system_limit={limit}"
        )
        raise OSError(f"{context} exceeds the real system path limit: {detail}; path={path}")
    if warn and length > OUTPUT_PATH_WARNING_LIMIT:
        warnings.warn(
            f"{context} approaches the system path limit "
            f"({length}>{OUTPUT_PATH_WARNING_LIMIT}) but remains valid "
            f"for this system ({limit - 1} maximum usable characters): {path}",
            RuntimeWarning,
            stacklevel=2,
        )
    return length


def ensure_output_directory(path: str | Path, *, context: str = "Output directory") -> Path:
    """Validate and create one concrete output directory."""
    directory = Path(path)
    validate_output_path(directory, context=context)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def prepare_output_file(path: str | Path, *, context: str = "Output file") -> Path:
    """Validate a concrete file destination and create its parent directory."""
    output = Path(path)
    validate_output_path(output, context=context)
    ensure_output_directory(output.parent, context=f"{context} parent directory")
    return output


def _legacy_relative(value) -> Path | None:
    text = str(value)
    windows = PureWindowsPath(text)
    if not windows.is_absolute():
        return None
    parts = list(windows.parts)
    for index, part in enumerate(parts):
        name = part.lower()
        remainder = parts[index + 1:]
        if name == LEGACY_PROJECT_NAME:
            return Path(*remainder)
        # The renamed project root is ``HMONet``, which is also a model output
        # directory name (for example ``checkpoints/.../HMONet/rational_affine``).
        # Rebase it only when the following component looks like a project
        # top-level entry; otherwise preserve the external absolute path.
        if name == PROJECT_NAME and (
            not remainder or remainder[0].lower() in PROJECT_TOP_LEVEL_NAMES
        ):
            return Path(*remainder)
    return None


def resolve_project_path(value, *, base: Path | None = None) -> Path:
    """Resolve project-relative paths and rebase legacy Windows project paths."""
    legacy = _legacy_relative(value)
    if legacy is not None:
        return (PROJECT_ROOT / legacy).resolve()
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return ((base or PROJECT_ROOT) / path).resolve()


def portable_project_path(value) -> str:
    """Serialize project-owned paths relative to the root with POSIX separators."""
    path = resolve_project_path(value)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)
