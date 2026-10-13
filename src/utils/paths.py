"""Cross-platform serialization and resolution of project-owned paths."""
from __future__ import annotations

from pathlib import Path, PureWindowsPath

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
        # directory name (for example ``checkpoints/.../HMONet/current``).
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
