"""Rewrite project-owned absolute paths as portable relative paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PureWindowsPath

import yaml


ROOT = Path(__file__).resolve().parents[2]
PROJECT_NAMES = {ROOT.name.lower(), "hmonet_p"}


def convert_string(value):
    """Convert a project-owned absolute path string to root-relative POSIX form."""
    windows_path = PureWindowsPath(value)
    if windows_path.is_absolute():
        parts = list(windows_path.parts)
        for index, part in enumerate(parts):
            if part.lower() in PROJECT_NAMES:
                return Path(*parts[index + 1 :]).as_posix()

    path = Path(value)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            pass
    return value


def convert(value):
    """Recursively convert path strings in JSON/YAML-compatible data."""
    if isinstance(value, dict):
        return {key: convert(item) for key, item in value.items()}
    if isinstance(value, list):
        return [convert(item) for item in value]
    if isinstance(value, str):
        return convert_string(value)
    return value


def migrate(path, dry_run=False):
    """Migrate one JSON or YAML file and report whether it changed."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            old = json.loads(path.read_text(encoding="utf-8"))
            new = convert(old)
            rendered = json.dumps(new, ensure_ascii=False, indent=2)
        else:
            old = yaml.safe_load(path.read_text(encoding="utf-8"))
            new = convert(old)
            rendered = yaml.safe_dump(new, allow_unicode=True, sort_keys=False)
    except (OSError, ValueError, TypeError, yaml.YAMLError, json.JSONDecodeError):
        return False

    if new == old:
        return False
    if not dry_run:
        path.write_text(rendered, encoding="utf-8")
    return True


def main(argv=None):
    """Migrate stored records under experiments, outputs and configs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    roots = (ROOT / "experiments", ROOT / "outputs", ROOT / "configs")
    files = [
        path
        for base in roots
        if base.exists()
        for path in base.rglob("*")
        if path.suffix.lower() in {".json", ".yaml", ".yml"}
    ]
    changed = [
        path.relative_to(ROOT).as_posix()
        for path in files
        if migrate(path, args.dry_run)
    ]
    report = {
        "scanned": len(files),
        "changed": len(changed),
        "files": changed,
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        report_path = ROOT / "portable_path_migration_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
