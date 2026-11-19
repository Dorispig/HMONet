"""Record generated-grid diagnostics before surfacing validation failures."""

import json
import warnings


def endpoint_indices(size):
    """Return the first and last sample IDs, deduplicating a single sample."""
    return (0,) if size == 1 else (0, size - 1)


def save_quality_audit(directory, name, grid_shape, samples, diagnostics, scope):
    illegal = [sample["sample_id"] for sample in samples if not sample["all_cells_legal"]]
    unconverged = [index for index, info in enumerate(diagnostics) if not info["converged"]]
    audit = {
        "dataset": name,
        "size": len(diagnostics),
        "grid_shape": list(grid_shape),
        "scope": scope,
        "checked_samples": len(samples),
        "illegal_samples": len(illegal),
        "illegal_sample_ids": illegal,
        "nonconverged_samples": len(unconverged),
        "nonconverged_sample_ids": unconverged,
        "samples": samples,
    }
    if len(grid_shape) == 2:
        for metric in ("self_intersecting", "overlapping_convex"):
            audit[f"{metric}_samples"] = sum(
                sample.get(f"{metric}_cells", 0) > 0 for sample in samples
            )
    (directory / "quality_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def warn_saved_dataset(directory, audit):
    if audit["illegal_samples"] or audit["nonconverged_samples"]:
        warnings.warn(
            f"Dataset saved at {directory}: "
            f"{audit['illegal_samples']}/{audit['checked_samples']} checked endpoint grids are invalid, "
            f"{audit['nonconverged_samples']}/{audit['size']} unconverged PDE samples. "
            f"See {directory / 'quality_audit.json'} for sample IDs and details.",
            RuntimeWarning,
            stacklevel=2,
        )
