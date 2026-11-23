"""Bounded-memory NumPy evaluation of the established hexahedral metrics."""
import numpy as np
from src.utils.quality_selection import HEX_METRICS, metric_names


def evaluate_hexahedral_grid_quality(grid, tolerance=1e-12, metrics=None):
    """Preserve z/y/x cell ordering and metric definitions; batch cell algebra."""
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 4 or grid.shape[-1] != 3 or min(grid.shape[:3]) < 2 or not np.isfinite(grid).all():
        raise ValueError("grid must be finite with shape (Nz,Ny,Nx,3)")
    selected = set(metric_names(metrics, HEX_METRICS))
    if not selected:
        return {}
    shape = np.array(grid.shape[:3]) - 1
    count = int(np.prod(shape))
    result = {name: np.empty(count) for name in HEX_METRICS if name in selected}
    need_scaled = bool(selected & {"minimum_scaled_jacobian", "flipped_cell"})
    need_lengths = bool(selected & {"edge_aspect_ratio", "degenerate_cell"})
    need_angles = bool(selected & {"max_angle", "min_angle", "skewness"})
    need_cosines = need_angles or "orthogonality" in selected
    need_center = bool(selected & {"center_jacobian", "center_condition_number", "cell_volume", "degenerate_cell"})
    need_det = bool(selected & {"center_jacobian", "cell_volume", "degenerate_cell"})
    for start in range(0, count, 32768):
        stop = min(count, start + 32768)
        k, remainder = np.divmod(np.arange(start, stop), shape[1] * shape[2])
        j, i = np.divmod(remainder, shape[2])
        c = np.stack([grid[k+ck, j+cj, i+ci] for ck in (0, 1) for cj in (0, 1) for ci in (0, 1)], axis=1).reshape(-1, 2, 2, 2, 3)
        values = {}
        scaled, cosines = [], []
        if need_scaled or need_cosines:
            for ck in (0, 1):
                for cj in (0, 1):
                    for ci in (0, 1):
                        eu = c[:, ck, cj, 1] - c[:, ck, cj, 0]
                        ev = c[:, ck, 1, ci] - c[:, ck, 0, ci]
                        ew = c[:, 1, cj, ci] - c[:, 0, cj, ci]
                        if need_scaled:
                            den = np.linalg.norm(eu, axis=-1) * np.linalg.norm(ev, axis=-1) * np.linalg.norm(ew, axis=-1)
                            det = np.linalg.det(np.stack((eu, ev, ew), axis=-1))
                            scaled.append(np.divide(det, den, out=np.full_like(den, -np.inf), where=den > tolerance))
                        if need_cosines:
                            outgoing = (eu*(1-2*ci), ev*(1-2*cj), ew*(1-2*ck))
                            for first, second in ((outgoing[0], outgoing[1]), (outgoing[0], outgoing[2]), (outgoing[1], outgoing[2])):
                                den = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
                                dot = np.einsum("ij,ij->i", first, second)
                                cosines.append(np.clip(np.divide(dot, den, out=np.full_like(den, np.nan), where=den > tolerance), -1., 1.))
        if need_scaled:
            sj = np.min(scaled, axis=0)
            values["minimum_scaled_jacobian"] = sj
            values["flipped_cell"] = (sj <= 0).astype(float)
        if need_cosines:
            cosine = np.stack(cosines, axis=1)
            finite = np.isfinite(cosine)
            any_finite = finite.any(axis=1)
            if "orthogonality" in selected:
                values["orthogonality"] = np.where(any_finite, 1 - np.max(np.where(finite, np.abs(cosine), -np.inf), axis=1), np.nan)
            if need_angles:
                angles = np.degrees(np.arccos(cosine))
                values["min_angle"] = np.where(any_finite, np.min(np.where(finite, angles, np.inf), axis=1), np.nan)
                values["max_angle"] = np.where(any_finite, np.max(np.where(finite, angles, -np.inf), axis=1), np.nan)
                values["skewness"] = np.where(any_finite, np.max(np.where(finite, np.abs(angles - 90), -np.inf), axis=1) / 90, np.nan)
        if need_lengths:
            lengths = np.concatenate([np.linalg.norm(a-b, axis=-1).reshape(stop-start, 4) for a,b in ((c[:, :, :, 1], c[:, :, :, 0]), (c[:, :, 1, :], c[:, :, 0, :]), (c[:, 1, :, :], c[:, 0, :, :]))], axis=1)
            shortest = lengths.min(axis=1)
            values["edge_aspect_ratio"] = lengths.max(axis=1) / np.maximum(shortest, tolerance)
        if need_center:
            eu = np.mean(c[:, :, :, 1] - c[:, :, :, 0], axis=(1, 2))
            ev = np.mean(c[:, :, 1, :] - c[:, :, 0, :], axis=(1, 2))
            ew = np.mean(c[:, 1, :, :] - c[:, 0, :, :], axis=(1, 2))
            jac = np.stack((eu, ev, ew), axis=-1)
            if need_det:
                det = np.linalg.det(jac)
                values["center_jacobian"] = det
                values["cell_volume"] = np.abs(det)
            if "center_condition_number" in selected:
                values["center_condition_number"] = np.linalg.cond(jac)
        if "degenerate_cell" in selected:
            values["degenerate_cell"] = ((np.abs(det) <= tolerance) | (shortest <= tolerance)).astype(float)
        for name in result:
            result[name][start:stop] = values[name]
    return result
