"""Audit the exact parameter-surface grids of the six 3-D benchmark families.

This is deliberately a boundary-only diagnostic.  It verifies that the
sampled face grids are locally nondegenerate and free of bow-tie quads, checks
that the TFI volume map reproduces those boundary samples, and performs a
triangle-level collision audit.  The paired y_min/y_max faces in every O-grid
family are an intentional duplicate seam and are excluded from collision
counts.

Run from the repository root:
    python scripts/visualize/audit_3d_boundary_surfaces.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.generation.make_dataset_parametric_3d import DATASET_FAMILIES
from src.data.geometry.boundary_generator_3d import generate_boundary
from src.data.mesh.sampling import sample_faces
from src.evaluation.quality_3d import boundary_surface_grids, surface_quad_quality_fields
from src.numerics.traditional.tfi_3d import tfi_from_surfaces


CANONICAL_3D = (
    ("cylindrical_domain_with_airfoil_hole", "3D-1"),
    ("sidewall_deformed_block", "3D-2"),
    ("rounded_u_groove_extrusion", "3D-3"),
    ("tapered_elliptic_bore_block", "3D-4"),
    ("bent_annular_duct", "3D-5"),
    ("lobed_annular_duct", "3D-6"),
)
SEAM_FACE_PAIR = frozenset(("y_min", "y_max"))
LOCAL_METRICS = (
    "jacobian",
    "cell_area",
    "self_intersection",
    "flipping_cell",
    "degenerate_cell",
)


def _parse_shape(value: str) -> tuple[int, int, int]:
    try:
        shape = tuple(int(item) for item in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must look like 11x31x11") from error
    if len(shape) != 3 or min(shape) < 3:
        raise argparse.ArgumentTypeError("shape must contain three integers >= 3")
    return shape


def _parameters(specification: dict, value: float) -> dict:
    return {
        **specification["fixed_parameters"],
        specification["parameter"]: float(value),
    }


def _boundary_mask(shape: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = shape
    mask = np.zeros((nz, ny, nx), dtype=bool)
    mask[[0, -1], :, :] = True
    mask[:, [0, -1], :] = True
    mask[:, :, [0, -1]] = True
    return mask


def _surface_triangles(surface: np.ndarray, face_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a structured physical face using its two quad diagonals."""
    lower_left = surface[:-1, :-1]
    lower_right = surface[:-1, 1:]
    upper_right = surface[1:, 1:]
    upper_left = surface[1:, :-1]
    first = np.stack((lower_left, lower_right, upper_right), axis=2).reshape(-1, 3, 3)
    second = np.stack((lower_left, upper_right, upper_left), axis=2).reshape(-1, 3, 3)
    triangles = np.concatenate((first, second), axis=0)
    return triangles, np.full(len(triangles), face_name, dtype=object)


def _share_vertex(first: np.ndarray, second: np.ndarray, tolerance: float) -> bool:
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=-1)
    return bool(np.any(distances <= tolerance))


def _triangles_intersect(first: np.ndarray, second: np.ndarray, tolerance: float) -> bool:
    """Use a separating-axis test, including coplanar in-plane axes."""
    first_edges = np.roll(first, -1, axis=0) - first
    second_edges = np.roll(second, -1, axis=0) - second
    first_normal = np.cross(first_edges[0], first_edges[1])
    second_normal = np.cross(second_edges[0], second_edges[1])
    axes = [first_normal, second_normal]
    axes.extend(np.cross(edge_a, edge_b) for edge_a in first_edges for edge_b in second_edges)
    # These in-plane axes are required when both triangles are coplanar.
    axes.extend(np.cross(first_normal, edge) for edge in first_edges)
    axes.extend(np.cross(second_normal, edge) for edge in second_edges)
    for axis in axes:
        length = np.linalg.norm(axis)
        if length <= tolerance:
            continue
        axis = axis / length
        projection_first = first @ axis
        projection_second = second @ axis
        if (
            projection_first.max() < projection_second.min() - tolerance
            or projection_second.max() < projection_first.min() - tolerance
        ):
            return False
    return True


def _collision_audit(surface_grids: dict[str, np.ndarray]) -> dict:
    """Find sampled triangle collisions excluding topology-valid contacts."""
    pieces = [_surface_triangles(grid, name) for name, grid in surface_grids.items()]
    triangles = np.concatenate([piece[0] for piece in pieces], axis=0)
    face_names = np.concatenate([piece[1] for piece in pieces], axis=0)
    bounds_min = triangles.min(axis=1)
    bounds_max = triangles.max(axis=1)
    scale = max(float(np.ptp(triangles.reshape(-1, 3), axis=0).max()), 1.0)
    tolerance = 1e-10 * scale
    candidates = 0
    tested = 0
    collision_count = 0
    hits: list[dict[str, object]] = []
    for first_index in range(len(triangles) - 1):
        overlap = np.all(
            (bounds_min[first_index + 1 :] <= bounds_max[first_index] + tolerance)
            & (bounds_max[first_index + 1 :] >= bounds_min[first_index] - tolerance),
            axis=1,
        )
        for second_index in np.flatnonzero(overlap) + first_index + 1:
            first_face, second_face = face_names[first_index], face_names[second_index]
            if frozenset((first_face, second_face)) == SEAM_FACE_PAIR:
                continue
            if _share_vertex(triangles[first_index], triangles[second_index], tolerance):
                continue
            candidates += 1
            if _triangles_intersect(triangles[first_index], triangles[second_index], tolerance):
                tested += 1
                collision_count += 1
                if len(hits) < 20:
                    hits.append(
                        {
                            "first_triangle": int(first_index),
                            "second_triangle": int(second_index),
                            "faces": [str(first_face), str(second_face)],
                        }
                    )
            else:
                tested += 1
    return {
        "triangle_count": int(len(triangles)),
        "nonadjacent_aabb_candidates": int(candidates),
        "triangle_pairs_tested": int(tested),
        "collision_pair_count": int(collision_count),
        "collision_examples": hits,
        "excluded_intentional_seam_pair": ["y_min", "y_max"],
        "tolerance": float(tolerance),
    }


def _local_quality(surface_grids: dict[str, np.ndarray]) -> dict:
    output = {}
    for name, surface in surface_grids.items():
        fields = surface_quad_quality_fields(surface, metrics=LOCAL_METRICS)
        output[name] = {
            "quad_count": int(len(fields["cell_area"])),
            "minimum_cell_area": float(np.min(fields["cell_area"])),
            "minimum_local_jacobian": float(np.min(fields["jacobian"])),
            "self_intersecting_quad_count": int(np.count_nonzero(fields["self_intersection"])),
            "flipping_quad_count": int(np.count_nonzero(fields["flipping_cell"])),
            "degenerate_quad_count": int(np.count_nonzero(fields["degenerate_cell"])),
        }
    return output


def _sample_surface_grids(boundary, shape: tuple[int, int, int]) -> tuple[dict, np.ndarray]:
    _, full = sample_faces(boundary, shape)
    return boundary_surface_grids(full), full


def _audit_case(family: str, label: str, value: float, shape, dense_shape) -> dict:
    specification = DATASET_FAMILIES[family]
    boundary = generate_boundary(specification["function"], **_parameters(specification, value))
    surface_grids, direct_boundary_grid = _sample_surface_grids(boundary, shape)
    dense_grids, _ = _sample_surface_grids(boundary, dense_shape)
    tfi_grid = tfi_from_surfaces(boundary, xi=shape[0], eta=shape[1], zeta=shape[2])
    mask = _boundary_mask(shape)
    return {
        "case": label,
        "family": family,
        "parameter": specification["parameter"],
        "parameter_value": float(value),
        "grid_shape": list(shape),
        "dense_local_grid_shape": list(dense_shape),
        "tfi_boundary_max_abs_error": float(
            np.max(np.abs(tfi_grid[mask] - direct_boundary_grid[mask]))
        ),
        "default_grid_local_face_quality": _local_quality(surface_grids),
        "dense_grid_local_face_quality": _local_quality(dense_grids),
        "default_grid_global_collision_audit": _collision_audit(surface_grids),
    }


def _case_summary(record: dict) -> dict:
    qualities = list(record["dense_grid_local_face_quality"].values())
    collisions = record["default_grid_global_collision_audit"]
    return {
        "case": record["case"],
        "parameter": record["parameter"],
        "value": record["parameter_value"],
        "tfi_boundary_error": record["tfi_boundary_max_abs_error"],
        "min_area_dense": min(item["minimum_cell_area"] for item in qualities),
        "min_local_jacobian_dense": min(item["minimum_local_jacobian"] for item in qualities),
        "local_self_intersections_dense": sum(item["self_intersecting_quad_count"] for item in qualities),
        "flipped_quads_dense": sum(item["flipping_quad_count"] for item in qualities),
        "degenerate_quads_dense": sum(item["degenerate_quad_count"] for item in qualities),
        "nonadjacent_triangle_collisions_default": collisions["collision_pair_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-shape", type=_parse_shape, default=(11, 31, 11))
    parser.add_argument("--dense-grid-shape", type=_parse_shape, default=(41, 121, 41))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/boundary_catalog/3d_boundary_surface_audit.json"),
    )
    arguments = parser.parse_args()
    records = []
    for family, label in CANONICAL_3D:
        specification = DATASET_FAMILIES[family]
        lower = min(interval[0] for interval in specification["extrapolation"])
        upper = max(interval[1] for interval in specification["extrapolation"])
        for endpoint, value in (("ext_min", lower), ("ext_max", upper)):
            record = _audit_case(
                family,
                f"{label}_{endpoint}",
                value,
                arguments.grid_shape,
                arguments.dense_grid_shape,
            )
            records.append(record)
            summary = _case_summary(record)
            print(
                "{case}: {parameter}={value:g}; boundary_error={tfi_boundary_error:.2e}; "
                "min_area={min_area_dense:.2e}; bowties={local_self_intersections_dense}; "
                "flipped={flipped_quads_dense}; degenerate={degenerate_quads_dense}; "
                "nonadjacent_collisions={nonadjacent_triangle_collisions_default}".format(**summary),
                flush=True,
            )
    payload = {
        "purpose": "Boundary parameter-surface audit; not an interior TFI-grid quality report.",
        "intentional_parameter_seam": {
            "faces": ["y_min", "y_max"],
            "description": "The two logical seam faces coincide physically in these O-grid geometries and are excluded from collision counting.",
        },
        "records": records,
        "summaries": [_case_summary(record) for record in records],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
