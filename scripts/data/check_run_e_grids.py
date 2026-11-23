"""Check run_e PDE caps and actual 3-D PDE meshes at selected depths/shapes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.data.generation.make_dataset_parametric_3d import (
    DATASET_FAMILIES, _audit_volume_grid, _validated_boundary_samples,
)
from src.data.geometry.boundary_generator_3d import rounded_u_notch_extrusion_boundary
from src.data.mesh.sampling import boundary_indices, sample_faces
from src.numerics.traditional.tfi_3d import tfi_from_surfaces
from src.numerics.traditional.winslow_3d import solve_winslow_3d


def grid_shape(text):
    try:
        shape = tuple(int(n) for n in text.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use NXxNYxNZ") from error
    if len(shape) != 3 or min(shape) < 3:
        raise argparse.ArgumentTypeError("Use NXxNYxNZ with counts >= 3")
    return shape


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values", nargs="+", type=float, default=[.55, 1.50, 1.52, 1.55])
    parser.add_argument("--grids", nargs="+", type=grid_shape,
                        default=[(21, 21, 11), (61, 21, 11), (51, 51, 11)])
    parser.add_argument("--winslow-max-iter", type=int, default=1000)
    parser.add_argument("--winslow-tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/3D/run_e_pde_caps/grid_validation.json")
    args = parser.parse_args(argv)
    records = []
    fixed = DATASET_FAMILIES["rounded_u_notch_extrusion"]["fixed_parameters"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for depth in args.values:
        started = time.perf_counter()
        geometry = rounded_u_notch_extrusion_boundary(depth=depth, **fixed)
        cap_seconds = time.perf_counter() - started
        for shape in args.grids:
            record = dict(depth=depth, shape=list(shape), cap_seconds=cap_seconds,
                          cap_grid=geometry.z_min.solver_info["grid_shape"],
                          volume_max_iter=args.winslow_max_iter,
                          volume_tolerance=args.winslow_tolerance)
            try:
                _validated_boundary_samples(geometry, shape, 0)
            except ValueError as error:
                record.update(status="invalid_faces", error=str(error))
            else:
                started = time.perf_counter()
                initial = tfi_from_surfaces(geometry, *shape)
                mesh, info = solve_winslow_3d(
                    geometry, *shape, initial_grid=initial,
                    max_iter=args.winslow_max_iter, tolerance=args.winslow_tolerance,
                    return_info=True,
                )
                elapsed = time.perf_counter() - started
                grid = mesh.reshape(*shape[::-1], 3)
                _, expected = sample_faces(geometry, shape)
                indices = boundary_indices(shape)
                error = np.max(np.abs(grid.reshape(-1, 3)[indices]
                                      - expected.reshape(-1, 3)[indices]))
                record.update(status="solved", seconds=elapsed, info=info,
                              quality=_audit_volume_grid(grid), boundary_max_error=float(error))
            records.append(record)
            # Save each finished case; failed prescribed faces are not passed to the solver.
            args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
            print(json.dumps(record), flush=True)
    return records


if __name__ == "__main__":
    main()
