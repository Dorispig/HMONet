"""Batch the registered 2-D/3-D parameter datasets without changing defaults."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NEW_FAMILIES = {
    2: ("symmetric_convex_tri", "gear_spline", "four_convex_arcs"),
    3: ("square_to_round_transition", "variable_area_s_duct", "twisted_blade_passage"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dimension", choices=("2", "3", "both"), default="both")
    parser.add_argument("--families", nargs="+", default=["all"],
                        help="all (default: all registered default families), new (optional subset), or names/prefixes")
    parser.add_argument("--split", choices=("ext", "int", "both"), default="ext")
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--version", type=int, help="Omitted: use each family's registered revision")
    parser.add_argument('--solver-backend', choices=('cpu','cuda'), default='cpu', help='3-D PDE generation backend; CPU by default')
    parser.add_argument('--gpus', type=int, nargs='+', help='CUDA-visible indices; requires --solver-backend cuda')
    parser.add_argument('--queue-log-root', type=Path, help='GPU logs root; default outputs/gpu_queue/generation')
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--winslow-max-iter", type=int, default=1000)
    parser.add_argument("--winslow-tolerance", type=float, default=1e-6)
    parser.add_argument("--timing-samples", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate existing 3-D datasets. Existing 2-D datasets are always skipped.")
    args = parser.parse_args(argv)
    if args.solver_backend == 'cuda' and args.dimension != '3':
        parser.error('GPU dataset generation currently requires --dimension 3')
    if args.gpus and args.solver_backend != 'cuda':
        parser.error('--gpus requires --solver-backend cuda')
    from src.data.generation import make_dataset_parametric_2d as two
    from src.data.generation import make_dataset_parametric_3d as three
    dimensions = (2, 3) if args.dimension == "both" else (int(args.dimension),)
    if args.dimension == "both" and args.families not in (["new"], ["all"]):
        parser.error("For named families specify --dimension 2 or 3")
    if args.size < 2 or args.workers < 1:
        parser.error("--size must be >= 2 and --workers must be >= 1")
    jobs = []
    for dimension in dimensions:
        generator = two if dimension == 2 else three
        registered = {item[0]: item for item in generator.SELECTED_DATASETS}
        families = (NEW_FAMILIES[dimension] if args.families == ["new"] else
                    tuple(registered) if args.families == ["all"] else
                    tuple(generator.resolve_family(name) for name in args.families))
        for family in dict.fromkeys(families):
            spec = generator.DATASET_FAMILIES[family]
            version = args.version if args.version is not None else registered.get(family, (None, None, 1))[2]
            for split in (("ext", "int") if args.split == "both" else (args.split,)):
                shape = spec["grid_shape"]
                name = generator.build_dataset_name(family, split, version, args.size, *shape)
                target = args.output_root / f"{dimension}D" / name
                print(f"{dimension}D: {name}; {spec['parameter']}={spec[generator.normalize_split(split)]}; destination={target}", flush=True)
                jobs.append((dimension, generator, family, split, version, shape, target))
    if args.dry_run:
        return jobs
    if args.solver_backend == 'cuda':
        three.generate_selected_datasets(
            selected=[(family, split, version, *shape)
                      for _, _, family, split, version, shape, _ in jobs],
            size=args.size, output_root=args.output_root/'3D',
            solver_backend='cuda', gpus=args.gpus, queue_log_root=args.queue_log_root,
            workers=args.workers, winslow_max_iter=args.winslow_max_iter,
            winslow_tolerance=args.winslow_tolerance, timing_samples=args.timing_samples,
            overwrite=args.force)
        return jobs
    for dimension, generator, family, split, version, shape, target in jobs:
        if target.exists() and (dimension == 2 or not args.force):
            required = ("dataset.npy", "boundary_geometry.npy") if dimension == 2 else ("metadata.json", "geometry.json", "boundary_samples.npy", "pde_mesh.npy")
            if not all((target / filename).is_file() for filename in required):
                raise FileExistsError(f"Incomplete dataset: {target}; inspect it before regeneration")
            if dimension == 3:
                generator.validate_existing_dataset(target)
            print(f"[skip] Dataset already exists: {target}", flush=True)
            continue
        common = dict(size=args.size, version=version, nx=shape[0], ny=shape[1],
                      workers=args.workers, winslow_max_iter=args.winslow_max_iter,
                      winslow_tolerance=args.winslow_tolerance, timing_samples=args.timing_samples)
        if dimension == 2:
            generator.generate_dataset(family, split, output_dir=target.parent, **common)
        else:
            generator.generate_family_dataset(family, split, nz=shape[2], output_root=target.parent, overwrite=args.force,
                solver_backend=args.solver_backend,gpus=args.gpus,
                queue_log_root=args.queue_log_root,
                **common)
    return jobs


if __name__ == "__main__":
    main()
