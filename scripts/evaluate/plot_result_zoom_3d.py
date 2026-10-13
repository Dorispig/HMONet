"""Plot 2-D section/3-D volume zooms from saved meshes; no inference required."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluation.zoom_3d import generate_zoom


def pair(text):
    try:
        values = tuple(float(v) for v in text.split(","))
        if len(values) != 2:
            raise ValueError()
        return values
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use two comma-separated numbers") from error


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result_dir", nargs="?", type=Path)
    p.add_argument("--batch-root", type=Path)
    p.add_argument("--mode", choices=["slice", "volume"], default="slice")
    p.add_argument("--slice", choices=["x_mid", "y_mid", "z_mid"], default="z_mid")
    p.add_argument(
        "--slice-coordinates",
        choices=["projected", "physical"],
        default="physical",
        help="Section display/selection frame: PCA U/V or physical YZ/XZ/XY for x/y/z sections",
    )
    p.add_argument("--position", type=float, default=0.5)
    p.add_argument(
        "--axis", choices=["x", "y", "z"], help="Legacy alias for section axis"
    )
    p.add_argument("--index", type=int, help="Exact section node-layer index")
    p.add_argument(
        "--region", nargs=4, type=int, help="Legacy half-open cell bounds: i0 i1 j0 j1"
    )
    for c in ("i", "j", "k"):
        p.add_argument(
            f"--{c}-range", type=pair, help="Half-open cell interval, e.g. 2,8"
        )
    for c in ("x", "y", "z"):
        p.add_argument(
            f"--{c}lim",
            type=pair,
            help="Slice: horizontal/vertical limits in --slice-coordinates frame; volume: physical XYZ limits",
        )
    p.add_argument("--cell-index", type=int)
    p.add_argument("--center-on", choices=["worst"])
    p.add_argument("--reference")
    p.add_argument("--cell-padding", type=int, default=2)
    p.add_argument("--methods", help="Comma-separated method names")
    p.add_argument("--metric")
    p.add_argument("--plot", choices=["mesh", "quality", "both"], default="both")
    p.add_argument("--color-scale", choices=["global", "local"], default="global")
    p.add_argument(
        "--volume-style",
        choices=["surface", "wireframe", "transparent"],
        default="surface",
    )
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--elevation", type=float, default=24.0)
    p.add_argument("--azimuth", type=float, default=-58.0)
    p.add_argument("--cut-axis", choices=["x", "y", "z"])
    p.add_argument("--cut-position", type=float)
    p.add_argument("--name")
    p.add_argument("--output-dir", type=Path)
    for flag, default in (
        ("save-png", True),
        ("save-pdf", False),
        ("show-axes", True),
        ("save-plots", True),
    ):
        p.add_argument(
            f"--{flag}", action=argparse.BooleanOptionalAction, default=default
        )
    p.add_argument("--force", action="store_true")
    return p


def main():
    parser = build_parser()
    args = vars(parser.parse_args())
    directory, batch = args.pop("result_dir"), args.pop("batch_root")
    if (directory is None) == (batch is None):
        parser.error("Provide either result_dir or --batch-root")
    if batch and args["output_dir"]:
        parser.error("--output-dir is only supported for one result")
    axis, region = args.pop("axis"), args.pop("region")
    if axis:
        args["slice"] = axis + "_mid"
    if region:
        if args["i_range"] or args["j_range"]:
            parser.error("--region cannot be combined with --i-range/--j-range")
        args["i_range"], args["j_range"] = region[:2], region[2:]
    if args["methods"]:
        args["methods"] = [m.strip() for m in args["methods"].split(",")]
    options = {k: v for k, v in args.items() if v is not None}
    directories = (
        [directory]
        if directory
        else sorted({p.parent for p in batch.rglob("comparison_mesh_data.npz")})
    )
    if not directories:
        parser.error("No saved comparison meshes found")
    failures = []
    for index, source in enumerate(directories, 1):
        try:
            output = generate_zoom(source, options)
            print(f"[{index}/{len(directories)}] Saved: {output}", flush=True)
        except (ValueError, OSError, KeyError) as error:
            if not batch:
                raise
            failures.append((str(source), str(error)))
            print(f"FAILED: {source}: {error}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
