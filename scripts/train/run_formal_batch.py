"""Train and evaluate all default formal data cases in one dimension."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(ROOT))

from src.utils.gpu_queue import GpuTask, parse_gpu_ids, run_gpu_queue

METHODS = "tfi,pde,meshonet,pimeshonet,hmonet,resmetamesh"


def run_resmetamesh_case(dimension, dataset, seed, run_tag, device, config_path, config):
    """Train and evaluate ResMetaMesh, then compare six saved meshes."""
    from scripts.train.train_evaluate import (
        existing_evaluation_commands,
        main as train_evaluate,
    )
    from importlib import import_module

    device_args = ["--device", device] if device else []
    print(f"[{dataset}] ResMetaMesh training and evaluation", flush=True)
    summaries = train_evaluate([
        "--dimension", str(dimension), "--dataset", dataset,
        "--models", "resmetamesh", "--evaluate-methods", "resmetamesh",
        "--checkpoint", "final", "--seed", str(seed), "--run-tag", run_tag,
        *device_args,
    ])
    [summary] = summaries
    checkpoint = Path(summary["final_checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    [command] = existing_evaluation_commands(
        dimension, [dataset], config_path, config, (), "best",
        METHODS.rsplit(",", 1)[0],
    )
    command[command.index("--methods") + 1] = METHODS
    print(f"[{dataset}] Six-method comparison from saved meshes", flush=True)
    evaluate = import_module(f"scripts.evaluate.evaluate_{dimension}d")
    evaluate.main([
        *command, "--resmetamesh-checkpoint", str(checkpoint),
        "--analysis-only", "--no-skip-missing-checkpoints", *device_args,
    ])
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--device", default=None)
    devices.add_argument("--gpus", nargs="+")
    parser.add_argument(
        "--resmetamesh-only", action="store_true",
        help="Train only ResMetaMesh and compare with five existing method results.",
    )
    parser.add_argument("--worker-dataset", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--run-tag", default="v1")
    parser.add_argument("--dry-run", action="store_true",
                        help="List tasks without opening datasets or starting training.")
    args = parser.parse_args(argv)

    from scripts.train.train_evaluate import validate_formal_run_tag
    args.run_tag = validate_formal_run_tag(args.run_tag)
    dimension = f"{args.dimension}D"
    config_path = ROOT / "configs" / dimension / "evaluate.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    datasets = config["evaluation"]["default_cases"]
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError(f"{config_path}: expected a nonempty list of unique default cases")
    if args.worker_dataset is not None:
        if (not args.resmetamesh_only or args.gpus
                or args.worker_dataset not in datasets
                or args.worker_dataset not in config["cases"]
                or Path(args.worker_dataset).name != args.worker_dataset):
            parser.error("Invalid ResMetaMesh batch worker arguments")
        return run_resmetamesh_case(
            args.dimension, args.worker_dataset, args.seed, args.run_tag,
            args.device, config_path, config,
        )

    commands = {}
    for dataset in datasets:
        if dataset not in config["cases"] or Path(dataset).name != dataset:
            raise ValueError(f"Invalid default case in {config_path}: {dataset!r}")
        command = [
            sys.executable, "-u", "scripts/train/train_evaluate.py",
            "--dimension", str(args.dimension), "--dataset", dataset,
            "--models", "formal-four", "--evaluate-methods", METHODS,
            "--seed", str(args.seed), "--run-tag", args.run_tag,
        ]
        if args.resmetamesh_only:
            command = [
                sys.executable, "-u", "scripts/train/run_formal_batch.py",
                "--dimension", str(args.dimension), "--resmetamesh-only",
                "--worker-dataset", dataset, "--seed", str(args.seed),
                "--run-tag", args.run_tag,
            ]
        if args.gpus or args.device:
            command += ["--device", "cuda:0" if args.gpus else args.device]
        commands[dataset] = command

    if args.gpus:
        log_root = ROOT / "outputs/gpu_queue/formal" / (
            Path(dimension) / "resmetamesh"
            if args.resmetamesh_only else Path("batch") / dimension
        ) / f"seed_{args.seed}" / args.run_tag
        records = run_gpu_queue(
            [GpuTask(name, command, log_relative_path=Path(name) / "run.log")
             for name, command in commands.items()],
            parse_gpu_ids(args.gpus), cwd=ROOT, log_root=log_root,
            continue_on_error=True, dry_run=args.dry_run,
        )
        if args.dry_run:
            return 0
        print(f"Queue summary: {log_root / 'queue_summary.json'}", flush=True)
        failed = [record["task"] for record in records if record["status"] != "completed"]
    elif args.dry_run:
        for command in commands.values():
            print(shlex.join(command))
        return 0
    else:
        failed = []
        for name, command in commands.items():
            print(f"[formal batch] {name}", flush=True)
            if subprocess.run(command, cwd=ROOT).returncode:
                failed.append(name)

    print("Failed datasets: " + (", ".join(failed) if failed else "none"), flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
