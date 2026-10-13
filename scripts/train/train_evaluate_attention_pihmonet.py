"""Train then immediately evaluate Attention-PI-HMONet in 2-D or 3-D."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from scripts.evaluate.evaluate_attention_pihmonet import evaluate_dataset
from scripts.train.train_attention_pihmonet import ROOT, _resolve, train_dataset


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--evaluate-config", type=Path)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--pde-backend", choices=("autograd", "finite_difference"))
    parser.add_argument("--grid-shapes")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    train_path = args.train_config or ROOT / f"configs/{args.dimension}D/attention_pihmonet_train.yaml"
    eval_path = args.evaluate_config or ROOT / f"configs/{args.dimension}D/attention_pihmonet_evaluate.yaml"
    train_config = yaml.safe_load(_resolve(train_path).read_text(encoding="utf-8-sig"))
    eval_config = yaml.safe_load(_resolve(eval_path).read_text(encoding="utf-8-sig"))
    datasets = args.dataset or train_config["training"].get("datasets", [])
    overrides = {key: value for key, value in {
        "num_iterations": args.iterations, "batch_size": args.batch_size,
        "seed": args.seed, "device": args.device,
    }.items() if value is not None}
    if args.pde_backend:
        train_config.setdefault("pde_loss", {})["backend"] = args.pde_backend
    if args.device:
        eval_config.setdefault("evaluation", {})["device"] = args.device
    if args.grid_shapes:
        eval_config.setdefault("evaluation", {})["grid_shapes"] = [
            [int(value) for value in item.split("x")] for item in args.grid_shapes.split(",")
        ]
    results = []
    for dataset in datasets:
        training = train_dataset(dataset, train_config, args.dimension, overrides)
        evaluation = evaluate_dataset(
            dataset, eval_config, args.dimension, training["checkpoint"]
        )
        results.append({"dataset": dataset, "training": training, "evaluation": evaluation})
    return results


if __name__ == "__main__":
    main()
