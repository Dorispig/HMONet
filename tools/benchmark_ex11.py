"""Report Ex11 model sizes and raw forward latency on one 101x101 query grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import HMONet2D, MeshONet2D, PIMeshONet2D, QGHMONet2D


VARIANTS = {
    "baseline": {},
    "e1_modified_mlp": {"modified_mlp_enabled": True},
    "e2_cross_attention": {"query_boundary_cross_attention_enabled": True},
    "e3_boundary_decoder": {"boundary_layer_decoder_enabled": True},
    "e4_decoupled_weights": {"decoupled_weight_decoder_enabled": True},
    "e5_qg_hmonet": {"model": "qghmonet"},
    "e6_modified_cross": {
        "modified_mlp_enabled": True,
        "query_boundary_cross_attention_enabled": True,
    },
    "e7_modified_cross_boundary": {
        "modified_mlp_enabled": True,
        "query_boundary_cross_attention_enabled": True,
        "boundary_layer_decoder_enabled": True,
    },
    "e8_full_factorized": {
        "modified_mlp_enabled": True,
        "query_boundary_cross_attention_enabled": True,
        "boundary_layer_decoder_enabled": True,
        "decoupled_weight_decoder_enabled": True,
    },
    "e9_meshonet_200k": {
        "model": "meshonet",
        "branch_dim": 40,
        "adjust_hidden_dim": 80,
        "branch_hidden_dim": 80,
        "trunk_hidden_dim": 80,
        "output_dim": 80,
        "branch_mlp_hidden_dim": 80,
    },
    "e10_pimeshonet_200k": {
        "model": "pimeshonet",
        "branch_dim": 40,
        "adjust_hidden_dim": 80,
        "branch_hidden_dim": 80,
        "trunk_hidden_dim": 80,
        "output_dim": 80,
    },
}


def square_boundary(size, device):
    t = torch.linspace(0, 1, size, device=device)
    bottom = torch.stack((t, torch.zeros_like(t)), -1)
    right = torch.stack((torch.ones_like(t[1:]), t[1:]), -1)
    top_t = t[:-1].flip(0)
    left_t = t[1:-1].flip(0)
    top = torch.stack((top_t, torch.ones_like(top_t)), -1)
    left = torch.stack((torch.zeros_like(left_t), left_t), -1)
    return torch.cat((bottom, right, top, left), 0)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/ex11/baseline/experiment.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    device = torch.device(args.device)
    config = yaml.safe_load(open(args.config, encoding="utf-8-sig"))
    common = dict(config["model"]["common"])
    base = dict(config["model"]["hmonet"])
    boundary = square_boundary(41, device).unsqueeze(0)
    query_axis = torch.linspace(0, 1, 101, device=device)
    eta, xi = torch.meshgrid(query_axis, query_axis, indexing="ij")
    xi, eta = xi.reshape(1, -1), eta.reshape(1, -1)
    rows = []
    for name, changes in VARIANTS.items():
        model_name = changes.get("model", "hmonet")
        if model_name == "meshonet":
            options = {
                **common,
                **config["model"].get("meshonet", {}),
                **{k: v for k, v in changes.items() if k != "model"},
            }
            cls = MeshONet2D
        elif model_name == "pimeshonet":
            options = {
                **common,
                **config["model"].get("pimeshonet", {}),
                **{k: v for k, v in changes.items() if k != "model"},
            }
            options.pop("architecture", None)
            cls = PIMeshONet2D
        else:
            options = {
                **common,
                **base,
                **{k: v for k, v in changes.items() if k != "model"},
            }
            options.pop("residual_regularization_weight", None)
            cls = QGHMONet2D if model_name == "qghmonet" else HMONet2D
        model = cls(41, 41, **options).to(device).eval()
        px, py = boundary.unbind(-1)
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(px, py, xi, eta)
            synchronize(device)
            samples = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                model(px, py, xi, eta)
                synchronize(device)
                samples.append(time.perf_counter() - started)
        rows.append({
            "model": name,
            "architecture": model_name,
            "capacity": {
                key: options[key]
                for key in (
                    "branch_dim", "trunk_dim", "adjust_hidden_dim",
                    "adjust_hidden_layer_num", "branch_hidden_dim",
                    "branch_hidden_layer_num", "trunk_hidden_dim",
                    "trunk_hidden_layer_num", "output_dim",
                    "branch_mlp_hidden_dim", "branch_mlp_hidden_layer_num",
                )
                if key in options
            },
            "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "forward_seconds_mean": sum(samples) / len(samples),
            "forward_seconds_minimum": min(samples),
        })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
