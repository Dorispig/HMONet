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

from src.models import HMONet2D, PIMeshONet2D, QGHMONet2D


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
    parser.add_argument("--repeats", type=int, default=10)
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
        options = {**common, **base, **{k: v for k, v in changes.items() if k != "model"}}
        options.pop("residual_regularization_weight", None)
        cls = QGHMONet2D if changes.get("model") == "qghmonet" else HMONet2D
        model = cls(41, 41, **options).to(device).eval()
        px, py = boundary.unbind(-1)
        with torch.inference_mode():
            for _ in range(3):
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
            "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "forward_seconds_mean": sum(samples) / len(samples),
            "forward_seconds_minimum": min(samples),
        })
    pi_options = {**common, **config["model"].get("pimeshonet", {})}
    pi_options.pop("architecture", None)
    pi = PIMeshONet2D(41, 41, **pi_options).to(device).eval()
    with torch.inference_mode():
        for _ in range(3):
            pi(px, py, xi, eta)
        synchronize(device)
        samples = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            pi(px, py, xi, eta)
            synchronize(device)
            samples.append(time.perf_counter() - started)
    rows.append({
        "model": "PI-MeshONet",
        "parameters": sum(p.numel() for p in pi.parameters() if p.requires_grad),
        "forward_seconds_mean": sum(samples) / len(samples),
        "forward_seconds_minimum": min(samples),
    })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
