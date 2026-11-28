"""Evaluate a trained ResMetaMesh decoder with one adaptation per geometry."""
from time import perf_counter

import torch

from src.models.resmetamesh import (
    ResMetaMesh, _synchronize, adapt_shape, decode_mesh, validate_training_options,
)

from .artifacts import digest, read_json


def generate_resmetamesh(geometry, shape, checkpoint, checkpoint_digest, device, options, runtime):
    model_key = ("resmetamesh-model", str(checkpoint), checkpoint_digest, str(device))
    if model_key not in runtime:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = saved.get("model_state_dict", saved)
        config = saved.get("config") or read_json(checkpoint.parent / "config.json")
        if "latents" not in state or not config or "resmetamesh_training" not in config:
            raise ValueError(f"Expected a ResMetaMesh checkpoint and training configuration: {checkpoint}")
        training = config["resmetamesh_training"]
        validate_training_options(training, len(shape))
        architecture = dict(config["model"]["resmetamesh"])
        if "hidden_layers" not in architecture:
            # Legacy custom configurations could omit the decoder depth.
            linear_layers = sum(key.startswith("decoder.") and key.endswith(".weight") for key in state)
            architecture["hidden_layers"] = linear_layers - 1
        model = ResMetaMesh(len(shape), len(state["latents"]), **architecture)
        model.load_state_dict(state)
        summary = read_json(checkpoint.parent / "training_summary.json", {})
        details = {
            "checkpoint": str(checkpoint),
            "epoch": saved.get("epoch"),
            "training_seconds": summary.get("training_time_seconds"),
        }
        runtime[model_key] = (
            model.to(device=device, dtype=torch.float32).eval().requires_grad_(False), training,
            int(config["training"]["seed"]), details,
        )
    model, training, seed, details = runtime[model_key]
    geometry_key = (model_key, digest(geometry.to_dict()))
    if geometry_key not in runtime:
        runtime[geometry_key] = adapt_shape(model, geometry, training, seed, device)
    (latent, left, right), adaptation = runtime[geometry_key]
    _synchronize(device)
    start = perf_counter()
    points = decode_mesh(model, geometry, shape, latent, (left, right), options["query_batch_size"])
    _synchronize(device)
    mesh_seconds = perf_counter() - start
    total = adaptation["adaptation_base_seconds"] + adaptation["adaptation_seconds"] + mesh_seconds
    return points, {
        **details,
        **adaptation,
        "inference_time_basis": "per_shape_adaptation_and_output",
        "output_grid_seconds": mesh_seconds,
        "generation_includes_per_shape_adaptation": True,
        "tfi_precision": "float32",
        "total_seconds": total,
    }
