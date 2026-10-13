"""Unified Attention-PI-HMONet trainer with a 2-D trainable-alpha strategy."""

from __future__ import annotations

import copy
import numpy as np
import torch
from tqdm import tqdm

from scripts.train.attention_pihmonet_common import (
    autograd_physics_losses,
    autograd_normalized_pde_loss,
    build_boundary_inputs,
    finite_difference_physics_losses,
    finite_difference_physics_losses_at_points,
    finite_difference_normalized_pde_loss,
    predict_attention,
    sample_boundary_logical_points,
)
from scripts.train.pi_anchor import (
    central_anchor_queries,
    masked_anchor_mse,
    tfi_anchor_targets,
)
from scripts.train.training_common import (
    BatchLossTotals,
    make_data_loader,
    safe_optimizer_step,
    seed_everything,
    update_learning_rate,
)
from scripts.train.training_state import TrainingState, prepare_training_config
from src.models.model import AttentionHMONet2D, AttentionHMONet3D


def _anchor_weight(initial, epoch, decay_end):
    """Linearly remove the optional TFI continuation loss."""
    initial, decay_end = float(initial), int(decay_end)
    if initial <= 0 or epoch >= decay_end:
        return 0.0
    return initial * (decay_end - epoch) / max(decay_end - 1, 1)


def _shared_interior_points(batch_size, count, dimension, margin, device, dtype):
    """Use one random logical point set for every geometry in a batch."""
    if not 0 <= margin < 0.5:
        raise ValueError("interior_margin must lie in [0, 0.5)")
    if margin == 0:
        margin = torch.finfo(dtype).eps
    points = margin + (1 - 2 * margin) * torch.rand(
        1, count, dimension, device=device, dtype=dtype
    )
    return points.expand(batch_size, -1, -1).clone().requires_grad_(True)


def train_attentionpihmonet(datasets, config, mode, args, device):
    """Train Attention-PI-HMONet through the common output/state pipeline."""
    name = "attentionpihmonet"
    config = copy.deepcopy(config)
    config.setdefault("training", {})["best_model_metric"] = "training_loss"
    config["training"].update(
        check_mse_convergence=False,
        check_max_point_error=False,
        check_p99_point_error=False,
        plateau_metric="train_loss",
    )
    config = prepare_training_config(config, name, args)
    training = config["training"]
    options = config[name + "_training"]
    shape = tuple(datasets[0].grid_shape)
    dimension = len(shape)
    seed_everything(int(training["seed"]))

    model_options = dict(config["model"].get("common", {}))
    model_options.update(config["model"].get(name, {}))
    model_class = AttentionHMONet2D if dimension == 2 else AttentionHMONet3D
    model = model_class(*shape, **model_options).to(device)
    if dimension == 2 and model.hmonet_mapping_mode != "current":
        raise ValueError(
            "The trainable boundary-alpha strategy requires 2-D current mapping"
        )
    model.boundary_sample_counts = options.get("boundary_sample_counts")
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    loader = make_data_loader(
        datasets, training["batch_size"], training["seed"], shuffle=True
    )

    backend = str(options.get("pde_backend", "finite_difference")).lower()
    if backend not in {"autograd", "finite_difference"}:
        raise ValueError("attentionpihmonet_training.pde_backend is invalid")
    count = int(options.get("interior_sample_count", 64 if dimension == 2 else 24))
    boundary_count = int(options.get("boundary_physics_sample_count", 64))
    boundary_loss_count = int(options.get("boundary_loss_sample_count", 64))
    if count < 1 or boundary_count < 2 * dimension or boundary_loss_count < 4:
        raise ValueError("interior/boundary physics sample counts are too small")
    margin = float(options.get("interior_margin", 0.0))
    if margin != 0.0:
        raise ValueError(
            "Attention-PI-HMONet autograd sampling now requires interior_margin=0"
        )
    boundary_offset = float(options.get("boundary_physics_offset", 0.01))
    boundary_tangential_margin = float(
        options.get("boundary_physics_tangential_margin", 0.05)
    )
    steps = options.get("fd_step") or [1 / (value - 1) for value in shape]
    if np.isscalar(steps):
        steps = [float(steps)] * dimension
    if len(steps) != dimension or min(map(float, steps)) <= 0:
        raise ValueError("fd_step must contain one positive value per dimension")
    if max(map(float, steps)) >= 0.5:
        raise ValueError("every fd_step must be smaller than 0.5")

    weights = {
        "pde": float(options.get("normalized_pde_loss_weight", 1.0)),
        "jacobian": float(options.get("jacobian_loss_weight", 0.01)),
        "anchor": float(options.get("anchor_loss_weight", 0.0)),
        "smoothness": float(options.get("smoothness_loss_weight", 1e-4)),
        "boundary_pde": float(options.get("boundary_pde_loss_weight", 1e-6)),
        "boundary_jacobian": float(
            options.get("boundary_jacobian_loss_weight", 0.01)
        ),
        "boundary_coordinate": float(options.get("boundary_loss_weight", 1.0)),
        "alpha_penalty": float(options.get("alpha_penalty_weight", 0.01)),
    }
    if dimension == 2:
        # The requested 2-D strategy uses only normalized PDE, exact-boundary
        # coordinate supervision and the trainable-alpha return penalty.
        disabled = (
            "jacobian", "anchor", "smoothness", "boundary_pde", "boundary_jacobian"
        )
        if any(weights[key] != 0.0 for key in disabled):
            raise ValueError(
                "The 2-D boundary-alpha strategy requires Jacobian, anchor, "
                "smoothness and boundary-physics loss weights to be zero"
            )
        for key in disabled:
            weights[key] = 0.0
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("Attention-PI-HMONet loss weights must be finite and nonnegative")
    if weights["pde"] <= 0:
        raise ValueError("normalized_pde_loss_weight must be positive")
    jacobian_minimum = float(options.get("jacobian_minimum", 1e-4))
    anchor_end = int(options.get("anchor_decay_end_epoch", 1000))
    anchor_count = int(options.get("anchor_points_per_axis", 3))
    anchor_margin = float(options.get("anchor_margin", 0.2))
    anchor_queries = None
    if weights["anchor"] > 0:
        anchor_queries = central_anchor_queries(
            dimension, anchor_count, anchor_margin, device, torch.float32
        )

    state = TrainingState(model, optimizer, datasets, config, mode, args, device)
    try:
        for epoch in range(1, training["num_epochs"] + 1):
            model.train()
            epoch_lr = optimizer.param_groups[0]["lr"]
            totals = BatchLossTotals()
            anchor_weight = _anchor_weight(weights["anchor"], epoch, anchor_end)
            progress = tqdm(
                loader,
                desc=f"Attention-PI-HMONet epoch {epoch}/{training['num_epochs']}",
                leave=True,
            )
            for batch in progress:
                geometries = batch.get("boundary")
                if geometries is None:
                    raise ValueError(
                        "Attention-PI-HMONet requires load_boundary=True"
                    )
                boundary_points = batch["boundary_points"].to(device)
                size, dtype = len(boundary_points), boundary_points.dtype
                boundary_inputs = build_boundary_inputs(
                    geometries,
                    shape,
                    device,
                    dtype,
                    model.boundary_sample_counts,
                )

                if backend == "autograd":
                    logical = _shared_interior_points(
                        size, count, dimension, 0.0, device, dtype
                    )
                    prediction = predict_attention(
                        model, boundary_inputs, logical, geometries
                    )
                    if dimension == 2:
                        pde_only = autograd_normalized_pde_loss(prediction, logical)
                        zero = pde_only.new_zeros(())
                        interior = {
                            "pde_normalized": pde_only,
                            "jacobian": zero,
                            "smoothness": zero,
                            "minimum_jacobian": zero.detach(),
                        }
                    else:
                        interior = autograd_physics_losses(
                            prediction, logical, jacobian_minimum
                        )
                else:
                    if dimension == 2:
                        pde_only = finite_difference_normalized_pde_loss(
                            model, boundary_inputs, geometries, size, count,
                            steps, device, dtype,
                        )
                        zero = pde_only.new_zeros(())
                        interior = {
                            "pde_normalized": pde_only,
                            "jacobian": zero,
                            "smoothness": zero,
                            "minimum_jacobian": zero.detach(),
                        }
                    else:
                        interior = finite_difference_physics_losses(
                            model,
                            boundary_inputs,
                            geometries,
                            size,
                            count,
                            steps,
                            device,
                            dtype,
                            jacobian_minimum,
                        )

                zero = interior["pde_normalized"].new_zeros(())
                boundary_pde = boundary_jacobian = zero
                boundary_minimum = zero.detach()
                boundary_coordinate = alpha_penalty = zero
                if dimension == 2:
                    boundary_logical = sample_boundary_logical_points(
                        size,
                        boundary_loss_count,
                        dimension,
                        device,
                        dtype,
                        offset=0.0,
                        tangential_margin=0.0,
                    ).detach()
                    boundary_prediction = predict_attention(
                        model, boundary_inputs, boundary_logical, geometries
                    )
                    with torch.no_grad():
                        boundary_target = model.evaluate_coons_points_from_edges(
                            *boundary_inputs,
                            boundary_logical[..., 0],
                            boundary_logical[..., 1],
                            boundary_geometry=geometries,
                        )
                    boundary_coordinate = (
                        boundary_prediction - boundary_target
                    ).square().sum(-1).mean()
                    alpha_penalty = (model.boundary_alpha - 1.0).square()
                elif weights["boundary_pde"] > 0 or weights["boundary_jacobian"] > 0:
                    if backend == "autograd":
                        boundary_logical = sample_boundary_logical_points(
                            size,
                            boundary_count,
                            dimension,
                            device,
                            dtype,
                            boundary_offset,
                            boundary_tangential_margin,
                        )
                        boundary_prediction = predict_attention(
                            model, boundary_inputs, boundary_logical, geometries
                        )
                        boundary_physics = autograd_physics_losses(
                            boundary_prediction, boundary_logical, jacobian_minimum
                        )
                    else:
                        # Each logical axis uses exactly its own finite-difference
                        # step as the excluded boundary-strip width.
                        boundary_logical = sample_boundary_logical_points(
                            1,
                            boundary_count,
                            dimension,
                            device,
                            dtype,
                            steps,
                            steps,
                        )
                        boundary_physics = finite_difference_physics_losses_at_points(
                            model,
                            boundary_inputs,
                            geometries,
                            size,
                            boundary_logical[0].detach(),
                            steps,
                            jacobian_minimum,
                        )
                    boundary_pde = boundary_physics["pde_normalized"]
                    boundary_jacobian = boundary_physics["jacobian"]
                    boundary_minimum = boundary_physics["minimum_jacobian"]

                anchor = zero
                valid_anchor_ratio = zero.detach()
                if anchor_weight > 0:
                    logical_anchor = anchor_queries.to(dtype=dtype)[None].expand(
                        size, -1, -1
                    )
                    anchor_prediction = predict_attention(
                        model, boundary_inputs, logical_anchor, geometries
                    )
                    with torch.no_grad():
                        target, valid = tfi_anchor_targets(
                            boundary_points, logical_anchor, shape
                        )
                    anchor = masked_anchor_mse(anchor_prediction, target, valid)
                    valid_anchor_ratio = valid.float().mean()

                if dimension == 2:
                    loss = (
                        weights["pde"] * interior["pde_normalized"]
                        + weights["boundary_coordinate"] * boundary_coordinate
                        + weights["alpha_penalty"] * alpha_penalty
                    )
                else:
                    loss = (
                        weights["pde"] * interior["pde_normalized"]
                        + weights["jacobian"] * interior["jacobian"]
                        + weights["smoothness"] * interior["smoothness"]
                        + anchor_weight * anchor
                        + weights["boundary_pde"] * boundary_pde
                        + weights["boundary_jacobian"] * boundary_jacobian
                    )
                if not safe_optimizer_step(
                    loss,
                    model,
                    optimizer,
                    training["gradient_clip_norm"],
                    training["skip_nonfinite_loss"],
                ):
                    continue
                alpha_metrics = (
                    {
                        "loss_boundary": boundary_coordinate,
                        "alpha_penalty": alpha_penalty,
                        "boundary_alpha": model.boundary_alpha.detach(),
                    }
                    if dimension == 2
                    else {}
                )
                totals.add(
                    size,
                    loss=loss,
                    loss_pde=interior["pde_normalized"],
                    loss_pde_normalized=interior["pde_normalized"],
                    jacobian=interior["jacobian"],
                    anchor=anchor,
                    anchor_weight=anchor_weight,
                    valid_anchor_ratio=valid_anchor_ratio,
                    grid_smoothness=interior["smoothness"],
                    boundary_pde=boundary_pde,
                    boundary_jacobian=boundary_jacobian,
                    minimum_sampled_jacobian=interior["minimum_jacobian"],
                    minimum_boundary_jacobian=boundary_minimum,
                    **alpha_metrics,
                )
                if dimension == 2:
                    progress.set_postfix(
                        loss=loss.item(),
                        pde=interior["pde_normalized"].item(),
                        boundary=boundary_coordinate.item(),
                        alpha=model.boundary_alpha.item(),
                        alpha_penalty=alpha_penalty.item(),
                    )
                else:
                    progress.set_postfix(
                        loss=loss.item(),
                        pde=interior["pde_normalized"].item(),
                        jac=interior["jacobian"].item(),
                        smooth=interior["smoothness"].item(),
                        boundary_pde=boundary_pde.item(),
                        boundary_jac=boundary_jacobian.item(),
                    )
            update_learning_rate(
                optimizer,
                epoch,
                options["lr_warmup_epochs"],
                options["lr_update_interval"],
                options["lr_decay_gamma"],
                options["lr_stop_below"],
            )
            if state.end_epoch(epoch, totals.means(), epoch_lr):
                break
        return state.finish()
    finally:
        if state.writer is not None:
            state.writer.close()
