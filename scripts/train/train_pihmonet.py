"""Self-supervised PI-HMONet training shared by 2-D and 3-D entry points."""

import copy
import numpy as np
import torch
from tqdm import tqdm

from src.models.model import HMONet2D, HMONet3D
from scripts.train.physics_loss import (
    stencil_queries,
    elliptic_residual,
    normalized_pde_loss,
    jacobian_positive_loss,
    grid_line_smoothness_loss,
    orthogonality_loss,
)
from scripts.train.pi_anchor import (
    central_anchor_queries,
    masked_anchor_mse,
    tfi_anchor_targets,
)
from scripts.train.training_common import (
    BatchLossTotals,
    make_data_loader,
    predict_coordinates,
    safe_optimizer_step,
    seed_everything,
    update_learning_rate,
)
from scripts.train.training_state import TrainingState, prepare_training_config


def _anchor_weight(initial_weight, epoch, decay_end_epoch):
    """Linearly decay from the initial weight at epoch 1 to zero at the end."""
    if initial_weight <= 0 or epoch >= decay_end_epoch:
        return 0.0
    if decay_end_epoch <= 1:
        return 0.0
    return initial_weight * (decay_end_epoch - epoch) / (decay_end_epoch - 1)


def train_pihmonet(datasets, config, mode, args, device):
    """Optimize PI-HMONet with finite-difference physics and optional priors."""
    config = copy.deepcopy(config)
    pi = config.setdefault("pihmonet_training", {})
    defaults = dict(
        pde_loss_weight=0.0,
        normalized_pde_loss_weight=1.0,
        interior_sample_count=128,
        fd_step=None,
        jacobian_loss_enabled=True,
        jacobian_loss_weight=0.1,
        jacobian_minimum=1e-4,
        anchor_loss_weight=0.01,
        anchor_decay_end_epoch=1000,
        anchor_points_per_axis=3,
        anchor_margin=0.2,
        smoothness_loss_weight=0.0,
        orthogonality_loss_weight=0.0,
    )
    for key, value in defaults.items():
        override = getattr(args, "pihmonet_" + key, None)
        if override is None:
            override = getattr(args, "pi_" + key, None)
        if override is None and key == "interior_sample_count":
            override = getattr(args, "interior_sample_count", None)
        if override is None and key in {
            "jacobian_loss_enabled",
            "jacobian_loss_weight",
            "jacobian_minimum",
        }:
            override = getattr(args, key, None)
        pi[key] = override if override is not None else pi.get(key, value)

    config.setdefault("training", {})["best_model_metric"] = "training_loss"
    config["training"].update(
        check_mse_convergence=False,
        check_max_point_error=False,
        check_p99_point_error=False,
        plateau_metric="train_loss",
    )
    config = prepare_training_config(config, "pihmonet", args)
    training = config["training"]
    pi = config["pihmonet_training"]
    shape = tuple(datasets[0].grid_shape)
    dimension = len(shape)

    weight_keys = (
        "pde_loss_weight",
        "normalized_pde_loss_weight",
        "jacobian_loss_weight",
        "anchor_loss_weight",
        "smoothness_loss_weight",
        "orthogonality_loss_weight",
    )
    weights = {key: float(pi[key]) for key in weight_keys}
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("PI-HMONet loss weights must be finite and nonnegative")
    count = int(pi["interior_sample_count"])
    if count < 1:
        raise ValueError("interior_sample_count must be positive")
    if not isinstance(pi["jacobian_loss_enabled"], bool):
        raise ValueError("jacobian_loss_enabled must be true or false")
    jacobian_minimum = float(pi["jacobian_minimum"])
    if not np.isfinite(jacobian_minimum) or jacobian_minimum < 0:
        raise ValueError("jacobian_minimum must be finite and nonnegative")
    anchor_end = int(pi["anchor_decay_end_epoch"])
    anchor_count = int(pi["anchor_points_per_axis"])
    anchor_margin = float(pi["anchor_margin"])
    if anchor_end < 1 or anchor_count < 1 or not 0 < anchor_margin < 0.5:
        raise ValueError("Invalid PI-HMONet anchor configuration")

    steps = pi["fd_step"]
    if steps is None:
        steps = [1 / (n - 1) for n in shape]
    if np.isscalar(steps):
        steps = [float(steps)] * dimension
    if len(steps) == 1:
        steps = steps * dimension
    if len(steps) != dimension or not all(
        np.isfinite(value) and 0 < value < 0.5 for value in steps
    ):
        raise ValueError("fd_step must have one value per axis in (0, 0.5)")
    pi["fd_step"] = list(map(float, steps))

    stencil_active = any(
        (
            weights["pde_loss_weight"] > 0,
            weights["normalized_pde_loss_weight"] > 0,
            pi["jacobian_loss_enabled"] and weights["jacobian_loss_weight"] > 0,
            weights["smoothness_loss_weight"] > 0,
            weights["orthogonality_loss_weight"] > 0,
        )
    )
    if not stencil_active and weights["anchor_loss_weight"] <= 0:
        raise ValueError("At least one PI-HMONet loss must be active")

    seed_everything(int(training["seed"]))
    options = {
        **config["model"].get("common", {}),
        **config["model"].get("pihmonet", {}),
    }
    if float(options.pop("residual_regularization_weight", 0.0)) != 0:
        raise ValueError("PI-HMONet residual_regularization_weight must be zero")
    model_class = HMONet2D if dimension == 2 else HMONet3D
    model = model_class(*shape, **options).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=pi["learning_rate"])
    loader = make_data_loader(
        datasets, training["batch_size"], training["seed"], shuffle=True
    )
    h = torch.tensor(steps, dtype=torch.float32, device=device)
    anchor_queries = central_anchor_queries(
        dimension,
        anchor_count,
        anchor_margin,
        device=device,
        dtype=torch.float32,
    )
    state = TrainingState(model, optimizer, datasets, config, mode, args, device)
    try:
        for epoch in range(1, training["num_epochs"] + 1):
            model.train()
            epoch_lr = optimizer.param_groups[0]["lr"]
            totals = BatchLossTotals()
            current_anchor_weight = _anchor_weight(
                weights["anchor_loss_weight"], epoch, anchor_end
            )
            progress = tqdm(
                loader,
                desc=f"PI-HMONet epoch {epoch}/{training['num_epochs']}",
                leave=True,
            )
            for batch in progress:
                boundary = batch["boundary_points"].to(device)
                size = len(boundary)
                if "boundary" not in batch:
                    raise ValueError(
                        "PI-HMONet requires load_boundary=True and exact parameter geometry"
                    )
                geometries = batch["boundary"]
                if dimension == 2:
                    from src.data.geometry.parametric_curve_2d import BoundaryCurveSet2D

                    geometries = BoundaryCurveSet2D.batch(
                        geometries, device=device, dtype=boundary.dtype
                    )

                query_parts = []
                stencil_size = 0
                if stencil_active:
                    centers = h + torch.rand(
                        count, dimension, device=device, dtype=boundary.dtype
                    ) * (1 - 2 * h)
                    stencil = stencil_queries(centers, h)
                    stencil_query = stencil.reshape(-1, dimension)
                    stencil_size = len(stencil_query)
                    query_parts.append(stencil_query)
                if current_anchor_weight > 0:
                    query_parts.append(anchor_queries.to(dtype=boundary.dtype))
                query = torch.cat(query_parts, 0)
                prediction = predict_coordinates(
                    model, boundary, query[None].expand(size, -1, -1), geometries
                )

                zero = prediction.sum() * 0.0
                raw_pde = normalized_pde = jacobian = smoothness = orthogonality = zero
                if stencil_active:
                    values = prediction[:, :stencil_size].reshape(
                        size, count, stencil.shape[-2], dimension
                    )
                    if weights["pde_loss_weight"] > 0:
                        residual = elliptic_residual(values, h)
                        raw_pde = residual.square().sum(-1).mean()
                    if weights["normalized_pde_loss_weight"] > 0:
                        normalized_pde = normalized_pde_loss(values, h)
                    if pi["jacobian_loss_enabled"] and weights["jacobian_loss_weight"] > 0:
                        jacobian = jacobian_positive_loss(
                            values, h, jacobian_minimum
                        )
                    if weights["smoothness_loss_weight"] > 0:
                        smoothness = grid_line_smoothness_loss(values, h)
                    if weights["orthogonality_loss_weight"] > 0:
                        orthogonality = orthogonality_loss(values, h)

                anchor = zero
                valid_anchor_ratio = zero.detach()
                if current_anchor_weight > 0:
                    anchor_prediction = prediction[:, stencil_size:]
                    with torch.no_grad():
                        target, valid = tfi_anchor_targets(
                            boundary, anchor_queries.to(dtype=boundary.dtype), shape
                        )
                    anchor = masked_anchor_mse(anchor_prediction, target, valid)
                    valid_anchor_ratio = valid.float().mean()

                loss = (
                    weights["pde_loss_weight"] * raw_pde
                    + weights["normalized_pde_loss_weight"] * normalized_pde
                    + weights["jacobian_loss_weight"] * jacobian
                    + current_anchor_weight * anchor
                    + weights["smoothness_loss_weight"] * smoothness
                    + weights["orthogonality_loss_weight"] * orthogonality
                )
                if not safe_optimizer_step(
                    loss,
                    model,
                    optimizer,
                    training["gradient_clip_norm"],
                    training["skip_nonfinite_loss"],
                ):
                    continue
                totals.add(
                    size,
                    loss=loss,
                    loss_pde=raw_pde,
                    loss_pde_normalized=normalized_pde,
                    jacobian=jacobian,
                    anchor=anchor,
                    anchor_weight=current_anchor_weight,
                    valid_anchor_ratio=valid_anchor_ratio,
                    grid_smoothness=smoothness,
                    orthogonality=orthogonality,
                )
                progress.set_postfix(
                    loss=loss.item(),
                    pde_norm=normalized_pde.item(),
                    jacobian=jacobian.item(),
                    anchor=anchor.item(),
                )
            update_learning_rate(
                optimizer,
                epoch,
                pi["lr_warmup_epochs"],
                pi["lr_update_interval"],
                pi["lr_decay_gamma"],
                pi["lr_stop_below"],
            )
            if state.end_epoch(epoch, totals.means(), epoch_lr):
                break
        return state.finish()
    finally:
        if state.writer is not None:
            state.writer.close()
