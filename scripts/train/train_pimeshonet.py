"""Self-supervised PI-MeshONet training, shared by 2-D and 3-D entry points."""

import copy
import numpy as np
import torch
from tqdm import tqdm
from src.models.pi_meshonet import PIMeshONet2D, PIMeshONet3D
from src.data.mesh.sampling import logical_grid, boundary_indices
from scripts.train.physics_loss import stencil_queries, elliptic_residual
from scripts.train.training_common import (
    BatchLossTotals,
    make_data_loader,
    predict_coordinates,
    safe_optimizer_step,
    seed_everything,
    update_learning_rate,
)
from scripts.train.training_state import TrainingState, prepare_training_config


def train_pimeshonet(datasets, config, mode, args, device):
    """Optimize boundary/PDE losses only; mesh labels are monitoring-only."""
    config = copy.deepcopy(config)
    pi = config.setdefault("pimeshonet_training", {})
    defaults = dict(
        pde_loss_weight=1e-4,
        boundary_loss_weight=1.0,
        interior_sample_count=128,
        fd_step=None,
    )
    for key, value in defaults.items():
        override = getattr(args, "pimeshonet_" + key, None)
        if override is None:
            override = getattr(args, "pi_" + key, None)
        if override is None and key == "interior_sample_count":
            override = getattr(args, "interior_sample_count", None)
        pi[key] = override if override is not None else pi.get(key, value)
    config.setdefault("training", {})["best_model_metric"] = "training_loss"
    # Labels may be reported, but must never select or stop a self-supervised run.
    config["training"].update(
        check_mse_convergence=False,
        check_max_point_error=False,
        check_p99_point_error=False,
        plateau_metric="train_loss",
    )
    config = prepare_training_config(config, "pimeshonet", args)
    training = config["training"]
    pi = config["pimeshonet_training"]
    shape = tuple(datasets[0].grid_shape)
    dimension = len(shape)
    count = int(pi["interior_sample_count"])
    weights = [float(pi["pde_loss_weight"]), float(pi["boundary_loss_weight"])]
    if count < 1 or not all(np.isfinite(w) and w > 0 for w in weights):
        raise ValueError(
            "PI-MeshONet requires positive finite PDE/boundary weights and sample count"
        )
    steps = pi["fd_step"]
    if steps is None:
        steps = [1 / (n - 1) for n in shape]
    if np.isscalar(steps):
        steps = [float(steps)] * dimension
    if len(steps) == 1:
        steps = steps * dimension
    if len(steps) != dimension or not all(
        np.isfinite(h) and 0 < h < 0.5 for h in steps
    ):
        raise ValueError(
            "fd_step must have one value per axis strictly between 0 and 0.5"
        )
    pi["fd_step"] = list(map(float, steps))
    seed_everything(int(training["seed"]))
    model_section = config.setdefault("model", {}).setdefault("pimeshonet", {})
    model_section["architecture"] = "paper_modified_mlp"
    options = {
        **config["model"].get("common", {}),
        **model_section,
    }
    options.pop("architecture", None)
    cls = PIMeshONet2D if dimension == 2 else PIMeshONet3D
    model = cls(*shape, **options).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=pi["learning_rate"])
    loader = make_data_loader(
        datasets, training["batch_size"], training["seed"], shuffle=True
    )
    boundary_query = torch.as_tensor(
        logical_grid(shape)[boundary_indices(shape)], dtype=torch.float32, device=device
    )
    h = torch.tensor(steps, dtype=torch.float32, device=device)
    state = TrainingState(model, optimizer, datasets, config, mode, args, device)
    try:
        for epoch in range(1, training["num_epochs"] + 1):
            model.train()
            epoch_lr = optimizer.param_groups[0]["lr"]
            totals = BatchLossTotals()
            progress = tqdm(
                loader,
                desc=f"PI-MeshONet epoch {epoch}/{training['num_epochs']}",
                leave=True,
            )
            for batch in progress:
                boundary = batch["boundary_points"].to(device)
                size = len(boundary)
                centers = h + torch.rand(count, dimension, device=device) * (1 - 2 * h)
                stencil = stencil_queries(centers, h)
                query = torch.cat([boundary_query, stencil.reshape(-1, dimension)], 0)
                prediction = predict_coordinates(
                    model, boundary, query[None].expand(size, -1, -1)
                )
                nb = len(boundary_query)
                boundary_loss = (prediction[:, :nb] - boundary).square().sum(-1).mean()
                values = prediction[:, nb:].reshape(
                    size, count, stencil.shape[-2], dimension
                )
                residual = elliptic_residual(values, h)
                pde_loss = residual.square().sum(-1).mean()
                loss = weights[0] * pde_loss + weights[1] * boundary_loss
                if not safe_optimizer_step(
                    loss,
                    model,
                    optimizer,
                    training["gradient_clip_norm"],
                    training["skip_nonfinite_loss"],
                ):
                    continue
                totals.add(
                    size, loss=loss, loss_pde=pde_loss, loss_boundary=boundary_loss
                )
                progress.set_postfix(
                    loss=loss.item(), pde=pde_loss.item(), boundary=boundary_loss.item()
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
