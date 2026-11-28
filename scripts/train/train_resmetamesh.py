"""Formal ResMetaMesh pre-training with indexed geometry latents."""
from functools import cached_property
from math import ceil, isfinite

import numpy as np
import torch

from src.data.mesh.sampling import boundary_indices
from src.engine.state import TrainingState, prepare_training_config
from src.engine.training import (
    BatchLossTotals, FIXED_GRID_MSE_SCALE, seed_everything,
    step_accumulated_gradients,
)
from src.models.resmetamesh import (
    ResMetaMesh, base_coordinates, base_derivatives, collocation_grid, pde_loss,
    logical_query_grid, validate_training_options,
)
from src.utils.progress import training_progress as tqdm


class ResMetaMeshTrainingState(TrainingState):
    @cached_property
    def _fixed_mse_data(self):
        dataset = self.datasets[0]
        interior = torch.ones(int(np.prod(self.shape)), dtype=torch.bool, device=self.device)
        interior[boundary_indices(self.shape)] = False
        query = logical_query_grid(self.shape, self.device)[interior]
        base = torch.stack([
            base_coordinates(geometry, self.shape, device=self.device)[interior]
            for geometry in dataset.boundary
        ])
        targets = torch.as_tensor(
            dataset.mesh_points.reshape(len(dataset), -1, len(self.shape)),
            device=self.device, dtype=torch.float32,
        )[:, interior]
        return query, base, targets

    @torch.inference_mode()
    def _measure(self):
        query, base, targets = self._fixed_mse_data
        point_errors = []
        measure_points = self.training.get("check_max_point_error", False) or self.training.get(
            "check_p99_point_error", False
        )
        squared_sum = base.new_zeros(())
        batch_size = self.training["batch_size"]
        for start in range(0, len(base), batch_size):
            rows = slice(start, start + batch_size)
            latent = self.model.latents[rows].unsqueeze(1)
            batch_query = query.unsqueeze(0).expand(len(latent), -1, -1)
            difference = base[rows] + self.model(batch_query, latent) - targets[rows]
            squared_sum += difference.square().sum()
            if measure_points:
                point_errors.append(torch.linalg.vector_norm(difference, dim=-1).flatten())
        mse = FIXED_GRID_MSE_SCALE * squared_sum.item() / targets.numel()
        if not isfinite(mse):
            raise FloatingPointError("Non-finite prediction during fixed-grid MSE")
        metrics = {"mse": mse}
        if point_errors:
            errors = torch.cat(point_errors)
            metrics.update(
                max_point_error=errors.max().item(),
                p99_point_error=torch.quantile(errors, 0.99).item(),
            )
        return metrics


def prepare_resmetamesh_config(config, args, dimension, dataset_size):
    config = prepare_training_config(config, "resmetamesh", args)
    options = config["resmetamesh_training"]
    hmonet = config.get("hmonet_training", {})
    options["fd_step"] = list(hmonet.get("pde_fd_step", [0.01] * dimension))
    options["correction_derivative_backend"] = "autograd"
    options["tfi_precision"] = "float32"
    validate_training_options(options, dimension)
    updates = options.get("pretrain_updates", 5000)
    if type(updates) is not int or updates < 1:
        raise ValueError("ResMetaMesh pretrain_updates must be a positive integer")
    options["pretrain_updates"] = updates
    # Algorithm 1 uses a fixed Adam budget, independent of the diagnostic MSE.
    if dataset_size < 1:
        raise ValueError("ResMetaMesh pre-training requires at least one geometry")
    training = config["training"]
    batches_per_epoch = ceil(dataset_size / training["batch_size"])
    epochs_for_budget = ceil(updates / batches_per_epoch)
    epoch_limit = getattr(args, "epochs", None)
    training.update(
        num_epochs=min(epoch_limit, epochs_for_budget) if epoch_limit is not None else epochs_for_budget,
        skip_nonfinite_loss=False,
        plateau_enabled=False,
        failure_plateau_enabled=False,
        check_training_loss_convergence=False,
        check_mse_convergence=False,
        check_max_point_error=False,
        check_p99_point_error=False,
    )
    return config


def accumulate_batch(
    model, optimizer, dataset, indices, query, shape, stencil, base_cache, options,
):
    optimizer.zero_grad(set_to_none=True)
    fd_step = options["fd_step"]
    for index in indices:
        if index not in base_cache:
            base_cache[index] = base_derivatives(dataset.boundary[index], shape, stencil, fd_step)

    batch_ids = torch.as_tensor(indices, device=query.device)
    base = [
        torch.stack(values)
        for values in zip(*(base_cache[index] for index in indices))
    ]
    regularizer = (
        options["latent_l2_weight"] * model.latents[batch_ids].square().sum(dim=-1).mean()
    )
    pde_total = query.new_zeros(())
    chunk = options["derivative_batch_size"]
    for start in range(0, len(query), chunk):
        part = slice(start, start + chunk)
        centers = query[part]
        # Independent coordinates keep each geometry's spatial derivatives separate.
        batched_query = centers.unsqueeze(0).expand(len(indices), -1, -1).clone()
        pde = pde_loss(
            model, batched_query, [value[:, part] for value in base],
            model.latents[batch_ids].unsqueeze(1),
        ) * (len(centers) / len(query))
        loss = pde + regularizer if start == 0 else pde
        loss.backward()
        pde_total += pde.detach()

    loss_pde, loss_latent = torch.stack((pde_total, regularizer.detach())).tolist()
    total = loss_pde + loss_latent
    if not isfinite(total):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Non-finite ResMetaMesh batch loss")
    return {"loss": total, "loss_pde": loss_pde, "loss_latent": loss_latent}


def train_resmetamesh(dataset, config, mode, args, device, *, on_update=None):
    config = prepare_resmetamesh_config(
        config, args, len(dataset.grid_shape), len(dataset),
    )
    options, settings = config["resmetamesh_training"], config["training"]
    seed = int(settings["seed"])
    seed_everything(seed)
    architecture = config["model"]["resmetamesh"]
    model = ResMetaMesh(len(dataset.grid_shape), len(dataset), **architecture).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    shape = tuple(options["collocation_shape"])
    centers, stencil = collocation_grid(shape, options["collocation_count"], seed, options["fd_step"])
    query = centers.to(device=device)
    stencil = stencil.to(device=device)
    state = ResMetaMeshTrainingState(model, optimizer, [dataset], config, mode, args, device)
    base_cache = {}
    target_updates = options["pretrain_updates"]
    batch_size = settings["batch_size"]
    max_epochs = settings["num_epochs"]
    rng = np.random.default_rng(seed)
    state.optimizer_updates = 0
    try:
        for epoch in range(1, max_epochs + 1):
            epoch_lr = optimizer.param_groups[0]["lr"]
            order = rng.permutation(len(dataset))
            totals = BatchLossTotals()
            batches_left = target_updates - state.optimizer_updates
            batch_starts = range(0, min(len(order), batches_left * batch_size), batch_size)
            with tqdm(
                batch_starts, desc=f"ResMetaMesh epoch {epoch}/{max_epochs}", leave=True,
            ) as progress:
                for start in progress:
                    indices = order[start:start + batch_size]
                    losses = accumulate_batch(
                        model, optimizer, dataset, indices, query, shape, stencil,
                        base_cache, options,
                    )
                    step_accumulated_gradients(
                        model, optimizer, settings["gradient_clip_norm"], skip_nonfinite_loss=False,
                    )
                    state.optimizer_updates += 1
                    if on_update is not None:
                        on_update(state, epoch)
                    totals.add(len(indices), **losses)
                    progress.set_postfix(**losses)
            if state.end_epoch(epoch, totals.means(), epoch_lr):
                break
            if state.optimizer_updates == target_updates:
                state.reason = "maximum_optimizer_updates"
        return state.finish()
    finally:
        if state.writer is not None:
            state.writer.close()
