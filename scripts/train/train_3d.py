"""Train dimension-specific operators with independent complete model routines."""

from __future__ import annotations

import argparse
import copy
import json
import time
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import yaml
from src.utils.progress import training_progress as tqdm

from scripts.train.model_selection import parse_model_selection
from src.engine.training import (
    BatchLossTotals,
    boundary_normal_loss,
    calculate_fixed_grid_mse_3d,
    hmonet_main_loss,
    interpolate_pde_mesh,
    jacobian_positive_loss_autograd,
    jacobian_positive_loss_finite_difference,
    jacobian_finite_difference_queries,
    logical_grid_and_interior_indices,
    make_data_loader,
    predict_coordinates,
    resolve_dataset_paths,
    safe_optimizer_step,
    sample_fixed_grid_batch,
    sample_shared_random_points,
    seed_everything,
    update_learning_rate,
)
from src.engine.state import (
    TrainingState,
    prepare_training_config,
    print_training_summaries,
)
from src.utils.config_validation import remove_deprecated_projection_config
from src.data.parametric_dataset import ParametricDataset3D
from src.data.mesh.sampling import (
    sample_surface_curvature_adaptive_points,
    structured_boundary_surface_curvature,
)
from src.losses.physics import (
    elliptic_grid_pde_residual,
    elliptic_grid_scale_invariant_pde_loss,
    normalized_mapping_barriers,
    stencil_queries as physics_stencil_queries,
)
from src.models import HMONet3D, MeshONet3D, QGHMONet3D
from src.models.mapping_modes import mapping_mode_choices
from src.data.mesh.topology import GenerateHexMeshBoundaryIndex


def _train_hmonet_physics_3d(
    model,
    optimizer,
    loader,
    datasets,
    config,
    mode,
    args,
    device,
    geometry_options,
):
    """Train a 3-D hard-boundary operator with the formal physics objective."""
    training = config["training"]
    sampling = config.get("hmonet_sampling", {})
    learning_rate = geometry_options
    shape = tuple(datasets[0].grid_shape)
    sample_count = int(geometry_options.get("random_loss_sample_count", 400))
    sampling_mode = str(
        geometry_options.get("random_loss_sampling_mode", "uniform")
    ).lower()
    global_count = int(
        geometry_options.get("random_loss_global_sample_count", sample_count)
    )
    adaptive_count = int(
        geometry_options.get("random_loss_curvature_sample_count", 0)
    )
    boundary_band_count = int(
        geometry_options.get("random_loss_boundary_band_sample_count", 0)
    )
    if sampling_mode not in {"uniform", "mixed_boundary"}:
        raise ValueError(
            "hmonet_training.random_loss_sampling_mode must be uniform or mixed_boundary"
        )
    if sampling_mode == "mixed_boundary":
        counts = (global_count, adaptive_count, boundary_band_count)
        if min(counts) < 0 or sum(counts) != sample_count:
            raise ValueError(
                "3-D mixed sampling counts must be nonnegative and sum to "
                "random_loss_sample_count"
            )
        if boundary_band_count:
            raise ValueError(
                "Formal 3-D training currently supports global and curvature-adaptive "
                "samples; boundary-band samples must be zero"
            )
    else:
        global_count, adaptive_count = sample_count, 0

    derivative_backend = str(
        geometry_options.get("random_loss_derivative_backend", "finite_difference")
    ).lower()
    boundary_scheme = str(
        geometry_options.get("random_loss_fd_boundary_scheme", "central")
    ).lower()
    if derivative_backend != "finite_difference" or boundary_scheme != "central":
        raise ValueError(
            "Formal 3-D HMONet currently requires central finite differences"
        )
    steps = torch.as_tensor(
        geometry_options.get("pde_fd_step", [0.01, 0.01, 0.01]),
        device=device,
        dtype=torch.float32,
    )
    if steps.shape != (3,) or bool(((steps <= 0) | (steps >= 0.5)).any()):
        raise ValueError("pde_fd_step must contain three values in (0, 0.5)")
    keep_inside = bool(
        geometry_options.get("random_loss_fd_stencil_inside", True)
    )
    margin = steps if keep_inside else torch.zeros_like(steps)

    pde_weight = float(geometry_options.get("physics_pde_loss_weight", 1.0))
    normalized_jacobian_weight = float(
        geometry_options.get("physics_normalized_jacobian_loss_weight", 0.0)
    )
    singular_value_weight = float(
        geometry_options.get("physics_singular_value_loss_weight", 0.0)
    )
    if min(pde_weight, normalized_jacobian_weight, singular_value_weight) < 0:
        raise ValueError("Physics loss weights must be nonnegative")
    if pde_weight + normalized_jacobian_weight + singular_value_weight <= 0:
        raise ValueError("At least one formal 3-D physics loss must be active")
    scale_invariant_pde = bool(
        geometry_options.get("physics_scale_invariant_pde_enabled", False)
    )
    jacobian_minimum = float(
        geometry_options.get("physics_normalized_jacobian_minimum", 0.02)
    )
    singular_minimum = float(
        geometry_options.get("physics_singular_value_minimum", 0.05)
    )
    reduction = str(
        geometry_options.get("physics_barrier_reduction", "all")
    ).lower()
    if reduction not in {"active", "all"}:
        raise ValueError("physics_barrier_reduction must be active or all")

    candidate_multiplier = int(sampling.get("candidate_multiplier", 10))
    curvature_quantile = float(sampling.get("curvature_quantile", 0.95))
    curvature_power = float(sampling.get("curvature_power", 1.0))
    sigma_min = float(sampling.get("distance_sigma_min", 0.02))
    sigma_max = float(sampling.get("distance_sigma_max", 0.10))
    uniform_probability = float(sampling.get("uniform_probability", 0.05))
    # Kept in the effective configuration for 2-D/3-D parity.  The 3-D anchor
    # sampler is already linear-memory and therefore does not need candidates.
    if candidate_multiplier < 1:
        raise ValueError("hmonet_sampling.candidate_multiplier must be positive")

    training["active_phase"] = "random_loss_combo"
    state = TrainingState(model, optimizer, datasets, config, mode, args, device)
    try:
        for epoch in range(1, training["num_epochs"] + 1):
            model.train()
            epoch_lr = optimizer.param_groups[0]["lr"]
            totals = BatchLossTotals()
            progress = tqdm(
                loader, desc=f"Epoch {epoch}/{training['num_epochs']}", leave=True
            )
            for batch in progress:
                boundary = batch["boundary_points"].to(device)
                geometries = batch["boundary"]
                point_parts = []
                if global_count:
                    point_parts.append(
                        sample_shared_random_points(
                            len(boundary),
                            global_count,
                            3,
                            device,
                            margin=margin,
                        ).to(dtype=boundary.dtype)
                    )
                if adaptive_count:
                    curvature, computational_boundary = (
                        structured_boundary_surface_curvature(
                            boundary,
                            shape,
                            quantile=curvature_quantile,
                        )
                    )
                    point_parts.append(
                        sample_surface_curvature_adaptive_points(
                            curvature,
                            computational_boundary,
                            adaptive_count,
                            curvature_power=curvature_power,
                            sigma_min=sigma_min,
                            sigma_max=sigma_max,
                            uniform_probability=uniform_probability,
                            margin=margin,
                        )
                    )
                centers = torch.cat(point_parts, dim=1)
                queries = physics_stencil_queries(centers, steps)
                values = predict_coordinates(
                    model,
                    boundary,
                    queries.flatten(-3, -2),
                    geometries,
                ).reshape(*queries.shape)

                if scale_invariant_pde:
                    pde = elliptic_grid_scale_invariant_pde_loss(values, steps)
                else:
                    pde = elliptic_grid_pde_residual(values, steps).square().sum(-1).mean()
                extents = boundary.amax(dim=-2) - boundary.amin(dim=-2)
                characteristic_length = extents.prod(dim=-1).clamp_min(1e-12).pow(1.0 / 3.0)
                normalized_jacobian, singular_value = normalized_mapping_barriers(
                    values,
                    steps,
                    characteristic_length,
                    jacobian_minimum=jacobian_minimum,
                    singular_value_minimum=singular_minimum,
                    reduction=reduction,
                )
                loss = (
                    pde_weight * pde
                    + normalized_jacobian_weight * normalized_jacobian
                    + singular_value_weight * singular_value
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
                    len(boundary),
                    loss=loss,
                    loss_pde=pde,
                    normalized_jacobian=normalized_jacobian,
                    singular_value=singular_value,
                )
                progress.set_postfix(
                    loss=loss.item(),
                    pde=pde.item(),
                    norm_j=normalized_jacobian.item(),
                    singular=singular_value.item(),
                )
            update_learning_rate(
                optimizer,
                epoch,
                learning_rate["lr_warmup_epochs"],
                learning_rate["lr_update_interval"],
                learning_rate["lr_decay_gamma"],
                learning_rate["lr_stop_below"],
            )
            if state.end_epoch(epoch, totals.means(), epoch_lr):
                break
        return state.finish()
    finally:
        if state.writer is not None:
            state.writer.close()


def train_hmonet(
    datasets, config, mode, args, device, *, model_name="hmonet", model_class=HMONet3D
):
    """Train formal self-supervised or archived supervised 3-D HMONet."""
    config = prepare_training_config(config, model_name, args)
    training = config["training"]
    geometry_options = config[f"{model_name}_training"]
    learning_rate = geometry_options
    shape = tuple(datasets[0].grid_shape)
    seed_everything(int(training["seed"]))
    model_options = dict(config["model"].get("common", {}))
    model_options.update(config["model"].get(model_name, {}))
    residual_weight = float(model_options.pop("residual_regularization_weight", 0.0))
    model_options.pop("projection_softplus_enabled", None)
    model_options["trunk_projection_coefficients_enabled"] = True
    if residual_weight < 0:
        raise ValueError("residual_regularization_weight must be nonnegative")
    model = model_class(*shape, **model_options).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate["learning_rate"]
    )
    loader = make_data_loader(
        datasets, training["batch_size"], training["seed"], shuffle=True
    )
    strategy = str(
        geometry_options.get("training_strategy", "supervised")
    ).lower()
    if strategy == "random_loss_combo_from_start":
        return _train_hmonet_physics_3d(
            model,
            optimizer,
            loader,
            datasets,
            config,
            mode,
            args,
            device,
            geometry_options,
        )
    if strategy != "supervised":
        raise ValueError(
            "3-D HMONet training_strategy must be supervised or "
            "random_loss_combo_from_start"
        )
    use_jacobian = bool(geometry_options.get("jacobian_loss_enabled", False))
    jacobian_method = geometry_options.setdefault(
        "jacobian_method", "finite_difference"
    )
    jacobian_step = float(geometry_options.setdefault("jacobian_fd_step", 1e-3))
    if jacobian_method not in ("autograd", "finite_difference"):
        raise ValueError("jacobian_method must be autograd or finite_difference")
    if not 0 < jacobian_step <= 1 / 3:
        raise ValueError("jacobian_fd_step must be in (0, 1/3]")
    use_fd_jacobian = use_jacobian and jacobian_method == "finite_difference"
    use_normal = bool(geometry_options.get("boundary_normal_loss_enabled", False))
    target_interpolation = str(
        geometry_options.get("target_interpolation", "bilinear")
    ).lower()
    if target_interpolation not in {"bilinear", "curved_boundary_cell"}:
        raise ValueError(
            "hmonet_training.target_interpolation must be bilinear or "
            "curved_boundary_cell"
        )
    point_loss_weighting = str(
        geometry_options.get("point_loss_weighting", "position")
    ).lower()
    if point_loss_weighting not in {"position", "uniform"}:
        raise ValueError(
            "hmonet_training.point_loss_weighting must be position or uniform"
        )
    fixed_count = 0
    random_count = int(training.get("sample_count", 400))
    if random_count < 1:
        raise ValueError("training.sample_count must be positive")
    state = TrainingState(model, optimizer, datasets, config, mode, args, device)
    try:
        for epoch in range(1, training["num_epochs"] + 1):
            model.train()
            epoch_lr = optimizer.param_groups[0]["lr"]
            totals = BatchLossTotals()
            progress = tqdm(
                loader, desc=f"Epoch {epoch}/{training['num_epochs']}", leave=True
            )
            for batch in progress:
                boundary = batch["boundary_points"].to(device)
                mesh = batch["mesh_points"].to(device)
                geometries = batch["boundary"]
                point_groups = []
                derivative_margin = jacobian_step if use_fd_jacobian else 0.0
                if fixed_count:
                    logical, target = sample_fixed_grid_batch(
                        mesh, shape, fixed_count, margin=derivative_margin
                    )
                    point_groups.append((True, logical, target))
                if random_count:
                    logical = sample_shared_random_points(
                        len(mesh), random_count, 3, device,
                        margin=derivative_margin,
                    )
                    target = interpolate_pde_mesh(
                        mesh,
                        logical,
                        shape,
                        target_interpolation=target_interpolation,
                        boundary_geometry=geometries,
                        idw_query_chunk_size=int(
                            geometry_options.get("idw_query_chunk_size", 512)
                        ),
                        idw_support_chunk_size=int(
                            geometry_options.get("idw_support_chunk_size", 2048)
                        ),
                    )
                    point_groups.append((False, logical, target))
                coordinate_loss = torch.zeros((), device=device)
                branch_losses = {}
                jacobian_terms = []
                displacement_sums = []
                displacement_count = 0
                for fixed, logical, target in point_groups:
                    if use_jacobian and not use_fd_jacobian:
                        logical = logical.detach().requires_grad_(True)
                    network_queries = logical
                    if use_fd_jacobian:
                        stencil_queries = jacobian_finite_difference_queries(
                            logical, jacobian_step
                        )
                        network_queries = stencil_queries.flatten(-3, -2)
                    if residual_weight > 0:
                        prediction, displacement = predict_coordinates(
                            model,
                            boundary,
                            network_queries,
                            geometries,
                            return_displacement=True,
                        )
                        if use_fd_jacobian:
                            displacement = displacement.reshape(*stencil_queries.shape)[
                                ..., 0, :
                            ]
                        displacement_sums.append(displacement.square().sum())
                        displacement_count += (
                            displacement.shape[0] * displacement.shape[1]
                        )
                    else:
                        prediction = predict_coordinates(
                            model, boundary, network_queries, geometries
                        )
                    if use_fd_jacobian:
                        stencil_prediction = prediction.reshape(*stencil_queries.shape)
                        prediction = stencil_prediction[..., 0, :]
                        jacobian_terms.append(
                            jacobian_positive_loss_finite_difference(
                                stencil_prediction,
                                logical,
                                float(geometry_options.get("jacobian_minimum", 1e-4)),
                                jacobian_step,
                            )
                        )
                    branch_loss = hmonet_main_loss(
                        prediction,
                        target,
                        logical,
                        fixed=fixed,
                        point_weighting=point_loss_weighting,
                    )
                    coordinate_loss = coordinate_loss + branch_loss
                    branch_name = (
                        "selected_internal_loss" if fixed else "random_internal_loss"
                    )
                    branch_losses[branch_name] = branch_loss
                    if use_jacobian and not use_fd_jacobian:
                        jacobian_terms.append(
                            jacobian_positive_loss_autograd(
                                prediction,
                                logical,
                                float(geometry_options.get("jacobian_minimum", 1e-4)),
                            )
                        )
                jacobian = (
                    torch.stack(jacobian_terms).mean()
                    if jacobian_terms
                    else torch.zeros((), device=device)
                )
                normal = torch.zeros((), device=device)
                if use_normal:
                    normal = boundary_normal_loss(
                        model,
                        boundary,
                        mesh,
                        shape,
                        3,
                        device,
                        int(geometry_options.get("boundary_normal_sample_count", 64)),
                        float(geometry_options.get("boundary_normal_offset", 0.01)),
                        geometry=geometries,
                    )
                regularization = (
                    torch.stack(displacement_sums).sum() / displacement_count
                    if displacement_sums
                    else torch.zeros((), device=device)
                )
                loss = (
                    coordinate_loss
                    + float(geometry_options.get("jacobian_loss_weight", 0.01))
                    * jacobian
                    + float(geometry_options.get("boundary_normal_loss_weight", 0.01))
                    * normal
                    + residual_weight * regularization
                )
                if not safe_optimizer_step(
                    loss,
                    model,
                    optimizer,
                    training["gradient_clip_norm"],
                    training["skip_nonfinite_loss"],
                ):
                    continue
                active_losses = {"loss": loss, "coordinate_loss": coordinate_loss}
                if use_jacobian:
                    active_losses["jacobian"] = jacobian
                if use_normal:
                    active_losses["boundary_normal"] = normal
                if residual_weight > 0:
                    active_losses["displacement_regularization"] = regularization
                totals.add(len(mesh), **active_losses)
                postfix = {"loss": loss.item()}
                postfix.update(
                    {key: value.item() for key, value in branch_losses.items()}
                )
                if use_jacobian:
                    postfix["jacobian_loss"] = jacobian.item()
                if use_normal:
                    postfix["boundary_normal_loss"] = normal.item()
                if residual_weight > 0:
                    postfix["displacement_regularization"] = regularization.item()
                progress.set_postfix(**postfix)
            update_learning_rate(
                optimizer,
                epoch,
                learning_rate["lr_warmup_epochs"],
                learning_rate["lr_update_interval"],
                learning_rate["lr_decay_gamma"],
                learning_rate["lr_stop_below"],
            )
            if state.end_epoch(epoch, totals.means(), epoch_lr):
                break
        return state.finish()
    finally:
        if state.writer is not None:
            state.writer.close()


def train_meshonet(datasets, config, mode, args, device):
    """Train MeshONet with batched interior and boundary predictions."""
    config = prepare_training_config(config, "meshonet", args)
    training = config["training"]
    mesh_training = config["meshonet_training"]
    learning_rate = mesh_training
    shape = tuple(datasets[0].grid_shape)
    seed_everything(int(training["seed"]))
    model_options = dict(config["model"].get("common", {}))
    model_options.update(config["model"].get("meshonet", {}))
    model = MeshONet3D(*shape, **model_options).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate["learning_rate"]
    )
    alpha = float(mesh_training.get("alpha", 1.0))
    beta = float(mesh_training.get("beta", 5.0))
    if alpha < 0 or beta < 0 or alpha + beta == 0:
        raise ValueError("alpha and beta must be nonnegative and not both zero")
    count = int(
        mesh_training.get("interior_sample_count", training.get("sample_count", 400))
    )
    if count < 1:
        raise ValueError("interior_sample_count must be positive")
    loader = make_data_loader(
        datasets, training["batch_size"], training["seed"], shuffle=True
    )
    grid, _ = logical_grid_and_interior_indices(shape, device)
    nx, ny, nz = shape
    faces = GenerateHexMeshBoundaryIndex(nz, ny, nx)
    indices = np.unique(np.concatenate([face.reshape(-1) for face in faces.values()]))
    computational_boundary = grid[
        torch.as_tensor(indices, dtype=torch.long, device=device)
    ]
    boundary_ids = torch.arange(len(indices), device=device)
    state = TrainingState(model, optimizer, datasets, config, mode, args, device)
    try:
        for epoch in range(1, training["num_epochs"] + 1):
            model.train()
            epoch_lr = optimizer.param_groups[0]["lr"]
            totals = BatchLossTotals()
            progress = tqdm(
                loader, desc=f"Epoch {epoch}/{training['num_epochs']}", leave=True
            )
            for batch in progress:
                boundary = batch["boundary_points"].to(device)
                mesh = batch["mesh_points"].to(device)
                boundary_query = computational_boundary.unsqueeze(0).expand(
                    len(mesh), -1, -1
                )
                boundary_prediction = predict_coordinates(
                    model, boundary, boundary_query
                )
                boundary_loss = (
                    (boundary_prediction - boundary[:, boundary_ids]).square().mean()
                )
                logical = sample_shared_random_points(len(mesh), count, 3, device)
                prediction = predict_coordinates(model, boundary, logical)
                target = interpolate_pde_mesh(mesh, logical, shape)
                interior_loss = (prediction - target).square().mean()
                loss = alpha * interior_loss + beta * boundary_loss
                if not safe_optimizer_step(
                    loss,
                    model,
                    optimizer,
                    training["gradient_clip_norm"],
                    training["skip_nonfinite_loss"],
                ):
                    continue
                totals.add(
                    len(mesh),
                    loss=loss,
                    coordinate_loss=interior_loss,
                    loss_interior=interior_loss,
                    loss_boundary=boundary_loss,
                )
                progress.set_postfix(
                    loss=loss.item(),
                    loss_interior=interior_loss.item(),
                    loss_boundary=boundary_loss.item(),
                )
            update_learning_rate(
                optimizer,
                epoch,
                learning_rate["lr_warmup_epochs"],
                learning_rate["lr_update_interval"],
                learning_rate["lr_decay_gamma"],
                learning_rate["lr_stop_below"],
            )
            if state.end_epoch(epoch, totals.means(), epoch_lr):
                break
        return state.finish()
    finally:
        if state.writer is not None:
            state.writer.close()


def train_pimeshonet(datasets, config, mode, args, device):
    """Train the self-supervised MeshONet architecture with finite differences."""
    from scripts.train.train_pimeshonet import train_pimeshonet as train_physics

    return train_physics(datasets, config, mode, args, device)


def train_models(names, paths, config, mode, args, device):
    """Train every selected model/dataset pair independently."""
    root = Path(__file__).resolve().parents[2]
    summaries = []
    started = time.perf_counter()
    for name in names:
        for path in paths:
            dataset_path = Path(path)
            dataset_name = (
                dataset_path.parent.name
                if dataset_path.is_file()
                else dataset_path.name
            )
            run_mode = mode
            if mode == "both":
                for candidate in ("interpolation", "extrapolation"):
                    candidates = resolve_dataset_paths(config, candidate, root)
                    if any(p.resolve() == dataset_path.resolve() for p in candidates):
                        run_mode = candidate
                        break
                if run_mode == "both":
                    run_mode = (
                        "interpolation"
                        if "_int_" in dataset_name
                        else "extrapolation" if "_ext_" in dataset_name else "custom"
                    )
            run_args = copy.copy(args)
            run_args.output_root = str(Path(args.output_root) / dataset_name)
            run_args.checkpoint_root = str(
                Path(args.output_root).parent.parent / "checkpoints"
            )
            run_args.config_archive_root = root / "configs" / "3D"
            run_args.dataset_name = dataset_name
            run_args.mode = run_mode
            data = ParametricDataset3D(
                path,
                load_boundary=(name in ("hmonet", "qghmonet")),
            )
            for axis, actual in zip(("nx", "ny", "nz"), data.grid_shape):
                requested = getattr(args, axis, None)
                if requested is not None and int(requested) != int(actual):
                    raise ValueError(
                        f"Explicit --{axis}={requested} differs from saved dataset "
                        f"grid {data.grid_shape}"
                    )
            run_config = copy.deepcopy(config)
            run_config.setdefault("grid", {}).update(
                dict(zip(("nx", "ny", "nz"), map(int, data.grid_shape)))
            )
            print(
                f"[{run_mode}] model={name}, dataset={dataset_name}, "
                f"grid={data.grid_shape} (read from dataset)",
                flush=True,
            )
            if name == "hmonet":
                train_hmonet([data], run_config, run_mode, run_args, device)
            elif name == "qghmonet":
                train_hmonet(
                    [data], run_config, run_mode, run_args, device,
                    model_name="qghmonet", model_class=QGHMONet3D,
                )
            elif name == "pimeshonet":
                train_pimeshonet([data], run_config, run_mode, run_args, device)
            elif name == "meshonet":
                train_meshonet([data], run_config, run_mode, run_args, device)
            else:
                raise ValueError(f"Unknown model: {name}")
            summary_path = Path(run_args.training_output_dir) / "training_summary.json"
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    print_training_summaries(summaries, time.perf_counter() - started, dimension=3)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/3D/train.yaml")
    parser.add_argument("--dataset")
    parser.add_argument(
        "--dataset-mode", choices=["interpolation", "extrapolation", "both"]
    )
    parser.add_argument(
        "--model",
        help=(
            "Model name, alias (both/pi/all), or a comma-separated model list, "
            "for example meshonet,pimeshonet"
        ),
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output-root", default="outputs/training/3D")
    parser.add_argument("--flat-run-root", help=argparse.SUPPRESS)
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--nz", type=int)
    parser.add_argument(
        "--coordinate-lift-mode",
        choices=["legacy", "paper", "power_fourier"],
        help="Coordinate encoding: legacy, published paper Lift-Layer, or power-Fourier.",
    )
    parser.add_argument("--coordinate-lift-max-power", type=int)
    parser.add_argument(
        "--coordinate-lift-q",
        type=int,
        help="Highest polynomial order q in the published MeshONet Lift-Layer.",
    )
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--best-model-metric", choices=["dataset_mse", "training_loss"])
    parser.add_argument("--hmonet-mapping-mode", choices=mapping_mode_choices())
    parser.add_argument(
        "--boundary-self-attention-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--weight-head-mlp-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use a geometry-coefficient MLP matching the Location MLP; default is enabled.",
    )
    parser.add_argument("--projection-mlp-hidden-dim", type=int)
    parser.add_argument("--attention-dim", type=int)
    parser.add_argument("--attention-heads", type=int)
    parser.add_argument("--attention-layers", type=int)
    parser.add_argument("--attention-feedforward-dim", type=int)
    parser.add_argument("--attention-dropout", type=float)
    for name in (
        "modified-mlp-enabled",
        "query-boundary-cross-attention-enabled",
        "boundary-layer-decoder-enabled",
        "decoupled-weight-decoder-enabled",
        "multiscale-boundary-encoder-enabled",
    ):
        parser.add_argument(
            "--" + name,
            dest=name.replace("-", "_"),
            action=argparse.BooleanOptionalAction,
            default=None,
        )
    parser.add_argument("--cross-attention-heads", type=int)
    parser.add_argument("--boundary-layer-width", type=float)
    parser.add_argument("--local-boundary-dim", type=int)
    parser.add_argument("--projection-coefficient-min", type=float)
    parser.add_argument("--residual-regularization-weight", type=float)
    parser.add_argument(
        "--jacobian-loss-enabled", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--jacobian-loss-weight", type=float)
    parser.add_argument("--jacobian-minimum", type=float)
    parser.add_argument("--jacobian-method", choices=["finite_difference", "autograd"])
    parser.add_argument("--jacobian-fd-step", type=float)
    parser.add_argument(
        "--target-interpolation",
        choices=["bilinear", "curved_boundary_cell"],
    )
    parser.add_argument("--idw-query-chunk-size", type=int)
    parser.add_argument("--idw-support-chunk-size", type=int)
    parser.add_argument(
        "--boundary-normal-loss-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--boundary-normal-loss-weight", type=float)
    parser.add_argument("--boundary-normal-sample-count", type=int)
    parser.add_argument("--boundary-normal-offset", type=float)
    parser.add_argument(
        "--check-max-point-error", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--check-p99-point-error", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--lr-warmup-epochs", type=int)
    parser.add_argument("--lr-update-interval", type=int)
    parser.add_argument("--lr-decay-gamma", type=float)
    parser.add_argument("--lr-stop-below", type=float)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--interior-sample-count", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument("--dataset-mse-interval", type=int)
    parser.add_argument("--fluctuation-quantiles", nargs=2, type=float)
    parser.add_argument("--max-point-error-tolerance", type=float)
    parser.add_argument("--p99-point-error-tolerance", type=float)
    parser.add_argument("--mse-tolerance", type=float)
    parser.add_argument("--mse-consecutive-epochs", type=int)
    parser.add_argument("--training-loss-tolerance", type=float)
    parser.add_argument("--training-loss-consecutive-epochs", type=int)
    parser.add_argument("--plateau-metric", choices=["mse", "train_loss"])
    parser.add_argument("--convergence-start-epoch", type=int)
    parser.add_argument("--plateau-window", type=int)
    parser.add_argument("--fluctuation-relative-tolerance", type=float)
    parser.add_argument("--fluctuation-absolute-minimum", type=float)
    parser.add_argument("--fluctuation-absolute-maximum", type=float)
    parser.add_argument("--plateau-near-best-relative-tolerance", type=float)
    parser.add_argument("--plateau-near-best-absolute-tolerance", type=float)
    parser.add_argument("--plateau-stable-consecutive-checks", type=int)
    parser.add_argument("--best-improvement-relative-tolerance", type=float)
    parser.add_argument("--success-plateau-patience", type=int)
    parser.add_argument("--plateau-log-interval", type=int)
    parser.add_argument("--failure-plateau-patience", type=int)
    parser.add_argument("--failure-plateau-metric-ratio", type=float)
    parser.add_argument("--failure-plateau-relative-fluctuation", type=float)
    parser.add_argument("--failure-plateau-min-lr-updates", type=int)
    for name in (
        "calculate-dataset-mse",
        "check-mse-convergence",
        "check-training-loss-convergence",
        "plateau-enabled",
        "failure-plateau-enabled",
        "save-best-immediately",
        "stop-file-enabled",
        "check-plateau-stability",
        "check-plateau-absolute-stability",
        "check-plateau-near-best",
        "check-plateau-consecutive-stability",
        "check-plateau-best-improvement",
        "skip-nonfinite-loss",
        "tensorboard-enabled",
    ):
        parser.add_argument(
            "--" + name,
            dest=name.replace("-", "_"),
            action=argparse.BooleanOptionalAction,
            default=None,
        )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check-convergence", dest="check_convergence", action="store_true"
    )
    group.add_argument(
        "--no-check-convergence", dest="check_convergence", action="store_false"
    )
    parser.set_defaults(check_convergence=None)
    parser.add_argument("--pi-pde-loss-weight", type=float)
    parser.add_argument("--pi-interior-sample-count", type=int)
    parser.add_argument("--pi-fd-step", type=float, nargs="+")
    parser.add_argument(
        "--pihmonet-mapping-mode", choices=mapping_mode_choices()
    )
    parser.add_argument(
        "--attentionpihmonet-mapping-mode", choices=mapping_mode_choices()
    )
    for prefix in ("pimeshonet", "pihmonet"):
        parser.add_argument(f"--{prefix}-pde-loss-weight", type=float)
        parser.add_argument(f"--{prefix}-interior-sample-count", type=int)
        parser.add_argument(f"--{prefix}-fd-step", type=float, nargs="+")
    parser.add_argument("--pimeshonet-boundary-loss-weight", type=float)
    parser.add_argument(
        "--pihmonet-jacobian-loss-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--pihmonet-jacobian-loss-weight", type=float)
    parser.add_argument("--pihmonet-jacobian-minimum", type=float)
    parser.add_argument("--pihmonet-normalized-pde-loss-weight", type=float)
    parser.add_argument("--pihmonet-anchor-loss-weight", type=float)
    parser.add_argument("--pihmonet-anchor-decay-end-epoch", type=int)
    parser.add_argument("--pihmonet-anchor-points-per-axis", type=int)
    parser.add_argument("--pihmonet-anchor-margin", type=float)
    parser.add_argument("--pihmonet-smoothness-loss-weight", type=float)
    parser.add_argument("--pihmonet-orthogonality-loss-weight", type=float)
    parser.add_argument("--attentionpihmonet-pde-backend", choices=["autograd", "finite_difference"])
    parser.add_argument("--attentionpihmonet-interior-sample-count", type=int)
    parser.add_argument("--attentionpihmonet-interior-margin", type=float)
    parser.add_argument("--attentionpihmonet-fd-step", type=float, nargs="+")
    parser.add_argument("--attentionpihmonet-normalized-pde-loss-weight", type=float)
    parser.add_argument("--attentionpihmonet-jacobian-loss-weight", type=float)
    parser.add_argument("--attentionpihmonet-jacobian-minimum", type=float)
    parser.add_argument("--attentionpihmonet-anchor-loss-weight", type=float)
    parser.add_argument("--attentionpihmonet-anchor-decay-end-epoch", type=int)
    parser.add_argument("--attentionpihmonet-anchor-points-per-axis", type=int)
    parser.add_argument("--attentionpihmonet-anchor-margin", type=float)
    parser.add_argument("--attentionpihmonet-smoothness-loss-weight", type=float)
    parser.add_argument("--attentionpihmonet-boundary-pde-loss-weight", type=float)
    parser.add_argument("--attentionpihmonet-boundary-jacobian-loss-weight", type=float)
    parser.add_argument("--attentionpihmonet-boundary-physics-sample-count", type=int)
    parser.add_argument("--attentionpihmonet-boundary-physics-offset", type=float)
    parser.add_argument("--attentionpihmonet-boundary-physics-tangential-margin", type=float)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    remove_deprecated_projection_config(config)
    training = config.setdefault("training", {})
    for axis in ("nx", "ny", "nz"):
        if getattr(args, axis) is not None:
            config.setdefault("grid", {})[axis] = getattr(args, axis)
    common_model = config.setdefault("model", {}).setdefault("common", {})
    if args.coordinate_lift_mode is not None:
        common_model["coordinate_lift_mode"] = args.coordinate_lift_mode
    if args.coordinate_lift_max_power is not None:
        if args.coordinate_lift_max_power < 1:
            parser.error("--coordinate-lift-max-power must be at least 1")
        common_model["coordinate_lift_max_power"] = args.coordinate_lift_max_power
    if args.coordinate_lift_q is not None:
        if args.coordinate_lift_q < 1:
            parser.error("--coordinate-lift-q must be at least 1")
        common_model["coordinate_lift_q"] = args.coordinate_lift_q
    overrides = {
        "num_epochs": "num_epochs",
        "seed": "seed",
        "device": "device",
        "best_model_metric": "best_model_metric",
        "sample_count": "sample_count",
        "dataset_mse_interval": "dataset_mse_interval",
        "fluctuation_quantiles": "fluctuation_quantiles",
        "max_point_error_tolerance": "max_point_error_tolerance",
        "p99_point_error_tolerance": "p99_point_error_tolerance",
        "mse_tolerance": "mse_tolerance",
        "mse_consecutive_epochs": "mse_consecutive_epochs",
        "training_loss_tolerance": "training_loss_tolerance",
        "training_loss_consecutive_epochs": "training_loss_consecutive_epochs",
        "plateau_metric": "plateau_metric",
        "convergence_start_epoch": "convergence_start_epoch",
        "plateau_window": "plateau_window",
        "fluctuation_relative_tolerance": "fluctuation_relative_tolerance",
        "fluctuation_absolute_minimum": "fluctuation_absolute_minimum",
        "fluctuation_absolute_maximum": "fluctuation_absolute_maximum",
        "plateau_near_best_relative_tolerance": "plateau_near_best_relative_tolerance",
        "plateau_near_best_absolute_tolerance": "plateau_near_best_absolute_tolerance",
        "plateau_stable_consecutive_checks": "plateau_stable_consecutive_checks",
        "best_improvement_relative_tolerance": "best_improvement_relative_tolerance",
        "success_plateau_patience": "success_plateau_patience",
        "plateau_log_interval": "plateau_log_interval",
        "failure_plateau_patience": "failure_plateau_patience",
        "failure_plateau_metric_ratio": "failure_plateau_metric_ratio",
        "failure_plateau_relative_fluctuation": "failure_plateau_relative_fluctuation",
        "failure_plateau_min_lr_updates": "failure_plateau_min_lr_updates",
        "batch_size": "batch_size",
        "gradient_clip_norm": "gradient_clip_norm",
    }
    for source, target in overrides.items():
        value = getattr(args, source)
        if value is not None:
            training[target] = value
    mesh_training = config.setdefault("meshonet_training", {})
    for source in ("interior_sample_count", "alpha", "beta"):
        if getattr(args, source) is not None:
            mesh_training[source] = getattr(args, source)
    for source in (
        "check_max_point_error",
        "check_p99_point_error",
        "calculate_dataset_mse",
        "check_mse_convergence",
        "check_training_loss_convergence",
        "plateau_enabled",
        "failure_plateau_enabled",
        "save_best_immediately",
        "stop_file_enabled",
        "check_plateau_stability",
        "check_plateau_absolute_stability",
        "check_plateau_near_best",
        "check_plateau_consecutive_stability",
        "check_plateau_best_improvement",
        "skip_nonfinite_loss",
        "tensorboard_enabled",
    ):
        if getattr(args, source) is not None:
            training[source] = getattr(args, source)
    if args.pihmonet_mapping_mode:
        config.setdefault("model", {}).setdefault("pihmonet", {})[
            "hmonet_mapping_mode"
        ] = args.pihmonet_mapping_mode
    if args.attentionpihmonet_mapping_mode:
        config.setdefault("model", {}).setdefault("attentionpihmonet", {})[
            "hmonet_mapping_mode"
        ] = args.attentionpihmonet_mapping_mode
    if args.hmonet_mapping_mode:
        config.setdefault("model", {}).setdefault("hmonet", {})[
            "hmonet_mapping_mode"
        ] = args.hmonet_mapping_mode
    hmonet_model = config.setdefault("model", {}).setdefault("hmonet", {})
    for option in (
        "boundary_self_attention_enabled",
        "projection_mlp_hidden_dim",
        "weight_head_mlp_enabled",
        "attention_dim",
        "attention_heads",
        "attention_layers",
        "attention_feedforward_dim",
        "attention_dropout",
        "modified_mlp_enabled",
        "query_boundary_cross_attention_enabled",
        "boundary_layer_decoder_enabled",
        "decoupled_weight_decoder_enabled",
        "multiscale_boundary_encoder_enabled",
        "cross_attention_heads",
        "boundary_layer_width",
        "local_boundary_dim",
    ):
        value = getattr(args, option)
        if value is not None:
            hmonet_model[option] = value
    if args.projection_coefficient_min is not None:
        config.setdefault("model", {}).setdefault("hmonet", {})[
            "projection_coefficient_min"
        ] = args.projection_coefficient_min
    if args.residual_regularization_weight is not None:
        config.setdefault("model", {}).setdefault("hmonet", {})[
            "residual_regularization_weight"
        ] = args.residual_regularization_weight
    htraining = config.setdefault("hmonet_training", {})
    for source in (
        "jacobian_loss_weight",
        "jacobian_minimum",
        "jacobian_method",
        "jacobian_fd_step",
        "target_interpolation",
        "idw_query_chunk_size",
        "idw_support_chunk_size",
        "boundary_normal_loss_weight",
        "boundary_normal_sample_count",
        "boundary_normal_offset",
    ):
        if getattr(args, source) is not None:
            htraining[source] = getattr(args, source)
    for source in ("jacobian_loss_enabled", "boundary_normal_loss_enabled"):
        if getattr(args, source) is not None:
            htraining[source] = getattr(args, source)
    attention_training = config.setdefault("attentionpihmonet_training", {})
    for key in (
        "pde_backend",
        "interior_sample_count",
        "interior_margin",
        "fd_step",
        "normalized_pde_loss_weight",
        "jacobian_loss_weight",
        "jacobian_minimum",
        "anchor_loss_weight",
        "anchor_decay_end_epoch",
        "anchor_points_per_axis",
        "anchor_margin",
        "smoothness_loss_weight",
        "boundary_pde_loss_weight",
        "boundary_jacobian_loss_weight",
        "boundary_physics_sample_count",
        "boundary_physics_offset",
        "boundary_physics_tangential_margin",
    ):
        value = getattr(args, f"attentionpihmonet_{key}")
        if value is not None:
            attention_training[key] = value
    mode = args.dataset_mode or training.get("dataset", "interpolation")
    paths = (
        [Path(args.dataset)]
        if args.dataset
        else resolve_dataset_paths(config, mode, root)
    )
    if not paths:
        raise ValueError("No dataset path supplied")
    choice = args.model or config.get("model", {}).get("name", "hmonet")
    names = parse_model_selection(choice)
    # Explicit CLI lift settings override model-specific YAML values as well
    # as the common block, matching the 2-D training entry point.
    for name in names:
        selected = config.setdefault("model", {}).setdefault(name, {})
        if args.coordinate_lift_mode is not None:
            selected["coordinate_lift_mode"] = args.coordinate_lift_mode
        if args.coordinate_lift_max_power is not None:
            selected["coordinate_lift_max_power"] = args.coordinate_lift_max_power
        if args.coordinate_lift_q is not None:
            selected["coordinate_lift_q"] = args.coordinate_lift_q
    device_name = training.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; select --device cpu")
    seed_everything(int(training.get("seed", 20260905)))
    return train_models(names, paths, config, mode, args, device)


if __name__ == "__main__":
    main()
