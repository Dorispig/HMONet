"""Train dimension-specific operators with independent complete model routines."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import yaml
from tqdm import tqdm

from scripts.train.training_common import (
    BatchLossTotals,
    boundary_normal_loss,
    calculate_fixed_grid_mse_2d,
    hmonet_main_loss,
    interpolate_pde_mesh,
    jacobian_positive_loss_autograd,
    jacobian_positive_loss_finite_difference,
    jacobian_finite_difference_queries,
    make_data_loader,
    predict_coordinates,
    resolve_dataset_paths,
    safe_optimizer_step,
    sample_fixed_grid_batch,
    sample_shared_random_points,
    seed_everything,
    update_learning_rate,
)
from scripts.train.training_state import (
    TrainingState,
    prepare_training_config,
    print_training_summaries,
)
from src.utils.config_validation import remove_deprecated_projection_config
from src.data.dataset import ParametricDataset2D, uniformly_spaced_subset
from src.models.model import HMONet2D, MeshONet2D
from src.data.mesh.sampling import (
    sample_curvature_distance_adaptive_points,
    sample_rectangle_boundary_uniform,
)
from src.utils.math import compute_discrete_boundary_curvature


def train_hmonet(datasets, config, mode, args, device):
    """Train HMONet with batched boundary objects and weighted coordinate targets."""
    config = prepare_training_config(config, "hmonet", args)
    training = config["training"]
    geometry_options = config["hmonet_training"]
    learning_rate = geometry_options
    shape = tuple(datasets[0].grid_shape)
    seed_everything(int(training["seed"]))
    model_options = dict(config["model"].get("common", {}))
    model_options.update(config["model"].get("hmonet", {}))
    residual_weight = float(model_options.pop("residual_regularization_weight", 0.0))
    if residual_weight < 0:
        raise ValueError("residual_regularization_weight must be nonnegative")
    model = HMONet2D(*shape, **model_options).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate["learning_rate"]
    )
    loader = make_data_loader(
        datasets, training["batch_size"], training["seed"], shuffle=True
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
        geometry_options.get("target_interpolation", "curved_boundary_cell")
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
    sampling = config.get("hmonet_sampling", {})
    fixed_count = int(sampling.get("fixed_grid_count", 0))
    sampling_mode = str(sampling.get("mode", "uniform")).lower()
    if sampling_mode not in {"uniform", "adaptive"}:
        raise ValueError("hmonet_sampling.mode must be uniform or adaptive")
    random_count = (
        int(sampling.get("global_random_count", 400))
        if sampling_mode == "uniform"
        else int(sampling.get("curvature_adaptive_count", 400))
    )
    if fixed_count < 0 or random_count < 0 or fixed_count + random_count == 0:
        raise ValueError("HMONet requires a positive fixed or random point count")
    computational_boundary = torch.as_tensor(
        sample_rectangle_boundary_uniform(*shape)[0],
        dtype=torch.float32,
        device=device,
    )
    curvature_quantile = float(sampling.get("curvature_quantile", 0.95))
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
                from src.data.geometry.parametric_curve_2d import BoundaryCurveSet2D

                geometries = BoundaryCurveSet2D.batch(
                    batch["boundary"], device=device, dtype=boundary.dtype
                )
                point_groups = []
                derivative_margin = jacobian_step if use_fd_jacobian else 0.0
                if fixed_count:
                    logical, target = sample_fixed_grid_batch(
                        mesh, shape, fixed_count, margin=derivative_margin
                    )
                    point_groups.append((True, logical, target))
                if random_count:
                    if sampling_mode == "uniform":
                        logical = sample_shared_random_points(
                            len(mesh), random_count, 2, device,
                            margin=derivative_margin,
                        )
                    else:
                        boundary_curvature = compute_discrete_boundary_curvature(
                            boundary, quantile=curvature_quantile
                        )
                        logical = sample_curvature_distance_adaptive_points(
                            boundary_curvature,
                            computational_boundary,
                            random_count,
                            candidate_multiplier=int(
                                sampling.get("candidate_multiplier", 10)
                            ),
                            curvature_power=float(
                                sampling.get("curvature_power", 1.0)
                            ),
                            sigma_min=float(
                                sampling.get("distance_sigma_min", 0.02)
                            ),
                            sigma_max=float(
                                sampling.get("distance_sigma_max", 0.10)
                            ),
                            uniform_probability=float(
                                sampling.get("uniform_probability", 0.05)
                            ),
                            margin=derivative_margin,
                        )
                    target = interpolate_pde_mesh(
                        mesh,
                        logical,
                        shape,
                        target_interpolation=target_interpolation,
                        boundary_geometry=geometries,
                        idw_neighbors=int(geometry_options.get("idw_neighbors", -1)),
                        idw_query_chunk_size=int(
                            geometry_options.get("idw_query_chunk_size", 128)
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
                        2,
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
                totals.add(
                    len(mesh),
                    loss=loss,
                    coordinate_loss=coordinate_loss,
                    jacobian=jacobian,
                    boundary_normal=normal,
                )
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
    model = MeshONet2D(*shape, **model_options).to(device)
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
    coordinates, indices = sample_rectangle_boundary_uniform(*shape)
    computational_boundary = torch.as_tensor(
        coordinates, dtype=torch.float32, device=device
    )
    boundary_ids = torch.as_tensor(indices, dtype=torch.long, device=device)
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
                # Convert component-wise MSE into mean squared Euclidean
                # distance per 2-D point, matching HMONet and dataset MSE.
                boundary_loss = 2.0 * (
                    (boundary_prediction - boundary[:, boundary_ids]).square().mean()
                )
                logical = sample_shared_random_points(len(mesh), count, 2, device)
                prediction = predict_coordinates(model, boundary, logical)
                target = interpolate_pde_mesh(mesh, logical, shape)
                interior_loss = 2.0 * (prediction - target).square().mean()
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


def train_pihmonet(datasets, config, mode, args, device):
    """Train hard-constrained HMONet using finite-difference PDE residuals."""
    from scripts.train.train_pihmonet import train_pihmonet as train_physics

    return train_physics(datasets, config, mode, args, device)


def train_attentionpihmonet(datasets, config, mode, args, device):
    """Train attention-encoded hard-constrained physics-informed HMONet."""
    from scripts.train.train_attentionpihmonet import train_attentionpihmonet as train_physics

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
            run_args.config_archive_root = root / "configs" / "2D"
            run_args.dataset_name = dataset_name
            run_args.mode = run_mode
            data = ParametricDataset2D(
                path,
                load_boundary=(name in ("hmonet", "pihmonet", "attentionpihmonet")),
            )
            subset_count = config.get("training", {}).get("dataset_sample_count")
            if subset_count is not None:
                data = uniformly_spaced_subset(data, int(subset_count))
                print(
                    f"Training subset: {len(data)}/{data.original_size} samples "
                    "selected by uniform index spacing",
                    flush=True,
                )
            for axis, actual in zip(("nx", "ny"), data.grid_shape):
                requested = getattr(args, axis, None)
                if requested is not None and int(requested) != int(actual):
                    raise ValueError(
                        f"Explicit --{axis}={requested} differs from saved dataset "
                        f"grid {data.grid_shape}"
                    )
            run_config = copy.deepcopy(config)
            run_config.setdefault("grid", {}).update(
                dict(zip(("nx", "ny"), map(int, data.grid_shape)))
            )
            print(
                f"[{run_mode}] model={name}, dataset={dataset_name}, "
                f"grid={data.grid_shape} (read from dataset)",
                flush=True,
            )
            if name == "hmonet":
                train_hmonet([data], run_config, run_mode, run_args, device)
            elif name == "pimeshonet":
                train_pimeshonet([data], run_config, run_mode, run_args, device)
            elif name == "pihmonet":
                train_pihmonet([data], run_config, run_mode, run_args, device)
            elif name == "attentionpihmonet":
                train_attentionpihmonet(
                    [data], run_config, run_mode, run_args, device
                )
            elif name == "meshonet":
                train_meshonet([data], run_config, run_mode, run_args, device)
            else:
                raise ValueError(f"Unknown model: {name}")
            summary_path = Path(run_args.training_output_dir) / "training_summary.json"
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    print_training_summaries(summaries, time.perf_counter() - started, dimension=2)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/2D/train.yaml")
    parser.add_argument("--dataset")
    parser.add_argument(
        "--dataset-mode", choices=["interpolation", "extrapolation", "both"]
    )
    parser.add_argument(
        "--model",
        choices=["hmonet", "meshonet", "pimeshonet", "pihmonet", "attentionpihmonet", "both", "pi", "all"],
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--dataset-sample-count",
        type=int,
        help="Use an evenly spaced in-memory subset of the saved training dataset.",
    )
    parser.add_argument("--output-root", default="outputs/training/2D")
    parser.add_argument("--flat-run-root", help=argparse.SUPPRESS)
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    # Operator-network capacity overrides. These options are primarily used by
    # the model-capacity ablation, but are also useful for reproducible manual
    # training runs without editing the YAML file.
    parser.add_argument("--branch-dim", type=int)
    parser.add_argument("--trunk-dim", type=int)
    parser.add_argument("--adjust-hidden-dim", type=int)
    parser.add_argument("--adjust-hidden-layer-num", type=int)
    parser.add_argument("--branch-hidden-dim", type=int)
    parser.add_argument("--branch-hidden-layer-num", type=int)
    parser.add_argument("--trunk-hidden-dim", type=int)
    parser.add_argument("--trunk-hidden-layer-num", type=int)
    parser.add_argument("--output-dim", type=int)
    parser.add_argument("--branch-mlp-hidden-dim", type=int)
    parser.add_argument("--branch-mlp-hidden-layer-num", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--best-model-metric", choices=["dataset_mse", "training_loss"])
    parser.add_argument("--hmonet-mapping-mode", choices=["current", "coons_residual"])
    parser.add_argument(
        "--trunk-projection-coefficients-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--projection-softplus-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--boundary-self-attention-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--projection-mlp-hidden-dim", type=int)
    parser.add_argument("--attention-dim", type=int)
    parser.add_argument("--attention-heads", type=int)
    parser.add_argument("--attention-layers", type=int)
    parser.add_argument("--attention-feedforward-dim", type=int)
    parser.add_argument("--attention-dropout", type=float)
    parser.add_argument("--projection-coefficient-min", type=float)
    parser.add_argument("--projection-transition-width", type=float)
    parser.add_argument("--projection-normal-smoothing", type=float)
    parser.add_argument("--projection-tangential-lift-beta", type=float)
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
    parser.add_argument("--idw-neighbors", type=int)
    parser.add_argument("--idw-query-chunk-size", type=int)
    parser.add_argument("--idw-support-chunk-size", type=int)
    parser.add_argument(
        "--point-loss-weighting",
        choices=["position", "uniform"],
        help="Use boundary-enhanced position weights or equal point weights.",
    )
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
    parser.add_argument("--fixed-grid-count", type=int)
    parser.add_argument("--global-random-count", type=int)
    parser.add_argument("--curvature-adaptive-count", type=int)
    parser.add_argument("--sampling-mode", choices=["uniform", "adaptive"])
    parser.add_argument("--adaptive-candidate-multiplier", type=int)
    parser.add_argument("--adaptive-curvature-quantile", type=float)
    parser.add_argument("--adaptive-curvature-power", type=float)
    parser.add_argument("--adaptive-distance-sigma-min", type=float)
    parser.add_argument("--adaptive-distance-sigma-max", type=float)
    parser.add_argument("--adaptive-uniform-probability", type=float)
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
        "--pihmonet-mapping-mode", choices=["current", "coons_residual"]
    )
    parser.add_argument(
        "--attentionpihmonet-mapping-mode", choices=["current", "coons_residual"]
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
    parser.add_argument("--attentionpihmonet-boundary-loss-weight", type=float)
    parser.add_argument("--attentionpihmonet-alpha-penalty-weight", type=float)
    parser.add_argument("--attentionpihmonet-boundary-loss-sample-count", type=int)
    parser.add_argument("--attentionpihmonet-boundary-alpha-initial-value", type=float)
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
    if args.nx:
        config.setdefault("grid", {})["nx"] = args.nx
    if args.ny:
        config.setdefault("grid", {})["ny"] = args.ny
    common_model = config.setdefault("model", {}).setdefault("common", {})
    for option in (
        "branch_dim",
        "trunk_dim",
        "adjust_hidden_dim",
        "adjust_hidden_layer_num",
        "branch_hidden_dim",
        "branch_hidden_layer_num",
        "trunk_hidden_dim",
        "trunk_hidden_layer_num",
        "output_dim",
        "branch_mlp_hidden_dim",
        "branch_mlp_hidden_layer_num",
    ):
        value = getattr(args, option)
        if value is not None:
            if value <= 0:
                parser.error(f"--{option.replace('_', '-')} must be positive")
            common_model[option] = value
    overrides = {
        "num_epochs": "num_epochs",
        "seed": "seed",
        "device": "device",
        "best_model_metric": "best_model_metric",
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
        "dataset_sample_count": "dataset_sample_count",
        "gradient_clip_norm": "gradient_clip_norm",
    }
    for source, target in overrides.items():
        value = getattr(args, source)
        if value is not None:
            training[target] = value
    if training.get("dataset_sample_count") is not None:
        if int(training["dataset_sample_count"]) <= 0:
            parser.error("--dataset-sample-count must be positive")
        training["dataset_sample_count"] = int(training["dataset_sample_count"])
        training["dataset_subset_mode"] = "uniform_index"
    sampling = config.setdefault("hmonet_sampling", {})
    sampling_overrides = {
        "fixed_grid_count": "fixed_grid_count",
        "global_random_count": "global_random_count",
        "curvature_adaptive_count": "curvature_adaptive_count",
        "sampling_mode": "mode",
        "adaptive_candidate_multiplier": "candidate_multiplier",
        "adaptive_curvature_quantile": "curvature_quantile",
        "adaptive_curvature_power": "curvature_power",
        "adaptive_distance_sigma_min": "distance_sigma_min",
        "adaptive_distance_sigma_max": "distance_sigma_max",
        "adaptive_uniform_probability": "uniform_probability",
    }
    for source, target in sampling_overrides.items():
        value = getattr(args, source)
        if value is not None:
            sampling[target] = value
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
        "trunk_projection_coefficients_enabled",
        "projection_softplus_enabled",
        "boundary_self_attention_enabled",
        "projection_mlp_hidden_dim",
        "attention_dim",
        "attention_heads",
        "attention_layers",
        "attention_feedforward_dim",
        "attention_dropout",
    ):
        value = getattr(args, option)
        if value is not None:
            hmonet_model[option] = value
    if args.projection_coefficient_min is not None:
        config.setdefault("model", {}).setdefault("hmonet", {})[
            "projection_coefficient_min"
        ] = args.projection_coefficient_min
    if args.projection_transition_width is not None:
        config.setdefault("model", {}).setdefault("hmonet", {})[
            "projection_transition_width"
        ] = args.projection_transition_width
    if args.projection_normal_smoothing is not None:
        config.setdefault("model", {}).setdefault("hmonet", {})[
            "projection_normal_smoothing"
        ] = args.projection_normal_smoothing
    if args.projection_tangential_lift_beta is not None:
        config.setdefault("model", {}).setdefault("hmonet", {})[
            "projection_tangential_lift_beta"
        ] = args.projection_tangential_lift_beta
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
        "idw_neighbors",
        "idw_query_chunk_size",
        "idw_support_chunk_size",
        "point_loss_weighting",
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
        "boundary_loss_weight",
        "alpha_penalty_weight",
        "boundary_loss_sample_count",
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
    alpha_initial = getattr(args, "attentionpihmonet_boundary_alpha_initial_value")
    if alpha_initial is not None:
        config.setdefault("model", {}).setdefault("attentionpihmonet", {})[
            "boundary_alpha_initial_value"
        ] = alpha_initial
    mode = args.dataset_mode or training.get("dataset", "interpolation")
    paths = (
        [Path(args.dataset)]
        if args.dataset
        else resolve_dataset_paths(config, mode, root)
    )
    if not paths:
        raise ValueError("No dataset path supplied")
    choice = args.model or config.get("model", {}).get("name", "hmonet")
    names = (
        ["hmonet", "meshonet", "pimeshonet", "pihmonet", "attentionpihmonet"]
        if choice == "all"
        else (
            ["hmonet", "meshonet"]
            if choice == "both"
            else ["pimeshonet", "pihmonet", "attentionpihmonet"]
            if choice == "pi"
            else [choice]
        )
    )
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
