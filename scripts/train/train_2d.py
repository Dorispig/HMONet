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
from src.utils.progress import training_progress as tqdm

from scripts.train.model_selection import parse_model_selection
from src.losses.physics import (
    elliptic_grid_pde_residual,
    jacobian_positive_loss as physics_jacobian_positive_loss,
    normalized_mapping_barriers,
    normalized_mapping_barriers_from_jacobian,
    elliptic_grid_scale_invariant_pde_loss,
    stencil_queries as physics_stencil_queries,
    winslow_energy_and_jacobian_barrier,
)
from src.numerics.differential import autograd_differential_terms
from scripts.train.pi_anchor import masked_anchor_mse, tfi_anchor_targets
from src.engine.training import (
    BatchLossTotals,
    boundary_normal_loss,
    calculate_fixed_grid_mse_2d,
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
from src.data.parametric_dataset import ParametricDataset2D, uniformly_spaced_subset
from src.models import HMONet2D, MeshONet2D
from src.data.mesh.sampling import (
    sample_curvature_distance_adaptive_points,
    sample_inward_boundary_band_points_2d,
    sample_rectangle_boundary_uniform,
)
from src.utils.math import compute_discrete_boundary_curvature


_HMONET_MULTISTAGE_STRATEGIES = {
    "supervised",
    "tfi_warmup_pde",
    "tfi_warmup_pde_jacobian",
    "supervised_then_fixed_mse_pde",
    "supervised_then_fixed_mse_pde_jacobian",
    "supervised_then_pde_from_convergence",
    "fixed_mse_pde_from_start",
    "supervised_then_fixed_mse_pde_from_convergence",
    "supervised_then_fixed_mse_pde_jacobian_from_convergence",
    "random_loss_combo_from_start",
    "random_mse_jac_then_mse_pde_jac",
    "random_mse_jac_then_pde_fixed_mse",
}


def _physics_phase_loss(
    *,
    phase,
    model,
    boundary,
    mesh,
    geometries,
    shape,
    device,
    pde_steps,
    tfi_count,
    pde_count,
    fixed_mse_count,
    pde_weight,
    jacobian_weight,
    jacobian_minimum,
    fixed_logical,
    fixed_indices,
    fixed_mse_weight=1.0,
    teacher=None,
    teacher_anchor_weight=0.0,
    teacher_anchor_count=128,
):
    """Evaluate one HMONet objective used by a multi-stage experiment.

    ``tfi_warmup`` learns a stable, boundary-exact transfinite initialization.
    The physics phases then use the same finite-difference stencil for the
    Winslow residual and, where requested, the positive-Jacobian penalty.
    """
    if phase == "tfi_warmup":
        logical = sample_shared_random_points(len(mesh), tfi_count, 2, device)
        target, valid = tfi_anchor_targets(boundary, logical, shape)
        prediction = predict_coordinates(model, boundary, logical, geometries)
        coordinate = masked_anchor_mse(prediction, target, valid)
        return coordinate, {
            "loss": coordinate,
            "tfi_mse": coordinate,
        }

    coordinate = None
    if phase in {"fixed_mse_physics", "fixed_mse_physics_jacobian"}:
        count = min(int(fixed_mse_count), int(fixed_logical.shape[0]))
        selected = torch.randperm(fixed_logical.shape[0], device=device)[:count]
        logical = fixed_logical[selected].unsqueeze(0).expand(len(mesh), -1, -1)
        target = mesh[:, fixed_indices[selected]]
        prediction = predict_coordinates(model, boundary, logical, geometries)
        # The two coordinate components form a squared Euclidean error.
        coordinate = 2.0 * (prediction - target).square().mean()

    centers = sample_shared_random_points(
        len(mesh), pde_count, 2, device, margin=float(pde_steps.max().item())
    )
    queries = physics_stencil_queries(centers, pde_steps)
    values = predict_coordinates(
        model, boundary, queries.flatten(-3, -2), geometries
    ).reshape(*queries.shape)
    pde = elliptic_grid_pde_residual(values, pde_steps).square().sum(-1).mean()
    use_jacobian = (
        phase in {"pde_jacobian", "fixed_mse_physics_jacobian"}
        and float(jacobian_weight) > 0.0
    )
    jacobian = (
        physics_jacobian_positive_loss(values, pde_steps, jacobian_minimum)
        if use_jacobian
        else None
    )
    loss = float(pde_weight) * pde
    if coordinate is not None:
        loss = loss + float(fixed_mse_weight) * coordinate
    if jacobian is not None:
        loss = loss + float(jacobian_weight) * jacobian
    teacher_anchor = None
    if teacher is not None and teacher_anchor_weight > 0:
        anchor_logical = sample_shared_random_points(
            len(mesh), teacher_anchor_count, 2, device
        )
        anchor_prediction = predict_coordinates(
            model, boundary, anchor_logical, geometries
        )
        with torch.no_grad():
            anchor_target = predict_coordinates(
                teacher, boundary, anchor_logical, geometries
            )
        teacher_anchor = 2.0 * (anchor_prediction - anchor_target).square().mean()
        loss = loss + float(teacher_anchor_weight) * teacher_anchor
    components = {
        "loss": loss,
        "loss_pde": pde,
    }
    if coordinate is not None:
        # This is the same two-coordinate MSE convention used by supervised
        # HMONet training, evaluated on the selected fixed PDE grid nodes.
        components["fixed_mse"] = coordinate
    if jacobian is not None:
        components["jacobian"] = jacobian
    if teacher_anchor is not None:
        components["teacher_anchor"] = teacher_anchor
    return loss, components


def _physics_progress_postfix(loss, components):
    """Format only the active losses, without an internal strategy label."""
    if all(
        key in components
        for key in ("loss_pde", "winslow", "jacobian_barrier")
    ):
        # The Winslow experiment has one fixed, compact terminal schema.  Its
        # Jacobian term is the positive-area barrier rather than the legacy
        # unnormalised Jacobian penalty, so expose it under the common
        # ``jacobian_loss`` label used in experiment reports.
        return {
            "loss": f"{loss.item():.3e}",
            "pde_loss": f"{components['loss_pde'].item():.3e}",
            "winslow_loss": f"{components['winslow'].item():.3e}",
            "jacobian_loss": f"{components['jacobian_barrier'].item():.3e}",
        }
    postfix = {"loss": f"{loss.item():.3e}"}
    if "tfi_mse" in components:
        postfix["tfi_mse"] = f"{components['tfi_mse'].item():.3e}"
    if "fixed_mse" in components:
        postfix["fixed_mse"] = f"{components['fixed_mse'].item():.3e}"
    if "random_mse" in components:
        postfix["random_mse"] = f"{components['random_mse'].item():.3e}"
    if "loss_pde" in components:
        postfix["pde_loss"] = f"{components['loss_pde'].item():.3e}"
    if "jacobian" in components:
        postfix["jacobian_loss"] = f"{components['jacobian'].item():.3e}"
    if "winslow" in components:
        postfix["winslow_loss"] = f"{components['winslow'].item():.3e}"
    if "jacobian_barrier" in components:
        postfix["jacobian_barrier"] = f"{components['jacobian_barrier'].item():.3e}"
    if "normalized_jacobian" in components:
        postfix["normalized_jacobian_loss"] = (
            f"{components['normalized_jacobian'].item():.3e}"
        )
    if "singular_value" in components:
        postfix["singular_value_loss"] = (
            f"{components['singular_value'].item():.3e}"
        )
    if "teacher_anchor" in components:
        postfix["teacher_anchor"] = f"{components['teacher_anchor'].item():.3e}"
    if "effective_pde_weight" in components:
        postfix["pde_weight"] = f"{components['effective_pde_weight'].item():.3e}"
    return postfix


def _gradient_l2_norm(loss, parameters):
    """Measure an unweighted loss gradient without consuming the training graph."""
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    return sum(
        (gradient.detach().square().sum() for gradient in gradients if gradient is not None),
        torch.zeros((), device=loss.device),
    ).sqrt().item()


@torch.no_grad()
def _finetune_mesh_quality(model, datasets, device, grid_size=101):
    """Check corner orientation and the worst scaled Jacobian on three geometries."""
    from src.data.geometry.parametric_curve_2d import BoundaryCurveSet2D

    query, _ = logical_grid_and_interior_indices((grid_size, grid_size), device)
    selected = sorted({0, len(datasets[0]) // 2, len(datasets[0]) - 1})
    invalid = 0
    minimum_scaled = float("inf")
    model.eval()
    for index in selected:
        sample = datasets[0][index]
        boundary = torch.as_tensor(
            sample["boundary_points"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        geometry = BoundaryCurveSet2D.batch(
            [sample["boundary"]], device=device, dtype=boundary.dtype
        )
        points = predict_coordinates(
            model, boundary, query.unsqueeze(0), geometry
        ).reshape(grid_size, grid_size, 2)
        p00, p10 = points[:-1, :-1], points[:-1, 1:]
        p01, p11 = points[1:, :-1], points[1:, 1:]
        coarse = torch.as_tensor(
            sample["mesh_points"], dtype=torch.float32, device=device
        ).reshape(datasets[0].grid_shape[1], datasets[0].grid_shape[0], 2)
        coarse_x = coarse[0, 1] - coarse[0, 0]
        coarse_y = coarse[1, 0] - coarse[0, 0]
        orientation = torch.sign(
            coarse_x[0] * coarse_y[1] - coarse_x[1] * coarse_y[0]
        )
        if orientation == 0:
            raise ValueError("Cannot determine validation mesh orientation")
        pairs = (
            (p10 - p00, p01 - p00),
            (p11 - p10, p00 - p10),
            (p01 - p11, p10 - p11),
            (p00 - p01, p11 - p01),
        )
        corner_crosses = []
        for a, b in pairs:
            cross = orientation * (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0])
            corner_crosses.append(cross)
            scaled = cross / (torch.linalg.vector_norm(a, dim=-1) *
                              torch.linalg.vector_norm(b, dim=-1)).clamp_min(1e-12)
            minimum_scaled = min(minimum_scaled, float(scaled.min().item()))
        invalid += int((torch.stack(corner_crosses, dim=0) <= 0).any(0).sum().item())
    return invalid, minimum_scaled


def _mixed_random_loss_points(
    *,
    batch_size,
    device,
    dtype,
    derivative_backend,
    steps,
    global_count,
    curvature_count,
    boundary_band_count,
    boundary_curvature,
    computational_boundary,
    candidate_multiplier,
    curvature_power,
    sigma_min,
    sigma_max,
    uniform_probability,
    boundary_min_step_multiple,
    boundary_max_step_multiple,
    boundary_min_distance,
    boundary_max_distance,
):
    """Build one shared 200/100/100-style physics point set."""
    counts = tuple(map(int, (global_count, curvature_count, boundary_band_count)))
    if min(counts) < 0 or sum(counts) < 1:
        raise ValueError("Mixed random-loss sample counts must be nonnegative and nonempty")
    if derivative_backend not in {"finite_difference", "autograd"}:
        raise ValueError("random-loss derivative backend must be finite_difference or autograd")

    finite_difference = derivative_backend == "finite_difference"
    margin = steps if finite_difference else 0.0
    parts = []
    if counts[0]:
        parts.append(
            sample_shared_random_points(
                batch_size, counts[0], 2, device, margin=margin
            ).to(dtype=dtype)
        )
    if counts[1]:
        parts.append(
            sample_curvature_distance_adaptive_points(
                boundary_curvature,
                computational_boundary,
                counts[1],
                candidate_multiplier=candidate_multiplier,
                curvature_power=curvature_power,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                uniform_probability=uniform_probability,
                margin=float(torch.as_tensor(margin).max().item()),
            ).to(dtype=dtype)
        )
    if counts[2]:
        if finite_difference:
            step = float(torch.as_tensor(steps).max().item())
            minimum = float(boundary_min_step_multiple) * step
            maximum = float(boundary_max_step_multiple) * step
            tangent_margin = step
        else:
            minimum = float(boundary_min_distance)
            maximum = float(boundary_max_distance)
            tangent_margin = 0.0
        parts.append(
            sample_inward_boundary_band_points_2d(
                batch_size,
                counts[2],
                min_distance=minimum,
                max_distance=maximum,
                device=device,
                dtype=dtype,
                tangent_margin=tangent_margin,
            )
        )
    return torch.cat(parts, dim=1)


def _autograd_scale_invariant_pde_and_jacobian(prediction, logical):
    """Return scale-invariant PDE residual and the exact mapping Jacobian."""
    terms = autograd_differential_terms(prediction, logical)
    metric_trace = terms["metric"].diagonal(dim1=-2, dim2=-1).sum(-1)
    exponent = 2 * prediction.shape[-1] - 1
    denominator = metric_trace.pow(exponent).clamp_min(1.0e-18)
    pde = (terms["residual"].square().sum(-1) / denominator).mean()
    return pde, terms["jacobian"]


def _random_loss_combo(
    *, model, boundary, mesh, geometries, shape, device, steps,
    sample_count, mse_weight, pde_weight, jacobian_weight,
    jacobian_minimum, target_interpolation, point_loss_weighting,
    idw_query_chunk_size, idw_support_chunk_size,
    winslow_weight=0.0, winslow_barrier_weight=0.0,
    winslow_minimum=1e-4, winslow_epsilon=1e-6,
    winslow_fd_step=1e-4,
    scale_invariant_pde=False,
    normalized_jacobian_weight=0.0,
    normalized_jacobian_minimum=0.02,
    singular_value_weight=0.0,
    singular_value_minimum=0.05,
    barrier_reduction="active",
    derivative_backend="finite_difference",
    mixed_sampling=False,
    global_count=0,
    curvature_count=0,
    boundary_band_count=0,
    boundary_curvature=None,
    computational_boundary=None,
    candidate_multiplier=10,
    curvature_power=1.0,
    sigma_min=0.02,
    sigma_max=0.10,
    uniform_probability=0.05,
    boundary_min_step_multiple=1.0,
    boundary_max_step_multiple=4.0,
    boundary_min_distance=0.0,
    boundary_max_distance=4.0e-4,
):
    """One-stage random-point MSE/PDE/Jacobian/Winslow ablation objective.

    Physics terms share one fresh point set each batch. Existing experiments
    retain uniform finite-difference sampling; the mixed mode combines global,
    curvature-adaptive and inward boundary-band points. MSE uses the selected
    interpolation.
    """
    if min(
        mse_weight, pde_weight, jacobian_weight, winslow_weight,
        winslow_barrier_weight, normalized_jacobian_weight,
        singular_value_weight,
    ) < 0:
        raise ValueError("Random loss weights must be nonnegative")
    if (
        mse_weight + pde_weight + jacobian_weight + winslow_weight
        + winslow_barrier_weight + normalized_jacobian_weight
        + singular_value_weight
    ) <= 0:
        raise ValueError("At least one random loss weight must be positive")
    if (
        normalized_jacobian_weight > 0 and normalized_jacobian_minimum <= 0
    ) or (singular_value_weight > 0 and singular_value_minimum <= 0):
        raise ValueError("Normalized Jacobian and singular-value minima must be positive")
    if (winslow_weight > 0 or winslow_barrier_weight > 0) and (
        winslow_minimum <= 0 or winslow_epsilon <= 0 or not 0 < winslow_fd_step <= 1 / 3
    ):
        raise ValueError("Winslow minimum and epsilon must be positive")
    if derivative_backend == "autograd" and (
        winslow_weight > 0 or winslow_barrier_weight > 0
    ):
        raise ValueError("Autograd random-loss mode does not support Winslow terms")
    components = {}
    loss = torch.zeros((), device=device)
    if mse_weight > 0:
        logical = sample_shared_random_points(len(mesh), sample_count, 2, device)
        target = interpolate_pde_mesh(
            mesh, logical, shape,
            target_interpolation=target_interpolation,
            boundary_geometry=geometries,
            idw_query_chunk_size=idw_query_chunk_size,
            idw_support_chunk_size=idw_support_chunk_size,
        )
        prediction = predict_coordinates(model, boundary, logical, geometries)
        mse = hmonet_main_loss(
            prediction, target, logical, point_weighting=point_loss_weighting
        )
        loss = loss + float(mse_weight) * mse
        components["random_mse"] = mse
    if (
        pde_weight > 0 or jacobian_weight > 0 or winslow_weight > 0
        or winslow_barrier_weight > 0 or normalized_jacobian_weight > 0
        or singular_value_weight > 0
    ):
        has_winslow = winslow_weight > 0 or winslow_barrier_weight > 0
        if mixed_sampling:
            if curvature_count > 0 and (
                boundary_curvature is None or computational_boundary is None
            ):
                raise ValueError("Mixed sampling requires boundary curvature and logical boundary")
            centers = _mixed_random_loss_points(
                batch_size=len(mesh),
                device=device,
                dtype=boundary.dtype,
                derivative_backend=derivative_backend,
                steps=steps,
                global_count=global_count,
                curvature_count=curvature_count,
                boundary_band_count=boundary_band_count,
                boundary_curvature=boundary_curvature,
                computational_boundary=computational_boundary,
                candidate_multiplier=candidate_multiplier,
                curvature_power=curvature_power,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                uniform_probability=uniform_probability,
                boundary_min_step_multiple=boundary_min_step_multiple,
                boundary_max_step_multiple=boundary_max_step_multiple,
                boundary_min_distance=boundary_min_distance,
                boundary_max_distance=boundary_max_distance,
            )
        else:
            margin = (
                torch.maximum(steps, torch.full_like(steps, float(winslow_fd_step)))
                if has_winslow else (steps if derivative_backend == "finite_difference" else 0.0)
            )
            centers = sample_shared_random_points(
                len(mesh), sample_count, 2, device, margin=margin
            )

        autograd_jacobian = None
        if derivative_backend == "autograd":
            centers = centers.detach().requires_grad_(True)
            prediction = predict_coordinates(model, boundary, centers, geometries)
            autograd_pde, autograd_jacobian = (
                _autograd_scale_invariant_pde_and_jacobian(prediction, centers)
            )
            values = None
        else:
            queries = physics_stencil_queries(centers, steps)
            if has_winslow:
                geometry_queries = jacobian_finite_difference_queries(
                    centers, winslow_fd_step
                )
                combined_queries = torch.cat((queries, geometry_queries), dim=-2)
                combined_values = predict_coordinates(
                    model, boundary, combined_queries.flatten(-3, -2), geometries
                ).reshape(*combined_queries.shape)
                values = combined_values[..., :queries.shape[-2], :]
                geometry_values = combined_values[..., queries.shape[-2]:, :]
            else:
                values = predict_coordinates(
                    model, boundary, queries.flatten(-3, -2), geometries
                ).reshape(*queries.shape)
        if pde_weight > 0:
            if derivative_backend == "autograd":
                if not scale_invariant_pde:
                    raise ValueError(
                        "Autograd random-loss mode currently requires scale-invariant PDE"
                    )
                pde = autograd_pde
            else:
                pde = (
                    elliptic_grid_scale_invariant_pde_loss(values, steps)
                    if scale_invariant_pde
                    else elliptic_grid_pde_residual(values, steps).square().sum(-1).mean()
                )
            loss = loss + float(pde_weight) * pde
            components["loss_pde"] = pde
        if jacobian_weight > 0:
            if derivative_backend == "autograd":
                determinant = torch.linalg.det(autograd_jacobian)
                jacobian = torch.relu(
                    float(jacobian_minimum) - determinant
                ).square().mean()
            else:
                jacobian = physics_jacobian_positive_loss(
                    values, steps, jacobian_minimum
                )
            loss = loss + float(jacobian_weight) * jacobian
            components["jacobian"] = jacobian
        if normalized_jacobian_weight > 0 or singular_value_weight > 0:
            extents = boundary.amax(dim=-2) - boundary.amin(dim=-2)
            characteristic_length = extents.prod(dim=-1).clamp_min(1e-12).sqrt()
            if derivative_backend == "autograd":
                normalized_jacobian, singular_value = (
                    normalized_mapping_barriers_from_jacobian(
                        autograd_jacobian,
                        characteristic_length,
                        jacobian_minimum=normalized_jacobian_minimum,
                        singular_value_minimum=singular_value_minimum,
                        reduction=barrier_reduction,
                    )
                )
            else:
                normalized_jacobian, singular_value = normalized_mapping_barriers(
                    values,
                    steps,
                    characteristic_length,
                    jacobian_minimum=normalized_jacobian_minimum,
                    singular_value_minimum=singular_value_minimum,
                    reduction=barrier_reduction,
                )
            if normalized_jacobian_weight > 0:
                loss = loss + float(normalized_jacobian_weight) * normalized_jacobian
                components["normalized_jacobian"] = normalized_jacobian
            if singular_value_weight > 0:
                loss = loss + float(singular_value_weight) * singular_value
                components["singular_value"] = singular_value
        if winslow_weight > 0 or winslow_barrier_weight > 0:
            winslow, barrier = winslow_energy_and_jacobian_barrier(
                geometry_values,
                torch.full_like(steps, float(winslow_fd_step)),
                winslow_minimum, winslow_epsilon,
            )
            if winslow_weight > 0:
                loss = loss + float(winslow_weight) * winslow
                components["winslow"] = winslow
            if winslow_barrier_weight > 0:
                loss = loss + float(winslow_barrier_weight) * barrier
                components["jacobian_barrier"] = barrier
    components["loss"] = loss
    return loss, components


def _is_successful_convergence(reason):
    """Return whether a standard supervised phase ended by a success rule."""
    return reason in {
        "converged_near_best_plateau",
        "metric_convergence",
        "training_loss_convergence",
    }


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
    model_options.pop("projection_softplus_enabled", None)
    model_options["trunk_projection_coefficients_enabled"] = True
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
    use_pde = bool(geometry_options.get("pde_loss_enabled", False))
    pde_weight = float(geometry_options.get("pde_loss_weight", 1.0e-4))
    pde_count = int(geometry_options.get("pde_interior_sample_count", 128))
    configured_pde_step = geometry_options.get("pde_fd_step")
    pde_steps = torch.as_tensor(
        configured_pde_step
        if configured_pde_step is not None
        else [1.0 / (size - 1) for size in shape],
        device=device,
        dtype=torch.float32,
    )
    if use_pde and (pde_weight < 0 or pde_count < 1):
        raise ValueError("PDE loss requires a nonnegative weight and positive sample count")
    if pde_steps.shape != (2,) or torch.any(pde_steps <= 0) or torch.any(pde_steps >= 0.5):
        raise ValueError("pde_fd_step must contain two values in (0, 0.5)")
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
    strategy = str(geometry_options.setdefault("training_strategy", "supervised")).lower()
    if strategy not in _HMONET_MULTISTAGE_STRATEGIES:
        raise ValueError(
            "hmonet_training.training_strategy must be one of "
            f"{sorted(_HMONET_MULTISTAGE_STRATEGIES)}"
        )
    warmup_epochs = int(geometry_options.setdefault("tfi_warmup_epochs", 5000))
    phase1_max_epochs = int(
        geometry_options.setdefault("phase1_max_epochs", training["num_epochs"])
    )
    phase2_max_epochs = int(
        geometry_options.setdefault("phase2_max_epochs", training["num_epochs"])
    )
    save_phase_best_checkpoints = bool(
        geometry_options.setdefault("save_phase_best_checkpoints", False)
    )
    reset_optimizer_on_phase_transition = bool(
        geometry_options.get("reset_optimizer_on_phase_transition", True)
    )
    physics_weight = float(geometry_options.setdefault("physics_pde_loss_weight", 1e-4))
    physics_jacobian_enabled = bool(
        geometry_options.setdefault("physics_jacobian_enabled", False)
    )
    physics_jacobian_weight = float(
        geometry_options.setdefault("physics_jacobian_loss_weight", 0.1)
    )
    physics_jacobian_minimum = float(
        geometry_options.setdefault("physics_jacobian_minimum", 1e-3)
    )
    physics_count = int(
        geometry_options.setdefault("physics_interior_sample_count", pde_count)
    )
    fixed_mse_count = int(
        geometry_options.setdefault("fixed_mse_sample_count", 400)
    )
    fixed_mse_weight = float(geometry_options.get("fixed_mse_loss_weight", 1.0))
    random_mse_weight = float(geometry_options.get("random_mse_loss_weight", 1.0))
    random_loss_count = int(geometry_options.get("random_loss_sample_count", 400))
    random_loss_derivative_backend = str(
        geometry_options.get("random_loss_derivative_backend", "finite_difference")
    ).lower()
    random_loss_sampling_mode = str(
        geometry_options.get("random_loss_sampling_mode", "uniform")
    ).lower()
    random_loss_global_count = int(
        geometry_options.get("random_loss_global_sample_count", random_loss_count)
    )
    random_loss_curvature_count = int(
        geometry_options.get("random_loss_curvature_sample_count", 0)
    )
    random_loss_boundary_count = int(
        geometry_options.get("random_loss_boundary_band_sample_count", 0)
    )
    random_loss_boundary_min_steps = float(
        geometry_options.get("random_loss_boundary_band_min_step_multiple", 1.0)
    )
    random_loss_boundary_max_steps = float(
        geometry_options.get("random_loss_boundary_band_max_step_multiple", 4.0)
    )
    random_loss_boundary_min_distance = float(
        geometry_options.get("random_loss_boundary_band_min_distance", 0.0)
    )
    random_loss_boundary_max_distance = float(
        geometry_options.get("random_loss_boundary_band_max_distance", 4.0e-4)
    )
    winslow_weight = float(geometry_options.get("physics_winslow_loss_weight", 0.0))
    winslow_barrier_weight = float(geometry_options.get("physics_winslow_jacobian_barrier_weight", 0.0))
    winslow_minimum = float(geometry_options.get("physics_winslow_jacobian_minimum", 1e-4))
    winslow_epsilon = float(geometry_options.get("physics_winslow_epsilon", 1e-6))
    scale_invariant_pde = bool(
        geometry_options.get("physics_scale_invariant_pde_enabled", False)
    )
    normalized_jacobian_weight = float(
        geometry_options.get("physics_normalized_jacobian_loss_weight", 0.0)
    )
    normalized_jacobian_minimum = float(
        geometry_options.get("physics_normalized_jacobian_minimum", 0.02)
    )
    singular_value_weight = float(
        geometry_options.get("physics_singular_value_loss_weight", 0.0)
    )
    singular_value_minimum = float(
        geometry_options.get("physics_singular_value_minimum", 0.05)
    )
    barrier_reduction = str(
        geometry_options.get("physics_barrier_reduction", "active")
    ).lower()
    if random_loss_derivative_backend not in {"finite_difference", "autograd"}:
        raise ValueError(
            "random_loss_derivative_backend must be finite_difference or autograd"
        )
    if barrier_reduction not in {"active", "all"}:
        raise ValueError("physics_barrier_reduction must be active or all")
    if random_loss_sampling_mode not in {"uniform", "mixed_boundary"}:
        raise ValueError("random_loss_sampling_mode must be uniform or mixed_boundary")
    if random_loss_sampling_mode == "mixed_boundary":
        mixed_counts = (
            random_loss_global_count,
            random_loss_curvature_count,
            random_loss_boundary_count,
        )
        if min(mixed_counts) < 0 or sum(mixed_counts) != random_loss_count:
            raise ValueError(
                "Mixed random-loss sample counts must be nonnegative and sum "
                "to random_loss_sample_count"
            )
        if not 1 <= random_loss_boundary_min_steps <= random_loss_boundary_max_steps:
            raise ValueError(
                "Finite-difference boundary-band step multiples must satisfy 1 <= min <= max"
            )
        if not (
            0 <= random_loss_boundary_min_distance
            <= random_loss_boundary_max_distance < 0.5
        ):
            raise ValueError(
                "Autograd boundary-band distances must satisfy 0 <= min <= max < 0.5"
            )
    controlled_finetune = strategy == "random_mse_jac_then_pde_fixed_mse"
    finetune_gradient_ratio = float(
        geometry_options.setdefault("finetune_pde_gradient_ratio", 0.1)
    ) if controlled_finetune else 0.0
    finetune_ramp_epochs = int(
        geometry_options.setdefault("finetune_pde_ramp_epochs", 1000)
    ) if controlled_finetune else 0
    finetune_lr_scale = float(
        geometry_options.setdefault("finetune_lr_scale", 0.1)
    ) if controlled_finetune else 1.0
    finetune_anchor_weight = float(
        geometry_options.setdefault("finetune_teacher_anchor_weight", 1.0)
    ) if controlled_finetune else 0.0
    finetune_pde_weight_cap = float(
        geometry_options.setdefault("finetune_pde_weight_cap", 1.0e-6)
    ) if controlled_finetune else 0.0
    if controlled_finetune and (
        finetune_gradient_ratio <= 0
        or finetune_ramp_epochs < 1
        or not 0 < finetune_lr_scale < 1
        or finetune_anchor_weight < 0
        or finetune_pde_weight_cap <= 0
        or fixed_mse_weight <= 0
        or not physics_jacobian_enabled
        or physics_jacobian_weight <= 0
    ):
        raise ValueError("Controlled phase-2 fine-tuning requires positive MSE, Jacobian, gradient ratio, ramp and PDE cap")
    if strategy in {
        "random_loss_combo_from_start",
        "random_mse_jac_then_mse_pde_jac",
        "random_mse_jac_then_pde_fixed_mse",
    } and (
        random_loss_count < 1
        or min(random_mse_weight, physics_weight, physics_jacobian_weight,
               winslow_weight, winslow_barrier_weight,
               normalized_jacobian_weight, singular_value_weight) < 0
        or random_mse_weight + physics_weight + physics_jacobian_weight
           + winslow_weight + winslow_barrier_weight
           + normalized_jacobian_weight + singular_value_weight <= 0
    ):
        raise ValueError("Random loss combination requires 400-style positive sampling and a nonzero loss")
    if min(
        warmup_epochs,
        phase1_max_epochs,
        phase2_max_epochs,
        physics_count,
        fixed_mse_count,
    ) < 1:
        raise ValueError("HMONet multi-stage epoch and sample counts must be positive")
    if min(
        physics_weight, physics_jacobian_weight, physics_jacobian_minimum,
        fixed_mse_weight, normalized_jacobian_weight, singular_value_weight,
    ) < 0:
        raise ValueError("HMONet multi-stage physics weights must be nonnegative")
    if normalized_jacobian_minimum <= 0 or singular_value_minimum <= 0:
        raise ValueError("Normalized Jacobian and singular-value minima must be positive")

    original_convergence = {
        key: int(training.get(key, 0))
        for key in (
            "convergence_start_epoch",
            "plateau_fluctuation_spread_start_epoch",
            "plateau_fluctuation_spread_activation_epoch",
        )
    }
    total_max_epochs = int(training["num_epochs"])
    if strategy in {
        "random_mse_jac_then_mse_pde_jac",
        "random_mse_jac_then_pde_fixed_mse",
    }:
        if phase1_max_epochs >= total_max_epochs:
            raise ValueError("The random-loss warmup must end before the training limit")
        active_phase = "random_combo_warmup"
        phase_start_epoch = 1
        phase_max_epochs = phase1_max_epochs
        # Complete the prescribed first stage before applying convergence rules.
        training["convergence_start_epoch"] = total_max_epochs + 1
    elif strategy.startswith("tfi_warmup"):
        training["num_epochs"] = total_max_epochs
        active_phase = "tfi_warmup"
        phase_start_epoch = 1
        phase_max_epochs = warmup_epochs
        # The warmup has an intentionally prescribed length.
        training["convergence_start_epoch"] = training["num_epochs"] + 1
    elif strategy in {"fixed_mse_pde_from_start", "random_loss_combo_from_start"}:
        active_phase = (
            "random_loss_combo" if strategy == "random_loss_combo_from_start"
            else "fixed_mse_physics"
        )
        phase_start_epoch = 1
        phase_max_epochs = total_max_epochs
    elif strategy.startswith("supervised_then"):
        training["num_epochs"] = total_max_epochs
        active_phase = "supervised"
        phase_start_epoch = 1
        phase_max_epochs = min(phase1_max_epochs, total_max_epochs)
    else:
        active_phase = "supervised"
        phase_start_epoch = 1
        phase_max_epochs = int(training["num_epochs"])
    training["active_phase"] = active_phase
    if strategy != "supervised":
        training["phase_transitions"] = [
            {"phase": active_phase, "start_epoch": phase_start_epoch}
        ]
    fixed_logical, fixed_indices = logical_grid_and_interior_indices(shape, device)
    fixed_logical = fixed_logical[fixed_indices]
    state = TrainingState(model, optimizer, datasets, config, mode, args, device)
    finetune_teacher = None
    finetune_pde_target_weight = None
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
                if active_phase != "supervised":
                    if active_phase in {"random_combo_warmup", "random_loss_combo"}:
                        random_loss_boundary_curvature = (
                            compute_discrete_boundary_curvature(
                                boundary, quantile=curvature_quantile
                            )
                            if random_loss_sampling_mode == "mixed_boundary"
                            and random_loss_curvature_count > 0
                            else None
                        )
                        loss, components = _random_loss_combo(
                            model=model, boundary=boundary, mesh=mesh,
                            geometries=geometries, shape=shape, device=device,
                            steps=pde_steps, sample_count=random_loss_count,
                            mse_weight=random_mse_weight,
                            pde_weight=(
                                0.0 if active_phase == "random_combo_warmup"
                                else physics_weight
                            ),
                            jacobian_weight=physics_jacobian_weight,
                            jacobian_minimum=physics_jacobian_minimum,
                            winslow_weight=winslow_weight if active_phase == "random_loss_combo" else 0.0,
                            winslow_barrier_weight=winslow_barrier_weight if active_phase == "random_loss_combo" else 0.0,
                            winslow_minimum=winslow_minimum,
                            winslow_epsilon=winslow_epsilon,
                            winslow_fd_step=jacobian_step,
                            scale_invariant_pde=scale_invariant_pde,
                            normalized_jacobian_weight=normalized_jacobian_weight,
                            normalized_jacobian_minimum=normalized_jacobian_minimum,
                            singular_value_weight=singular_value_weight,
                            singular_value_minimum=singular_value_minimum,
                            barrier_reduction=barrier_reduction,
                            derivative_backend=random_loss_derivative_backend,
                            mixed_sampling=(
                                random_loss_sampling_mode == "mixed_boundary"
                            ),
                            global_count=random_loss_global_count,
                            curvature_count=random_loss_curvature_count,
                            boundary_band_count=random_loss_boundary_count,
                            boundary_curvature=random_loss_boundary_curvature,
                            computational_boundary=computational_boundary,
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
                            boundary_min_step_multiple=random_loss_boundary_min_steps,
                            boundary_max_step_multiple=random_loss_boundary_max_steps,
                            boundary_min_distance=random_loss_boundary_min_distance,
                            boundary_max_distance=random_loss_boundary_max_distance,
                            target_interpolation=target_interpolation,
                            point_loss_weighting=point_loss_weighting,
                            idw_query_chunk_size=int(geometry_options.get("idw_query_chunk_size", 128)),
                            idw_support_chunk_size=int(geometry_options.get("idw_support_chunk_size", 2048)),
                        )
                        if not safe_optimizer_step(
                            loss, model, optimizer,
                            training["gradient_clip_norm"],
                            training["skip_nonfinite_loss"],
                        ):
                            continue
                        totals.add(len(mesh), **components)
                        progress.set_postfix(
                            _physics_progress_postfix(loss, components)
                        )
                        continue
                    phase_kind = active_phase
                    if active_phase == "pde":
                        phase_kind = (
                            "pde_jacobian" if physics_jacobian_enabled else "pde"
                        )
                    elif active_phase == "fixed_mse_physics":
                        phase_kind = (
                            "fixed_mse_physics_jacobian"
                            if physics_jacobian_enabled
                            else "fixed_mse_physics"
                        )
                    loss, components = _physics_phase_loss(
                        phase=phase_kind,
                        model=model,
                        boundary=boundary,
                        mesh=mesh,
                        geometries=geometries,
                        shape=shape,
                        device=device,
                        pde_steps=pde_steps,
                        tfi_count=random_count,
                        pde_count=physics_count,
                        fixed_mse_count=fixed_mse_count,
                        pde_weight=(0.0 if controlled_finetune else physics_weight),
                        jacobian_weight=physics_jacobian_weight,
                        jacobian_minimum=physics_jacobian_minimum,
                        fixed_logical=fixed_logical,
                        fixed_indices=fixed_indices,
                        fixed_mse_weight=fixed_mse_weight,
                        teacher=finetune_teacher,
                        teacher_anchor_weight=finetune_anchor_weight if finetune_teacher is not None else 0.0,
                    )
                    if controlled_finetune:
                        if finetune_pde_target_weight is None:
                            parameters = tuple(p for p in model.parameters() if p.requires_grad)
                            mse_norm = _gradient_l2_norm(components["fixed_mse"], parameters)
                            pde_norm = _gradient_l2_norm(components["loss_pde"], parameters)
                            if not torch.isfinite(torch.tensor([mse_norm, pde_norm])).all() or pde_norm <= 0:
                                raise FloatingPointError("Cannot calibrate stage-2 PDE gradient")
                            finetune_pde_target_weight = min(
                                finetune_pde_weight_cap,
                                finetune_gradient_ratio * fixed_mse_weight * mse_norm / pde_norm,
                            )
                            print(
                                f"Stage-2 gradient calibration: MSE={mse_norm:.3e}, "
                                f"PDE={pde_norm:.3e}, target PDE weight="
                                f"{finetune_pde_target_weight:.3e}",
                                flush=True,
                            )
                        ramp = min(1.0, (epoch - phase_start_epoch + 1) / finetune_ramp_epochs)
                        effective_pde_weight = finetune_pde_target_weight * ramp
                        loss = loss + effective_pde_weight * components["loss_pde"]
                        components["loss"] = loss
                        components["effective_pde_weight"] = loss.new_tensor(effective_pde_weight)
                    if not safe_optimizer_step(
                        loss,
                        model,
                        optimizer,
                        training["gradient_clip_norm"],
                        training["skip_nonfinite_loss"],
                    ):
                        continue
                    totals.add(len(mesh), **components)
                    progress.set_postfix(
                        _physics_progress_postfix(loss, components)
                    )
                    continue
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
                pde = torch.zeros((), device=device)
                if use_pde and pde_weight > 0:
                    pde_centers = sample_shared_random_points(
                        len(mesh),
                        pde_count,
                        2,
                        device,
                        margin=float(pde_steps.max().item()),
                    )
                    pde_queries = physics_stencil_queries(pde_centers, pde_steps)
                    pde_prediction = predict_coordinates(
                        model,
                        boundary,
                        pde_queries.flatten(-3, -2),
                        geometries,
                    ).reshape(*pde_queries.shape)
                    pde = elliptic_grid_pde_residual(pde_prediction, pde_steps).square().sum(-1).mean()
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
                    + pde_weight * pde
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
                active_losses = {
                    "loss": loss,
                    "coordinate_loss": coordinate_loss,
                }
                if use_jacobian:
                    active_losses["jacobian"] = jacobian
                if use_normal:
                    active_losses["boundary_normal"] = normal
                if use_pde and pde_weight > 0:
                    active_losses["loss_pde"] = pde
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
                if use_pde and pde_weight > 0:
                    postfix["pde_loss"] = pde.item()
                if residual_weight > 0:
                    postfix["displacement_regularization"] = regularization.item()
                progress.set_postfix(**postfix)
            phase_epoch = epoch - phase_start_epoch + 1
            scheduler_epoch = (
                phase_epoch if reset_optimizer_on_phase_transition else epoch
            )
            update_learning_rate(
                optimizer,
                scheduler_epoch,
                learning_rate["lr_warmup_epochs"],
                learning_rate["lr_update_interval"],
                learning_rate["lr_decay_gamma"],
                learning_rate["lr_stop_below"],
            )
            should_stop = state.end_epoch(epoch, totals.means(), epoch_lr)
            phase_complete = phase_epoch >= phase_max_epochs

            def begin_phase(next_phase):
                nonlocal active_phase, phase_start_epoch, phase_max_epochs, optimizer
                nonlocal finetune_teacher, finetune_pde_target_weight
                if save_phase_best_checkpoints:
                    state.save_phase_best("phase1", active_phase)
                if controlled_finetune:
                    if state.best_state is None:
                        raise RuntimeError("Stage 1 has no best checkpoint for fine-tuning")
                    model.load_state_dict(state.best_state)
                    finetune_teacher = copy.deepcopy(model).eval()
                    for parameter in finetune_teacher.parameters():
                        parameter.requires_grad_(False)
                    finetune_pde_target_weight = None
                active_phase = next_phase
                phase_start_epoch = epoch + 1
                phase_max_epochs = min(
                    phase2_max_epochs,
                    max(1, int(training["num_epochs"]) - epoch),
                )
                if reset_optimizer_on_phase_transition:
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=learning_rate["learning_rate"]
                    )
                    state.optimizer = optimizer
                elif controlled_finetune:
                    # A restored best model must not inherit Adam moments from
                    # the different weights at the final warmup epoch.
                    current_lr = optimizer.param_groups[0]["lr"]
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=current_lr * finetune_lr_scale
                    )
                    state.optimizer = optimizer
                state.begin_next_phase(
                    next_phase,
                    phase_start_epoch,
                    reset_best=False,
                )
                for key, offset in original_convergence.items():
                    training[key] = phase_start_epoch - 1 + offset
                print(
                    f"Transitioning HMONet to phase '{next_phase}' at epoch "
                    f"{phase_start_epoch}; optimizer "
                    f"{'recreated at reduced current LR' if controlled_finetune else ('reset' if reset_optimizer_on_phase_transition else 'retained')} "
                    "and convergence window reset.",
                    flush=True,
                )

            if active_phase == "random_combo_warmup" and phase_complete:
                begin_phase(
                    "fixed_mse_physics"
                    if strategy == "random_mse_jac_then_pde_fixed_mse"
                    else "random_loss_combo"
                )
                continue
            if active_phase == "tfi_warmup" and phase_complete:
                begin_phase("pde")
                continue
            if active_phase == "supervised" and strategy.startswith("supervised_then"):
                if should_stop and _is_successful_convergence(state.reason):
                    next_phase = (
                        "pde"
                        if strategy == "supervised_then_pde_from_convergence"
                        else "fixed_mse_physics"
                    )
                    begin_phase(next_phase)
                    continue
                if phase_complete:
                    state.reason = "phase1_maximum_epochs"
                    break
                if should_stop:
                    break
            if active_phase != "supervised" and (should_stop or phase_complete):
                if phase_complete and not should_stop:
                    state.reason = "phase_maximum_epochs"
                break
            if active_phase == "supervised" and should_stop:
                break
        if controlled_finetune and finetune_teacher is not None:
            current_weights = copy.deepcopy(model.state_dict())
            phase1_quality = _finetune_mesh_quality(
                finetune_teacher, datasets, device
            )
            model.load_state_dict(state.best_state)
            candidate_quality = _finetune_mesh_quality(model, datasets, device)
            model.load_state_dict(current_weights)
            accepted = (
                candidate_quality[0] <= phase1_quality[0]
                and candidate_quality[1] >= phase1_quality[1] - 0.02
            )
            print(
                "Stage-2 quality gate (101x101, three geometries): "
                f"phase1={phase1_quality}, candidate={candidate_quality}, "
                f"accepted={accepted}", flush=True,
            )
            if not accepted:
                state.best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in finetune_teacher.state_dict().items()
                }
                record = state.phase_best_records["phase1"]
                state.best = float(record["metric_value"])
                state.best_epoch = int(record["epoch"])
                print("Restored phase-1 best model as the final best checkpoint.", flush=True)
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
        help=(
            "Model name, alias (both/pi/all), or a comma-separated model list, "
            "for example meshonet,pimeshonet"
        ),
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
    parser.add_argument(
        "--coordinate-lift-mode",
        choices=["legacy", "power_fourier"],
        help="Coordinate encoding: preserve the existing layout or use power-Fourier features.",
    )
    parser.add_argument("--coordinate-lift-max-power", type=int)
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
    parser.add_argument("--hmonet-mapping-mode", choices=["current", "coons_residual", "corner_corrected"])
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
        "--hmonet-training-strategy",
        choices=sorted(_HMONET_MULTISTAGE_STRATEGIES),
    )
    parser.add_argument("--hmonet-tfi-warmup-epochs", type=int)
    parser.add_argument("--hmonet-phase1-max-epochs", type=int)
    parser.add_argument("--hmonet-phase2-max-epochs", type=int)
    parser.add_argument(
        "--hmonet-save-phase-best-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Save the first-stage best model separately while selecting the "
            "final best model over all epochs from both training stages."
        ),
    )
    parser.add_argument(
        "--hmonet-reset-optimizer-on-phase-transition",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Recreate Adam at a phase transition. Disable to retain optimizer "
            "moments, the current learning rate, and the global LR schedule."
        ),
    )
    parser.add_argument("--hmonet-fixed-mse-sample-count", type=int)
    parser.add_argument("--hmonet-fixed-mse-loss-weight", type=float)
    parser.add_argument("--hmonet-finetune-pde-gradient-ratio", type=float)
    parser.add_argument("--hmonet-finetune-pde-ramp-epochs", type=int)
    parser.add_argument("--hmonet-finetune-lr-scale", type=float)
    parser.add_argument("--hmonet-finetune-teacher-anchor-weight", type=float)
    parser.add_argument("--hmonet-finetune-pde-weight-cap", type=float)
    parser.add_argument("--hmonet-physics-pde-loss-weight", type=float)
    parser.add_argument(
        "--hmonet-physics-jacobian-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--hmonet-physics-jacobian-loss-weight", type=float)
    parser.add_argument("--hmonet-physics-jacobian-minimum", type=float)
    parser.add_argument("--hmonet-physics-winslow-loss-weight", type=float)
    parser.add_argument("--hmonet-physics-winslow-jacobian-barrier-weight", type=float)
    parser.add_argument("--hmonet-physics-winslow-jacobian-minimum", type=float)
    parser.add_argument("--hmonet-physics-winslow-epsilon", type=float)
    parser.add_argument(
        "--hmonet-physics-scale-invariant-pde-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--hmonet-physics-normalized-jacobian-loss-weight", type=float)
    parser.add_argument("--hmonet-physics-normalized-jacobian-minimum", type=float)
    parser.add_argument("--hmonet-physics-singular-value-loss-weight", type=float)
    parser.add_argument("--hmonet-physics-singular-value-minimum", type=float)
    parser.add_argument(
        "--hmonet-physics-barrier-reduction",
        choices=["active", "all"],
    )
    parser.add_argument("--hmonet-physics-interior-sample-count", type=int)
    parser.add_argument("--hmonet-random-mse-loss-weight", type=float)
    parser.add_argument("--hmonet-random-loss-sample-count", type=int)
    parser.add_argument(
        "--hmonet-random-loss-derivative-backend",
        choices=["finite_difference", "autograd"],
    )
    parser.add_argument(
        "--hmonet-random-loss-sampling-mode",
        choices=["uniform", "mixed_boundary"],
    )
    parser.add_argument("--hmonet-random-loss-global-sample-count", type=int)
    parser.add_argument("--hmonet-random-loss-curvature-sample-count", type=int)
    parser.add_argument("--hmonet-random-loss-boundary-band-sample-count", type=int)
    parser.add_argument("--hmonet-random-loss-boundary-band-min-step-multiple", type=float)
    parser.add_argument("--hmonet-random-loss-boundary-band-max-step-multiple", type=float)
    parser.add_argument("--hmonet-random-loss-boundary-band-min-distance", type=float)
    parser.add_argument("--hmonet-random-loss-boundary-band-max-distance", type=float)
    parser.add_argument(
        "--pde-loss-enabled", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--pde-loss-weight", type=float)
    parser.add_argument("--pde-interior-sample-count", type=int)
    parser.add_argument("--pde-fd-step", type=float, nargs=2)
    parser.add_argument(
        "--target-interpolation",
        choices=["bilinear", "curved_boundary_cell"],
    )
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
        "coordinate_lift_max_power",
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
    if args.coordinate_lift_mode is not None:
        common_model["coordinate_lift_mode"] = args.coordinate_lift_mode
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
        "boundary_self_attention_enabled",
        "projection_mlp_hidden_dim",
        "weight_head_mlp_enabled",
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
        "pde_loss_weight",
        "pde_interior_sample_count",
        "pde_fd_step",
        "target_interpolation",
        "idw_query_chunk_size",
        "idw_support_chunk_size",
        "point_loss_weighting",
        "boundary_normal_loss_weight",
        "boundary_normal_sample_count",
        "boundary_normal_offset",
    ):
        if getattr(args, source) is not None:
            htraining[source] = getattr(args, source)
    hmonet_strategy_overrides = {
        "hmonet_training_strategy": "training_strategy",
        "hmonet_tfi_warmup_epochs": "tfi_warmup_epochs",
        "hmonet_phase1_max_epochs": "phase1_max_epochs",
        "hmonet_phase2_max_epochs": "phase2_max_epochs",
        "hmonet_save_phase_best_checkpoints": "save_phase_best_checkpoints",
        "hmonet_reset_optimizer_on_phase_transition": "reset_optimizer_on_phase_transition",
        "hmonet_fixed_mse_sample_count": "fixed_mse_sample_count",
        "hmonet_fixed_mse_loss_weight": "fixed_mse_loss_weight",
        "hmonet_finetune_pde_gradient_ratio": "finetune_pde_gradient_ratio",
        "hmonet_finetune_pde_ramp_epochs": "finetune_pde_ramp_epochs",
        "hmonet_finetune_lr_scale": "finetune_lr_scale",
        "hmonet_finetune_teacher_anchor_weight": "finetune_teacher_anchor_weight",
        "hmonet_finetune_pde_weight_cap": "finetune_pde_weight_cap",
        "hmonet_physics_pde_loss_weight": "physics_pde_loss_weight",
        "hmonet_physics_jacobian_loss_weight": "physics_jacobian_loss_weight",
        "hmonet_physics_jacobian_minimum": "physics_jacobian_minimum",
        "hmonet_physics_winslow_loss_weight": "physics_winslow_loss_weight",
        "hmonet_physics_winslow_jacobian_barrier_weight": "physics_winslow_jacobian_barrier_weight",
        "hmonet_physics_winslow_jacobian_minimum": "physics_winslow_jacobian_minimum",
        "hmonet_physics_winslow_epsilon": "physics_winslow_epsilon",
        "hmonet_physics_scale_invariant_pde_enabled": "physics_scale_invariant_pde_enabled",
        "hmonet_physics_normalized_jacobian_loss_weight": "physics_normalized_jacobian_loss_weight",
        "hmonet_physics_normalized_jacobian_minimum": "physics_normalized_jacobian_minimum",
        "hmonet_physics_singular_value_loss_weight": "physics_singular_value_loss_weight",
        "hmonet_physics_singular_value_minimum": "physics_singular_value_minimum",
        "hmonet_physics_barrier_reduction": "physics_barrier_reduction",
        "hmonet_physics_interior_sample_count": "physics_interior_sample_count",
        "hmonet_random_mse_loss_weight": "random_mse_loss_weight",
        "hmonet_random_loss_sample_count": "random_loss_sample_count",
        "hmonet_random_loss_derivative_backend": "random_loss_derivative_backend",
        "hmonet_random_loss_sampling_mode": "random_loss_sampling_mode",
        "hmonet_random_loss_global_sample_count": "random_loss_global_sample_count",
        "hmonet_random_loss_curvature_sample_count": "random_loss_curvature_sample_count",
        "hmonet_random_loss_boundary_band_sample_count": "random_loss_boundary_band_sample_count",
        "hmonet_random_loss_boundary_band_min_step_multiple": "random_loss_boundary_band_min_step_multiple",
        "hmonet_random_loss_boundary_band_max_step_multiple": "random_loss_boundary_band_max_step_multiple",
        "hmonet_random_loss_boundary_band_min_distance": "random_loss_boundary_band_min_distance",
        "hmonet_random_loss_boundary_band_max_distance": "random_loss_boundary_band_max_distance",
        "hmonet_physics_jacobian_enabled": "physics_jacobian_enabled",
    }
    for source, target in hmonet_strategy_overrides.items():
        value = getattr(args, source)
        if value is not None:
            htraining[target] = value
    for source in (
        "jacobian_loss_enabled",
        "pde_loss_enabled",
        "boundary_normal_loss_enabled",
    ):
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
    names = parse_model_selection(choice)
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
