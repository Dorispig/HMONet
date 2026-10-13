"""Early validation for training and evaluation YAML configurations."""

from __future__ import annotations

import numpy as np

import warnings


def remove_deprecated_projection_config(config):
    """Accept saved legacy options while keeping new model settings unambiguous."""
    hmonet = config.get("model", {}).get("hmonet", {})
    if "boundary_curve_interpolation" in hmonet:
        hmonet.pop("boundary_curve_interpolation")
        warnings.warn(
            "boundary_curve_interpolation is deprecated and has been removed. "
            "HMONet evaluates supplied parameter boundaries directly; otherwise "
            "it interpolates the sampled boundary linearly.",
            UserWarning,
            stacklevel=2,
        )


def validate_train_config(config: dict) -> None:
    for section in ("datasets", "grid", "model", "training"):
        if section not in config:
            raise ValueError(f"train config is missing section: {section}")
    remove_deprecated_projection_config(config)
    if config["model"].get("name") not in {
        "hmonet",
        "meshonet",
        "pimeshonet",
        "pihmonet",
        "attentionpihmonet",
        "both",
        "pi",
        "all",
    }:
        raise ValueError(
            "model.name must be hmonet, meshonet, pimeshonet, pihmonet, "
            "attentionpihmonet, both, pi or all"
        )
    if not isinstance(config["model"].get("common", {}), dict):
        raise ValueError("model.common must be a mapping")
    hmonet = config["model"].get("hmonet", {})
    mapping_mode = (
        str(hmonet.get("hmonet_mapping_mode", "current"))
        .strip()
        .lower()
        .replace("-", "_")
    )
    if mapping_mode not in {"current", "coons_residual"}:
        raise ValueError(
            "model.hmonet.hmonet_mapping_mode must be current or coons_residual"
        )
    if float(hmonet.get("residual_regularization_weight", 0.0)) < 0:
        raise ValueError(
            "model.hmonet.residual_regularization_weight must be non-negative"
        )
    for option in (
        "trunk_projection_coefficients_enabled",
        "projection_softplus_enabled",
        "boundary_self_attention_enabled",
    ):
        if not isinstance(hmonet.get(option, False), bool):
            raise ValueError(f"model.hmonet.{option} must be true or false")
    if int(hmonet.get("projection_mlp_hidden_dim", 64)) <= 0:
        raise ValueError("model.hmonet.projection_mlp_hidden_dim must be positive")
    attention_dim = int(hmonet.get("attention_dim", 64))
    attention_heads = int(hmonet.get("attention_heads", 4))
    if attention_dim <= 0 or attention_heads <= 0 or attention_dim % attention_heads:
        raise ValueError(
            "model.hmonet.attention_dim must be positive and divisible by attention_heads"
        )
    if int(hmonet.get("attention_layers", 2)) <= 0:
        raise ValueError("model.hmonet.attention_layers must be positive")
    pi_mapping = (
        config["model"].get("pihmonet", {}).get("hmonet_mapping_mode", "current")
    )
    if pi_mapping not in {"current", "coons_residual"}:
        raise ValueError(
            "model.pihmonet.hmonet_mapping_mode must be current or coons_residual"
        )
    attention_mapping = config["model"].get("attentionpihmonet", {}).get(
        "hmonet_mapping_mode", "current"
    )
    if attention_mapping not in {"current", "coons_residual"}:
        raise ValueError(
            "model.attentionpihmonet.hmonet_mapping_mode must be current or "
            "coons_residual"
        )
    if min(int(config["grid"]["nx"]), int(config["grid"]["ny"])) < 2:
        raise ValueError("grid.nx and grid.ny must be at least 2")
    metric = config["training"].get("best_model_metric", "dataset_mse")
    if metric not in {"dataset_mse", "training_loss"}:
        raise ValueError(
            "training.best_model_metric must be dataset_mse or training_loss"
        )
    if not isinstance(config["training"].get("tensorboard_enabled", True), bool):
        raise ValueError("training.tensorboard_enabled must be true or false")
    legacy_training = config["training"]
    learning_rate_defaults = {
        "learning_rate": 0.0004,
        "lr_warmup_epochs": 0,
        "lr_update_interval": 500,
        "lr_decay_gamma": 0.9,
        "lr_stop_below": 1e-5,
    }
    for section_name in (
        "hmonet_training",
        "meshonet_training",
        "pimeshonet_training",
        "pihmonet_training",
        "attentionpihmonet_training",
    ):
        section = config.get(section_name, {})
        values = {
            key: section.get(key, legacy_training.get(key, default))
            for key, default in learning_rate_defaults.items()
        }
        if float(values["learning_rate"]) <= 0:
            raise ValueError(f"{section_name}.learning_rate must be positive")
        if int(values["lr_warmup_epochs"]) < 0:
            raise ValueError(f"{section_name}.lr_warmup_epochs must be nonnegative")
        if int(values["lr_update_interval"]) < 1:
            raise ValueError(f"{section_name}.lr_update_interval must be positive")
        if not 0 < float(values["lr_decay_gamma"]) <= 1:
            raise ValueError(f"{section_name}.lr_decay_gamma must be in (0, 1]")
        if float(values["lr_stop_below"]) <= 0:
            raise ValueError(f"{section_name}.lr_stop_below must be positive")
    if int(config["training"].get("mse_consecutive_epochs", 30)) < 1:
        raise ValueError("training.mse_consecutive_epochs must be positive")
    training = config["training"]
    if training.get("plateau_metric", "mse") not in {"mse", "train_loss"}:
        raise ValueError("training.plateau_metric must be mse or train_loss")
    if float(training.get("training_loss_tolerance", 1e-6)) < 0:
        raise ValueError("training.training_loss_tolerance must be non-negative")
    if int(training.get("training_loss_consecutive_epochs", 30)) < 1:
        raise ValueError("training.training_loss_consecutive_epochs must be positive")
    if int(training.get("convergence_start_epoch", 3000)) < 1:
        raise ValueError("training.convergence_start_epoch must be positive")
    if int(training.get("plateau_window", 1000)) < 2:
        raise ValueError("training.plateau_window must be at least 2")
    if int(training.get("success_plateau_patience", 1000)) < 1:
        raise ValueError("training.success_plateau_patience must be positive")
    if int(training.get("plateau_log_interval", 100)) < 1:
        raise ValueError("training.plateau_log_interval must be positive")
    quantiles = training.get("fluctuation_quantiles", [0.2, 0.8])
    if not isinstance(quantiles, (list, tuple)) or len(quantiles) != 2:
        raise ValueError("training.fluctuation_quantiles must have two values")
    lower_quantile, upper_quantile = map(float, quantiles)
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError(
            "training fluctuation quantiles require " "0 <= lower < upper <= 1"
        )
    for name in (
        "fluctuation_relative_tolerance",
        "fluctuation_absolute_minimum",
        "fluctuation_absolute_maximum",
        "best_improvement_relative_tolerance",
        "gradient_clip_norm",
    ):
        if float(training.get(name, 0.0)) < 0:
            raise ValueError(f"training.{name} must be non-negative")
    absolute_minimum = float(training.get("fluctuation_absolute_minimum", 1e-6))
    absolute_maximum = float(training.get("fluctuation_absolute_maximum", 1e-5))
    if absolute_minimum > absolute_maximum:
        raise ValueError(
            "training.fluctuation_absolute_minimum must not exceed "
            "training.fluctuation_absolute_maximum"
        )
    for name in ("check_mse_convergence", "check_training_loss_convergence"):
        if name in training and not isinstance(training[name], bool):
            raise ValueError(f"training.{name} must be true or false")
    for section_name in (
        "hmonet_training",
        "meshonet_training",
        "pimeshonet_training",
        "pihmonet_training",
        "attentionpihmonet_training",
    ):
        section = config.get(section_name, {})
        for name in (
            "check_plateau_stability",
            "check_plateau_best_improvement",
        ):
            if name in section and not isinstance(section[name], bool):
                raise ValueError(f"{section_name}.{name} must be true or false")
    hmonet_training = config.get("hmonet_training", {})
    if hmonet_training.get("jacobian_method", "finite_difference") not in {
        "finite_difference",
        "autograd",
    }:
        raise ValueError(
            "hmonet_training.jacobian_method must be finite_difference or autograd"
        )
    if not 0 < float(hmonet_training.get("jacobian_fd_step", 1e-3)) <= 1 / 3:
        raise ValueError("hmonet_training.jacobian_fd_step must be in (0, 1/3]")
    if hmonet_training.get("target_interpolation", "curved_boundary_cell") not in {
        "bilinear",
        "curved_boundary_cell",
    }:
        raise ValueError(
            "hmonet_training.target_interpolation must be bilinear or "
            "curved_boundary_cell"
        )
    if hmonet_training.get("point_loss_weighting", "position") not in {
        "position",
        "uniform",
    }:
        raise ValueError(
            "hmonet_training.point_loss_weighting must be position or uniform"
        )
    idw_neighbors = int(hmonet_training.get("idw_neighbors", -1))
    if idw_neighbors == 0 or idw_neighbors < -1:
        raise ValueError("hmonet_training.idw_neighbors must be -1 or positive")
    for name, default in (
        ("idw_query_chunk_size", 512),
        ("idw_support_chunk_size", 2048),
    ):
        if int(hmonet_training.get(name, default)) < 1:
            raise ValueError(f"hmonet_training.{name} must be positive")
    for name in ("jacobian_loss_enabled", "boundary_normal_loss_enabled"):
        if name in hmonet_training and not isinstance(hmonet_training[name], bool):
            raise ValueError(f"hmonet_training.{name} must be true or false")
    for name in (
        "jacobian_loss_weight",
        "jacobian_minimum",
        "boundary_normal_loss_weight",
    ):
        if float(hmonet_training.get(name, 0.0)) < 0:
            raise ValueError(f"hmonet_training.{name} must be non-negative")
    if int(hmonet_training.get("boundary_normal_sample_count", 64)) < 1:
        raise ValueError(
            "hmonet_training.boundary_normal_sample_count must be positive"
        )
    boundary_normal_offset = float(hmonet_training.get("boundary_normal_offset", 0.01))
    if not 0 < boundary_normal_offset < 0.5:
        raise ValueError("hmonet_training.boundary_normal_offset must be in (0, 0.5)")
    sampling = config.get("hmonet_sampling", {})
    if str(sampling.get("mode", "uniform")).lower() not in {"uniform", "adaptive"}:
        raise ValueError("hmonet_sampling.mode must be uniform or adaptive")
    sample_counts = {
        name: int(sampling.get(name, default))
        for name, default in (
            ("fixed_grid_count", 200),
            ("global_random_count", 100),
            ("curvature_adaptive_count", 100),
        )
    }
    for name, count in sample_counts.items():
        if count < 0:
            raise ValueError(f"hmonet_sampling.{name} must be non-negative")
    if not any(sample_counts.values()):
        raise ValueError("at least one HMONet sampling count must be positive")
    if int(sampling.get("candidate_multiplier", 10)) < 1:
        raise ValueError("hmonet_sampling.candidate_multiplier must be positive")
    sigma_min = float(sampling.get("distance_sigma_min", 0.02))
    sigma_max = float(sampling.get("distance_sigma_max", 0.10))
    if not 0 < sigma_min <= sigma_max:
        raise ValueError(
            "hmonet_sampling requires 0 < distance_sigma_min <= distance_sigma_max"
        )
    if not 0 < float(sampling.get("curvature_quantile", 0.95)) <= 1:
        raise ValueError("hmonet_sampling.curvature_quantile must be in (0, 1]")
    if float(sampling.get("curvature_power", 1.0)) < 0:
        raise ValueError("hmonet_sampling.curvature_power must be non-negative")
    if float(sampling.get("uniform_probability", 0.05)) < 0:
        raise ValueError("hmonet_sampling.uniform_probability must be non-negative")

    pihmonet_training = config.get("pihmonet_training", {})
    if "jacobian_loss_enabled" in pihmonet_training and not isinstance(
        pihmonet_training["jacobian_loss_enabled"], bool
    ):
        raise ValueError("pihmonet_training.jacobian_loss_enabled must be true or false")
    for name in ("jacobian_loss_weight", "jacobian_minimum"):
        value = float(pihmonet_training.get(name, 0.01 if name.endswith("weight") else 1e-4))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"pihmonet_training.{name} must be finite and non-negative")

    attention = config.get("attentionpihmonet_training", {})
    alpha_initial = float(
        config.get("model", {}).get("attentionpihmonet", {}).get(
            "boundary_alpha_initial_value", 1.01
        )
    )
    if not np.isfinite(alpha_initial):
        raise ValueError(
            "model.attentionpihmonet.boundary_alpha_initial_value must be finite"
        )
    if attention.get("pde_backend", "finite_difference") not in {
        "autograd",
        "finite_difference",
    }:
        raise ValueError(
            "attentionpihmonet_training.pde_backend must be autograd or "
            "finite_difference"
        )
    for name in (
        "normalized_pde_loss_weight",
        "jacobian_loss_weight",
        "jacobian_minimum",
        "smoothness_loss_weight",
        "boundary_pde_loss_weight",
        "boundary_jacobian_loss_weight",
        "anchor_loss_weight",
        "boundary_loss_weight",
        "alpha_penalty_weight",
    ):
        value = float(attention.get(name, 0.0))
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"attentionpihmonet_training.{name} must be finite and non-negative"
            )
    if int(attention.get("interior_sample_count", 1)) < 1:
        raise ValueError(
            "attentionpihmonet_training.interior_sample_count must be positive"
        )
    if float(attention.get("interior_margin", 0.0)) != 0.0:
        raise ValueError(
            "attentionpihmonet_training.interior_margin must be 0 so autograd "
            "samples the complete open computational domain"
        )
    if int(attention.get("boundary_physics_sample_count", 4)) < 4:
        raise ValueError(
            "attentionpihmonet_training.boundary_physics_sample_count is too small"
        )
    if int(attention.get("boundary_loss_sample_count", 4)) < 4:
        raise ValueError(
            "attentionpihmonet_training.boundary_loss_sample_count is too small"
        )
    offset = float(attention.get("boundary_physics_offset", 0.0))
    if not 0 <= offset < 0.5:
        raise ValueError(
            "attentionpihmonet_training.boundary_physics_offset must be in [0, 0.5)"
        )
    tangential_margin = float(
        attention.get("boundary_physics_tangential_margin", 0.05)
    )
    if not 0 <= tangential_margin < 0.5:
        raise ValueError(
            "attentionpihmonet_training.boundary_physics_tangential_margin must "
            "be in [0, 0.5)"
        )


def validate_evaluate_config(config: dict) -> None:
    if "cases" not in config or "evaluation" not in config:
        raise ValueError("evaluate config requires cases and evaluation sections")
    evaluation = config["evaluation"]
    methods = evaluation.get("methods", [])
    if not isinstance(methods, list) or set(methods) - {
        "hmonet",
        "meshonet",
        "pimeshonet",
        "pihmonet",
        "attentionpihmonet",
        "tfi",
        "pde",
    }:
        raise ValueError(
            "evaluation.methods must contain hmonet, meshonet, tfi and/or pde"
        )
    for key in (
        "hmonet_boundary_projection",
        "meshonet_boundary_projection",
        "pimeshonet_boundary_projection",
        "pihmonet_boundary_projection",
        "attentionpihmonet_boundary_projection",
    ):
        if evaluation.get(key, "direct") not in {"none", "direct", "inverse_distance"}:
            raise ValueError(
                f"evaluation.{key} must be none, direct or inverse_distance"
            )
    if evaluation.get("model", "hmonet") not in {
        "hmonet",
        "meshonet",
        "pimeshonet",
        "pihmonet",
        "attentionpihmonet",
        "both",
        "pi",
        "all",
    }:
        raise ValueError(
            "evaluation.model must be hmonet, meshonet, pimeshonet, pihmonet, both or all"
        )
    if evaluation.get("hmonet_sampling_mode", "adaptive") not in {
        "adaptive",
        "non_adaptive",
        "adp",
        "non_adp",
        "both",
    }:
        raise ValueError(
            "evaluation.hmonet_sampling_mode must be "
            "adaptive/adp, non_adaptive/non_adp, or both"
        )
    if evaluation.get("hmonet_mapping_mode", "current") not in {
        "current",
        "coons_residual",
        "both",
    }:
        raise ValueError(
            "evaluation.hmonet_mapping_mode must be current, " "coons_residual, or both"
        )
    if evaluation.get("attentionpihmonet_mapping_mode", "current") not in {
        "current",
        "coons_residual",
        "both",
    }:
        raise ValueError(
            "evaluation.attentionpihmonet_mapping_mode must be current, "
            "coons_residual, or both"
        )
    if not isinstance(evaluation.get("include_tfi", True), bool):
        raise ValueError("evaluation.include_tfi must be true or false")
    for key, default in (
        ("save_plots", True),
        ("save_png", True),
        ("save_pdf", False),
        ("save_distribution_plots", True),
    ):
        if not isinstance(evaluation.get(key, default), bool):
            raise ValueError(f"evaluation.{key} must be true or false")
    if float(evaluation.get("inverse_distance_power", 2.0)) <= 0:
        raise ValueError("evaluation.inverse_distance_power must be positive")
    pde_iterations = evaluation.get("hmonet_pde_iterations", 0)
    if (
        isinstance(pde_iterations, bool)
        or not isinstance(pde_iterations, int)
        or pde_iterations < 0
    ):
        raise ValueError("evaluation.hmonet_pde_iterations must be an integer >= 0")
    if not isinstance(evaluation.get("reuse_generated_results", True), bool):
        raise ValueError("evaluation.reuse_generated_results must be true or false")
    if not isinstance(evaluation.get("reuse_analysis_results", True), bool):
        raise ValueError("evaluation.reuse_analysis_results must be true or false")
    warmup_runs = evaluation.get("inference_warmup_runs", 1)
    timing_repeats = evaluation.get("inference_timing_repeats", 3)
    if (
        isinstance(warmup_runs, bool)
        or not isinstance(warmup_runs, int)
        or warmup_runs < 0
    ):
        raise ValueError("evaluation.inference_warmup_runs must be an integer >= 0")
    if (
        isinstance(timing_repeats, bool)
        or not isinstance(timing_repeats, int)
        or timing_repeats < 1
    ):
        raise ValueError("evaluation.inference_timing_repeats must be an integer >= 1")
    for name, case in config["cases"].items():
        for field in (
            "boundary_type",
            "split",
            "parameter",
            "values",
            "train_grid",
            "grid_shapes",
        ):
            if field not in case:
                raise ValueError(f"case {name} is missing field: {field}")
        if len(case["train_grid"]) != 2 or min(map(int, case["train_grid"])) < 2:
            raise ValueError(f"case {name}.train_grid must contain two values >= 2")
        for grid in case["grid_shapes"]:
            if isinstance(grid, dict):
                if case["boundary_type"] == "double_semiellipse_hole":
                    required = {"n_half_top", "n_right", "ny"}
                else:
                    required = {"nx", "ny"}
                missing = required.difference(grid)
                if missing:
                    raise ValueError(
                        f"case {name} grid shape is missing: {', '.join(sorted(missing))}"
                    )
                if min(int(grid[field]) for field in required) < 2:
                    raise ValueError(f"case {name} grid shape values must be >= 2")
            elif len(grid) != 2 or min(map(int, grid)) < 2:
                raise ValueError(
                    f"case {name} grid shapes must contain two values >= 2"
                )


