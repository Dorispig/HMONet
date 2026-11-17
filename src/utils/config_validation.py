"""Early validation for training and evaluation YAML configurations."""

from __future__ import annotations

import numpy as np

import warnings

from src.models.mapping_modes import normalize_mapping_mode


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
        "qghmonet",
        "meshonet",
        "pimeshonet",
        "both",
        "pi",
        "all",
    }:
        raise ValueError(
            "model.name must be hmonet, qghmonet, meshonet, pimeshonet, "
            "both, pi or all"
        )
    if not isinstance(config["model"].get("common", {}), dict):
        raise ValueError("model.common must be a mapping")
    common_model = config["model"].get("common", {})
    lift_mode = (
        str(common_model.get("coordinate_lift_mode", "legacy"))
        .strip()
        .lower()
        .replace("-", "_")
    )
    if lift_mode not in {"legacy", "paper", "power_fourier"}:
        raise ValueError(
            "model.common.coordinate_lift_mode must be legacy, paper or power_fourier"
        )
    if int(common_model.get("coordinate_lift_max_power", 1)) < 1:
        raise ValueError(
            "model.common.coordinate_lift_max_power must be at least 1"
        )
    for model_name in ("hmonet", "qghmonet", "meshonet", "pimeshonet"):
        options = dict(common_model)
        options.update(config["model"].get(model_name, {}))
        mode = str(options.get("coordinate_lift_mode", "legacy")).lower()
        if mode not in {"legacy", "paper", "power_fourier"}:
            raise ValueError(
                f"model.{model_name}.coordinate_lift_mode must be legacy, "
                "paper or power_fourier"
            )
        if int(options.get("coordinate_lift_q", 6)) < 1:
            raise ValueError(
                f"model.{model_name}.coordinate_lift_q must be at least 1"
            )
    hmonet = config["model"].get("hmonet", {})
    normalize_mapping_mode(
        hmonet.get("hmonet_mapping_mode", "rational_affine")
    )
    if float(hmonet.get("residual_regularization_weight", 0.0)) < 0:
        raise ValueError(
            "model.hmonet.residual_regularization_weight must be non-negative"
        )
    for option in (
        "weight_head_mlp_enabled",
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
        "qghmonet_training",
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
    if not 0 < float(hmonet_training.get("jacobian_fd_step", 0.01)) <= 1 / 3:
        raise ValueError("hmonet_training.jacobian_fd_step must be in (0, 1/3]")
    if hmonet_training.get("target_interpolation", "bilinear") not in {
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
    for name, default in (
        ("idw_query_chunk_size", 512),
        ("idw_support_chunk_size", 2048),
    ):
        if int(hmonet_training.get(name, default)) < 1:
            raise ValueError(f"hmonet_training.{name} must be positive")
    for name in (
        "jacobian_loss_enabled",
        "boundary_normal_loss_enabled",
        "physics_scale_invariant_pde_enabled",
        "random_loss_fd_stencil_inside",
    ):
        if name in hmonet_training and not isinstance(hmonet_training[name], bool):
            raise ValueError(f"hmonet_training.{name} must be true or false")
    for name in (
        "jacobian_loss_weight",
        "jacobian_minimum",
        "boundary_normal_loss_weight",
        "physics_normalized_jacobian_loss_weight",
        "physics_singular_value_loss_weight",
    ):
        if float(hmonet_training.get(name, 0.0)) < 0:
            raise ValueError(f"hmonet_training.{name} must be non-negative")
    for name, default in (
        ("physics_normalized_jacobian_minimum", 0.02),
        ("physics_singular_value_minimum", 0.05),
    ):
        if float(hmonet_training.get(name, default)) <= 0:
            raise ValueError(f"hmonet_training.{name} must be positive")
    random_backend = str(
        hmonet_training.get("random_loss_derivative_backend", "finite_difference")
    ).lower()
    if random_backend not in {"finite_difference", "autograd"}:
        raise ValueError(
            "hmonet_training.random_loss_derivative_backend must be "
            "finite_difference or autograd"
        )
    if "pde_fd_step" in hmonet_training:
        dimension = 3 if "nz" in config.get("grid", {}) else 2
        pde_steps = np.asarray(hmonet_training["pde_fd_step"], dtype=float)
        if pde_steps.shape != (dimension,) or not np.all(
            (pde_steps > 0) & (pde_steps < 0.5)
        ):
            raise ValueError(
                f"hmonet_training.pde_fd_step must contain {dimension} values "
                "in (0, 0.5)"
            )
    random_fd_scheme = str(
        hmonet_training.get("random_loss_fd_boundary_scheme", "central")
    ).lower()
    if random_fd_scheme not in {"central", "one_sided"}:
        raise ValueError(
            "hmonet_training.random_loss_fd_boundary_scheme must be central or one_sided"
        )
    if random_backend != "finite_difference" and random_fd_scheme != "central":
        raise ValueError(
            "one_sided random-loss FD boundary scheme requires finite_difference backend"
        )
    random_sampling = str(
        hmonet_training.get("random_loss_sampling_mode", "uniform")
    ).lower()
    if random_sampling not in {"uniform", "mixed_boundary"}:
        raise ValueError(
            "hmonet_training.random_loss_sampling_mode must be uniform or mixed_boundary"
        )
    if random_sampling == "mixed_boundary":
        total = int(hmonet_training.get("random_loss_sample_count", 400))
        counts = tuple(
            int(hmonet_training.get(name, default))
            for name, default in (
                ("random_loss_global_sample_count", 200),
                ("random_loss_curvature_sample_count", 100),
                ("random_loss_boundary_band_sample_count", 100),
            )
        )
        if min(counts) < 0 or sum(counts) != total:
            raise ValueError(
                "mixed random-loss sample counts must be nonnegative and sum "
                "to random_loss_sample_count"
            )
        min_steps = float(
            hmonet_training.get("random_loss_boundary_band_min_step_multiple", 1.0)
        )
        max_steps = float(
            hmonet_training.get("random_loss_boundary_band_max_step_multiple", 4.0)
        )
        if not 1 <= min_steps <= max_steps:
            raise ValueError(
                "random-loss boundary step multiples must satisfy 1 <= min <= max"
            )
        min_distance = float(
            hmonet_training.get("random_loss_boundary_band_min_distance", 0.0)
        )
        max_distance = float(
            hmonet_training.get("random_loss_boundary_band_max_distance", 4e-4)
        )
        if not 0 <= min_distance <= max_distance < 0.5:
            raise ValueError(
                "random-loss boundary distances must satisfy 0 <= min <= max < 0.5"
            )
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



def _validate_grid_counts(name: str, value, dimension: int) -> None:
    """Validate a fixed-size evaluation grid without assuming a dimension."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must contain {dimension} integer values")
    if len(value) != dimension or min(map(int, value)) < 2:
        raise ValueError(f"{name} must contain {dimension} values >= 2")


def _infer_evaluation_dimension(cases: dict) -> int:
    """Infer the 2-D/3-D case convention for backwards-compatible validation."""
    for case in cases.values():
        if "grid_shape" in case:
            return 3
        if "train_grid" in case or "grid_shapes" in case:
            return 2
    raise ValueError("evaluate config must define at least one case grid")


def validate_evaluate_config(config: dict, *, dimension: int | None = None) -> None:
    """Validate either the 2-D or the 3-D evaluation YAML schema.

    Two-dimensional cases use ``boundary_type``, ``train_grid`` and a list of
    ``grid_shapes``.  Three-dimensional cases use ``boundary`` and one
    ``grid_shape``.  Keeping both schemas explicit prevents a valid 3-D YAML
    from being rejected by legacy 2-D-only checks.
    """
    if "cases" not in config or "evaluation" not in config:
        raise ValueError("evaluate config requires cases and evaluation sections")
    if not isinstance(config["cases"], dict) or not config["cases"]:
        raise ValueError("evaluate config requires a non-empty cases mapping")
    if dimension is None:
        dimension = _infer_evaluation_dimension(config["cases"])
    if dimension not in (2, 3):
        raise ValueError("evaluation dimension must be 2 or 3")
    evaluation = config["evaluation"]
    methods = evaluation.get("methods", [])
    if not isinstance(methods, list) or not methods or set(methods) - {
        "hmonet",
        "meshonet",
        "pimeshonet",
        "qghmonet",
        "tfi",
        "pde",
    }:
        raise ValueError(
            "evaluation.methods must be a non-empty list drawn from tfi, pde, "
            "meshonet, pimeshonet, hmonet and qghmonet"
        )
    for key in (
        "hmonet_boundary_projection",
        "qghmonet_boundary_projection",
        "meshonet_boundary_projection",
        "pimeshonet_boundary_projection",
    ):
        if evaluation.get(key, "direct") not in {"none", "direct", "inverse_distance"}:
            raise ValueError(
                f"evaluation.{key} must be none, direct or inverse_distance"
            )
    if evaluation.get("model", "hmonet") not in {
        "hmonet",
        "qghmonet",
        "meshonet",
        "pimeshonet",
        "both",
        "pi",
        "all",
    }:
        raise ValueError(
            "evaluation.model must be hmonet, qghmonet, meshonet, pimeshonet, both or all"
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
    normalize_mapping_mode(
        evaluation.get("hmonet_mapping_mode", "rational_affine"),
        allow_both=True,
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
        if not isinstance(case, dict):
            raise ValueError(f"case {name} must be a mapping")
        if dimension == 3:
            for field in ("boundary", "parameter", "values", "grid_shape"):
                if field not in case:
                    raise ValueError(f"case {name} is missing field: {field}")
            _validate_grid_counts(f"case {name}.grid_shape", case["grid_shape"], 3)
        else:
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
            _validate_grid_counts(f"case {name}.train_grid", case["train_grid"], 2)
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
                else:
                    _validate_grid_counts(f"case {name}.grid_shapes", grid, 2)


