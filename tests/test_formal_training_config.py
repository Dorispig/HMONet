from pathlib import Path

import yaml

from src.models import (
    HMONet2D,
    HMONet3D,
    MeshONet2D,
    MeshONet3D,
    PIMeshONet2D,
    PIMeshONet3D,
)
from src.utils.config_validation import validate_train_config


ROOT = Path(__file__).resolve().parents[1]


def _formal_config():
    path = ROOT / "configs" / "2D" / "train.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _formal_config_3d():
    path = ROOT / "configs" / "3D" / "train.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _options(config, name):
    options = dict(config["model"]["common"])
    options.update(config["model"].get(name, {}))
    options.pop("residual_regularization_weight", None)
    return options


def test_formal_hmonet_is_pure_normalized_pde_self_supervision():
    config = _formal_config()
    validate_train_config(config)
    h = config["hmonet_training"]

    assert h["training_strategy"] == "random_loss_combo_from_start"
    assert h["best_model_metric"] == "training_loss"
    assert h["plateau_metric"] == "train_loss"
    assert h["random_mse_loss_weight"] == 0.0
    assert h["physics_pde_loss_weight"] == 1.0
    assert h["physics_scale_invariant_pde_enabled"] is True
    assert h["physics_normalized_jacobian_loss_weight"] == 0.0
    assert h["physics_singular_value_loss_weight"] == 0.0
    assert h["pde_fd_step"] == [0.01, 0.01]
    assert h["jacobian_fd_step"] == 0.01
    assert h["random_loss_sampling_mode"] == "uniform"
    assert h["random_loss_curvature_sample_count"] == 0
    assert h["random_loss_global_sample_count"] == 400
    assert h["random_loss_boundary_band_sample_count"] == 0

    removed_supervised_options = {
        "target_interpolation",
        "idw_query_chunk_size",
        "idw_support_chunk_size",
        "point_loss_weighting",
        "jacobian_loss_enabled",
        "jacobian_loss_weight",
        "pde_loss_enabled",
        "pde_loss_weight",
        "boundary_normal_loss_enabled",
        "boundary_normal_loss_weight",
    }
    assert removed_supervised_options.isdisjoint(h)
    assert config["model"]["common"]["coordinate_lift_mode"] == "legacy"


def test_ex7_baseline_keeps_the_frozen_coordinate_lift_ablation_configuration():
    formal = _formal_config()["hmonet_training"]
    baseline_path = ROOT / "experiments" / "ex7" / "baseline" / "experiment.yaml"
    baseline_config = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    baseline = baseline_config["hmonet_training"]
    inherited_keys = (
        "learning_rate",
        "lr_warmup_epochs",
        "lr_update_interval",
        "lr_decay_gamma",
        "lr_stop_below",
        "training_strategy",
        "random_loss_sample_count",
        "random_mse_loss_weight",
        "physics_pde_loss_weight",
        "physics_scale_invariant_pde_enabled",
        "physics_normalized_jacobian_loss_weight",
        "physics_normalized_jacobian_minimum",
        "physics_singular_value_loss_weight",
        "physics_singular_value_minimum",
        "physics_barrier_reduction",
        "pde_fd_step",
        "jacobian_fd_step",
        "random_loss_derivative_backend",
        "random_loss_fd_stencil_inside",
        "random_loss_fd_boundary_scheme",
    )
    for key in inherited_keys:
        assert baseline[key] == formal[key]
    # Ex7 deliberately isolates coordinate augmentation and keeps its P=2
    # baseline even when the formal HMONet default changes.
    assert baseline_config["model"]["common"]["coordinate_lift_mode"] == "power_fourier"
    assert baseline_config["model"]["common"]["coordinate_lift_max_power"] == 2
    assert baseline["random_loss_sampling_mode"] == "mixed_boundary"
    assert baseline["random_loss_global_sample_count"] == 200
    assert baseline["random_loss_curvature_sample_count"] == 200
    assert baseline["random_loss_boundary_band_sample_count"] == 0


def test_formal_3d_hmonet_matches_2d_physics_objective_and_sampling():
    config_2d = _formal_config()
    config_3d = _formal_config_3d()
    validate_train_config(config_3d)
    h2 = config_2d["hmonet_training"]
    h3 = config_3d["hmonet_training"]

    for key in (
        "training_strategy",
        "best_model_metric",
        "plateau_metric",
        "learning_rate",
        "physics_pde_loss_weight",
        "physics_scale_invariant_pde_enabled",
        "physics_normalized_jacobian_loss_weight",
        "physics_normalized_jacobian_minimum",
        "physics_singular_value_loss_weight",
        "physics_singular_value_minimum",
        "physics_barrier_reduction",
        "random_loss_sampling_mode",
        "random_loss_global_sample_count",
        "random_loss_curvature_sample_count",
        "random_loss_boundary_band_sample_count",
    ):
        assert h3[key] == h2[key]
    assert h3["pde_fd_step"] == [0.01, 0.01, 0.01]
    assert h3["jacobian_fd_step"] == 0.01
    assert config_3d["training"]["seed"] == config_2d["training"]["seed"]
    assert config_3d["hmonet_sampling"] == config_2d["hmonet_sampling"]


def test_formal_operator_capacities_match_selected_architecture_suite_sizes():
    config = _formal_config()
    shape = (41, 41)
    models = {
        "hmonet": HMONet2D(*shape, **_options(config, "hmonet")),
        "meshonet": MeshONet2D(*shape, **_options(config, "meshonet")),
        "pimeshonet": PIMeshONet2D(*shape, **_options(config, "pimeshonet")),
    }
    counts = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    }
    assert counts == {
        "hmonet": 168074,
        "meshonet": 208722,
        "pimeshonet": 192562,
    }

    for name in ("hmonet", "meshonet", "pimeshonet"):
        assert config["model"].get(name, {}).get(
            "coordinate_lift_mode", config["model"]["common"]["coordinate_lift_mode"]
        ) == "legacy"

    config_3d = _formal_config_3d()
    shape_3d = (11, 21, 11)
    models_3d = {
        "hmonet": HMONet3D(*shape_3d, **_options(config_3d, "hmonet")),
        "meshonet": MeshONet3D(*shape_3d, **_options(config_3d, "meshonet")),
        "pimeshonet": PIMeshONet3D(*shape_3d, **_options(config_3d, "pimeshonet")),
    }
    counts_3d = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models_3d.items()
    }
    assert counts_3d == {
        "hmonet": 415215,
        "meshonet": 488443,
        "pimeshonet": 472443,
    }
