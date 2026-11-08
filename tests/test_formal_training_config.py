from pathlib import Path

import yaml

from src.models import HMONet2D, MeshONet2D, PIMeshONet2D
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


def test_formal_hmonet_is_pure_self_supervised_derivative_suite_objective():
    config = _formal_config()
    validate_train_config(config)
    h = config["hmonet_training"]

    assert h["training_strategy"] == "random_loss_combo_from_start"
    assert h["best_model_metric"] == "training_loss"
    assert h["plateau_metric"] == "train_loss"
    assert h["physics_pde_loss_weight"] == 1.0
    assert h["physics_scale_invariant_pde_enabled"] is True
    assert h["physics_normalized_jacobian_loss_weight"] == 0.01
    assert h["physics_singular_value_loss_weight"] == 0.05
    assert h["random_loss_curvature_sample_count"] == 200
    assert h["random_loss_global_sample_count"] == 200
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
    assert config_3d["training"]["seed"] == config_2d["training"]["seed"]
    assert config_3d["hmonet_sampling"] == config_2d["hmonet_sampling"]


def test_formal_operator_capacities_match_selected_architecture_suite_sizes():
    config = _formal_config()
    shape = (61, 21)
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
        "hmonet": 167690,
        "meshonet": 208562,
        "pimeshonet": 192082,
    }

    assert config["model"]["hmonet"].get(
        "coordinate_lift_mode", config["model"]["common"]["coordinate_lift_mode"]
    ) == "power_fourier"
    for name in ("meshonet", "pimeshonet"):
        assert config["model"][name]["coordinate_lift_mode"] == "paper"
        assert config["model"][name]["coordinate_lift_q"] == 6
