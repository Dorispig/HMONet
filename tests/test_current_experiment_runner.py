"""Regression tests for the compact ex1--ex4 experiment runner."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiments import run_ablation_2d
from scripts.experiments.run_ablation_2d import (
    Variant,
    _load_yaml_mapping,
    _variant_options,
    evaluated_methods,
    reject_embedded_baseline_variants,
    variant_model,
    variant_training_overrides,
    variants_from_definition,
)
from scripts.experiments.run_ablations_2d import (
    _forward_common_arguments,
    _natural_experiment_key,
    resolve_ablations,
)


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_directories_use_natural_numeric_order():
    values = [Path("e10_model"), Path("e2_model"), Path("e1_model")]
    assert [path.name for path in sorted(values, key=_natural_experiment_key)] == [
        "e1_model", "e2_model", "e10_model"
    ]


def test_only_consecutive_suite_names_are_accepted():
    for value in ("ex1", "ex2", "ex4"):
        assert run_ablation_2d.validate_experiment_set_name(value) == value
    for value in ("ex0", "ex1_1", "ex1.1", "experiment1"):
        with pytest.raises(ValueError):
            run_ablation_2d.validate_experiment_set_name(value)


def test_run_tag_is_portable_and_bounded_for_output_paths():
    assert run_ablation_2d.validate_run_tag("v1") == "v1"
    assert run_ablation_2d.validate_run_tag("paper.v1-2026_11") == "paper.v1-2026_11"
    with pytest.raises(ValueError):
        run_ablation_2d.validate_run_tag("bad/tag")
    with pytest.raises(ValueError):
        run_ablation_2d.validate_run_tag("x" * 25)


def test_suite_local_all_and_baseline_selection():
    available = ("rational_affine", "coons_residual")
    assert resolve_ablations(["all"], available) == list(available)
    assert resolve_ablations(["baseline", "all"], available) == ["baseline", *available]
    assert resolve_ablations(["coons_residual"], available) == ["coons_residual"]


def test_every_retained_experiment_declares_a_yaml_variant():
    for suite in ("ex1", "ex2", "ex3", "ex4"):
        for path in (ROOT / "experiments" / suite).iterdir():
            definition = path / "experiment.yaml"
            if path.name == "baseline" or not definition.is_file():
                continue
            variants = variants_from_definition(_load_yaml_mapping(definition))
            assert variants, definition
            assert all(variant.name != "baseline" for variant in variants)


def test_ex2_adaptive_400_uses_only_curvature_points():
    definition = ROOT / "experiments/ex2/adaptive_400/experiment.yaml"
    variant = variants_from_definition(_load_yaml_mapping(definition))[0]
    options = _variant_options(variant.overrides)
    assert options["hmonet_random_loss_sample_count"] == "400"
    assert options["hmonet_random_loss_global_sample_count"] == "0"
    assert options["hmonet_random_loss_curvature_sample_count"] == "400"
    assert options["hmonet_random_loss_boundary_band_sample_count"] == "0"


def test_ex2_sampling_variants_change_only_global_adaptive_split():
    expected = {
        "global300_adaptive100": ("300", "100"),
        "global100_adaptive300": ("100", "300"),
    }
    for name, (global_count, adaptive_count) in expected.items():
        definition = ROOT / "experiments/ex2" / name / "experiment.yaml"
        options = _variant_options(
            variants_from_definition(_load_yaml_mapping(definition))[0].overrides
        )
        assert options["hmonet_random_loss_global_sample_count"] == global_count
        assert options["hmonet_random_loss_curvature_sample_count"] == adaptive_count
        assert options["hmonet_random_loss_boundary_band_sample_count"] == "0"


def test_operator_variants_select_their_own_model():
    for model in ("meshonet", "pimeshonet"):
        variant = Variant(model, model, ("--model", model, "--plateau-metric", "train_loss"))
        assert variant_model(variant) == model
        assert variant_training_overrides(variant) == ["--plateau-metric", "train_loss"]
        assert evaluated_methods(
            {"methods": ["tfi", "pde", "hmonet"]}, {"model": model}
        ) == ["tfi", "pde", model]


def test_embedded_baseline_variant_is_rejected():
    with pytest.raises(ValueError, match="baseline"):
        reject_embedded_baseline_variants(
            [Variant("baseline", "baseline", ())],
            Path("experiments/ex1/rational_affine/experiment.yaml"),
        )


def test_architecture_boolean_overrides_do_not_consume_the_next_flag():
    variant = Variant(
        "architecture", True,
        ("--lr", "0.0004", "--modified-mlp-enabled", "--boundary-layer-decoder-enabled"),
    )
    assert variant_training_overrides(variant) == [
        "--lr", "0.0004", "--modified-mlp-enabled", "--boundary-layer-decoder-enabled"
    ]


def test_common_arguments_keep_dataset_seed_grid_and_tag():
    args = SimpleNamespace(
        config=Path("config.yaml"), baseline_config=Path("baseline.yaml"),
        evaluate_config=Path("evaluate.yaml"), experiment_root=Path("experiments/ex1"),
        run_tag="paper", dataset=["one"], seeds=[1],
        evaluation_grid_shapes=["101x101"], evaluation_values=[0.5],
        evaluation_plots=True, device="cpu", dry_run=True, train_only=False,
        evaluate_only=False, force=False, continue_on_error=False,
    )
    command = _forward_common_arguments(args, "rational_affine")
    for value in ("one", "101x101", "paper", "rational_affine"):
        assert value in command
