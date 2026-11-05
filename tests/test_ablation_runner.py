"""Ablation selection, parameter variants, output profiles and batch-runner tests."""

from types import SimpleNamespace
from unittest.mock import patch
from scripts.experiments import run_ablation_2d
from scripts.experiments.run_ablations_2d import (
    ALL_ABLATIONS,
    _forward_common_arguments,
    _natural_experiment_key,
    _shared_declared_config,
    main,
    resolve_ablations,
)
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from scripts.experiments.run_ablation_2d import (
    _variant_options,
    plot_training_histories,
    reject_embedded_baseline_variants,
    run_initial_learning_rate_experiment,
    run_point_weighting_experiment,
    run_training_sample_count_experiment,
    Variant,
)
from scripts.experiments.run_ablation_2d import (
    evaluate_one,
    evaluated_methods,
    variant_model,
    variant_training_overrides,
    write_profiled_metric_outputs,
)
import pytest


class TestBatchAblationRunner:
    def test_experiment_directories_use_natural_numeric_order(self):
        values = [Path("e10_model"), Path("e2_model"), Path("e1_model")]
        assert (
            [path.name for path in sorted(values, key=_natural_experiment_key)]
        ) == (["e1_model", "e2_model", "e10_model"])

    def test_operator_model_variants_use_their_own_training_and_evaluation(self):
        for model in ("meshonet", "pimeshonet"):
            variant = Variant(
                model,
                model,
                ("--model", model, "--plateau-metric", "train_loss"),
            )
            assert (variant_model(variant)) == (model)
            assert (variant_training_overrides(variant)) == (
                ["--plateau-metric", "train_loss"]
            )
            assert (
                evaluated_methods(
                    {"methods": ["tfi", "pde", "hmonet"]},
                    {"model": model},
                )
            ) == (["tfi", "pde", model])

    def test_experiment_definition_cannot_embed_baseline_variant(self):
        with pytest.raises(ValueError, match="never train or evaluate baselines"):
            reject_embedded_baseline_variants(
                [Variant("baseline", "baseline", ())],
                Path("experiments/ex9/e7/experiment.yaml"),
            )

    def test_regular_experiment_variant_passes_baseline_guard(self):
        reject_embedded_baseline_variants(
            [Variant("e7_pde_winslow_jac", "physics", ())],
            Path("experiments/ex9/e7/experiment.yaml"),
        )

    def test_phase_transition_optimizer_can_be_retained_by_ablation(self):
        options = _variant_options(("--no-hmonet-reset-optimizer-on-phase-transition",))
        assert (options["hmonet_reset_optimizer_on_phase_transition"]) is (False)

    def test_ex11_raw_pde_disables_only_scale_invariant_normalization(self):
        variant = Variant(
            "e11_raw_pde",
            "raw_elliptic_pde_residual_loss",
            ("--lr", "0.0004", "--no-hmonet-physics-scale-invariant-pde-enabled"),
        )
        assert (variant_training_overrides(variant)) == (
            ["--lr", "0.0004", "--no-hmonet-physics-scale-invariant-pde-enabled"]
        )
        options = _variant_options(variant.overrides)
        assert (options["lr"]) == ("0.0004")
        assert (options["hmonet_physics_scale_invariant_pde_enabled"]) is (False)

    def test_ex11_boolean_architecture_switches_are_forwarded_without_values(self):
        flags = (
            "--modified-mlp-enabled",
            "--query-boundary-cross-attention-enabled",
            "--boundary-layer-decoder-enabled",
            "--decoupled-weight-decoder-enabled",
            "--multiscale-boundary-encoder-enabled",
        )
        for flag in flags:
            variant = Variant(
                "architecture_switch",
                True,
                ("--lr", "0.0004", flag),
            )
            assert (variant_training_overrides(variant)) == (["--lr", "0.0004", flag])
            assert (
                _variant_options(variant.overrides)[flag[2:].replace("-", "_")]
            ) is (True)

    def test_model_override_is_removed_without_dropping_following_boolean(self):
        variant = Variant(
            "qg",
            "qghmonet",
            (
                "--model",
                "qghmonet",
                "--lr",
                "0.0004",
                "--modified-mlp-enabled",
            ),
        )
        assert (variant_training_overrides(variant)) == (
            ["--lr", "0.0004", "--modified-mlp-enabled"]
        )

    def test_numbered_subsuite_name_is_supported(self):
        assert (run_ablation_2d.validate_experiment_set_name("ex8_1")) == ("ex8_1")
        for invalid in ("ex0_1", "ex8_", "ex8_a", "ex8.1", "experiment8_1"):
            with pytest.raises(ValueError):
                run_ablation_2d.validate_experiment_set_name(invalid)

    def test_subsuite_can_declare_an_external_read_only_baseline(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "ex8/baseline/experiment.yaml"
            baseline.parent.mkdir(parents=True)
            baseline.write_text("model: {}\n", encoding="utf-8")
            experiment = root / "ex8_1/physics_training_strategy/experiment.yaml"
            experiment.parent.mkdir(parents=True)
            experiment.write_text(
                "baseline_config: ../../ex8/baseline/experiment.yaml\n",
                encoding="utf-8",
            )
            resolved = _shared_declared_config(
                root / "ex8_1", ["physics_training_strategy"], "baseline_config"
            )
            assert (resolved) == (baseline.resolve())

    def test_all_and_selected_experiments(self):
        assert (resolve_ablations(["all"])) == (list(ALL_ABLATIONS))
        selected = ["jacobian_enabled", "training_sample_count"]
        assert (resolve_ablations(selected)) == (selected)

    def test_ex10_adaptive_400_uses_only_curvature_adaptive_points(self):
        definition = (
            run_ablation_2d.ROOT
            / "experiments/ex10/adaptive_400/experiment.yaml"
        )
        variants = run_ablation_2d.variants_from_definition(
            run_ablation_2d._load_yaml_mapping(definition)
        )
        assert len(variants) == 1
        assert variants[0].name == "adaptive_400"
        options = _variant_options(variants[0].overrides)
        assert options["hmonet_random_loss_sampling_mode"] == "mixed_boundary"
        assert options["hmonet_random_loss_sample_count"] == "400"
        assert options["hmonet_random_loss_global_sample_count"] == "0"
        assert options["hmonet_random_loss_curvature_sample_count"] == "400"
        assert options["hmonet_random_loss_boundary_band_sample_count"] == "0"

    def test_baseline_can_precede_one_experiment_or_all(self):
        assert (resolve_ablations(["baseline", "jacobian_enabled"])) == (
            ["baseline", "jacobian_enabled"]
        )
        assert (resolve_ablations(["all", "baseline"])) == (
            ["baseline", *ALL_ABLATIONS]
        )
        assert (
            resolve_ablations(["baseline", "all"], ("e1_mse_pde", "e7_pde_winslow_jac"))
        ) == (["baseline", "e1_mse_pde", "e7_pde_winslow_jac"])

    def test_invalid_selections_are_rejected(self):
        for values in (
            ["all", "normal_loss"],
            ["baseline", "all", "normal_loss"],
            ["unknown"],
            ["normal_loss"] * 2,
        ):
            with pytest.raises(ValueError):
                resolve_ablations(values)

    def test_baseline_error_is_deferred_until_all_selected_ablations_finish(self):
        error = run_ablation_2d.BaselineCompatibilityError("baseline mismatch")
        with patch(
            "scripts.experiments.run_ablations_2d.run_ablation_2d.main",
            side_effect=[error, []],
        ) as run:
            with pytest.raises(run_ablation_2d.BaselineCompatibilityError):
                main(
                    [
                        "--ablations",
                        "jacobian_enabled",
                        "normal_loss",
                        "--dry-run",
                    ]
                )
        assert (run.call_count) == (2)

    def test_common_options_are_forwarded(self):
        args = SimpleNamespace(
            config="a.yaml",
            train_config="t.yaml",
            evaluate_config="e.yaml",
            experiment_root="experiments",
            run_tag="paper",
            dataset=["one", "two"],
            seeds=[1, 2, 3],
            evaluation_grid_shapes=["101x101"],
            device="cuda",
            dry_run=True,
            train_only=False,
            evaluate_only=False,
            force=False,
            continue_on_error=True,
        )
        command = _forward_common_arguments(args, "normal_loss")
        assert ("normal_loss") in (command)
        assert (command.count("--dataset")) == (2)
        assert ("--seeds") in (command)
        assert ("--evaluation-grid-shapes") in (command)
        assert ("101x101") in (command)
        assert ("--dry-run") in (command)
        assert ("--continue-on-error") in (command)


class TestInitialLearningRateAblation:
    def test_variants_only_override_initial_learning_rate(self):
        variants = run_initial_learning_rate_experiment({}, [0.0001, 0.0004, 0.001])
        assert ([item.value for item in variants]) == ([0.0001, 0.0004, 0.001])
        for variant in variants:
            assert (variant.overrides) == ("--lr", str(variant.value))

    def test_invalid_or_empty_ranges_are_rejected(self):
        for values in ([], [0.0], [-0.001], [float("nan")]):
            with pytest.raises(ValueError):
                run_initial_learning_rate_experiment({}, values)

    def test_training_sample_count_variants(self):
        variants = run_training_sample_count_experiment({})
        assert ([item.value for item in variants]) == ([50, 100, 200, 400, 800, 1200])
        for variant in variants:
            assert (variant.overrides) == ("--global-random-count", str(variant.value))

    def test_invalid_training_sample_counts_are_rejected(self):
        for values in ([], [0], [-1], [100.5], [100, 100]):
            with pytest.raises(ValueError):
                run_training_sample_count_experiment({}, values)

    def test_point_weighting_variants(self):
        variants = run_point_weighting_experiment({})
        assert ([item.name for item in variants]) == (
            ["position_weighted", "uniform_weight"]
        )
        assert (variants[0].overrides) == ()
        assert (variants[1].overrides) == ("--point-loss-weighting", "uniform")

    def test_all_variants_share_loss_and_mse_comparison_figures(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "initial_learning_rate"
            for variant, scale in (("lr_1e-4", 1.0), ("lr_0p0004", 0.5)):
                for seed in (1, 2):
                    run = root / "dataset" / variant / f"seed_{seed}" / "paper"
                    training = run / "training"
                    training.mkdir(parents=True)
                    (run / "run_manifest.json").write_text(
                        json.dumps(
                            {
                                "dataset": "dataset",
                                "variant": variant,
                                "seed": seed,
                                "run_tag": "paper",
                            }
                        ),
                        encoding="utf-8",
                    )
                    (training / "training_summary.json").write_text(
                        json.dumps(
                            {
                                "training_loss_history": [
                                    scale,
                                    scale * 0.8,
                                    scale * 0.6,
                                ],
                                "dataset_mse_epochs": [1, 2, 3],
                                "dataset_mse_history": [scale * 2, scale, scale * 0.5],
                                "completed_epochs": 3,
                                "training_time_seconds": 6.0 * scale + seed,
                            }
                        ),
                        encoding="utf-8",
                    )
            plot_training_histories(root, "paper")
            figures = root / "dataset" / "figures" / "paper" / "training"
            for stem in ("training_loss_comparison", "dataset_mse_comparison"):
                assert (figures / f"{stem}.png").is_file()
                csv_text = (figures / f"{stem}.csv").read_text(encoding="utf-8-sig")
                assert ("lr_1e-4") in (csv_text)
                assert ("lr_0p0004") in (csv_text)
            assert (figures / "parameter_vs_iteration_time.png").is_file()
            assert (figures / "parameter_vs_total_training_time.png").is_file()
            assert (figures / "parameter_vs_iteration_count.png").is_file()
            statistics = (figures / "parameter_training_statistics.csv").read_text(
                encoding="utf-8-sig"
            )
            assert ("average_epoch_seconds_mean") in (statistics)
            assert ("completed_epochs_mean") in (statistics)
            assert ("total_training_seconds_mean") in (statistics)


class TestProfiledAblationOutput:
    def test_evaluation_uses_only_default_51_grid(self):
        root = Path(__file__).resolve().parents[1]
        dataset = root / "data/raw/2D/air_t_int_0_1000_21_20"
        with TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                evaluate_config=root / "configs/2D/evaluate.yaml",
                device=None,
            )
            summary = {
                "best_checkpoint": str(Path(temporary) / "best.pt"),
                "hmonet_mapping_mode": "current",
            }
            with patch("scripts.evaluate.evaluate_2d.main", return_value=None) as main:
                command = evaluate_one(args, dataset, summary, Path(temporary) / "eval")
            assert (main.call_count) == (1)
            assert ("--save-plots") in (command)
            index = command.index("--grid-shapes")
            assert (command[index + 1]) == ("51x51")

    def test_metric_categories_are_split_inside_each_grid(self):
        rows = []
        for grid in ("51x51",):
            for variant, value in (("jac_off", 0.0), ("jac_on", 1.0)):
                rows.append(
                    {
                        "ablation": "jacobian_enabled",
                        "dataset": "air_t_int_0_1000_21_20",
                        "variant": variant,
                        "variant_value": value,
                        "seed": 1,
                        "run_tag": "paper",
                        "method": "HMONet",
                        "parameter": "t",
                        "value": 0.3,
                        "grid_shape": grid,
                        "result_file": "result.json",
                        "invalid_cell_fraction": value,
                        "self_intersection__affected_cell_fraction": value,
                        "flipping_cell__affected_cell_fraction": value,
                        "jacobian__minimum": 1.0 - value,
                        "scaled_jacobian__minimum": 1.0 - value,
                        "fixed_grid_best_mse": value + 0.1,
                        "orthogonality__mean": 0.9 - value * 0.1,
                    }
                )
        with TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "jacobian_enabled"
            write_profiled_metric_outputs(rows, experiment, "paper")
            for grid in ("51x51",):
                aggregate = experiment / "air_t_int_0_1000_21_20/aggregate/paper" / grid
                for category in (
                    "important_metrics",
                    "secondary_metrics",
                    "other_metrics",
                ):
                    assert (aggregate / f"{category}.csv").is_file()
                figures = experiment / "air_t_int_0_1000_21_20/figures/paper" / grid
                assert (
                    figures
                    / "important_metrics/invalid_cell_fraction/parameter_0.3.png"
                ).is_file()
                assert (
                    figures / "secondary_metrics/fixed_grid_best_mse/parameter_0.3.png"
                ).is_file()
                assert (
                    figures / "other_metrics/orthogonality__mean/parameter_0.3.png"
                ).is_file()
