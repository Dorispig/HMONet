"""Regression checks for the unified quantile-based convergence rules."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from scripts.train.training_common import simplified_plateau_status
from scripts.train.training_state import TrainingState, prepare_training_config


class SimplifiedPlateauTests(unittest.TestCase):
    def check(self, history, previous=None):
        return simplified_plateau_status(
            history, 3, 2, 0.1, 1e-6, 1e-5, 0.005, 0.2, 1e-7, previous
        )

    @staticmethod
    def make_state(training):
        state = TrainingState.__new__(TrainingState)
        state.training = training
        state.learning_rate = {"lr_warmup_epochs": 0, "lr_update_interval": 1}
        state.plateau_history = []
        state.long_fluctuation_history = []
        state.long_fluctuation_active = False
        state.plateau = None
        state.writer = None
        state.loss_hits = 0
        state.mse_hits = 0
        state.best = 1.0e-4
        state.best_epoch = 1
        state.reference_best = None
        state.stale = 0
        state.stable_count = 0
        state.failure_plateau_best_window = float("inf")
        state.failure_plateau_bad_checks = 0
        state.failure_plateau = None
        state.reason = "maximum_epochs"
        return state

    def test_patience_and_outlier(self):
        status = self.check([1e-4, 1e-8, 1e-4])
        self.assertEqual(status["historical_best"], 1e-4)
        self.assertFalse(status["passes"])
        status = self.check([1e-4] * 3, status)
        self.assertEqual(status["stale"], 1)
        status = self.check([1e-4] * 3, status)
        self.assertTrue(status["passes"])

    def test_noise_does_not_erase_no_improvement_history(self):
        base = self.check([1e-4] * 3)
        base = self.check([1e-4] * 3, base)
        noisy = self.check([1e-4, 1e-2, 1e-4], base)
        self.assertEqual(noisy["epochs_without_significant_improvement"], 2)
        self.assertEqual(noisy["consecutive_stable_count"], 0)
        self.assertFalse(noisy["passes"])
        degraded = self.check([2e-4] * 3, base)
        self.assertEqual(degraded["epochs_without_significant_improvement"], 2)
        self.assertFalse(degraded["near_best_passes"])
        self.assertFalse(degraded["passes"])
        improved = self.check([8e-5] * 3, base)
        self.assertEqual(improved["epochs_without_significant_improvement"], 0)
        self.assertTrue(improved["significant_improvement"])
        self.assertEqual(improved["historical_best"], 8e-5)

    def test_absolute_floor_and_incomplete_window(self):
        self.assertIsNone(self.check([0]))
        self.assertTrue(self.check([0, 1e-7, 0])["stability_passes"])

    def test_improvement_uses_lower_quantile_and_near_best_uses_median(self):
        base = self.check([1.0, 1.0, 1.0])
        improved = self.check([0.8, 1.0, 1.0], base)
        self.assertEqual(improved["improvement_basis"], "window_lower_quantile")
        self.assertEqual(improved["historical_best_basis"], "window_median")
        self.assertTrue(improved["significant_improvement"])
        self.assertAlmostEqual(improved["historical_best"], 1.0)
        self.assertTrue(improved["near_best_passes"])

    def test_window_improvement_accumulates_against_significant_reference(self):
        base = self.check([1.0, 1.0, 1.0])
        small_one = self.check([0.996, 0.996, 0.996], base)
        accumulated = self.check([0.994, 0.994, 0.994], small_one)
        self.assertFalse(small_one["significant_improvement"])
        self.assertTrue(accumulated["significant_improvement"])
        self.assertAlmostEqual(accumulated["reference_best"], 0.994)

    def test_default_schema_is_shared_in_both_dimensions(self):
        root = Path(__file__).resolve().parents[1]
        obsolete = {
            "best_metric_plateau_enabled",
            "best_metric_plateau_require_near_best",
            "best_metric_plateau_patience",
            "plateau_start_epoch",
            "plateau_patience",
            "failure_plateau_start_epoch",
            "failure_plateau_window",
        }
        for dimension in ("2D", "3D"):
            config = yaml.safe_load(
                (root / "configs" / dimension / "train.yaml").read_text(
                    encoding="utf-8"
                )
            )
            for model in ("hmonet", "meshonet", "pimeshonet", "pihmonet"):
                training = prepare_training_config(config, model, SimpleNamespace())[
                    "training"
                ]
                self.assertEqual(training["num_epochs"], 50000)
                self.assertEqual(training["convergence_start_epoch"], 3000)
                self.assertEqual(training["success_plateau_patience"], 1000)
                self.assertEqual(training["failure_plateau_metric_ratio"], 1.5)
                self.assertEqual(training["plateau_window"], 500)
                self.assertFalse(obsolete.intersection(training))
                self.assertFalse(training["check_mse_convergence"])
                self.assertFalse(training["check_training_loss_convergence"])

    def test_success_cannot_stop_before_earliest_epoch(self):
        training = self._training("hmonet")
        training.update(
            plateau_window=3,
            success_plateau_patience=2,
            plateau_stable_consecutive_checks=2,
            convergence_start_epoch=5,
            plateau_fluctuation_spread_enabled=False,
        )
        state = self.make_state(training)
        for epoch in range(1, 5):
            state._check_plateau(epoch, 1.0e-4)
        self.assertEqual(state.stale, 0)
        self.assertEqual(state.reason, "maximum_epochs")
        state._check_plateau(5, 1.0e-4)
        self.assertEqual(state.stale, 1)
        self.assertEqual(state.reason, "maximum_epochs")
        state._check_plateau(6, 1.0e-4)
        self.assertEqual(state.reason, "converged_near_best_plateau")

    def test_long_fluctuation_is_an_alternative_success_branch(self):
        training = self._training("hmonet")
        training.update(
            plateau_window=3,
            success_plateau_patience=2,
            plateau_stable_consecutive_checks=100,
            convergence_start_epoch=1,
            plateau_fluctuation_spread_start_epoch=1,
            plateau_fluctuation_spread_activation=1.0,
            plateau_fluctuation_spread_activation_epoch=10,
            plateau_fluctuation_spread_window=3,
            plateau_fluctuation_spread_tolerance=0.1,
        )
        state = self.make_state(training)
        for epoch in range(1, 5):
            state._check_plateau(epoch, 1.0e-4)
        self.assertEqual(state.stale, 1)
        self.assertEqual(state.reason, "maximum_epochs")
        state._check_plateau(5, 1.0e-4)
        self.assertTrue(state.plateau["long_fluctuation_passes"])
        self.assertEqual(state.reason, "converged_near_best_plateau")

    def test_failure_uses_shared_window_ratio_and_earliest_epoch(self):
        training = {
            "failure_plateau_enabled": True,
            "convergence_start_epoch": 4,
            "plateau_window": 3,
            "failure_plateau_patience": 2,
            "failure_plateau_metric_ratio": 1.5,
            "failure_plateau_relative_fluctuation": 0.15,
            "failure_plateau_min_lr_updates": 0,
            "fluctuation_quantiles": [0.1, 0.9],
        }
        state = self.make_state(training)
        for epoch, value in enumerate((1.0, 1.0, 1.0, 2.0, 2.0), start=1):
            state.plateau_history.append(value)
            self.assertFalse(state._failure_plateau_ready(epoch))
        state.plateau_history.append(2.0)
        self.assertFalse(state._failure_plateau_ready(6))
        state.plateau_history.append(2.0)
        self.assertTrue(state._failure_plateau_ready(7))
        self.assertAlmostEqual(
            state.failure_plateau["historical_best_window_median"], 1.0
        )
        self.assertGreaterEqual(state.failure_plateau["metric_ratio"], 2.0)

    def test_legacy_convergence_fields_are_removed(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (root / "configs/2D/train.yaml").read_text(encoding="utf-8")
        )
        config["training"].update(
            plateau_fluctuation_spread_min_samples=5,
            best_metric_plateau_enabled=True,
            best_metric_plateau_patience=9,
            plateau_start_epoch=7,
            plateau_patience=8,
            failure_plateau_start_epoch=6,
            failure_plateau_window=5,
        )
        training = prepare_training_config(config, "hmonet", SimpleNamespace())[
            "training"
        ]
        for key in (
            "plateau_fluctuation_spread_min_samples",
            "best_metric_plateau_enabled",
            "best_metric_plateau_patience",
            "plateau_start_epoch",
            "plateau_patience",
            "failure_plateau_start_epoch",
            "failure_plateau_window",
        ):
            self.assertNotIn(key, training)

    def test_plateau_switch_applies_to_all_models(self):
        for dimension in ("2D", "3D"):
            for model in ("hmonet", "meshonet", "pimeshonet", "pihmonet"):
                training = self._training(model, dimension)
                training.update(
                    plateau_window=3,
                    success_plateau_patience=2,
            plateau_stable_consecutive_checks=2,
                    convergence_start_epoch=1,
                    plateau_fluctuation_spread_enabled=False,
                )
                state = self.make_state(training)
                for epoch in range(1, 6):
                    state._check_plateau(epoch, 1e-4)
                self.assertEqual(state.reason, "converged_near_best_plateau")
                state.training["plateau_enabled"] = False
                state.reason = "maximum_epochs"
                state._check_plateau(5, 1e-4)
                self.assertEqual(state.reason, "maximum_epochs")

    @staticmethod
    def _training(model, dimension="2D"):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (root / "configs" / dimension / "train.yaml").read_text(
                encoding="utf-8"
            )
        )
        return prepare_training_config(config, model, SimpleNamespace())["training"]


if __name__ == "__main__":
    unittest.main()



