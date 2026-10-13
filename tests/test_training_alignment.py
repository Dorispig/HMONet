"""Numerical and integration regressions for legacy-aligned training behavior."""

import contextlib
import copy
import csv
import io
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.training_common import (
    BatchLossTotals,
    calculate_fixed_grid_metrics,
    hmonet_main_loss,
    interpolate_pde_mesh,
    logical_grid_and_interior_indices,
    make_data_loader,
    safe_optimizer_step,
    sample_fixed_grid_batch,
    sample_shared_random_points,
    update_learning_rate,
)
from scripts.train.training_state import TrainingState, prepare_training_config
from src.data.dataset import ParametricDataset2D, ParametricDataset3D
from src.models.model import HMONet2D, MeshONet2D
from src.utils.config_validation import remove_deprecated_projection_config

SMALL_MODEL = {
    "branch_dim": 4,
    "trunk_dim": 2,
    "adjust_hidden_dim": 4,
    "branch_hidden_dim": 4,
    "trunk_hidden_dim": 4,
    "adjust_hidden_layer_num": 1,
    "branch_hidden_layer_num": 1,
    "trunk_hidden_layer_num": 1,
    "output_dim": 4,
}


class ArrayDataset:
    def __init__(self, dimension=2, size=9):
        self.grid_shape = (4, 3) if dimension == 2 else (4, 3, 3)
        grid, _ = logical_grid_and_interior_indices(self.grid_shape, "cpu")
        self.mesh_points = np.repeat(grid.numpy()[None], size, axis=0)
        self.boundary_points = np.zeros((size, 1, dimension), dtype=np.float32)
        self.boundary_points[:, 0, 0] = np.arange(size)
        self.root = Path("array_dataset")

    def __len__(self):
        return len(self.mesh_points)

    def __getitem__(self, index):
        return {
            "boundary_points": self.boundary_points[index],
            "mesh_points": self.mesh_points[index],
            "boundary": {"sample": index},
        }


class IdentityModel(torch.nn.Module):
    def forward(self, *values, **kwargs):
        dimension = len(values) // 2
        return tuple(value.unsqueeze(-1) for value in values[dimension:])


class TrainingMathTests(unittest.TestCase):
    def test_shuffle_is_reproducible_and_keeps_geometry_aligned(self):
        data = ArrayDataset()

        def orders(loader):
            epochs = []
            for _ in range(2):
                order = []
                for batch in loader:
                    ids = batch["boundary_points"][:, 0, 0].int().tolist()
                    self.assertEqual(
                        ids, [item["sample"] for item in batch["boundary"]]
                    )
                    order.extend(ids)
                epochs.append(order)
            return epochs

        first = orders(make_data_loader([data], 4, seed=23, shuffle=True))
        repeated = orders(make_data_loader([data], 4, seed=23, shuffle=True))
        self.assertEqual(first, repeated)
        self.assertNotEqual(first[0], first[1])
        self.assertNotEqual(first[0], list(range(len(data))))
        self.assertEqual(sorted(first[0]), list(range(len(data))))

    def test_batch_shares_random_points_but_each_batch_is_fresh(self):
        for dimension in (2, 3):
            generator = torch.Generator().manual_seed(37)
            first = sample_shared_random_points(3, 8, dimension, "cpu", generator)
            second = sample_shared_random_points(3, 8, dimension, "cpu", generator)
            torch.testing.assert_close(first[0], first[2], rtol=0, atol=0)
            self.assertFalse(torch.equal(first, second))
            self.assertTrue(bool(((first > 0) & (first < 1)).all()))
            replay = sample_shared_random_points(
                3, 8, dimension, "cpu", torch.Generator().manual_seed(37)
            )
            torch.testing.assert_close(first, replay, rtol=0, atol=0)

    def test_finite_difference_sampling_uses_exact_axis_margins(self):
        points = sample_shared_random_points(
            2, 2048, 2, "cpu", margin=(0.07, 0.13)
        )
        self.assertTrue(bool((points[..., 0] >= 0.07).all()))
        self.assertTrue(bool((points[..., 0] <= 0.93).all()))
        self.assertTrue(bool((points[..., 1] >= 0.13).all()))
        self.assertTrue(bool((points[..., 1] <= 0.87).all()))
        torch.testing.assert_close(points[0], points[1], rtol=0, atol=0)

        full = sample_shared_random_points(1, 20000, 2, "cpu", margin=0.0)
        self.assertTrue(bool(((full > 0) & (full < 1)).all()))
        self.assertTrue(bool((full[..., 0] < 0.07).any()))
        self.assertTrue(bool((full[..., 1] > 0.87).any()))

    def test_pde_interpolation_uses_each_samples_mesh(self):
        for dimension in (2, 3):
            shape = (5,) * dimension
            grid, _ = logical_grid_and_interior_indices(shape, "cpu")
            meshes = torch.stack((grid, 2 * grid + 1))
            query = sample_shared_random_points(2, 13, dimension, "cpu")
            expected = torch.stack((query[0], 2 * query[1] + 1))
            torch.testing.assert_close(
                interpolate_pde_mesh(meshes, query, shape), expected
            )

    def test_both_hmonet_2d_losses_are_twice_coordinate_mse(self):
        q = torch.tensor([[[0.25, 0.75], [0.5, 0.5]]])
        error = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        for fixed, coefficient in ((False, 16), (True, 32)):
            weights = 1 + coefficient * ((q - 0.5) ** 4).sum(-1)
            expected = (weights.square() * error.square().sum(-1)).mean()
            actual = hmonet_main_loss(error, torch.zeros_like(error), q, fixed)
            torch.testing.assert_close(actual, expected)

    def test_uniform_point_weighting_uses_equal_weights(self):
        q = torch.tensor([[[0.01, 0.01], [0.5, 0.5], [0.99, 0.99]]])
        error = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        expected = error.square().sum(dim=-1).mean()
        actual = hmonet_main_loss(
            error,
            torch.zeros_like(error),
            q,
            point_weighting="uniform",
        )
        torch.testing.assert_close(actual, expected)

    def test_invalid_point_weighting_is_rejected(self):
        value = torch.zeros(1, 1, 2)
        with self.assertRaises(ValueError):
            hmonet_main_loss(value, value, value, point_weighting="unknown")

    def test_fixed_samples_gather_exact_mesh_nodes(self):
        shape = (5, 4)
        grid, interior = logical_grid_and_interior_indices(shape, "cpu")
        meshes = torch.stack((grid, grid + 3))
        query, targets = sample_fixed_grid_batch(meshes, shape, len(interior))
        torch.testing.assert_close(query[0], query[1])
        torch.testing.assert_close(targets[0], query[0])
        torch.testing.assert_close(targets[1], query[1] + 3)
        with self.assertRaises(ValueError):
            sample_fixed_grid_batch(meshes, shape, len(interior) + 1)

        inset_query, _ = sample_fixed_grid_batch(meshes, shape, 2, margin=0.3)
        self.assertTrue(bool(((inset_query >= 0.3) & (inset_query <= 0.7)).all()))

    def test_epoch_loss_weights_the_short_last_batch(self):
        totals = BatchLossTotals()
        totals.add(4, loss=2.0, coordinate_loss=1.0)
        totals.add(1, loss=12.0, coordinate_loss=6.0)
        self.assertEqual(totals.means()["loss"], 4.0)
        self.assertEqual(totals.means()["coordinate_loss"], 2.0)
        with self.assertRaises(FloatingPointError):
            BatchLossTotals().means()

    def test_mse_excludes_boundary_and_doubles_coordinate_mean(self):
        for dimension in (2, 3):
            data = ArrayDataset(dimension, size=3)
            _, interior = logical_grid_and_interior_indices(data.grid_shape, "cpu")
            boundary = np.setdiff1d(
                np.arange(data.mesh_points.shape[1]), interior.numpy()
            )
            data.mesh_points[:, boundary] += 1000
            model = IdentityModel()
            metrics = calculate_fixed_grid_metrics(
                model, [data], data.grid_shape, "cpu", batch_size=2
            )
            self.assertEqual(metrics["mse"], 0.0)
            self.assertTrue(model.training)
            delta = np.arange(1, dimension + 1, dtype=np.float32)
            data.mesh_points[:, interior] += delta
            metrics = calculate_fixed_grid_metrics(
                model,
                [data],
                data.grid_shape,
                "cpu",
                batch_size=2,
                calculate_point_errors=True,
            )
            self.assertAlmostEqual(
                metrics["mse"], 2 * float((delta**2).mean()), places=6
            )
            self.assertAlmostEqual(
                metrics["max_point_error"], float(np.linalg.norm(delta)), places=6
            )
            self.assertAlmostEqual(
                metrics["p99_point_error"], float(np.linalg.norm(delta)), places=6
            )

    def test_learning_rate_can_cross_stop_threshold_once(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=1.2e-5)
        update_learning_rate(optimizer, 1, 0, 1, gamma=0.5, stop_below=1e-5)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 6e-6)
        update_learning_rate(optimizer, 2, 0, 1, gamma=0.5, stop_below=1e-5)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 6e-6)

    def test_nonfinite_loss_and_gradient_skip_or_raise_without_step(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters())
        for skip in (True, False):
            loss = model.weight.sum() * float("nan")
            if skip:
                self.assertFalse(safe_optimizer_step(loss, model, optimizer, 1.0, True))
            else:
                with self.assertRaises(FloatingPointError):
                    safe_optimizer_step(loss, model, optimizer, 1.0, False)
        with torch.no_grad():
            model.weight.zero_()
        for clip in (1.0, None):
            loss = model.weight.sqrt().sum()
            self.assertFalse(safe_optimizer_step(loss, model, optimizer, clip, True))
            self.assertEqual(model.weight.item(), 0.0)
        self.assertEqual(len(optimizer.state), 0)

    def test_meshonet_batched_forward_matches_individual_predictions(self):
        model = MeshONet2D(4, 3, **SMALL_MODEL)
        boundary = torch.rand(3, 10, 2)
        query = sample_shared_random_points(3, 7, 2, "cpu")
        batch = torch.cat(model(*boundary.unbind(-1), *query.unbind(-1)), dim=-1)
        rows = torch.stack(
            [
                torch.cat(model(*boundary[i].unbind(-1), *query[i].unbind(-1)), dim=-1)
                for i in range(3)
            ]
        )
        torch.testing.assert_close(batch, rows)

    def test_linear_fallback_matches_parametric_rectangles_for_batched_queries(self):
        from src.data.geometry.boundary_generator_2d import rectangular_boundary
        from src.data.mesh.sampling import sample_rectangle_boundary_uniform
        from scripts.tfi.tfi_2d import tfi_from_parameter_grid

        geometry = rectangular_boundary(x_min=0, x_max=1, y_min=0, y_max=1)
        q_boundary, _ = sample_rectangle_boundary_uniform(5, 4)
        boundary = torch.as_tensor(
            tfi_from_parameter_grid(geometry, q_boundary), dtype=torch.float32
        )
        boundary = torch.stack((boundary, 2 * boundary))
        model = HMONet2D(5, 4, **SMALL_MODEL)
        query = torch.tensor(
            [[[0.0, 0.0], [1.0, 1.0], [0.2, 0.7]], [[0.0, 1.0], [1.0, 0.0], [0.9, 0.3]]]
        )
        projection = model.project(*boundary.unbind(-1), *query.unbind(-1))
        expected = torch.stack(
            (
                torch.stack((torch.zeros_like(query[..., 1]), query[..., 1]), dim=-1),
                torch.stack((torch.ones_like(query[..., 1]), query[..., 1]), dim=-1),
                torch.stack((query[..., 0], torch.zeros_like(query[..., 0])), dim=-1),
                torch.stack((query[..., 0], torch.ones_like(query[..., 0])), dim=-1),
            ),
            dim=-2,
        )
        expected[1] *= 2
        torch.testing.assert_close(projection, expected)


class TrainingControlTests(unittest.TestCase):
    def test_model_plateau_flags_and_command_line_override(self):
        cfg = {
            "hmonet_training": {"check_plateau_stability": True},
            "meshonet_training": {"check_plateau_stability": False},
        }
        args = SimpleNamespace()
        self.assertTrue(
            prepare_training_config(cfg, "hmonet", args)["training"][
                "check_plateau_stability"
            ]
        )
        self.assertFalse(
            prepare_training_config(cfg, "meshonet", args)["training"][
                "check_plateau_stability"
            ]
        )
        args.check_plateau_stability = False
        self.assertFalse(
            prepare_training_config(cfg, "hmonet", args)["training"][
                "check_plateau_stability"
            ]
        )

    def test_plateau_and_mse_stop_conditions_are_applied(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(
                output_root=temp, checkpoint_root=str(Path(temp) / "ck")
            )
            for use_plateau in (True, False):
                config = prepare_training_config(
                    {
                        "training": {
                            "num_epochs": 6,
                            "plateau_enabled": use_plateau,
                            "plateau_metric": "mse",
                            "plateau_window": 2,
                            "convergence_start_epoch": 2,
                            "success_plateau_patience": 1,
                            "plateau_stable_consecutive_checks": 1,
                            "plateau_fluctuation_spread_enabled": False,
                        }
                    },
                    "hmonet",
                    args,
                )
                model = torch.nn.Linear(1, 1)
                state = TrainingState(
                    model,
                    torch.optim.Adam(model.parameters()),
                    [ArrayDataset()],
                    config,
                    "interpolation",
                    args,
                    "cpu",
                )
                state._measure = lambda: {"mse": 0.1}
                for epoch in range(1, 5):
                    if state.end_epoch(epoch, {"loss": 0.1}, 0.001):
                        break
                self.assertEqual(
                    state.reason,
                    "converged_near_best_plateau" if use_plateau else "maximum_epochs",
                )
                if use_plateau:
                    self.assertEqual(epoch, 3)
            config = prepare_training_config(
                {
                    "training": {
                        "num_epochs": 6,
                        "plateau_enabled": False,
                        "check_mse_convergence": True,
                        "mse_tolerance": 0.2,
                        "mse_consecutive_epochs": 2,
                        "dataset_mse_interval": 3,
                        "convergence_start_epoch": 1,
                    }
                },
                "hmonet",
                args,
            )
            state = TrainingState(
                model,
                torch.optim.Adam(model.parameters()),
                [ArrayDataset()],
                config,
                "interpolation",
                args,
                "cpu",
            )
            state._measure = lambda: {"mse": 0.1}
            for epoch in range(1, 5):
                if state.end_epoch(epoch, {"loss": 0.1}, 0.001):
                    break
            self.assertEqual(epoch, 3)
            self.assertEqual(state.mse_epochs, [1, 3])
            self.assertEqual(state.reason, "metric_convergence")

    def test_deprecated_projection_option_is_removed_at_config_boundary(self):
        cfg = {"model": {"hmonet": {"boundary_curve_interpolation": "fitted_curve"}}}
        with self.assertWarns(UserWarning):
            remove_deprecated_projection_config(cfg)
        self.assertNotIn("boundary_curve_interpolation", cfg["model"]["hmonet"])

    def test_both_dataset_modes_are_labeled_separately(self):
        from scripts.train import train_2d, train_3d

        paths = [ROOT / "dummy_int", ROOT / "dummy_ext"]
        cfg = {
            "datasets": {"interpolation": str(paths[0]), "extrapolation": str(paths[1])}
        }
        args = SimpleNamespace(output_root="outputs/training/2D")
        with tempfile.TemporaryDirectory() as temporary:

            def save_fixture_summary(data, config, mode, run_args, device):
                folder = Path(temporary) / run_args.dataset_name
                folder.mkdir(exist_ok=True)
                run_args.training_output_dir = str(folder)
                (folder / "training_summary.json").write_text(
                    json.dumps(
                        {"dataset": mode, "dataset_name": run_args.dataset_name}
                    ),
                    encoding="utf-8",
                )

            for dimension, module in ((2, train_2d), (3, train_3d)):
                with (
                    patch.object(
                        module,
                        f"ParametricDataset{dimension}D",
                        return_value=ArrayDataset(),
                    ),
                    patch.object(
                        module, "train_hmonet", side_effect=save_fixture_summary
                    ) as htrain,
                    patch.object(
                        module, "train_meshonet", side_effect=save_fixture_summary
                    ) as mtrain,
                    patch.object(module, "print_training_summaries") as report,
                ):
                    module.train_models(
                        ["hmonet", "meshonet"], paths, cfg, "both", args, "cpu"
                    )
                    for train in (htrain, mtrain):
                        self.assertEqual(
                            [call.args[2] for call in train.call_args_list],
                            ["interpolation", "extrapolation"],
                        )
                    report.assert_called_once()
                    self.assertEqual(report.call_args.kwargs["dimension"], dimension)
                    self.assertEqual(
                        [item["dataset"] for item in report.call_args.args[0]],
                        ["interpolation", "extrapolation"] * 2,
                    )


class TrainingIntegrationTests(unittest.TestCase):
    def test_both_dimensions_models_and_projection_modes_save_legacy_outputs(self):
        from src.data.generation.make_dataset_parametric_2d import generate_explicit_dataset
        from src.data.generation.make_dataset_parametric_3d import generate_dataset
        from scripts.train import train_2d, train_3d

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with contextlib.redirect_stdout(io.StringIO()):
                p2 = generate_explicit_dataset(
                    "rect",
                    "rectangle",
                    [{}, {}, {}],
                    grid_shape=(5, 4),
                    output_root=root,
                )
                p3 = generate_dataset(
                    "box",
                    "box",
                    [{}, {}, {}],
                    grid_shape=(4, 4, 4),
                    sample_count=2,
                    output_root=root,
                )
            for dimension, path, dataset_class, module in (
                (2, p2, ParametricDataset2D, train_2d),
                (3, p3, ParametricDataset3D, train_3d),
            ):
                for name, mapping in (
                    ("hmonet", "current"),
                    ("hmonet", "coons_residual"),
                    ("meshonet", "current"),
                ):
                    config = {
                        "model": {
                            "common": dict(SMALL_MODEL, trunk_dim=dimension),
                            "hmonet": {"hmonet_mapping_mode": mapping},
                        },
                        "training": {
                            "num_epochs": 2,
                            "batch_size": 2,
                            "sample_count": 4,
                            "learning_rate": 0.001,
                            "tensorboard_enabled": True,
                        },
                        "hmonet_sampling": {
                            "global_random_count": 4,
                            "fixed_grid_count": 2,
                        },
                        "hmonet_training": {
                            "jacobian_loss_enabled": True,
                            "boundary_normal_loss_enabled": True,
                            "boundary_normal_sample_count": 4,
                        },
                        "meshonet_training": {"interior_sample_count": 4},
                    }
                    if mapping == "coons_residual":
                        config["model"]["hmonet"][
                            "residual_regularization_weight"
                        ] = 0.01
                    data = dataset_class(path, load_boundary=(name == "hmonet"))
                    args = SimpleNamespace(
                        epochs=2,
                        batch_size=2,
                        output_root=str(root / f"run{dimension}_{mapping}"),
                        checkpoint_root=str(root / "checkpoints"),
                        dataset_name=path.name,
                        mode="interpolation",
                    )
                    stdout = io.StringIO()
                    with (
                        warnings.catch_warnings(),
                        contextlib.redirect_stdout(stdout),
                        contextlib.redirect_stderr(io.StringIO()),
                    ):
                        warnings.simplefilter("ignore", DeprecationWarning)
                        history = getattr(module, f"train_{name}")(
                            [data], config, "interpolation", args, torch.device("cpu")
                        )
                    self.assertEqual(len(history), 2)
                    self.assertIn("Epoch 2/2: training_loss=", stdout.getvalue())
                    folder = Path(args.training_output_dir)
                    summary = json.loads((folder / "training_summary.json").read_text())
                    self.assertEqual(summary["dataset_mse_epochs"], [1, 2])
                    self.assertEqual(summary["dataset_mse_scale"], 2.0)
                    for key in (
                        "training_time_seconds",
                        "convergence_epoch",
                        "best_training_loss",
                        "best_model_dataset_mse",
                        "final_plateau_count",
                        "final_mse_consecutive_count",
                        "best_checkpoint",
                        "final_checkpoint",
                    ):
                        self.assertIn(key, summary)
                    self.assertTrue(np.isfinite(summary["best_model_dataset_mse"]))
                    with (folder / "training_history.csv").open() as stream:
                        columns = next(csv.reader(stream))
                    expected = ["epoch", "training_loss"]
                    if name == "hmonet":
                        expected.append("jacobian_loss")
                    expected.append("dataset_mse")
                    self.assertEqual(columns, expected)
                    for filename in (
                        "config.json",
                        "training_loss.png",
                        "dataset_mse.png",
                        "mse_history.csv",
                    ):
                        self.assertTrue((folder / filename).exists())
                    expected_files = {
                        "config.json",
                        "training_history.csv",
                        "mse_history.csv",
                        "training_loss.png",
                        "dataset_mse.png",
                        "training_summary.json",
                        "training_summary.csv",
                        "tensorboard",
                    }

                    self.assertEqual(
                        {path.name for path in folder.iterdir()}, expected_files
                    )
                    from tensorboard.backend.event_processing.event_accumulator import (
                        EventAccumulator,
                    )

                    primary = EventAccumulator(summary["tensorboard_log_dir"]).Reload()
                    mirror = EventAccumulator(
                        summary["tensorboard_mirror_log_dir"]
                    ).Reload()
                    self.assertEqual(primary.Tags(), mirror.Tags())
                    for tag in primary.Tags()["scalars"]:
                        self.assertEqual(primary.Scalars(tag), mirror.Scalars(tag))
                    for tag in primary.Tags()["tensors"]:
                        self.assertEqual(primary.Tensors(tag), mirror.Tensors(tag))
                    self.assertIn("run/summary/text_summary", primary.Tags()["tensors"])
                    self.assertIn("summary/best_metric_value", primary.Tags()["scalars"])
                    self.assertTrue(Path(summary["config_yaml"]).is_file())
                    checkpoint = Path(summary["checkpoint_dir"])
                    for filename in (
                        "best.pt",
                        "best_model.pth",
                        "final.pt",
                        "final_model.pth",
                        "checkpoint_metadata.json",
                    ):
                        self.assertTrue((checkpoint / filename).exists())
                    payload = torch.load(
                        checkpoint / "final.pt", map_location="cpu", weights_only=True
                    )
                    self.assertEqual(payload["model_name"], name)
                    self.assertTrue(
                        all(
                            torch.isfinite(value).all()
                            for value in payload["model_state_dict"].values()
                        )
                    )


if __name__ == "__main__":
    unittest.main()
