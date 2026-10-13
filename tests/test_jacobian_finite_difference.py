"""Check domain-safe finite differences and both complete HMONet training paths."""

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from scripts.train.training_common import (
    jacobian_finite_difference_queries,
    jacobian_positive_loss,
    jacobian_positive_loss_autograd,
    jacobian_positive_loss_finite_difference,
)
from tests.test_paired_dataset_training import SMALL_MODEL


class JacobianFiniteDifferenceTests(unittest.TestCase):
    def test_quadratic_values_and_parameter_gradients_match_autograd(self):
        for dim in (2, 3):
            q = torch.tensor([0.0, 0.0001, 0.3, 0.9999, 1.0], dtype=torch.float64)
            q = q[:, None].expand(-1, dim).clone()[None].requires_grad_(True)
            a = torch.tensor(-0.7, dtype=torch.float64, requires_grad=True)

            def mapping(points):
                parts = list(points.unbind(-1))
                parts[0] = (
                    a * parts[0] + 0.2 * parts[0].square() + 0.1 * parts[1].square()
                )
                return torch.stack(parts, dim=-1)

            queries = jacobian_finite_difference_queries(q, 0.01)
            self.assertTrue(((queries >= 0) & (queries <= 1)).all())
            self.assertEqual(queries.shape, (1, 5, 1 + 2 * dim, dim))
            fd = jacobian_positive_loss_finite_difference(
                mapping(queries), q, 0.1, 0.01
            )
            ad = jacobian_positive_loss_autograd(mapping(q), q, 0.1)
            torch.testing.assert_close(fd, ad, rtol=1e-10, atol=1e-10)
            fd_gradient = torch.autograd.grad(fd, a, retain_graph=True)[0]
            ad_gradient = torch.autograd.grad(ad, a, retain_graph=True)[0]
            torch.testing.assert_close(fd_gradient, ad_gradient, rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(jacobian_positive_loss(mapping(q), q, 0.1), ad)

    def test_fd_penalty_can_optimize_without_spatial_autograd(self):
        for dim in (2, 3):
            q = torch.full((1, 10, dim), 0.5, dtype=torch.float64)
            queries = jacobian_finite_difference_queries(q)
            a = torch.nn.Parameter(torch.tensor(-0.5, dtype=torch.float64))
            optimizer = torch.optim.SGD([a], lr=0.2)
            losses = []
            with patch(
                "torch.autograd.grad", side_effect=AssertionError("Spatial AD used")
            ):
                for _ in range(12):
                    parts = list(queries.unbind(-1))
                    parts[0] = a * parts[0]
                    loss = jacobian_positive_loss_finite_difference(
                        torch.stack(parts, -1), q, 0.1
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
            self.assertLess(losses[-1], losses[0] * 0.001)

    def test_invalid_steps_and_training_cli(self):
        q = torch.zeros(1, 2, 2)
        for step in (0, -1, 0.4, 0.51, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                jacobian_finite_difference_queries(q, step)
        domain_points = torch.rand(2, 100, 3, dtype=torch.float64)
        for step in (1e-3, 0.2, 1 / 3):
            stencil = jacobian_finite_difference_queries(domain_points, step)
            self.assertTrue(((stencil >= 0) & (stencil <= 1)).all())
        for dim in (2, 3):
            module = importlib.import_module(f"scripts.train.train_{dim}d")
            with patch.object(module, "train_models", return_value=[]) as dispatch:
                module.main(
                    [
                        "--config",
                        str(
                            Path(__file__).resolve().parents[1]
                            / f"configs/{dim}D/train.yaml"
                        ),
                        "--model",
                        "hmonet",
                        "--device",
                        "cpu",
                        "--jacobian-method",
                        "autograd",
                        "--jacobian-fd-step",
                        "0.002",
                    ]
                )
                config = dispatch.call_args.args[2]
                self.assertEqual(
                    config["hmonet_training"]["jacobian_method"], "autograd"
                )
                self.assertEqual(config["hmonet_training"]["jacobian_fd_step"], 0.002)

    def test_complete_training_both_dimensions_and_methods(self):
        from src.data.generation.make_dataset_parametric_2d import generate_explicit_dataset
        from src.data.generation.make_dataset_parametric_3d import generate_dataset
        from src.data.dataset import ParametricDataset2D, ParametricDataset3D

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dim in (2, 3):
                shape = (4,) * dim
                if dim == 2:
                    out = generate_explicit_dataset(
                        "jac2", "rectangle", [{}], grid_shape=shape, output_root=root
                    )
                    data = ParametricDataset2D(out, load_boundary=True)
                else:
                    out = generate_dataset(
                        "jac3", "box", [{}], grid_shape=shape, output_root=root
                    )
                    data = ParametricDataset3D(out, load_boundary=True)
                train = importlib.import_module(
                    f"scripts.train.train_{dim}d"
                ).train_hmonet
                for method in ("finite_difference", "autograd"):
                    for mapping in ("current", "coons_residual"):
                        with self.subTest(
                            dimension=dim, method=method, mapping=mapping
                        ):
                            config = dict(
                                model=dict(
                                    common={**SMALL_MODEL, "trunk_dim": dim},
                                    hmonet=dict(
                                        hmonet_mapping_mode=mapping,
                                        residual_regularization_weight=0.01,
                                    ),
                                ),
                                training=dict(
                                    num_epochs=1,
                                    batch_size=1,
                                    tensorboard_enabled=False,
                                ),
                                hmonet_training=dict(
                                    jacobian_loss_enabled=True,
                                    jacobian_minimum=10.0,
                                    jacobian_method=method,
                                ),
                                hmonet_sampling=dict(
                                    fixed_grid_count=2, global_random_count=4
                                ),
                            )
                            args = SimpleNamespace(
                                output_root=root / "training",
                                checkpoint_root=root / "checkpoints",
                                dataset_name=out.name,
                                mode="interpolation",
                            )
                            if method == "finite_difference":
                                with patch(
                                    "torch.autograd.grad",
                                    side_effect=AssertionError("Spatial AD used"),
                                ):
                                    history = train(
                                        [data],
                                        config,
                                        "interpolation",
                                        args,
                                        torch.device("cpu"),
                                    )
                            else:
                                history = train(
                                    [data],
                                    config,
                                    "interpolation",
                                    args,
                                    torch.device("cpu"),
                                )
                            self.assertTrue(np.isfinite(history[0]["train_loss"]))
                            summary = json.loads(
                                (
                                    Path(args.training_output_dir)
                                    / "training_summary.json"
                                ).read_text()
                            )
                            self.assertEqual(summary["jacobian_method"], method)
                            self.assertEqual(summary["jacobian_fd_step"], 0.001)


if __name__ == "__main__":
    unittest.main()
