"""Numerical and end-to-end tests of self-supervised PI-MeshONet."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import yaml
from scripts.train.physics_loss import stencil_queries, elliptic_residual
from scripts.train.train_pimeshonet import train_pimeshonet
from src.evaluation.inference import load_network
from tests.test_paired_dataset_training import SMALL_MODEL


class PhysicsTests(unittest.TestCase):
    def test_analytic_residual_and_parameter_gradient(self):
        for dim in (2, 3):
            q = torch.full((2, 7, dim), 0.5, dtype=torch.float64)
            h = torch.full((dim,), 0.05, dtype=torch.float64)
            points = stencil_queries(q, h)
            a = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
            components = list(points.unbind(-1))
            components[0] = components[0] + a * components[1].square()
            result = elliptic_residual(torch.stack(components, -1), h)
            expected = torch.zeros_like(result)
            expected[..., 0] = 2 * a
            torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)
            loss = result.square().sum(-1).mean()
            loss.backward()
            self.assertAlmostEqual(a.grad.item(), 8 * a.item(), places=8)
            self.assertTrue(torch.all((points >= 0) & (points <= 1)))

    def test_mixed_derivatives_against_autograd_reference(self):
        torch.manual_seed(41)
        for dim in (2, 3):
            linear = torch.randn(dim, dim, dtype=torch.float64)
            quadratic = torch.randn(dim, dim, dim, dtype=torch.float64)
            quadratic = (quadratic + quadratic.transpose(-1, -2)) / 2

            def mapping(x):
                return linear @ x + torch.einsum("kij,i,j->k", quadratic, x, x)

            q = torch.rand(dim, dtype=torch.float64) * 0.5 + 0.25
            first = torch.autograd.functional.jacobian(mapping, q).T
            metric = first @ first.T
            cofactor = torch.linalg.det(metric) * torch.linalg.inv(metric)
            expected = (cofactor[None] * (2 * quadratic)).sum((-1, -2))
            offsets = stencil_queries(q[None], [0.02] * dim)[0]
            values = torch.stack([mapping(x) for x in offsets])
            actual = elliptic_residual(values, [0.02] * dim)
            torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)

    def test_pde_loss_can_optimize(self):
        for dim in (2, 3):
            a = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
            optim = torch.optim.SGD([a], lr=0.05)
            points = stencil_queries(
                torch.full((5, dim), 0.5, dtype=torch.float64), [0.05] * dim
            )
            losses = []
            for _ in range(10):
                outputs = list(points.unbind(-1))
                outputs[0] = outputs[0] + a * outputs[1].square()
                loss = (
                    elliptic_residual(torch.stack(outputs, -1), [0.05] * dim)
                    .square()
                    .sum(-1)
                    .mean()
                )
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.item())
            self.assertLess(losses[-1], losses[0] * 0.01)

    def test_train_checkpoint_evaluate_both_dimensions(self):
        from src.data.generation.make_dataset_parametric_2d import generate_explicit_dataset
        from src.data.generation.make_dataset_parametric_3d import generate_dataset
        from src.data.dataset import ParametricDataset2D, ParametricDataset3D
        from src.evaluation.configuration import load_configuration
        from scripts.evaluate.evaluation_common import run_evaluation

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dim in (2, 3):
                shape = (4,) * dim
                name = f"physics_{dim}d"
                if dim == 2:
                    out = generate_explicit_dataset(
                        name, "rectangle", [{}], grid_shape=shape, output_root=root
                    )
                else:
                    out = generate_dataset(
                        name, "box", [{}], grid_shape=shape, output_root=root
                    )
                cls = ParametricDataset2D if dim == 2 else ParametricDataset3D
                data = cls(out, load_boundary=False)
                config = dict(
                    model=dict(common={**SMALL_MODEL, "trunk_dim": dim}),
                    training=dict(
                        num_epochs=2,
                        batch_size=1,
                        learning_rate=0.001,
                        tensorboard_enabled=False,
                        best_model_metric="dataset_mse",
                    ),
                    pimeshonet_training=dict(
                        interior_sample_count=4, fd_step=[0.1] * dim
                    ),
                )
                args = SimpleNamespace(
                    output_root=root / "training" / name,
                    checkpoint_root=root / "checkpoints",
                    config_archive_root=root / "configs" / f"{dim}D",
                    dataset_name=name,
                    mode="interpolation",
                )
                # No input-coordinate automatic derivatives may be requested.
                with patch(
                    "torch.autograd.grad", side_effect=AssertionError("spatial AD used")
                ):
                    history = train_pimeshonet(
                        [data], config, "interpolation", args, torch.device("cpu")
                    )
                self.assertTrue(np.isfinite(history[-1]["loss_pde"]))
                checkpoint = Path(args.checkpoint_output_dir) / "best.pt"
                self.assertIn("PI-MeshONet", checkpoint.parts)
                self.assertIn("PI-MeshONet", Path(args.training_output_dir).parts)
                model, _, _ = load_network(
                    checkpoint, "PI-MeshONet", dim, "current", torch.device("cpu")
                )
                saved = torch.load(checkpoint, weights_only=False)
                self.assertEqual(
                    saved["config"]["training"]["best_model_metric"], "training_loss"
                )
                self.assertTrue(
                    list(
                        (root / "configs" / f"{dim}D" / name / "PI-MeshONet").glob(
                            "*/train.yaml"
                        )
                    )
                )
                # Deliberately corrupt interior labels: optimization must be identical.
                original_mesh = data.mesh_points.copy()
                data.mesh_points *= 7
                second_history = train_pimeshonet(
                    [data], config, "interpolation", args, torch.device("cpu")
                )
                second_saved = torch.load(
                    Path(args.checkpoint_output_dir) / "best.pt", weights_only=False
                )
                self.assertEqual(
                    [r["train_loss"] for r in history],
                    [r["train_loss"] for r in second_history],
                )
                for key, value in saved["model_state_dict"].items():
                    torch.testing.assert_close(
                        value, second_saved["model_state_dict"][key], rtol=0, atol=0
                    )
                data.mesh_points[:] = original_mesh
                ec = root / f"eval{dim}.yaml"
                ec.write_text(
                    yaml.safe_dump(
                        dict(
                            cases={
                                name: dict(
                                    boundary_type="rectangle" if dim == 2 else "box",
                                    values=[0],
                                    grid_shapes=[list(shape)],
                                )
                            },
                            evaluation=dict(
                                save_plots=False,
                                methods=["tfi", "pimeshonet"],
                                skip_missing_checkpoints=False,
                            ),
                        )
                    ),
                    encoding="utf-8",
                )
                ea, cfg = load_configuration(
                    dim,
                    [
                        "--config",
                        str(ec),
                        "--case",
                        name,
                        "--pimeshonet-checkpoint",
                        str(checkpoint),
                        "--device",
                        "cpu",
                        "--output-root",
                        str(root / "results"),
                        "--inference-warmup-runs",
                        "0",
                        "--inference-timing-repeats",
                        "3",
                    ],
                )
                run_evaluation(dim, ea, cfg)
                result_paths = list((root / "results").rglob("results.json"))
                relevant = [p for p in result_paths if name in p.parts]
                self.assertTrue(relevant)
                result = json.loads(relevant[0].read_text())
                self.assertIn("PI-MeshONet", result["timings"])
                self.assertIn("PI-MeshONet", result["metrics"])
                self.assertTrue(
                    any(
                        r["method"] == "PI-MeshONet" and r["stage"] == "raw"
                        for r in result["boundary_errors"]
                    )
                )


if __name__ == "__main__":
    unittest.main()
