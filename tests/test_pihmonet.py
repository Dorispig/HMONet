"""Numerical and end-to-end tests of self-supervised PI-HMONet."""

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
from scripts.train.train_pihmonet import train_pihmonet
from src.evaluation.inference import load_network
from tests.test_paired_dataset_training import SMALL_MODEL


class HardPhysicsTests(unittest.TestCase):
    def test_curved_batched_pde_gradients_and_optimization(self):
        from src.models.model import HMONet2D, HMONet3D
        from src.data.geometry.boundary_generator_2d import circle_airfoil_boundary
        from src.data.geometry.boundary_generator_3d import cylindrical_domain_with_airfoil_hole
        from src.data.generation.make_dataset_parametric_2d import to_computational_boundary
        from src.data.geometry.parametric_curve_2d import BoundaryCurveSet2D
        from src.data.mesh.sampling import (
            sample_boundary,
            logical_grid,
            boundary_indices,
        )
        from scripts.train.training_common import predict_coordinates

        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for dim in (2, 3):
                shape = (5,) * dim
                geometries = [
                    (
                        to_computational_boundary(circle_airfoil_boundary(t=t))
                        if dim == 2
                        else cylindrical_domain_with_airfoil_hole(thickness=t)
                    )
                    for t in (0.1, 0.15)
                ]
                boundary = torch.tensor(
                    np.stack([sample_boundary(g, shape) for g in geometries]),
                    dtype=torch.float32,
                    device=device,
                )
                geometry_batch = (
                    BoundaryCurveSet2D.batch(
                        geometries, device=device, dtype=boundary.dtype
                    )
                    if dim == 2
                    else geometries
                )
                # Evaluate unsampled boundary points too, not only the network's sensors.
                fine_shape = (7,) * dim
                boundary_queries = torch.tensor(
                    logical_grid(fine_shape)[boundary_indices(fine_shape)],
                    dtype=torch.float32,
                    device=device,
                )
                exact = torch.tensor(
                    np.stack([sample_boundary(g, fine_shape) for g in geometries]),
                    dtype=torch.float32,
                    device=device,
                )
                for mapping in ("current", "coons_residual"):
                    with self.subTest(device=device, dimension=dim, mapping=mapping):
                        torch.manual_seed(123)
                        cls = HMONet2D if dim == 2 else HMONet3D
                        model = cls(
                            *shape,
                            **{**SMALL_MODEL, "trunk_dim": dim},
                            hmonet_mapping_mode=mapping,
                        ).to(device)
                        h = torch.full((dim,), 0.03, device=device)
                        centers = 0.1 + 0.8 * torch.rand(12, dim, device=device)
                        stencil = stencil_queries(centers, h)
                        query = stencil.reshape(1, -1, dim).expand(2, -1, -1)
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                        losses = []
                        for step in range(12):
                            predicted = predict_coordinates(
                                model, boundary, query, geometry_batch
                            )
                            residual = elliptic_residual(
                                predicted.reshape(2, 12, stencil.shape[-2], dim), h
                            )
                            loss = residual.square().sum(-1).mean()
                            optimizer.zero_grad()
                            loss.backward()
                            gradients = [
                                p.grad for p in model.parameters() if p.grad is not None
                            ]
                            self.assertTrue(
                                all(torch.isfinite(g).all() for g in gradients)
                            )
                            self.assertGreater(
                                sum(g.abs().sum().item() for g in gradients), 0
                            )
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            losses.append(loss.item())
                        self.assertLess(min(losses[-4:]), losses[0])
                        with torch.no_grad():
                            predicted_boundary = predict_coordinates(
                                model,
                                boundary,
                                boundary_queries[None].expand(2, -1, -1),
                                geometry_batch,
                            )
                        torch.testing.assert_close(
                            predicted_boundary, exact, atol=2e-5, rtol=1e-5
                        )
                        print(
                            f"PI-HMONet {device} {dim}D {mapping}: PDE {losses[0]:.6g} -> {losses[-1]:.6g}"
                        )

    def test_command_line_models_and_independent_mappings(self):
        from src.evaluation.configuration import build_parser, evaluation_options
        from scripts.evaluate.evaluation_common import mapping_combinations
        from scripts.train.train_evaluate import evaluation_commands
        import importlib

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
                        "all",
                        "--device",
                        "cpu",
                        "--pihmonet-mapping-mode",
                        "coons_residual",
                        "--pihmonet-pde-loss-weight",
                        "0.2",
                    ]
                )
                names, paths, config, mode, args, device = dispatch.call_args.args
                self.assertEqual(
                    names,
                    [
                        "hmonet",
                        "meshonet",
                        "pimeshonet",
                        "pihmonet",
                        "attentionpihmonet",
                    ],
                )
                self.assertEqual(
                    config["model"]["pihmonet"]["hmonet_mapping_mode"], "coons_residual"
                )
                self.assertEqual(args.pihmonet_pde_loss_weight, 0.2)
        for dim in (2, 3):
            args = build_parser(dim).parse_args(
                ["--model", "all", "--pihmonet-mapping-mode", "both"]
            )
            options = evaluation_options({}, {}, args)
            self.assertIn("PI-HMONet", options["methods"])
            combinations = mapping_combinations(options)
            self.assertEqual(
                [(h, o["pihmonet_mapping_mode"]) for h, o in combinations],
                [("current", "current"), ("current", "coons_residual")],
            )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "best.pt"
            checkpoint.touch()
            command = evaluation_commands(
                [
                    dict(
                        dataset_name="sample",
                        model="pihmonet",
                        best_checkpoint=str(checkpoint),
                        hmonet_mapping_mode="coons_residual",
                    )
                ],
                "eval.yaml",
                {"sample": {}},
                "best",
            )[0]
            self.assertIn("--pihmonet-checkpoint", command)
            self.assertIn("--pihmonet-mapping-mode", command)
            self.assertIn("coons_residual", command)

    def test_train_checkpoint_evaluate_both_dimensions(self):
        from src.data.generation.make_dataset_parametric_2d import generate_explicit_dataset
        from src.data.generation.make_dataset_parametric_3d import generate_dataset
        from src.data.dataset import ParametricDataset2D, ParametricDataset3D
        from src.evaluation.configuration import load_configuration
        from scripts.evaluate.evaluation_common import run_evaluation

        temporary_root = Path.home() / ".hmonet_test_tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="pih_", dir=temporary_root) as temporary:
            root = Path(temporary)
            for dim, mapping in (
                (2, "current"),
                (2, "coons_residual"),
                (3, "current"),
                (3, "coons_residual"),
            ):
                shape = (4,) * dim
                name = f"physics_{dim}d_{mapping}"
                if dim == 2:
                    out = generate_explicit_dataset(
                        name, "rectangle", [{}], grid_shape=shape, output_root=root
                    )
                else:
                    out = generate_dataset(
                        name, "box", [{}], grid_shape=shape, output_root=root
                    )
                cls = ParametricDataset2D if dim == 2 else ParametricDataset3D
                data = cls(out, load_boundary=True)
                config = dict(
                    model=dict(
                        common={**SMALL_MODEL, "trunk_dim": dim},
                        pihmonet=dict(hmonet_mapping_mode=mapping),
                    ),
                    training=dict(
                        num_epochs=2,
                        batch_size=1,
                        learning_rate=0.001,
                        tensorboard_enabled=False,
                        best_model_metric="dataset_mse",
                    ),
                    pihmonet_training=dict(
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
                    history = train_pihmonet(
                        [data], config, "interpolation", args, torch.device("cpu")
                    )
                self.assertTrue(np.isfinite(history[-1]["loss_pde"]))
                self.assertNotIn("loss_boundary", history[-1])
                final = history[-1]
                expected_loss = (
                    final["loss_pde_normalized"]
                    + 0.1 * final["jacobian"]
                    + final["anchor_weight"] * final["anchor"]
                )
                np.testing.assert_allclose(
                    final["train_loss"],
                    expected_loss,
                    rtol=1e-6,
                    atol=1e-12,
                )
                checkpoint = Path(args.checkpoint_output_dir) / "best.pt"
                self.assertIn("PI-HMONet", checkpoint.parts)
                self.assertIn("PI-HMONet", Path(args.training_output_dir).parts)
                model, _, _ = load_network(
                    checkpoint, "PI-HMONet", dim, mapping, torch.device("cpu")
                )
                saved = torch.load(checkpoint, weights_only=False)
                self.assertEqual(
                    saved["config"]["training"]["best_model_metric"], "training_loss"
                )
                self.assertTrue(
                    list(
                        (root / "configs" / f"{dim}D" / name / "PI-HMONet").glob(
                            "*/*/train.yaml"
                        )
                    )
                )
                # Deliberately corrupt interior labels: optimization must be identical.
                original_mesh = data.mesh_points.copy()
                data.mesh_points *= 7
                second_history = train_pihmonet(
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
                                save_plots=(mapping == "current"),
                                quality_metrics=["scaled_jacobian"],
                                methods=["tfi", "pihmonet"],
                                pihmonet_mapping_mode=mapping,
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
                        "--pihmonet-checkpoint",
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
                self.assertIn("PI-HMONet", result["timings"])
                self.assertIn("PI-HMONet", result["metrics"])
                raw_error = next(
                    v
                    for v in result["boundary_errors"]
                    if v["method"] == "PI-HMONet" and v["stage"] == "raw"
                )
                self.assertLess(raw_error["boundary_max_error"], 1e-6)
                self.assertTrue((Path(result["comparison_directory"])).is_dir())
                self.assertTrue(
                    any(
                        r["method"] == "PI-HMONet" and r["stage"] == "raw"
                        for r in result["boundary_errors"]
                    )
                )


if __name__ == "__main__":
    unittest.main()

