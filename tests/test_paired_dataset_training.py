"""Exercise both complete 2-D training routines on the paired dataset format."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_2d import train_hmonet, train_meshonet
from src.data.dataset import ParametricDataset2D
from src.data.generation.make_dataset_parametric_2d import generate_circle_airfoil_t_dataset
from src.models.model import HMONet2D

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
    "branch_mlp_hidden_dim": 4,
    "branch_mlp_hidden_layer_num": 1,
}


class PairedDatasetTrainingTests(unittest.TestCase):
    def test_both_trainers_save_finite_fixed_grid_mse_and_checkpoints(self):
        torch.manual_seed(17)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with contextlib.redirect_stdout(io.StringIO()):
                output = generate_circle_airfoil_t_dataset(
                    "training_test",
                    [(0.08, 0.12)],
                    count=2,
                    nx=7,
                    ny=5,
                    output_root=root / "data",
                )
            config = {
                "model": {"common": SMALL_MODEL},
                "training": {
                    "calculate_dataset_mse": True,
                    "tensorboard_enabled": False,
                    "batch_size": 2,
                    "num_epochs": 1,
                    "learning_rate": 0.001,
                },
                "hmonet_sampling": {"global_random_count": 8},
                "meshonet_training": {"interior_sample_count": 8},
            }
            args = SimpleNamespace(
                epochs=1,
                batch_size=2,
                output_root=str(root / "training"),
                checkpoint_root=str(root / "checkpoints"),
                dataset_name="training_test",
                mode="interpolation",
            )
            for name, train in (("hmonet", train_hmonet), ("meshonet", train_meshonet)):
                dataset = ParametricDataset2D(output, load_boundary=(name == "hmonet"))
                with contextlib.redirect_stdout(io.StringIO()):
                    history = train(
                        [dataset], config, "interpolation", args, torch.device("cpu")
                    )
                self.assertEqual(len(history), 1)
                self.assertTrue(np.isfinite(history[0]["train_loss"]))
                summary_path = Path(args.training_output_dir) / "training_summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                self.assertTrue(np.isfinite(summary["final_dataset_mse"]))
                checkpoint_dir = Path(summary["checkpoint_dir"])
                for filename in (
                    "best.pt",
                    "best_model.pth",
                    "final.pt",
                    "final_model.pth",
                ):
                    self.assertTrue((checkpoint_dir / filename).is_file())

            geometry = ParametricDataset2D(output)[0]["boundary"]
            points = torch.as_tensor(dataset.boundary_points[:1])
            query = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
            model = HMONet2D(7, 5, **SMALL_MODEL)
            prediction = torch.cat(
                model(
                    points[..., 0],
                    points[..., 1],
                    query[..., 0],
                    query[..., 1],
                    boundary_geometry=geometry,
                ),
                dim=-1,
            )
            expected = np.stack(
                [
                    geometry.bottom.evaluate(0),
                    geometry.bottom.evaluate(1),
                    geometry.top.evaluate(0),
                    geometry.top.evaluate(1),
                ]
            )
            np.testing.assert_allclose(
                prediction.detach().numpy()[0], expected, atol=1e-7
            )
            prediction.square().sum().backward()
            self.assertTrue(
                all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                )
            )


if __name__ == "__main__":
    unittest.main()
