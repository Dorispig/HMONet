"""Network-capacity ablation configuration and parameter-count tests."""

from pathlib import Path

import yaml
from unittest.mock import patch

from scripts.experiments.run_ablation_2d import _variant_options
from scripts.train import train_2d
from src.models.model import MeshONet2D
from src.models.pi_meshonet import PIMeshONet2D

ROOT = Path(__file__).resolve().parents[1]


class TestModelCapacityAblation:
    def test_architecture_suite_comparison_models_are_about_200k(self):
        baseline = yaml.safe_load(
            (ROOT / "experiments/ex3/baseline/experiment.yaml").read_text(
                encoding="utf-8"
            )
        )
        cases = (
            ("e9_meshonet_200k", "meshonet", MeshONet2D, 208242),
            ("e10_pimeshonet_200k", "pimeshonet", PIMeshONet2D, 191122),
        )
        for experiment, model_name, model_class, expected in cases:
            definition = yaml.safe_load(
                (ROOT / f"experiments/ex3/{experiment}/experiment.yaml").read_text(
                    encoding="utf-8"
                )
            )
            overrides = _variant_options(
                tuple(map(str, definition["variants"][0]["overrides"]))
            )
            overrides.pop("model")
            for key in ("lr", "best_model_metric", "plateau_metric"):
                overrides.pop(key, None)
            options = {
                **baseline["model"]["common"],
                **baseline["model"][model_name],
                **{key: int(value) for key, value in overrides.items()},
            }
            options.pop("architecture", None)
            model = model_class(41, 41, **options)
            assert (sum(parameter.numel() for parameter in model.parameters())) == (
                expected
            )

    def test_cli_capacity_overrides_selected_model_section(self):
        with patch.object(train_2d, "train_models", return_value=[]) as dispatch:
            train_2d.main(
                [
                    "--config",
                    str(ROOT / "experiments/ex3/baseline/experiment.yaml"),
                    "--dataset",
                    str(ROOT),
                    "--model",
                    "meshonet",
                    "--device",
                    "cpu",
                    "--branch-dim",
                    "40",
                    "--adjust-hidden-dim",
                    "80",
                    "--branch-hidden-dim",
                    "80",
                    "--trunk-hidden-dim",
                    "80",
                    "--output-dim",
                    "80",
                ]
            )
        model_config = dispatch.call_args.args[2]["model"]["meshonet"]
        assert (model_config["branch_dim"]) == (40)
        for key in (
            "adjust_hidden_dim",
            "branch_hidden_dim",
            "trunk_hidden_dim",
            "output_dim",
        ):
            assert (model_config[key]) == (80)

