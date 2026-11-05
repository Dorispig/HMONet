"""Network-capacity ablation configuration and parameter-count tests."""

from pathlib import Path

import yaml
from unittest.mock import patch

from scripts.experiments.run_ablation_2d import _variant_options
from scripts.train import train_2d
from src.models.model import HMONet2D, MeshONet2D
from src.models.pi_meshonet import PIMeshONet2D

ROOT = Path(__file__).resolve().parents[1]


def _options(overrides):
    result = {}
    index = 0
    while index < len(overrides):
        flag = overrides[index]
        if flag.startswith("--no-"):
            result[flag[5:].replace("-", "_")] = False
            index += 1
        elif flag.endswith("-enabled"):
            result[flag[2:].replace("-", "_")] = True
            index += 1
        else:
            result[flag[2:].replace("-", "_")] = int(overrides[index + 1])
            index += 2
    return result


class TestModelCapacityAblation:
    def test_ex11_comparison_models_are_about_200k(self):
        baseline = yaml.safe_load(
            (ROOT / "experiments/ex11/baseline/experiment.yaml").read_text(
                encoding="utf-8"
            )
        )
        cases = (
            ("e9_meshonet_200k", "meshonet", MeshONet2D, 208242),
            ("e10_pimeshonet_200k", "pimeshonet", PIMeshONet2D, 191122),
        )
        for experiment, model_name, model_class, expected in cases:
            definition = yaml.safe_load(
                (ROOT / f"experiments/ex11/{experiment}/experiment.yaml").read_text(
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
                    str(ROOT / "experiments/ex11/baseline/experiment.yaml"),
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

    def test_ex4_uses_the_compact_baseline(self):
        config = yaml.safe_load(
            (ROOT / "experiments/ex4/baseline/experiment.yaml").read_text(
                encoding="utf-8"
            )
        )
        options = dict(config["model"]["common"])
        model = HMONet2D(21, 20, **options)
        assert (sum(p.numel() for p in model.parameters())) == (157450)

    def test_ex5_capacity_ladder_reaches_below_meshonet(self):
        baseline = yaml.safe_load(
            (ROOT / "experiments/ex5/baseline/experiment.yaml").read_text(
                encoding="utf-8"
            )
        )
        definition = yaml.safe_load(
            (ROOT / "experiments/ex5/model_capacity/experiment.yaml").read_text(
                encoding="utf-8"
            )
        )
        counts = []
        for variant in definition["variants"]:
            options = dict(baseline["model"]["common"])
            options.update(_options(variant["overrides"]))
            model = HMONet2D(21, 20, **options)
            counts.append(sum(p.numel() for p in model.parameters()))
        # Counts reflect the current trunk-modulated geometry head; historical
        # labels in the YAML are descriptive names, not fingerprint inputs.
        assert (counts) == ([190926, 92506, 43634, 18776])
        meshonet = MeshONet2D(21, 20)
        meshonet_count = sum(p.numel() for p in meshonet.parameters())
        assert (counts[-1]) <= (round(meshonet_count * 0.8) + 2)
