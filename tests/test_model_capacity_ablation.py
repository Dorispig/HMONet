"""Network-capacity ablation configuration and parameter-count tests."""

from pathlib import Path
import unittest

import yaml

from src.models.model import HMONet2D, MeshONet2D


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


class ModelCapacityAblationTests(unittest.TestCase):
    def test_ex4_uses_the_compact_baseline(self):
        config = yaml.safe_load(
            (ROOT / "experiments/ex4/baseline/experiment.yaml").read_text(
                encoding="utf-8"
            )
        )
        options = dict(config["model"]["common"])
        model = HMONet2D(21, 20, **options)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 177770)

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
        self.assertEqual(counts, [552198, 177770, 80086, 40046, 15188])
        meshonet = MeshONet2D(21, 20)
        meshonet_count = sum(p.numel() for p in meshonet.parameters())
        self.assertLessEqual(counts[-1], round(meshonet_count * 0.8) + 2)


if __name__ == "__main__":
    unittest.main()
