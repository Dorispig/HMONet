"""Training-dataset size ablation tests."""

from pathlib import Path
import unittest

import numpy as np
import yaml

from src.data.dataset import uniformly_spaced_subset


ROOT = Path(__file__).resolve().parents[1]


class _Dataset:
    def __init__(self, size, load_boundary=True):
        self.root = Path("memory_dataset")
        self.grid_shape = (2, 2)
        self.load_boundary = load_boundary
        self.boundary_points = np.arange(size * 4).reshape(size, 2, 2)
        self.mesh_points = np.arange(size * 8).reshape(size, 4, 2)
        if load_boundary:
            self.boundary = list(range(size))

    def __len__(self):
        return len(self.boundary_points)


class DatasetSizeAblationTests(unittest.TestCase):
    def test_uniform_subset_spans_full_ordered_dataset(self):
        source = _Dataset(1000)
        subset = uniformly_spaced_subset(source, 250)
        self.assertEqual(len(subset), 250)
        self.assertEqual(subset.subset_indices[0], 0)
        self.assertEqual(subset.subset_indices[-1], 999)
        self.assertEqual(len(np.unique(subset.subset_indices)), 250)
        gaps = np.diff(subset.subset_indices)
        self.assertLessEqual(int(gaps.max() - gaps.min()), 1)
        self.assertEqual(subset.boundary[0], 0)
        self.assertEqual(subset.boundary[-1], 999)

    def test_ex6_defines_requested_dataset_sizes(self):
        definition = yaml.safe_load(
            (ROOT / "experiments/ex6/dataset_size/experiment.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            [variant["value"] for variant in definition["variants"]],
            [1000, 800, 500, 250],
        )

    def test_invalid_subset_sizes_are_rejected(self):
        source = _Dataset(10, load_boundary=False)
        for size in (0, 11):
            with self.subTest(size=size), self.assertRaises(ValueError):
                uniformly_spaced_subset(source, size)


if __name__ == "__main__":
    unittest.main()
