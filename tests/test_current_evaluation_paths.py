"""Evaluation-path tests for the compact experiment layout."""

from pathlib import Path
from scripts.experiments.evaluation_paths import (
    EVALUATION_PATH_LIMIT,
    evaluation_collection_root,
    evaluation_result_root,
)


def test_result_layout_contains_no_hash_layer():
    dataset_root = Path("C:/hmonet_test/ex3/e8_full_factorized/run_ext_1_500_41_41")
    result = evaluation_result_root(
        dataset_root, "e8_full_factorized", 20261030, "physics_v1", "101x101", 1.52
    )
    relative = result.relative_to(evaluation_collection_root(dataset_root))
    assert relative.parts == (
        "seed_20261030", "physics_v1", "e8_full_factorized", "101x101", "1.52"
    )
    assert all(
        len(part) != 6 or not all(c in "0123456789abcdef" for c in part)
        for part in relative.parts
    )


def test_longest_retained_variant_fits_windows_path_budget():
    root = Path(__file__).resolve().parents[1]
    dataset_root = root / "experiments/ex3/e7_modified_cross_boundary/run_ext_1_500_41_41"
    result = evaluation_result_root(
        dataset_root, "e7_modified_cross_boundary", 20261030,
        "physics_v1", "401x401", 1.55,
    )
    longest = result / "p/pi/p/q/q_sj.png"
    assert len(str(longest.resolve())) <= EVALUATION_PATH_LIMIT
