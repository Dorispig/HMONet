"""Evaluation-path tests for the compact experiment layout."""

from pathlib import Path
from scripts.experiments.evaluation_paths import (
    EVALUATION_PATH_LIMIT,
    evaluation_collection_root,
    evaluation_result_root,
    evaluation_staging_root,
    staging_experiment_name,
)


def test_result_layout_contains_no_hash_layer():
    dataset_root = Path("C:/hmonet_test/ex3/e8_full_factorized/run_ext_1_500_41_41")
    result = evaluation_result_root(
        dataset_root, "e8_full_factorized", 20261030, "physics_v1", "101x101", 1.52
    )
    relative = result.relative_to(evaluation_collection_root(dataset_root))
    assert relative.parts == (
        "seed_20261030", "physics_v1", "e8", "101x101", "1.52"
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


def test_staging_layout_uses_short_alias_and_task_identity():
    root = Path("C:/hmonet_test/experiments/ex6")
    first = evaluation_staging_root(
        root, "air_t_ext_1_500_21_21", "e1_6_fd0_008", 20261030, "v1", "101x101"
    )
    second = evaluation_staging_root(
        root, "air_t_ext_1_500_21_21", "e1_7_fd0_02", 20261030, "v1", "101x101"
    )
    other_seed = evaluation_staging_root(
        root, "air_t_ext_1_500_21_21", "e1_6_fd0_008", 20260917, "v1", "101x101"
    )
    assert len({first, second, other_seed}) == 3
    assert first.parts[-6:] == (
        ".eval_work", "air_t_ext_1_500_21_21", "e1_6",
        "seed_20261030", "v1", "101x101",
    )
    assert ".eval_work" in first.parts
    assert len(str(first.resolve())) + 1 + 96 <= EVALUATION_PATH_LIMIT


def test_staging_alias_honours_declared_alias_and_checkpoint_suffix(tmp_path):
    root = tmp_path / "experiments" / "ex1"
    definition = root / "rational_affine" / "experiment.yaml"
    definition.parent.mkdir(parents=True)
    definition.write_text("cli_alias: e1_1\n", encoding="utf-8")
    assert staging_experiment_name(root, "rational_affine") == "e1_1"
    assert staging_experiment_name(root, "rational_affine__phase2") == "e1_1__phase2"
