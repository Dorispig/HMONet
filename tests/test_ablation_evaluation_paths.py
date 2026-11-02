import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.experiments.evaluation_paths import (
    EVALUATION_PATH_LIMIT,
    evaluation_collection_root,
    evaluation_result_root,
    evaluation_staging_root,
    validate_evaluation_destination,
)

TEST_TEMP_ROOT = Path.home() / ".hmonet_test_tmp"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


class AblationEvaluationPathTests(unittest.TestCase):
    def test_known_long_variants_stay_within_windows_budget(self):
        experiments = Path(__file__).resolve().parents[1] / "experiments"
        cases = (
            ("sampling_mode", "adaptive"),
            ("target_interpolation", "bilinear"),
            ("training_sample_count", "samples_1200"),
        )
        for ablation, variant in cases:
            dataset_root = experiments / ablation / "air_t_int_0_1000_41_20"
            result = evaluation_result_root(
                dataset_root,
                variant,
                20261031,
                "paper_v1",
                "101x101",
                0.325,
            )
            longest = result / "plots/PI-MeshONet/processed/quality/quality_scaled_jacobian.png"
            self.assertLessEqual(len(str(longest.resolve())), EVALUATION_PATH_LIMIT)

    def test_all_experiment_suite_families_fit_the_path_budget(self):
        experiments = Path(__file__).resolve().parents[1] / "experiments"
        cases = (
            ("ex1/point_weighting", "position_weighted"),
            ("ex2/boundary_self_attention", "attention_on"),
            ("ex2/trunk_geometry_coefficients", "trunk_on"),
            ("ex3/jacobian_interpolation", "jacobian_curved"),
            ("ex4/jacobian_interpolation", "jacobian_curved"),
            ("ex5/model_capacity", "compact_178k"),
            ("ex6/dataset_size", "samples_1000"),
        )
        for experiment, variant in cases:
            dataset_root = (
                experiments
                / experiment
                / "smooth_opposing_depression_ext_0_1000_61_21"
            )
            result = evaluation_result_root(
                dataset_root,
                variant,
                20261031,
                "paper_v1",
                "201x201",
                0.325,
            )
            longest = result / "plots/PI-MeshONet/processed/quality/quality_scaled_jacobian.png"
            self.assertLessEqual(len(str(longest.resolve())), EVALUATION_PATH_LIMIT)

    def test_staging_is_one_deterministic_suite_directory(self):
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            experiment = Path(temporary) / "experiments/target_interpolation"
            staging = evaluation_staging_root(
                experiment,
                "air_t_int_0_1000_41_20",
                "curved_boundary_cell",
                20260802,
                "paper_v1",
                "101x101",
            )
            self.assertEqual(staging, experiment.parent / ".eval_work")
            self.assertNotIn("air_t_int_0_1000_41_20", staging.parts)
            self.assertNotRegex(staging.name, r"^[0-9a-f]{12}$")

    def test_current_dataset_name_is_preserved_in_formal_results(self):
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            dataset = "air_t_ext_0_1000_41_20"
            dataset_root = (
                Path(temporary) / "experiments/ex3/jacobian_interpolation" / dataset
            )
            collection = evaluation_collection_root(dataset_root)
            self.assertIn(dataset, collection.parts)
            self.assertEqual(
                collection,
                dataset_root.resolve().parents[1] / "eval" / dataset,
            )

    def test_physics_strategy_uses_readable_experiment_and_run_names(self):
        experiments = Path(__file__).resolve().parents[1] / "experiments"
        dataset_root = (
            experiments
            / "ex8/physics_training_strategy/run_ext_0_1000_61_21"
        )
        result = evaluation_result_root(
            dataset_root,
            "e2_nojac_then_mse_pde__phase1",
            20261030,
            "physics_v1",
            "101x101",
            1.52,
        )
        relative = result.relative_to(experiments / "ex8")
        self.assertEqual(relative.parts[:4], (
            "eval", "run_ext_0_1000_61_21", "seed_20261030", "physics_v1"
        ))
        self.assertEqual(relative.parts[4], "e2_nojac_then_mse_pde__phase1")
        self.assertNotRegex(str(relative), r"[0-9a-f]{6}$")

    def test_destination_validation_checks_actual_nested_artifacts(self):
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            source = root / "source"
            nested = source / "comparisons/grid_quality"
            nested.mkdir(parents=True)
            (nested / "quality.json").write_text("{}", encoding="utf-8")
            target = root / "experiments/eval/dataset/jac/on/1/paper/51x51/0.5"
            validate_evaluation_destination(source, target)

    def test_baseline_collection_omits_duplicate_variant_folder(self):
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            dataset_root = Path(temporary) / "experiments/baseline/dataset"
            collection = evaluation_collection_root(dataset_root)
            result = evaluation_result_root(
                dataset_root, "baseline", 1, "paper", "51x51", 0.5
            )
            self.assertEqual(
                collection,
                dataset_root.resolve().parents[1] / "eval" / "dataset",
            )
            self.assertEqual(
                result.relative_to(collection).parts[:3],
                ("seed_1", "paper", "baseline"),
            )

    def test_single_variant_suite_omits_duplicate_family_folder(self):
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            experiments = Path(temporary) / "experiments"
            dataset_root = (
                experiments
                / "ex9/e9_norm_pde_jac_svd/run_ext_0_1000_61_21"
            )
            collection = evaluation_collection_root(dataset_root)
            result = evaluation_result_root(
                dataset_root,
                "e9_norm_pde_jac_svd",
                20261030,
                "physics_v1",
                "101x101",
                1.55,
            )
            relative = result.relative_to(collection)
            self.assertEqual(
                collection,
                experiments / "ex9/eval/run_ext_0_1000_61_21",
            )
            self.assertEqual(
                relative.parts[:3],
                ("seed_20261030", "physics_v1", "e9_norm_pde_jac_svd"),
            )
            self.assertNotEqual(relative.parts[:2], ("e9", "e9"))

    def test_long_public_names_are_not_replaced_by_hash_suffixes(self):
        with TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            dataset_root = (
                Path(temporary)
                / "experiments/ex11/e8_full_factorized/run_ext_1_500_41_41"
            )
            result = evaluation_result_root(
                dataset_root,
                "e8_full_factorized",
                20261030,
                "physics_v1",
                "101x101",
                1.22,
            )
            collection = evaluation_collection_root(dataset_root)
            relative = result.relative_to(collection)
            self.assertEqual(
                relative.parts,
                (
                    "seed_20261030",
                    "physics_v1",
                    "e8_full_factorized",
                    "101x101",
                    "1.22",
                ),
            )
            self.assertNotRegex(str(relative), r"-[0-9a-f]{6}(?:\\|$)")


if __name__ == "__main__":
    unittest.main()
