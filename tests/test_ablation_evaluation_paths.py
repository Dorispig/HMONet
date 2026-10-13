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
            longest = result / ("x" * 128)
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
            longest = result / ("x" * 120)
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
            self.assertEqual(result.relative_to(collection).parts[0], "1")


if __name__ == "__main__":
    unittest.main()
