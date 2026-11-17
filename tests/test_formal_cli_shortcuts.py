"""Regression tests for the compact formal training/evaluation CLI."""

from pathlib import Path

import pytest
import yaml

from src.engine.training import resolve_explicit_dataset_path
from src.evaluation.configuration import DEFAULTS, build_parser, compact_grid_shapes, csv_values
from src.utils.config_validation import validate_evaluate_config


def test_explicit_dataset_name_resolves_under_the_dimension_data_root(tmp_path):
    dataset = tmp_path / "data" / "raw" / "3D" / "swb_a_ext_1_500_11_21_11"
    dataset.mkdir(parents=True)

    assert resolve_explicit_dataset_path(
        "swb_a_ext_1_500_11_21_11", dimension=3, project_root=tmp_path
    ) == dataset.resolve()
    assert resolve_explicit_dataset_path(
        dataset, dimension=3, project_root=tmp_path
    ) == dataset.resolve()
    with pytest.raises(FileNotFoundError, match="data/raw/3D"):
        resolve_explicit_dataset_path("missing", dimension=3, project_root=tmp_path)


def test_compact_grid_and_value_forms_match_between_dimensions():
    args_2d = build_parser(2).parse_args(
        ["--case", "air_t_ext_1_500_21_21", "--grids", "101", "201", "--values", "0.62", "0.65"]
    )
    args_3d = build_parser(3).parse_args(
        ["--case", "swb_a_ext_1_500_11_21_11", "--grids", "21", "31", "--values", "0.02,0.34"]
    )

    assert compact_grid_shapes(args_2d.grids, 2) == [(101, 101), (201, 201)]
    assert compact_grid_shapes(args_3d.grids, 3) == [(21, 21, 21), (31, 31, 31)]
    assert csv_values(args_2d.values) == [0.62, 0.65]
    assert csv_values(args_3d.values) == [0.02, 0.34]


def test_formal_pipeline_accepts_legacy_comma_values():
    import scripts.train.train_evaluate as train_evaluate

    # The formal wrapper forwards values to the shared evaluator; comma
    # spelling remains valid while the recommended form uses spaces.
    parser = train_evaluate.argparse.ArgumentParser(add_help=False)
    parser.add_argument("--values", nargs="+")
    args = parser.parse_args(["--values", "0.62,0.65"])
    assert csv_values(args.values) == [0.62, 0.65]


def test_non_square_and_non_cubic_shapes_remain_available():
    assert compact_grid_shapes("101x81", 2) == [(101, 81)]
    assert compact_grid_shapes("11x31x11", 3) == [(11, 31, 11)]


def test_both_evaluation_yaml_schemas_validate_and_default_to_all_models():
    root = Path(__file__).resolve().parents[1]
    for dimension in (2, 3):
        config = yaml.safe_load(
            (root / f"configs/{dimension}D/evaluate.yaml").read_text(encoding="utf-8")
        )
        validate_evaluate_config(config, dimension=dimension)
    assert set(DEFAULTS["methods"]) == {
        "tfi",
        "pde",
        "meshonet",
        "pimeshonet",
        "hmonet",
        "qghmonet",
    }


def test_formal_preflight_validates_explicit_3d_shapes_and_values():
    from scripts.train.train_evaluate import validate_formal_evaluation_request

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/3D/evaluate.yaml").read_text(encoding="utf-8")
    )
    name = "swb_a_ext_1_500_11_21_11"
    records = validate_formal_evaluation_request(
        3,
        ["--grids", "11x21x11", "21x41x21", "--values", "0.02,0.34"],
        config["cases"],
        [name],
    )
    assert records == [
        {
            "name": name,
            "grids": ["11x21x11", "21x41x21"],
            "values": ["0.02", "0.34"],
        }
    ]
    with pytest.raises(ValueError, match="Invalid grid size"):
        validate_formal_evaluation_request(
            3, ["--grids", "not-a-grid"], config["cases"], [name]
        )
