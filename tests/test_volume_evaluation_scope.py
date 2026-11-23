"""Volume work is independently optional; sections remain intrinsic 2-D metrics."""
from pathlib import Path
from unittest.mock import patch
import pytest
from src.evaluation.configuration import build_parser, evaluation_options
from src.utils.quality_selection import quality_settings
from tests.test_evaluation_pipeline import TestEvaluationPipeline as Pipeline


def test_default_scope_and_cli():
    parser = build_parser(3)
    options = evaluation_options({}, {}, parser.parse_args([]))
    assert options["middle_sections"] == ["z_mid"]
    assert quality_settings(options, 3)["volume_evaluation"] is True
    args = parser.parse_args(["--no-volume-evaluation", "--middle-sections", "x_mid", "z_mid"])
    settings = quality_settings(evaluation_options({}, {}, args), 3)
    assert settings["metrics"] == []
    assert settings["surface_metrics"]
    assert settings["middle_sections"] == ["x_mid", "z_mid"]


def test_formal_preflight_rejects_volume_flag_in_2d():
    from scripts.train.train_evaluate import validate_formal_evaluation_request
    with pytest.raises(ValueError, match="only available for 3-D"):
        validate_formal_evaluation_request(2, ["--no-volume-evaluation"], {}, [])


class TestVolumeScope:
    setup_method = Pipeline.setup_method
    teardown_method = Pipeline.teardown_method
    configuration = Pipeline.configuration
    run_evaluation = Pipeline.run_evaluation

    @pytest.mark.parametrize("sections", [("z_mid",), ("x_mid", "y_mid"), ("none",)])
    def test_no_volume_skips_expensive_evaluator_and_volume_figures(self, sections):
        config = self.configuration(3, ["tfi"], save_plots=True)
        with patch("src.evaluation.analysis.evaluate_hexahedral_grid_quality", side_effect=AssertionError("volume must not run")), patch("src.evaluation.analysis.plot_volumes", side_effect=AssertionError("volume must not plot")):
            result = self.run_evaluation(3, config, "--no-volume-evaluation", "--middle-sections", *sections)[0]
        group = Path(result["comparison_directory"])
        assert not (group / "volume_mesh_comparison.png").exists()
        assert result["metrics"] == {"TFI": {}}
        for section in ("x_mid", "y_mid", "z_mid"):
            assert (group / "boundary_faces" / section / "comparison_quality_data.npz").exists() == (section in sections)
