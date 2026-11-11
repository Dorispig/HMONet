"""Unit tests for opt-in quality-figure selection."""

import pytest

from src.utils.quality_selection import (
    QUAD_METRICS,
    quality_settings,
    selected_quality_plot_metrics,
)


def _options(**overrides):
    return {
        "compute_grid_quality": True,
        "quality_metrics": "both",
        "quality_switches": {},
        "quality_plot_metrics": [],
        **overrides,
    }


def test_quality_plots_are_disabled_by_default_without_disabling_summaries():
    settings = quality_settings(_options(), 2)
    assert settings["metrics"]
    assert settings["plot_metrics"] == []


def test_quality_plot_selection_is_limited_to_explicit_metrics():
    options = _options(quality_plot_metrics=["scaled_jacobian", "max_angle"])
    assert selected_quality_plot_metrics(options, 2) == (
        "max_angle",
        "scaled_jacobian",
    )


def test_quality_plot_all_expands_to_current_numeric_selection():
    options = _options(quality_plot_metrics=["all"])
    assert selected_quality_plot_metrics(options, 2) == tuple(
        metric for metric in QUAD_METRICS if metric != "cell_area"
    )


def test_quality_plot_metric_must_be_known():
    with pytest.raises(ValueError, match="Unknown quality plot metrics"):
        selected_quality_plot_metrics(_options(quality_plot_metrics=["wrong"]), 2)
