"""Canonical metric names and selection shared by evaluation and mesh utilities."""

QUAD_METRICS = (
    "max_angle",
    "min_angle",
    "jacobian",
    "scaled_jacobian",
    "skewness",
    "aspect_ratio",
    "orthogonality",
    "cell_area",
    "concave_vertex_count",
    "self_intersection",
    "intersecting_cell",
    "flipping_cell",
    "degenerate_cell",
)
HEX_METRICS = (
    "minimum_scaled_jacobian",
    "center_jacobian",
    "center_condition_number",
    "edge_aspect_ratio",
    "max_angle",
    "min_angle",
    "skewness",
    "orthogonality",
    "cell_volume",
    "flipped_cell",
    "degenerate_cell",
)
ALL_METRICS = tuple(dict.fromkeys((*QUAD_METRICS, *HEX_METRICS)))


def _normalize_plot_selection(selection):
    """Return a stable requested-quality-plot selection.

    Quality *calculation* and quality *visualization* intentionally use
    independent controls.  Evaluation always retains the numeric summaries;
    rendered metric maps can be very large and are therefore opt-in.
    """
    if selection is None:
        return ()
    if isinstance(selection, str):
        selection = [item.strip().lower() for item in selection.split(",") if item.strip()]
    return tuple(
        token.strip().lower()
        for item in selection
        for token in str(item).split(",")
        if token.strip()
    )


def metric_names(selection, available):
    """Resolve both/all, none, or explicit names; reject misspelled metrics."""
    if selection is None:
        return tuple(available)
    if isinstance(selection, str):
        selection = [
            item.strip().lower() for item in selection.split(",") if item.strip()
        ]
    if list(selection) in (["both"], ["all"]):
        return tuple(available)
    if list(selection) == ["none"]:
        return ()
    unknown = set(selection) - set(available)
    if unknown:
        raise ValueError(
            f"Unknown quality metrics {sorted(unknown)}; choose {list(available)}"
        )
    return tuple(name for name in available if name in selection)


def selected_quality_metrics(options, dimension=2, *, surface=False):
    """Apply the master switch, selection, and per-metric boolean gates.

    Three-dimensional evaluations accept both volume and middle-section names.
    A shared name (for example min_angle) controls both types of output.
    """
    available = QUAD_METRICS if dimension == 2 else ALL_METRICS
    selected = tuple(
        name
        for name in metric_names(options.get("quality_metrics", "both"), available)
        if name != "cell_area"
    )
    switches = options.get("quality_switches") or {}
    unknown = set(switches) - set(available)
    if unknown:
        raise ValueError(f"Unknown quality switches: {sorted(unknown)}")
    if any(type(value) is not bool for value in switches.values()):
        raise ValueError("Each quality_switches value must be true or false")
    if type(options.get("compute_grid_quality", True)) is not bool:
        raise ValueError("compute_grid_quality must be true or false")
    if not options.get("compute_grid_quality", True):
        return ()
    applicable = QUAD_METRICS if dimension == 2 or surface else HEX_METRICS
    return tuple(
        name for name in applicable if name in selected and switches.get(name, True)
    )


def selected_quality_plot_metrics(options, dimension=2, *, surface=False):
    """Resolve opt-in metric figures without changing numeric summaries.

    ``--quality-plots all`` requests every metric currently calculated for the
    applicable mesh type.  A named request is validated against the available
    metrics and intersected with the active numeric-quality selection; this
    keeps a deliberately disabled metric from silently reappearing as a plot.
    """
    requested = _normalize_plot_selection(options.get("quality_plot_metrics"))
    if not requested or requested == ("none",):
        return ()
    applicable = QUAD_METRICS if dimension == 2 or surface else HEX_METRICS
    unknown = set(requested) - {"all", "none"} - set(applicable)
    if unknown:
        raise ValueError(
            f"Unknown quality plot metrics {sorted(unknown)}; choose "
            f"'all' or {list(applicable)}"
        )
    if "all" in requested:
        if len(requested) != 1:
            raise ValueError("--quality-plots all cannot be combined with named metrics")
        return selected_quality_metrics(options, dimension, surface=surface)
    if "none" in requested:
        if len(requested) != 1:
            raise ValueError("--quality-plots none cannot be combined with named metrics")
        return ()
    calculated = set(selected_quality_metrics(options, dimension, surface=surface))
    return tuple(name for name in applicable if name in requested and name in calculated)


def quality_settings(options, dimension):
    """Resolved selections for metadata and analysis-cache invalidation."""
    return {
        "compute_grid_quality": options.get("compute_grid_quality", True),
        "metrics": list(selected_quality_metrics(options, dimension)),
        "surface_metrics": (
            list(selected_quality_metrics(options, dimension, surface=True))
            if dimension == 3
            else []
        ),
        "plot_metrics": list(selected_quality_plot_metrics(options, dimension)),
        "surface_plot_metrics": (
            list(selected_quality_plot_metrics(options, dimension, surface=True))
            if dimension == 3
            else []
        ),
    }
