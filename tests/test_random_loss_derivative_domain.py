from pathlib import Path

import torch
import yaml

from scripts.experiments.run_ablation_2d import (
    _load_yaml_mapping,
    variant_training_overrides,
    variants_from_definition,
)
from scripts.train.train_2d import _mixed_random_loss_points
from src.utils.config import load_yaml_mapping
from src.numerics.differential import (
    boundary_aware_finite_difference_derivatives,
    boundary_aware_stencil_queries,
)


ROOT = Path(__file__).resolve().parents[1]


def _sample(*, keep_inside):
    torch.manual_seed(20261106)
    return _mixed_random_loss_points(
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        derivative_backend="finite_difference",
        steps=torch.tensor([0.1, 0.1]),
        global_count=2000,
        curvature_count=0,
        boundary_band_count=0,
        boundary_curvature=None,
        computational_boundary=None,
        candidate_multiplier=10,
        curvature_power=1.0,
        sigma_min=0.02,
        sigma_max=0.10,
        uniform_probability=0.05,
        boundary_min_step_multiple=1.0,
        boundary_max_step_multiple=4.0,
        boundary_min_distance=0.0,
        boundary_max_distance=4.0e-4,
        fd_stencil_inside=keep_inside,
    )


def test_finite_difference_baseline_keeps_stencil_inside_domain():
    points = _sample(keep_inside=True)
    assert torch.all(points >= 0.1)
    assert torch.all(points <= 0.9)


def test_full_open_domain_allows_finite_difference_queries_to_cross_boundary():
    points = _sample(keep_inside=False)
    assert torch.all(points > 0.0)
    assert torch.all(points < 1.0)
    assert torch.any((points < 0.1) | (points > 0.9))


def test_formal_defaults_and_derivative_baseline_use_200_global_plus_200_adaptive():
    formal = yaml.safe_load((ROOT / "configs/2D/train.yaml").read_text(encoding="utf-8"))
    baseline = load_yaml_mapping(ROOT / "experiments/ex4/baseline/experiment.yaml")
    for config in (formal, baseline):
        assert config["model"]["hmonet"]["hmonet_mapping_mode"] == "corner_corrected_affine"
        assert config["hmonet_sampling"]["mode"] == "adaptive"
        assert config["hmonet_sampling"]["global_random_count"] == 200
        assert config["hmonet_sampling"]["curvature_adaptive_count"] == 200
    training = baseline["hmonet_training"]
    assert training["random_loss_global_sample_count"] == 200
    assert training["random_loss_curvature_sample_count"] == 200
    assert training["random_loss_boundary_band_sample_count"] == 0
    assert training["random_loss_fd_stencil_inside"] is True


def test_derivative_variants_change_only_the_requested_setting():
    e1 = variants_from_definition(
        _load_yaml_mapping(
            ROOT / "experiments/ex4/e1_fd_full_open_domain/experiment.yaml"
        )
    )[0]
    e2 = variants_from_definition(
        _load_yaml_mapping(ROOT / "experiments/ex4/e2_autograd/experiment.yaml")
    )[0]
    e3 = variants_from_definition(
        _load_yaml_mapping(
            ROOT / "experiments/ex4/e3_fd_one_sided_boundary/experiment.yaml"
        )
    )[0]
    assert variant_training_overrides(e1) == [
        "--no-hmonet-random-loss-fd-stencil-inside"
    ]
    assert variant_training_overrides(e2) == [
        "--hmonet-random-loss-derivative-backend",
        "autograd",
    ]
    assert variant_training_overrides(e3) == [
        "--no-hmonet-random-loss-fd-stencil-inside",
        "--hmonet-random-loss-fd-boundary-scheme",
        "one_sided",
    ]


def test_boundary_aware_stencil_stays_inside_and_is_exact_for_quadratics():
    centers = torch.tensor(
        [[[0.003, 0.40], [0.997, 0.60], [0.40, 0.50]]], dtype=torch.float64
    )
    steps = torch.tensor([0.01, 0.01], dtype=torch.float64)
    queries = boundary_aware_stencil_queries(centers, steps)
    assert torch.all(queries >= 0.0)
    assert torch.all(queries <= 1.0)

    xi, eta = queries[..., 0], queries[..., 1]
    values = torch.stack((xi.square() + 3 * xi * eta, eta.square()), -1)
    first, second, mixed = boundary_aware_finite_difference_derivatives(
        values, centers, steps
    )
    expected_first = torch.stack(
        (
            torch.stack((2 * centers[..., 0] + 3 * centers[..., 1],
                         torch.zeros_like(centers[..., 0])), -1),
            torch.stack((3 * centers[..., 0], 2 * centers[..., 1]), -1),
        ), -2,
    )
    expected_second = torch.tensor(
        [2.0, 0.0, 0.0, 2.0], dtype=torch.float64
    ).reshape(1, 1, 2, 2).expand_as(second)
    torch.testing.assert_close(first, expected_first, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(second, expected_second, rtol=1e-10, atol=1e-10)
    expected_mixed = torch.tensor([3.0, 0.0], dtype=torch.float64).reshape(1, 1, 2)
    torch.testing.assert_close(
        mixed[0], expected_mixed.expand_as(mixed[0]), rtol=1e-10, atol=1e-10
    )
