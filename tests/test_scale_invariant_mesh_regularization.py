import torch

from scripts.train.physics_loss import (
    normalized_mapping_barriers,
    normalized_mapping_barriers_from_jacobian,
    scale_invariant_pde_loss,
    stencil_queries,
)
import pytest


class TestScaleInvariantMeshRegularization:
    def setup_method(self):
        self.steps = torch.tensor([0.01, 0.01], dtype=torch.float64)
        centers = torch.tensor([[0.3, 0.4], [0.6, 0.7]], dtype=torch.float64)
        self.queries = stencil_queries(centers, self.steps)

    def test_normalized_pde_is_uniform_scale_invariant(self):
        def mapping(scale):
            q = self.queries
            return scale * torch.stack(
                (
                    q[..., 0] + 0.2 * q[..., 0].square() + 0.05 * q[..., 0] * q[..., 1],
                    q[..., 1] + 0.15 * q[..., 1].square(),
                ),
                -1,
            )

        first = scale_invariant_pde_loss(mapping(1.0), self.steps)
        second = scale_invariant_pde_loss(mapping(3.0), self.steps)
        torch.testing.assert_close(first, second, rtol=1e-7, atol=1e-10)

    def test_valid_identity_mapping_has_zero_barriers(self):
        values = self.queries.unsqueeze(0)
        jacobian, singular = normalized_mapping_barriers(
            values, self.steps, torch.tensor([1.0], dtype=torch.float64)
        )
        assert (float(jacobian)) == (0.0)
        assert (float(singular)) == (0.0)

    def test_collapsed_mapping_activates_both_barriers(self):
        values = (0.001 * self.queries).unsqueeze(0)
        jacobian, singular = normalized_mapping_barriers(
            values, self.steps, torch.tensor([1.0], dtype=torch.float64)
        )
        assert (float(jacobian)) > (0.0)
        assert (float(singular)) > (0.0)

    def test_barriers_average_only_nonzero_violations(self):
        first = torch.stack(
            (
                torch.zeros(2, 2, dtype=torch.float64),
                torch.eye(2, dtype=torch.float64),
            )
        ).unsqueeze(0)
        jacobian, singular = normalized_mapping_barriers_from_jacobian(
            first,
            torch.tensor([1.0], dtype=torch.float64),
            jacobian_minimum=0.02,
            singular_value_minimum=0.05,
        )
        assert (float(jacobian)) == pytest.approx(1.0, abs=10**-12)
        assert (float(singular)) == pytest.approx(1.0, abs=10**-12)

    def test_barriers_can_average_over_all_samples(self):
        first = torch.stack(
            (
                torch.zeros(2, 2, dtype=torch.float64),
                torch.eye(2, dtype=torch.float64),
            )
        ).unsqueeze(0)
        jacobian, singular = normalized_mapping_barriers_from_jacobian(
            first,
            torch.tensor([1.0], dtype=torch.float64),
            jacobian_minimum=0.02,
            singular_value_minimum=0.05,
            reduction="all",
        )
        assert (float(jacobian)) == pytest.approx(0.5, abs=10**-12)
        assert (float(singular)) == pytest.approx(0.5, abs=10**-12)
