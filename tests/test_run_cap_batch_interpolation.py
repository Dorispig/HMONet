"""Batched run_e interpolation must match the previous per-geometry formula."""
import numpy as np
import pytest
import torch

from src.data.geometry.pde_cap_3d import (
    RunPdeCapMap, _batch_spline_residual, _cubic_basis,
)


def loop_residual(caps, u, v):
    """Frozen pre-optimization formula, retained only as a test reference."""
    result = []
    for index, cap in enumerate(caps):
        tu, tv, coefficients = cap.tensors(u)
        iu, bu = _cubic_basis(u[index], tu)
        iv, bv = _cubic_basis(v[index], tv)
        local = coefficients[iu[..., :, None], iv[..., None, :], :]
        result.append((local * bu[..., :, None, None] * bv[..., None, :, None]).sum((-3, -2)))
    return torch.stack(result)


def synthetic_caps(heterogeneous):
    rng = np.random.default_rng(19)
    caps = []
    for index in range(4):
        n = 13 if heterogeneous and index % 2 else 11
        # Include equal-size, unequal-knot cases as well as unequal resolutions.
        exponent = 1.2 if heterogeneous and index == 2 else 1.
        tu = np.r_[np.zeros(4), np.linspace(0, 1, n-2)[1:-1] ** exponent, np.ones(4)]
        tv = np.r_[np.zeros(4), np.linspace(0, 1, 10)[1:-1], np.ones(4)]
        coefficients = rng.normal(size=(len(tu)-4, len(tv)-4, 2)) * .01
        coefficients[[0, -1]] = 0
        coefficients[:, [0, -1]] = 0
        caps.append(RunPdeCapMap(
            dict(width=3., height=2., notch_width=.9, depth=1., radius=.25),
            tu, tv, coefficients, {},
        ))
    return caps


@pytest.mark.parametrize("heterogeneous", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_values_and_two_query_derivatives_match_loop(heterogeneous, dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    caps = synthetic_caps(heterogeneous)
    rng = np.random.default_rng(23)
    query = torch.tensor(rng.uniform(.01, .99, (4, 79, 2)), device=device, dtype=dtype)
    query[:, :4] = query.new_tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
    collected = []
    for implementation in (loop_residual, _batch_spline_residual):
        q = query.detach().clone().requires_grad_()
        value = implementation(caps, *q.unbind(-1))
        first = torch.autograd.grad(value.square().sum(), q, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), q)[0]
        collected.append((value, first, second))
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    for expected, actual in zip(*collected):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    # The four exact edges of the residual remain zero.
    assert torch.count_nonzero(collected[1][0][:, :4]) == 0


def test_batch64_distinct_coefficients_and_order():
    source = synthetic_caps(False)
    caps = []
    for index in range(64):
        base = source[(index * 7) % 4]
        caps.append(RunPdeCapMap(base.parameters, base.knots_u, base.knots_v,
                                base.coefficients * (1 + .01*index), {}))
    q = torch.rand(64, 37, 2, dtype=torch.float64)
    torch.testing.assert_close(_batch_spline_residual(caps, *q.unbind(-1)),
                               loop_residual(caps, *q.unbind(-1)), atol=2e-14, rtol=2e-14)
