"""Direct evaluation has no optimization setup and preserves batch semantics."""

import unittest
from unittest.mock import patch
import numpy as np
import torch
from tests.test_inference_geometry import boundary, small_model
from src.data.mesh.sampling import logical_grid
from src.data.geometry.parametric_curve_2d import BoundaryCurveSet2D
from src.evaluation.inference import predict, predict_batch
from src.evaluation.configuration import load_configuration, evaluation_options


class EvaluationOptimizationTests(unittest.TestCase):
    def test_batched_predictions_equal_individual_results(self):
        options = dict(
            query_batch_size=11, inference_warmup_runs=1, inference_timing_repeats=2
        )
        for dimension in (2, 3):
            for mapping in ("current", "coons_residual"):
                model, shape = small_model(dimension, mapping)
                geometries = [boundary(dimension, t) for t in (0.2, 0.32)]
                q = logical_grid((4,) * dimension)
                actual, timing = predict_batch(
                    model, "HMONet", geometries, shape, q, torch.device("cpu"), options
                )
                for i, geometry in enumerate(geometries):
                    expected, _ = predict(
                        model,
                        "HMONet",
                        geometry,
                        shape,
                        q,
                        torch.device("cpu"),
                        options,
                    )
                    np.testing.assert_allclose(
                        actual[i], expected, atol=2e-6, rtol=1e-5
                    )
                self.assertEqual(timing["inference_batch_size"], 2)
                self.assertEqual(
                    timing["inference_time_basis"], "per_item_batch_average"
                )
                self.assertAlmostEqual(
                    timing["batch_inference_seconds"],
                    2 * timing["network_inference_seconds"],
                )

    def test_no_preparation_or_graph_and_exact_warmup_count(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device_name in devices:
            device = torch.device(device_name)
            for dimension in (2, 3):
                for mapping in ("current", "coons_residual"):
                    model, shape = small_model(dimension, mapping)
                    model.to(device)
                    geometry = boundary(dimension, 0.3)
                    query = logical_grid((3,) * dimension)
                    options = dict(
                        query_batch_size=1000,
                        inference_warmup_runs=0,
                        inference_timing_repeats=2,
                        curve_kernel_backend="auto",
                    )
                    with patch.object(
                        BoundaryCurveSet2D,
                        "batch",
                        side_effect=AssertionError("no coefficient setup"),
                    ), patch(
                        "torch.cuda.CUDAGraph",
                        side_effect=AssertionError("no graph setup"),
                    ), patch.object(
                        model, "forward", wraps=model.forward
                    ) as forward:
                        actual, timing = predict(
                            model, "HMONet", geometry, shape, query, device, options
                        )
                    self.assertEqual(forward.call_count, 2)
                    self.assertTrue(np.isfinite(actual).all())
                    self.assertEqual(timing["effective_inference_warmup_runs"], 0)
                    self.assertFalse(
                        any(
                            "setup" in k or "graph" in k or "preparation" in k
                            for k in timing
                        )
                    )

    def test_updated_configs_have_no_setup_options(self):
        for dimension in (2, 3):
            args, config = load_configuration(dimension, [])
            options = evaluation_options(config, {}, args)
            self.assertNotIn("hmonet_geometry_mode", options)
            self.assertNotIn("inference_cuda_graph", options)


if __name__ == "__main__":
    unittest.main()
