"""Regression checks for direct geometry projection in training and inference."""

import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.data.geometry.boundary_generator_2d import circle_airfoil_boundary
from src.data.geometry.boundary_generator_3d import cylindrical_domain_with_airfoil_hole
from src.data.generation.make_dataset_parametric_2d import to_computational_boundary
from src.data.mesh.sampling import logical_grid, sample_boundary
from src.evaluation.inference import predict
from src.models.model import HMONet2D, HMONet3D


def boundary(dimension, thickness):
    if dimension == 2:
        return to_computational_boundary(circle_airfoil_boundary(t=thickness))
    return cylindrical_domain_with_airfoil_hole(thickness=thickness, z_min=1.3)


def small_model(dimension, mapping):
    shape = (7, 6) if dimension == 2 else (5, 4, 3)
    cls = HMONet2D if dimension == 2 else HMONet3D
    torch.manual_seed(17)
    model = cls(
        *shape,
        branch_dim=4,
        trunk_dim=dimension,
        adjust_hidden_dim=4,
        branch_hidden_dim=4,
        trunk_hidden_dim=4,
        adjust_hidden_layer_num=1,
        branch_hidden_layer_num=1,
        trunk_hidden_layer_num=1,
        output_dim=4,
        branch_mlp_hidden_dim=4,
        branch_mlp_hidden_layer_num=1,
        hmonet_mapping_mode=mapping,
    ).eval()
    return model, shape


class InferenceGeometryTests(unittest.TestCase):
    def test_single_geometry_attention_hmonet_keeps_batch_axis(self):
        shape = (7, 6)
        model = HMONet2D(
            *shape,
            branch_dim=4,
            trunk_dim=2,
            adjust_hidden_dim=4,
            branch_hidden_dim=4,
            trunk_hidden_dim=4,
            adjust_hidden_layer_num=1,
            branch_hidden_layer_num=1,
            trunk_hidden_layer_num=1,
            output_dim=4,
            branch_mlp_hidden_dim=4,
            branch_mlp_hidden_layer_num=1,
            boundary_self_attention_enabled=True,
            attention_dim=4,
            attention_heads=1,
            attention_layers=1,
            attention_feedforward_dim=8,
        ).eval()
        geometry = boundary(2, 0.3)
        query = logical_grid((4, 4))
        actual, _ = predict(
            model,
            "HMONet",
            geometry,
            shape,
            query,
            torch.device("cpu"),
            dict(
                query_batch_size=11,
                inference_warmup_runs=0,
                inference_timing_repeats=1,
            ),
        )
        self.assertEqual(actual.shape, (len(query), 2))

    def test_each_warmup_and_measured_forward_evaluates_geometry(self):
        options = dict(
            query_batch_size=11, inference_warmup_runs=2, inference_timing_repeats=3
        )
        for dimension in (2, 3):
            for mapping in ("current", "coons_residual"):
                with self.subTest(dimension=dimension, mapping=mapping):
                    model, shape = small_model(dimension, mapping)
                    geometry = boundary(dimension, 0.3)
                    query = logical_grid((4,) * dimension)
                    # Count actual forwards as well as projections so preparation
                    # outside forward or duplicate Coons face evaluation is caught.
                    with patch.object(model, "project", wraps=model.project) as project:
                        with patch.object(
                            model, "forward", wraps=model.forward
                        ) as forward:
                            actual, timing = predict(
                                model,
                                "HMONet",
                                geometry,
                                shape,
                                query,
                                torch.device("cpu"),
                                options,
                            )
                        count = math.ceil(len(query) / options["query_batch_size"]) * 5
                        self.assertEqual(forward.call_count, count)
                        self.assertEqual(project.call_count, count)
                    with torch.inference_mode():
                        b = torch.tensor(
                            sample_boundary(geometry, shape), dtype=torch.float32
                        )
                        q = torch.tensor(query, dtype=torch.float32)
                        expected = torch.cat(
                            model(
                                *b.unbind(-1), *q.unbind(-1), boundary_geometry=geometry
                            ),
                            -1,
                        ).numpy()
                    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
                    self.assertGreater(timing["network_inference_seconds"], 0)
                    self.assertEqual(len(timing["inference_samples_seconds"]), 3)
                    self.assertFalse(any("precompute" in key for key in timing))

    def test_batched_training_with_exact_geometry_and_linear_fallback(self):
        for dimension in (2, 3):
            for mapping in ("current", "coons_residual"):
                model, shape = small_model(dimension, mapping)
                model.train()
                geometries = [boundary(dimension, t) for t in (0.2, 0.32)]
                b = torch.tensor(
                    np.stack([sample_boundary(g, shape) for g in geometries]),
                    dtype=torch.float32,
                )
                q = torch.rand(2, 8, dimension)
                for geometry in (geometries, None):
                    with self.subTest(
                        dimension=dimension, mapping=mapping, exact=geometry is not None
                    ):
                        model.zero_grad(set_to_none=True)
                        outputs = model(
                            *b.unbind(-1), *q.unbind(-1), boundary_geometry=geometry
                        )
                        sum(x.square().mean() for x in outputs).backward()
                        gradients = [
                            p.grad for p in model.parameters() if p.grad is not None
                        ]
                        self.assertTrue(gradients)
                        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))


if __name__ == "__main__":
    unittest.main()

