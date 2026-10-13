"""Dataset-family geometry, evaluation configuration and deterministic generation tests."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
import yaml
from src.data.generation import make_dataset_parametric_2d as gen
from src.data.dataset import ParametricDataset2D, read_dataset_metadata
from src.data.mesh.sampling import sample_boundary, boundary_indices
from src.evaluation.configuration import create_geometry


class DatasetFamilyTests(unittest.TestCase):
    def test_family_defaults_and_endpoints(self):
        for family, prefix, grid in [
            ("rect_top_semiellipse", "rse", [20, 20]),
            ("eccentric_semiannulus", "esa", [20, 20]),
            ("symmetric_convex_tri", "tri", [21, 20]),
        ]:
            spec = gen.DATASET_FAMILIES[family]
            self.assertEqual(spec["grid_shape"], grid)
            for split in ("int", "ext"):
                with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    path = gen.generate_dataset(
                        family, split, size=4, output_dir=tmp, timing_samples=0
                    )
                    self.assertEqual(
                        path.name, f"{prefix}_{split}_0_4_{grid[0]}_{grid[1]}"
                    )
                    meta = read_dataset_metadata(path)
                    self.assertTrue(meta["all_pde_converged"])
                    self.assertEqual(
                        meta["interval_sample_counts"],
                        [2, 2] if split == "int" else [4],
                    )
                    numeric = np.load(path / "dataset.npy", allow_pickle=True)
                    geometric = np.load(
                        path / "boundary_geometry.npy", allow_pickle=True
                    )
                    for n, g in zip(numeric, geometric):
                        self.assertEqual(n["sample_id"], g["sample_id"])
                        self.assertEqual(n["dataset_id"], g["dataset_id"])
                        value = g["parameters"][spec["parameter"]]
                        boundary, _ = create_geometry(
                            2,
                            {"boundary_type": family, "parameter": spec["parameter"]},
                            value,
                        )
                        expected = sample_boundary(boundary, grid)
                        np.testing.assert_allclose(
                            n["boundary_points"], expected, atol=1e-7
                        )
                        np.testing.assert_allclose(
                            n["mesh_points"][boundary_indices(grid)],
                            expected,
                            atol=1e-7,
                        )
                    for need in (True, False):
                        dataset = ParametricDataset2D(path, load_boundary=need)
                        self.assertEqual(len(dataset), 4)

    def test_evaluation_cases_use_reference_values_and_canonical_names(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (root / "configs/2D/evaluate.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(list(config["cases"]), sorted(config["cases"]))
        expected = {
            "rse_int_0_1000_20_20": [0.225, 0.25, 0.275],
            "rse_ext_0_1000_20_20": [0.42, 0.46, 0.50, 0.55, 0.60],
            "esa_int_0_1000_20_20": [0.475, 0.50, 0.525],
            "esa_ext_0_1000_20_20": [0.72, 0.75],
            "tri_int_0_1000_21_20": [0.225, 0.25, 0.275],
            "tri_ext_0_1000_21_20": [0.42, 0.46, 0.50],
        }
        self.assertTrue(
            (
                set(expected)
                | {"air_t_int_0_1000_21_20", "air_t_ext_0_1000_21_20"}
            ).issubset(
                config["cases"]
            )
        )
        for name, values in expected.items():
            case = config["cases"][name]
            self.assertEqual(case["values"], values)
            for value in values:
                boundary, parameters = create_geometry(2, case, value)
                self.assertEqual(parameters[case["parameter"]], value)
                self.assertTrue(
                    np.isfinite(sample_boundary(boundary, case["train_grid"])).all()
                )

    def test_apex_and_semicircle_geometry(self):
        tri = gen.DATASET_GENERATORS["symmetric_convex_tri"](protrusion_distance=0.3)
        np.testing.assert_allclose(tri.bottom(0.5), [0.5, -0.3])
        np.testing.assert_allclose(tri.top(0.5), [0.5, 1.3])
        esa = gen.DATASET_GENERATORS["eccentric_semiannulus"](inner_center_x=0.7)
        np.testing.assert_allclose(esa.bottom(0.5), [0.7, 0.1])
        np.testing.assert_allclose(esa.top(0.5), [0.5, 0.5])
        rse = gen.DATASET_GENERATORS["rect_top_semiellipse"](semi_minor_axis=0.0)
        np.testing.assert_allclose(rse.top.sample(20)[:, 1], 1.0)


FAMILIES = [
    "smooth_constricted_channel",
    "variable_width_s_channel",
    "variable_width_s_channel_bend",
    "gear_spline",
    "rounded_u_notch_compact",
]


class StressDatasetTests(unittest.TestCase):
    def test_evaluation_configuration(self):
        root = Path(__file__).resolve().parents[1]
        cases = yaml.safe_load(
            (root / "configs/2D/evaluate.yaml").read_text(encoding="utf-8")
        )["cases"]
        self.assertEqual(list(cases), sorted(cases))
        for family in FAMILIES:
            spec = gen.DATASET_FAMILIES[family]
            for split in ["interpolation", "extrapolation"]:
                case = cases[
                    gen.build_dataset_name(
                        family, split, 0, 1000, *spec["grid_shape"]
                    )
                ]
                self.assertEqual(case["parameter"], spec["parameter"])
                self.assertEqual(case["fixed_parameters"], spec["fixed_parameters"])
                for value in case["values"]:
                    self.assertFalse(any(a <= value <= b for a, b in spec[split]))
                    boundary, _ = create_geometry(2, case, value)
                    boundary.validate()

    def test_parallel_identity_and_numerics(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(
            io.StringIO()
        ):
            paths = []
            for workers in [1, 2]:
                paths.append(
                    gen.generate_dataset(
                        FAMILIES[0],
                        "int",
                        size=4,
                        nx=9,
                        ny=9,
                        workers=workers,
                        timing_samples=0,
                        output_dir=Path(tmp) / str(workers),
                    )
                )
            data = [np.load(p / "dataset.npy", allow_pickle=True) for p in paths]
            for i in range(4):
                np.testing.assert_allclose(
                    data[0][i]["mesh_points"], data[1][i]["mesh_points"]
                )
            for path in paths:
                geom = np.load(path / "boundary_geometry.npy", allow_pickle=True)
                for i, row in enumerate(geom):
                    self.assertEqual(row["sample_id"], i)
                    self.assertEqual(
                        row["dataset_id"], data[paths.index(path)][i]["dataset_id"]
                    )
                self.assertEqual(read_dataset_metadata(path)["size"], 4)
                self.assertEqual(len(ParametricDataset2D(path, load_boundary=True)), 4)
                self.assertEqual(len(ParametricDataset2D(path, load_boundary=False)), 4)


if __name__ == "__main__":
    unittest.main()
