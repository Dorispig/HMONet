"""Verify shared boundary orientation across analytic and spline geometries."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tfi.tfi_2d import tfi_from_curves
from src.data.geometry.boundary_generator_2d import (
    BOUNDARY_GENERATORS,
    circle_airfoil_boundary,
    rectangular_boundary,
)
from src.data.generation.make_dataset_parametric_2d import to_computational_boundary
from src.data.mesh.topology import GenerateQuadMeshBoundaryIndex
from src.data.geometry.parametric_curve_2d import (
    BoundaryCurveSet2D,
    BSplineCurve,
    CircleArc,
    EllipseArc,
    FittedCurve,
    LineSegment,
    ParametricCurve,
)


class BoundaryOrientationTests(unittest.TestCase):
    def test_all_registered_boundaries_preserve_contour_and_tfi_edges(self):
        nx, ny = 21, 11
        counts = dict(bottom=nx, right=ny, top=nx, left=ny)
        ids = GenerateQuadMeshBoundaryIndex(ny, nx)
        for name, factory in BOUNDARY_GENERATORS.items():
            with self.subTest(boundary=name):
                source = factory()
                original = json.dumps(source.to_dict(), sort_keys=True)
                result = to_computational_boundary(source)
                self.assertEqual(result.parameter_order, "computational")
                self.assertTrue(result.validate())
                np.testing.assert_allclose(
                    result.sample_closed(counts), source.sample_closed(counts), atol=1e-12
                )
                self.assertEqual(original, json.dumps(source.to_dict(), sort_keys=True))
                again = to_computational_boundary(result)
                self.assertEqual(result.to_dict(), again.to_dict())
                loaded = BoundaryCurveSet2D.from_dict(result.to_dict())
                np.testing.assert_allclose(
                    loaded.sample_closed(counts), result.sample_closed(counts), atol=1e-12
                )
                grid = tfi_from_curves(loaded, nx=nx, ny=ny)
                np.testing.assert_allclose(
                    grid.reshape(-1, 2)[ids],
                    loaded.sample_closed(counts)[:-1],
                    atol=1e-12,
                )

    def test_custom_perimeter_input_can_override_missing_metadata(self):
        reference = rectangular_boundary()
        perimeter = BoundaryCurveSet2D(
            reference.bottom,
            reference.right,
            reference.top.reverse(),
            reference.left.reverse(),
            validate=False,
        )
        with self.assertRaisesRegex(ValueError, "corner mismatch"):
            to_computational_boundary(perimeter)
        result = to_computational_boundary(perimeter, source_order="perimeter")
        for side in BoundaryCurveSet2D.names:
            np.testing.assert_allclose(
                getattr(result, side).sample(17), getattr(reference, side).sample(17)
            )
        self.assertEqual(perimeter.metadata, {})

    def test_arc_and_spline_reversal_preserves_the_original_curve(self):
        curves = [
            BSplineCurve([[0, 0], [0.2, 1], [0.8, -0.2], [1, 0]]),
            FittedCurve([[0, 0], [0.3, 1], [1, 0]]),
        ]
        for start, end, clockwise in (
            (np.pi, 0, False),
            (0, np.pi, False),
            (0.3, 2.1, True),
            (0.3, -2.1, True),
            (0, 2 * np.pi, False),
        ):
            curves.extend([
                CircleArc((0.1, -0.3), 2, start, end, clockwise=clockwise),
                EllipseArc((0.1, -0.3), 2, 1, start, end, rotation=0.4, clockwise=clockwise),
            ])
        parameters = np.linspace(0, 1, 31)
        for curve in curves:
            with self.subTest(curve=type(curve).__name__):
                reverse = curve.reverse()
                expected = curve.evaluate(1 - parameters)
                np.testing.assert_allclose(reverse.evaluate(parameters), expected, atol=1e-12)
                loaded = ParametricCurve.from_dict(reverse.to_dict())
                np.testing.assert_allclose(loaded.evaluate(parameters), expected, atol=1e-12)
                np.testing.assert_allclose(
                    reverse.reverse().evaluate(parameters), curve.evaluate(parameters), atol=1e-12
                )

    def test_legacy_circle_metadata_and_visualization_remain_compatible(self):
        import matplotlib.pyplot as plt
        from src.utils.visualize import plot_boundary_curve_set_2d

        descriptor = circle_airfoil_boundary().to_dict()
        descriptor["metadata"].pop("parameter_order")
        source = BoundaryCurveSet2D.from_dict(descriptor)
        self.assertEqual(source.parameter_order, "perimeter")
        result = to_computational_boundary(source)
        self.assertEqual(result.metadata["topology"], "circle_airfoil")
        figure, axes = plt.subplots(1, 2)
        try:
            for boundary, axis in zip((source, result), axes):
                plot_boundary_curve_set_2d(boundary, nx=21, ny=11, ax=axis)
            np.testing.assert_allclose(
                axes[0].lines[0].get_xydata(), axes[1].lines[0].get_xydata(), atol=1e-12
            )
        finally:
            plt.close(figure)

    def test_invalid_boundaries_are_rejected(self):
        source = rectangular_boundary()
        with self.assertRaisesRegex(ValueError, "source_order"):
            to_computational_boundary(source, source_order="guess")
        with self.assertRaises(TypeError):
            to_computational_boundary("rectangle")
        with self.assertRaisesRegex(ValueError, "tolerance"):
            to_computational_boundary(source, tolerance=0)
        for start in ([3, 3], [np.nan, 0]):
            malformed = BoundaryCurveSet2D(
                source.bottom,
                source.right,
                source.top,
                LineSegment(start, [-1, 1]),
                validate=False,
            )
            with self.assertRaises(ValueError):
                to_computational_boundary(malformed)


if __name__ == "__main__":
    unittest.main()
