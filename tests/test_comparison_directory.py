"""Comparison output paths remain compact and collision safe."""

from pathlib import Path

from scripts.evaluate.evaluation_common import comparison_directory


class TestComparisonDirectory:
    def options(self, pimeshonet_projection="none"):
        return {
            "hmonet_boundary_projection": "none",
            "meshonet_boundary_projection": "none",
            "pimeshonet_boundary_projection": pimeshonet_projection,
        }

    def test_default_five_method_path_has_no_extra_pi_layer(self):
        names = ["TFI", "PDE_Winslow", "MeshONet", "PI-MeshONet", "HMONet"]
        actual = comparison_directory(Path("result"), "rational_affine", self.options(), names)
        expected = (
            Path("result")
            / "comparisons"
            / "RAP"
            / "H_raw__M_raw"
            / "TFI_PDE_M_PI_H"
        )
        assert (actual) == (expected)
        assert ("PI_raw") not in (actual.parts)

    def test_nondefault_pi_projection_is_encoded_without_extra_level(self):
        names = ["TFI", "MeshONet", "PI-MeshONet", "HMONet"]
        actual = comparison_directory(
            Path("result"), "rational_affine", self.options("direct"), names
        )
        assert (actual.parent.name) == ("H_raw__M_raw__PI_direct")
        assert (actual.name) == ("TFI_M_PI_H")

    def test_pi_only_comparison_keeps_one_projection_directory(self):
        actual = comparison_directory(
            Path("result"), "rational_affine", self.options(), ["TFI", "PI-MeshONet"]
        )
        assert (actual) == (
            Path("result") / "comparisons" / "PI_raw" / "TFI_PI"
        )
