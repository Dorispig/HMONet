"""Comparison output paths remain compact and collision safe."""

from pathlib import Path

from scripts.evaluate.evaluation_common import comparison_directory


class TestComparisonDirectory:
    def options(self, pimeshonet_projection="none"):
        return {
            "hmonet_boundary_projection": "none",
            "meshonet_boundary_projection": "none",
            "pimeshonet_boundary_projection": pimeshonet_projection,
            "pihmonet_mapping_mode": "rational_affine",
            "pihmonet_boundary_projection": "none",
            "attentionpihmonet_mapping_mode": "rational_affine",
        }

    def test_default_five_method_path_has_no_extra_pi_layer(self):
        names = ["TFI", "PDE_Winslow", "MeshONet", "PI-MeshONet", "HMONet"]
        actual = comparison_directory(Path("result"), "rational_affine", self.options(), names)
        expected = (
            Path("result")
            / "comparisons"
            / "rational_affine"
            / "hmonet_none__meshonet_none"
            / "tfi_pde_meshonet_pimeshonet_hmonet"
        )
        assert (actual) == (expected)
        assert ("pimeshonet_none") not in (actual.parts)

    def test_nondefault_pi_projection_is_encoded_without_extra_level(self):
        names = ["TFI", "MeshONet", "PI-MeshONet", "HMONet"]
        actual = comparison_directory(
            Path("result"), "rational_affine", self.options("direct"), names
        )
        assert (actual.parent.name) == ("hmonet_none__meshonet_none__pimeshonet_direct")
        assert (actual.name) == ("tfi_meshonet_pimeshonet_hmonet")

    def test_pi_only_comparison_keeps_one_projection_directory(self):
        actual = comparison_directory(
            Path("result"), "rational_affine", self.options(), ["TFI", "PI-MeshONet"]
        )
        assert (actual) == (
            Path("result") / "comparisons" / "pimeshonet_none" / "tfi_pimeshonet"
        )
