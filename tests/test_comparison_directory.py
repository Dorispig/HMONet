"""Comparison output paths remain compact and collision safe."""

from pathlib import Path
import unittest

from scripts.evaluate.evaluation_common import comparison_directory


class ComparisonDirectoryTests(unittest.TestCase):
    def options(self, pimeshonet_projection="none"):
        return {
            "hmonet_boundary_projection": "none",
            "meshonet_boundary_projection": "none",
            "pimeshonet_boundary_projection": pimeshonet_projection,
            "pihmonet_mapping_mode": "current",
            "pihmonet_boundary_projection": "none",
            "attentionpihmonet_mapping_mode": "current",
        }

    def test_default_five_method_path_has_no_extra_pi_layer(self):
        names = ["TFI", "PDE_Winslow", "MeshONet", "PI-MeshONet", "HMONet"]
        actual = comparison_directory(Path("result"), "current", self.options(), names)
        expected = (
            Path("result")
            / "comparisons"
            / "current"
            / "hmonet_none__meshonet_none"
            / "tfi_pde_meshonet_pimeshonet_hmonet"
        )
        self.assertEqual(actual, expected)
        self.assertNotIn("pimeshonet_none", actual.parts)

    def test_nondefault_pi_projection_is_encoded_without_extra_level(self):
        names = ["TFI", "MeshONet", "PI-MeshONet", "HMONet"]
        actual = comparison_directory(
            Path("result"), "current", self.options("direct"), names
        )
        self.assertEqual(
            actual.parent.name,
            "hmonet_none__meshonet_none__pimeshonet_direct",
        )
        self.assertEqual(actual.name, "tfi_meshonet_pimeshonet_hmonet")

    def test_pi_only_comparison_keeps_one_projection_directory(self):
        actual = comparison_directory(
            Path("result"), "current", self.options(), ["TFI", "PI-MeshONet"]
        )
        self.assertEqual(
            actual,
            Path("result")
            / "comparisons"
            / "pimeshonet_none"
            / "tfi_pimeshonet",
        )


if __name__ == "__main__":
    unittest.main()
