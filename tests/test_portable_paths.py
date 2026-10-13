import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.utils.paths import PROJECT_ROOT, portable_project_path, resolve_project_path


class PortableProjectPathTests(unittest.TestCase):
    def test_project_path_serializes_with_posix_separators(self):
        path = PROJECT_ROOT / "experiments" / "baseline" / "experiment.yaml"
        self.assertEqual(
            portable_project_path(path),
            "experiments/baseline/experiment.yaml",
        )

    def test_relative_path_resolves_from_project_root(self):
        expected = PROJECT_ROOT / "configs" / "2D" / "train.yaml"
        self.assertEqual(
            resolve_project_path("configs/2D/train.yaml"),
            expected.resolve(),
        )

    def test_legacy_windows_path_is_rebased(self):
        legacy = r"E:\projects\HMONet_p\experiments\baseline\experiment.yaml"
        expected = PROJECT_ROOT / "experiments" / "baseline" / "experiment.yaml"
        self.assertEqual(resolve_project_path(legacy), expected.resolve())
        self.assertEqual(resolve_project_path(Path(legacy)), expected.resolve())

    def test_external_hmonet_model_directory_is_not_rebased(self):
        with TemporaryDirectory() as temporary:
            external = (
                Path(temporary)
                / "checkpoints"
                / "dataset"
                / "HMONet"
                / "current"
                / "best.pt"
            )
            self.assertEqual(resolve_project_path(external), external.resolve())
            self.assertEqual(portable_project_path(external), str(external.resolve()))


if __name__ == "__main__":
    unittest.main()

