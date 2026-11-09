"""Formal evaluation retention keeps summaries/images and drops heavy details."""

from pathlib import Path
from tempfile import TemporaryDirectory

from src.evaluation.retention import compact_evaluation_directory


class TestFormalEvaluationRetention:
    def test_compaction_keeps_results_and_images_only(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "results.json").write_text("{}", encoding="utf-8")
            (root / "generated_meshes.npz").write_bytes(b"mesh")
            detail = root / "TFI/grid_quality"
            detail.mkdir(parents=True)
            (detail / "quality_summary.csv").write_text("metric,value\n")
            (detail / "quality_scaled_jacobian.png").write_bytes(b"png")

            compact_evaluation_directory(root)

            assert (root / "results.json").is_file()
            assert (root / "p/tfi/q/q_sj.png").is_file()
            assert not (root / "generated_meshes.npz").exists()
            assert not (detail / "quality_summary.csv").exists()
