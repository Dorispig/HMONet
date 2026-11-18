"""Formal evaluation retention keeps summaries/images and drops heavy details."""

from pathlib import Path
from tempfile import TemporaryDirectory

from src.evaluation.retention import compact_evaluation_directory, readable_image_path


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
            assert (root / "plots/TFI/quality/quality_scaled_jacobian.png").is_file()
            assert not (root / "generated_meshes.npz").exists()
            assert not (detail / "quality_summary.csv").exists()

    def test_compaction_replaces_stale_public_images(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "results.json").write_text("{}", encoding="utf-8")
            stale = root / "plots/comparison/methods/mesh_comparison_overview.png"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"old")
            current = root / "comparisons/group/mesh_comparison_overview.png"
            current.parent.mkdir(parents=True)
            current.write_bytes(b"new")

            compact_evaluation_directory(root)

            public = root / "plots/comparison/methods/mesh_comparison_overview.png"
            assert public.read_bytes() == b"new"

    def test_middle_section_images_keep_their_logical_section_name(self):
        root = Path("evaluation")
        result = readable_image_path(
            root,
            Path(
                "comparisons/group/boundary_faces/y_mid/"
                "middle_section_mesh_overview.png"
            ),
        )
        assert result == (
            root
            / "plots/comparison/methods/middle_sections/y_mid/"
            "middle_section_mesh_overview.png"
        )
