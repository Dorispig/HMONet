"""Batch CLI keeps registered revisions/ranges and never starts on dry-run."""
from unittest.mock import patch
from scripts.data.generate_datasets import main
from src.data.generation import make_dataset_parametric_2d as two
from src.data.generation import make_dataset_parametric_3d as three


def test_default_batch_selects_all_eighteen_families():
    jobs = main(["--dry-run"])
    assert len(jobs) == 18
    assert [tuple(j[5]) for j in jobs] == [(41, 41)] * 9 + [
        (41, 41, 11) if j[2] == 'rounded_u_notch_extrusion' else (21, 21, 11)
        for j in jobs[9:]]


def test_new_batch_has_six_families_and_registered_shapes():
    with patch.object(two, "generate_dataset", side_effect=AssertionError), patch.object(three, "generate_family_dataset", side_effect=AssertionError):
        jobs = main(["--families", "new", "--dry-run"])
    assert len(jobs) == 6
    assert [tuple(j[5]) for j in jobs] == [(41, 41)] * 3 + [(21, 21, 11)] * 3


def test_all_both_splits_use_name_version_one():
    jobs = main(["--families", "all", "--split", "both", "--dry-run"])
    assert len(jobs) == 36
    assert all(job[4] == 1 for job in jobs)


def test_short_names_select_only_requested_family():
    jobs = main(["--dimension", "3", "--families", "tbp_a", "srt_r", "--dry-run"])
    assert [j[2] for j in jobs] == ["twisted_blade_passage", "square_to_round_transition"]
