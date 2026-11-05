"""Tests for arbitrary command-line model combinations."""

from scripts.train.model_selection import parse_model_selection
from scripts.train.train_evaluate import extract_evaluation_model
import pytest


class TestModelSelection:
    def test_arbitrary_order_is_preserved(self):
        assert (parse_model_selection("pimeshonet,meshonet,hmonet")) == (
            ["pimeshonet", "meshonet", "hmonet"]
        )

    def test_aliases_expand_and_duplicates_are_removed(self):
        assert (parse_model_selection("both,pimeshonet,hmonet")) == (
            ["hmonet", "meshonet", "pimeshonet"]
        )

    def test_unknown_model_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown model"):
            parse_model_selection("hmonet,unknown")

    def test_evaluation_model_list_adds_traditional_methods(self):
        methods, remaining = extract_evaluation_model(
            ["--model", "meshonet,pimeshonet", "--grid-shapes", "51x51"]
        )
        assert (methods) == (["tfi", "pde", "meshonet", "pimeshonet"])
        assert (remaining) == (["--grid-shapes", "51x51"])
