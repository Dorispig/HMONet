"""Tests for arbitrary command-line model combinations."""

import unittest

from scripts.train.model_selection import parse_model_selection
from scripts.train.train_evaluate import extract_evaluation_model


class ModelSelectionTests(unittest.TestCase):
    def test_arbitrary_order_is_preserved(self):
        self.assertEqual(
            parse_model_selection("pimeshonet,meshonet,hmonet"),
            ["pimeshonet", "meshonet", "hmonet"],
        )

    def test_aliases_expand_and_duplicates_are_removed(self):
        self.assertEqual(
            parse_model_selection("both,pimeshonet,hmonet"),
            ["hmonet", "meshonet", "pimeshonet"],
        )

    def test_unknown_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            parse_model_selection("hmonet,unknown")

    def test_evaluation_model_list_adds_traditional_methods(self):
        methods, remaining = extract_evaluation_model(
            ["--model", "meshonet,pimeshonet", "--grid-shapes", "51x51"]
        )
        self.assertEqual(methods, ["tfi", "pde", "meshonet", "pimeshonet"])
        self.assertEqual(remaining, ["--grid-shapes", "51x51"])


if __name__ == "__main__":
    unittest.main()
