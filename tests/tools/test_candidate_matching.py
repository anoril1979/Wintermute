"""Tests for candidate promotion in the ingest tool (fuzzy matching)."""

from __future__ import annotations

import unittest

from src.tools.ingest_tool import _similar_names


class SimilarNamesTest(unittest.TestCase):
    AVAILABLE = [
        "Dark Earth - Le marcheur (Gazette #1).pdf",
        "Dark Earth - Le marcheur (Gazette #1)-16-18.pdf",
        "Dark Earth - Le marcheur (Gazette #1)-17.pdf",
        "Pyramides.pdf",
        "Artefacts.pdf",
    ]

    def test_truncated_reference_finds_full_document_first(self):
        # The real-world case: the LLM truncated the name before the '#'.
        candidates = _similar_names(
            "Dark Earth - Le marcheur (Gazette.pdf", self.AVAILABLE
        )
        self.assertEqual(candidates[0], "Dark Earth - Le marcheur (Gazette #1).pdf")
        self.assertIn("Dark Earth - Le marcheur (Gazette #1)-17.pdf", candidates)

    def test_typo_tolerated(self):
        candidates = _similar_names("Drak Earth - Le marcheur.pdf", self.AVAILABLE)
        self.assertEqual(candidates[0], "Dark Earth - Le marcheur (Gazette #1).pdf")

    def test_title_style_partial_reference(self):
        # Word containment: "Gazette #1" must reach the Gazette files.
        candidates = _similar_names("Gazette #1", self.AVAILABLE)
        self.assertTrue(
            all("Gazette" in name for name in candidates),
            f"unexpected candidates: {candidates}",
        )
        self.assertIn("Dark Earth - Le marcheur (Gazette #1).pdf", candidates)

    def test_unrelated_reference_yields_nothing(self):
        self.assertEqual(_similar_names("Totally Unrelated Doc.pdf", self.AVAILABLE), [])

    def test_unrelated_files_not_promoted(self):
        candidates = _similar_names(
            "Dark Earth - Le marcheur (Gazette.pdf", self.AVAILABLE
        )
        self.assertNotIn("Artefacts.pdf", candidates)
        self.assertNotIn("Pyramides.pdf", candidates)

    def test_deterministic_and_capped(self):
        first = _similar_names("Dark Earth - Le marcheur (Gazette.pdf", self.AVAILABLE)
        second = _similar_names("Dark Earth - Le marcheur (Gazette.pdf", self.AVAILABLE)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 5)

    def test_empty_reference(self):
        self.assertEqual(_similar_names("   ", self.AVAILABLE), [])


if __name__ == "__main__":
    unittest.main()
