"""Tests for the shared string helpers (src/helpers/strings.py).

One canonical slug/fold implementation; the knowledge layer's three
call sites (entity ids, knowledge-base filenames, lookup match keys)
are thin wrappers whose exact behaviors are pinned here.
"""

from __future__ import annotations

import unittest

from src.helpers.strings import fold, slugify


class SlugifyTest(unittest.TestCase):
    def test_accents_and_spaces(self):
        self.assertEqual(slugify("Édmond Dantès"), "edmond_dantes")

    def test_joiner_flavor(self):
        self.assertEqual(slugify("Édmond Dantès", joiner="-"), "edmond-dantes")

    def test_collapses_punctuation_runs(self):
        self.assertEqual(slugify("Joe  le   Clodo!"), "joe_le_clodo")
        self.assertEqual(slugify("Sombre-Terre"), "sombre_terre")

    def test_empty_and_degenerate_input(self):
        self.assertEqual(slugify("###"), "")
        self.assertEqual(slugify(""), "")
        # Callers own the fallback (the knowledge store uses 'unnamed').
        self.assertEqual(slugify("###", joiner="-") or "unnamed", "unnamed")


class FoldTest(unittest.TestCase):
    def test_case_accent_punctuation_insensitive(self):
        self.assertEqual(fold("Épée de Vif-Argent"), "epee de vif argent")
        self.assertEqual(
            fold("épée de vif argent"), fold("ÉPÉE DE VIF-ARGENT")
        )

    def test_runs_collapse_to_single_spaces(self):
        self.assertEqual(fold("  Joe   le   Clodo!  "), "joe le clodo")


if __name__ == "__main__":
    unittest.main()
