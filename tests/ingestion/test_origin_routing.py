"""Tests for document-origin routing (canon / community / rpg).

The origin is governance metadata decided BEFORE storage: user-stated
wins, a confident filename inference second, otherwise the router asks
(``ORIGIN_REQUIRED``). The LLM only echoes the user's words; inference is
Python's job.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from src.ingestion.ingestion_router import (
    ROUTER_NEEDS_CLARIFICATION,
    ROUTER_PROCEED,
    IngestionFacts,
    IngestionRouter,
    apply_decision_table,
    infer_origin,
)
from src.ingestion.models import ClarificationKind, parse_intent


def _facts(**overrides) -> IngestionFacts:
    base = dict(file_name="meow.pdf", found=True,
                source_path=Path("data/sources/pdf/meow.pdf"))
    base.update(overrides)
    return IngestionFacts(**base)


class InferOriginTest(unittest.TestCase):
    def test_canon_markers(self):
        self.assertEqual(infer_origin("Dark Earth - Rules Source Book.pdf"), "canon")
        self.assertEqual(infer_origin("core_rulebook_v2.pdf"), "canon")
        self.assertEqual(infer_origin("Livres de Base.pdf"), "canon")

    def test_community_markers(self):
        self.assertEqual(infer_origin("Dark Earth, Gazette #3.pdf"), "community")
        self.assertEqual(infer_origin("fanzine_hiver.pdf"), "community")

    def test_rpg_markers(self):
        self.assertEqual(infer_origin("ma_campagne_du_Nord.pdf"), "rpg")
        self.assertEqual(infer_origin("session notes RPG.pdf"), "rpg")

    def test_word_boundary_no_false_positive(self):
        self.assertIsNone(infer_origin("porcelain_rules.pdf"))  # "rpg" not inside a word
        self.assertIsNone(infer_origin("symphony_no3.pdf"))     # "no" is not a marker anyway

    def test_no_marker_refuses(self):
        self.assertIsNone(infer_origin("meow.pdf"))

    def test_conflicting_markers_refuse(self):
        # A "canon gazette" is a contradiction, not a guess.
        self.assertIsNone(infer_origin("canon gazette.pdf"))

    def test_empty_name_refuses(self):
        self.assertIsNone(infer_origin(""))
        self.assertIsNone(infer_origin(None))


class DecisionTableOriginTest(unittest.TestCase):
    def _intent(self, **overrides):
        from src.ingestion.models import IngestionIntent

        data = dict(valid=True, document="meow.pdf")
        data.update(overrides)
        return IngestionIntent(**data)

    def test_stated_origin_wins_and_sets_flag(self):
        intent = self._intent(origin="rpg")
        decision = apply_decision_table(_facts(), intent)
        self.assertEqual(decision.status, ROUTER_PROCEED)
        self.assertEqual(decision.flags["document_origin"], "rpg")

    def test_confident_inference_sets_flag(self):
        facts = _facts(file_name="Dark Earth - Rules Source Book.pdf")
        decision = apply_decision_table(facts, self._intent())
        self.assertEqual(decision.status, ROUTER_PROCEED)
        self.assertEqual(decision.flags["document_origin"], "canon")
        self.assertTrue(facts.origin_known)

    def test_unknown_origin_requires_clarification(self):
        decision = apply_decision_table(_facts(), self._intent())
        self.assertEqual(decision.status, ROUTER_NEEDS_CLARIFICATION)
        self.assertEqual(decision.clarification, ClarificationKind.ORIGIN_REQUIRED)
        # The three choices must be proposed.
        joined = "\n".join(decision.suggestions).lower()
        for kind in ("canon", "community", "rpg"):
            self.assertIn(kind, joined)

    def test_conflicting_markers_require_clarification(self):
        facts = _facts(file_name="canon gazette.pdf")
        decision = apply_decision_table(facts, self._intent())
        self.assertEqual(decision.status, ROUTER_NEEDS_CLARIFICATION)
        self.assertEqual(decision.clarification, ClarificationKind.ORIGIN_REQUIRED)

    def test_stated_origin_beats_inference(self):
        # The user says rpg even though the name looks canon: user wins.
        intent = self._intent(origin="rpg")
        facts = _facts(file_name="Dark Earth - Rules Source Book.pdf")
        decision = apply_decision_table(facts, intent)
        self.assertEqual(decision.flags["document_origin"], "rpg")


class IntentParsingTest(unittest.TestCase):
    def test_stated_origin_is_accepted(self):
        intent = parse_intent(
            '{"valid": true, "document": "g.pdf", "origin": "Community"}'
        )
        self.assertEqual(intent.origin, "community")  # normalized

    def test_unknown_origin_rejected(self):
        with self.assertRaises(Exception):
            parse_intent('{"valid": true, "document": "g.pdf", "origin": "official"}')

    def test_missing_origin_is_none(self):
        intent = parse_intent('{"valid": true, "document": "g.pdf"}')
        self.assertIsNone(intent.origin)


class RouterFlowTest(unittest.TestCase):
    """End-to-end through the router with a mocked LLM (hermetic)."""

    def _router_with_llm(self, answers):
        router = IngestionRouter()
        llm = mock.MagicMock()
        llm.complete.side_effect = list(answers)
        with mock.patch.object(IngestionRouter, "_llm", return_value=llm):
            yield router

    def _patch_facts(self, stored_origin=None):
        import contextlib

        import src.ingestion.ingestion_router as ir
        from src.graphs import ingestion_routing_graph as graph_module

        stack = contextlib.ExitStack()
        facts = IngestionFacts(
            file_name="meow.pdf", found=True,
            source_path=Path("data/sources/pdf/meow.pdf"),
            stored_origin=stored_origin)
        stack.enter_context(mock.patch.object(
            ir, "gather_facts", return_value=facts))
        stack.enter_context(mock.patch.object(
            graph_module, "gather_facts", return_value=facts))
        return stack

    def test_missing_origin_asks_the_user(self):
        for router in self._router_with_llm(
            ['{"valid": true, "document": "meow.pdf"}']
        ):
            with self._patch_facts(stored_origin=None):
                decision = router.route("ingest meow.pdf")
        self.assertEqual(decision.status, ROUTER_NEEDS_CLARIFICATION)
        self.assertEqual(decision.clarification, ClarificationKind.ORIGIN_REQUIRED)
        self.assertTrue(decision.question)  # wording present (fallback ok)

    def test_stated_origin_proceeds(self):
        for router in self._router_with_llm(
            ['{"valid": true, "document": "meow.pdf", "origin": "community"}']
        ):
            with self._patch_facts(stored_origin=None):
                decision = router.route("ingest meow.pdf, it is fan-made")
        self.assertEqual(decision.status, ROUTER_PROCEED)
        self.assertEqual(decision.flags["document_origin"], "community")

    def test_stored_origin_means_no_question(self):
        """A re-ingestion keeps the origin decided at the first ingestion."""
        for router in self._router_with_llm(
            ['{"valid": true, "document": "meow.pdf"}']
        ):
            with self._patch_facts(stored_origin="rpg"):
                decision = router.route("re-ingest meow.pdf")
        self.assertEqual(decision.status, ROUTER_PROCEED)
        self.assertEqual(decision.flags["document_origin"], "rpg")

    def test_stated_origin_overrides_stored(self):
        """The user may CORRECT the record: a stated origin wins."""
        for router in self._router_with_llm(
            ['{"valid": true, "document": "meow.pdf", "origin": "community"}']
        ):
            with self._patch_facts(stored_origin="canon"):
                decision = router.route("re-ingest meow.pdf, actually it is fan-made")
        self.assertEqual(decision.status, ROUTER_PROCEED)
        self.assertEqual(decision.flags["document_origin"], "community")


if __name__ == "__main__":
    unittest.main()
