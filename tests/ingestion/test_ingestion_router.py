"""Tests for the LLM-backed ingestion router and its routing graph.

Covers the division of labor frozen in src/ingestion/ingestion_router.py:

* ``parse_intent`` — LLM answer → validated :class:`IngestionIntent`;
* keyword fallback (LLM unavailable) — never invents an intent;
* ``apply_decision_table`` — pure (facts, intent) → flags / clarify;
* ``IngestionRouter.route`` — full flow with a mocked LLM client;
* ``IngestionRoutingGraph.run`` — same flow, fully traced.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from src.ingestion import ingestion_router as ir
from src.ingestion.ingestion_router import (
    ROUTER_NEEDS_CLARIFICATION,
    ROUTER_PROCEED,
    IngestionFacts,
    IngestionRouter,
    apply_decision_table,
)
from src.ingestion.models import (
    ClarificationKind,
    IngestionIntent,
    parse_intent,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _intent(**overrides) -> IngestionIntent:
    """A valid intent naming a document (tests override single fields)."""
    base = dict(valid=True, document="meow.pdf", force=False,
                redo_summaries=False, reason="test intent")
    base.update(overrides)
    return IngestionIntent(**base)


def _facts(**overrides) -> IngestionFacts:
    """Facts for a document known to the stores (tests override fields)."""
    base = dict(file_name="meow.pdf", found=True,
                extraction_job="already_done", canonical_json=True,
                summarization_job="already_done", summarized_json=True,
                summaries_stale=False)
    base.update(overrides)
    return IngestionFacts(**base)


def _found_facts(**overrides) -> IngestionFacts:
    """Facts for a document present in the documents tree."""
    overrides.setdefault("source_path", Path("data/sources/pdf/meow.pdf"))
    return _facts(**overrides)


def _patch_fact_gathering():
    """Patch gather_facts to fixed facts, in every namespace that binds it.

    The router calls the module-global (``src.ingestion.ingestion_router``)
    while the routing graph imported its own reference at module load —
    both must be patched.
    """
    from src.graphs import ingestion_routing_graph as graph_module

    return _MultiPatch([
        mock.patch.object(ir, "gather_facts", return_value=_found_facts()),
        mock.patch.object(graph_module, "gather_facts",
                          return_value=_found_facts()),
    ])


class _MultiPatch:
    """Start/stop several patchers as one context manager."""

    def __init__(self, patchers) -> None:
        self._patchers = patchers

    def __enter__(self):
        for patcher in self._patchers:
            patcher.start()
        return self

    def __exit__(self, *exc):
        for patcher in reversed(self._patchers):
            patcher.stop()
        return False


def _router_with_llm(answers) -> IngestionRouter:
    """A router whose LLM client returns the given answers in order."""
    router = IngestionRouter()
    client = mock.MagicMock()
    client.complete.side_effect = answers
    patcher = mock.patch.object(router, "_llm", return_value=client)
    patcher.start()
    # No addCleanup: the router is local to the calling test; the patch is
    # only held on the local instance attribute. _llm is resolved at call
    # time, so releasing the patcher is unnecessary for correctness.
    return router


# ---------------------------------------------------------------------------
# parse_intent
# ---------------------------------------------------------------------------

class ParseIntentTest(unittest.TestCase):
    """The LLM's raw answer is validated into a strict pydantic model."""

    def test_plain_json_answer(self):
        raw = '{"valid": true, "document": "meow.pdf", "force": false}'
        intent = parse_intent(raw)
        self.assertTrue(intent.valid)
        self.assertEqual(intent.document, "meow.pdf")
        self.assertFalse(intent.force)

    def test_fenced_json_answer(self):
        raw = (
            "Here is my analysis:\n"
            "```json\n"
            '{"valid": true, "document": "Dark Earth.pdf", "redo_summaries": true}'
            "\n```\n"
        )
        intent = parse_intent(raw)
        self.assertTrue(intent.valid)
        self.assertEqual(intent.document, "Dark Earth.pdf")
        self.assertTrue(intent.redo_summaries)

    def test_prose_wrapped_answer(self):
        raw = 'Sure! {"valid": true, "document": "meow.pdf"} hope this helps'
        intent = parse_intent(raw)
        self.assertTrue(intent.valid)

    def test_invalid_intent_needs_a_clarification_kind(self):
        raw = '{"valid": false, "reason": "not an ingestion order"}'
        with self.assertRaises(ValueError):
            parse_intent(raw)

    def test_invalid_intent_with_clarification_is_valid_model(self):
        raw = (
            '{"valid": false, "clarification": "request_unclear", '
            '"reason": "gibberish"}'
        )
        intent = parse_intent(raw)
        self.assertFalse(intent.valid)
        self.assertEqual(intent.clarification, ClarificationKind.REQUEST_UNCLEAR)

    def test_document_must_be_a_bare_name(self):
        """Paths, drive letters and traversal are rejected (Python resolves)."""
        for bad in ("data/sources/pdf/meow.pdf", "C:\\meow.pdf", "../meow.pdf"):
            with self.subTest(bad=bad):
                raw = '{"valid": true, "document": "%s"}' % bad
                with self.assertRaises(ValueError):
                    parse_intent(raw)

    def test_unknown_fields_rejected(self):
        raw = '{"valid": true, "document": "meow.pdf", "option": "x"}'
        with self.assertRaises(ValueError):
            parse_intent(raw)

    def test_empty_answer_rejected(self):
        with self.assertRaises(ValueError):
            parse_intent("")

    def test_no_json_rejected(self):
        with self.assertRaises(ValueError):
            parse_intent("I could not understand the request.")


# ---------------------------------------------------------------------------
# Keyword fallback (LLM unavailable)
# ---------------------------------------------------------------------------

class KeywordFallbackTest(unittest.TestCase):
    """Without an LLM the router classifies deterministically or asks."""

    def test_force_keywords(self):
        intent = IngestionRouter()._keyword_intent("please reingest 'meow.pdf'")
        self.assertTrue(intent.valid)
        self.assertEqual(intent.document, "meow.pdf")
        self.assertTrue(intent.force)

    def test_redo_summaries_keywords(self):
        intent = IngestionRouter()._keyword_intent(
            "re-ingest meow.pdf and force summarization"
        )
        self.assertTrue(intent.valid)
        self.assertTrue(intent.redo_summaries)

    def test_quoted_document_with_spaces(self):
        intent = IngestionRouter()._keyword_intent(
            "ingest 'Dark Earth - Le marcheur (Gazette #1).pdf'"
        )
        self.assertEqual(
            intent.document, "Dark Earth - Le marcheur (Gazette #1).pdf"
        )

    def test_no_document_means_unclear_intent(self):
        intent = IngestionRouter()._keyword_intent("do the thing")
        self.assertFalse(intent.valid)
        self.assertEqual(intent.clarification, ClarificationKind.REQUEST_UNCLEAR)

    def test_fallback_never_sets_question(self):
        """The wording belongs to the clarification step, not the intent."""
        intent = IngestionRouter()._keyword_intent("ingest meow.pdf")
        self.assertIsNone(intent.question)


# ---------------------------------------------------------------------------
# Decision table
# ---------------------------------------------------------------------------

class DecisionTableTest(unittest.TestCase):
    """Pure (facts, intent) → flags / clarify, deterministic."""

    def test_plain_reingest_proceeds(self):
        decision = apply_decision_table(_found_facts(), _intent())
        self.assertEqual(decision.status, ROUTER_PROCEED)
        self.assertEqual(
            decision.flags,
            {"force_extraction": False, "force_summarization": False},
        )

    def test_force_implies_force_summarization(self):
        decision = apply_decision_table(_found_facts(), _intent(force=True))
        self.assertEqual(decision.status, ROUTER_PROCEED)
        self.assertTrue(decision.flags["force_extraction"])
        self.assertTrue(decision.flags["force_summarization"])

    def test_redo_summaries_only(self):
        decision = apply_decision_table(
            _found_facts(), _intent(redo_summaries=True)
        )
        self.assertTrue(decision.flags["force_summarization"])
        self.assertFalse(decision.flags["force_extraction"])

    def test_stale_fingerprints_force_summarization(self):
        decision = apply_decision_table(_found_facts(summaries_stale=True), _intent())
        self.assertTrue(decision.flags["force_summarization"])

    def test_valid_intent_unknown_document_clarifies(self):
        decision = apply_decision_table(_facts(found=False), _intent())
        self.assertEqual(decision.status, ROUTER_NEEDS_CLARIFICATION)
        self.assertEqual(
            decision.clarification, ClarificationKind.DOCUMENT_NOT_FOUND
        )

    def test_invalid_intent_clarifies_even_with_document(self):
        """The router never guesses a document from an unclear utterance."""
        decision = apply_decision_table(
            _found_facts(),
            _intent(valid=False, clarification=ClarificationKind.REQUEST_UNCLEAR,
                    reason="gibberish"),
        )
        self.assertEqual(decision.status, ROUTER_NEEDS_CLARIFICATION)
        self.assertEqual(
            decision.clarification, ClarificationKind.REQUEST_UNCLEAR
        )

    def test_orphaned_summaries_clarify(self):
        decision = apply_decision_table(
            _found_facts(summarized_json=True,
                         summarization_job="already_done",
                         canonical_json=False),
            _intent(),
        )
        self.assertEqual(decision.status, ROUTER_NEEDS_CLARIFICATION)
        self.assertEqual(decision.clarification, ClarificationKind.STATE_CONFLICT)

    def test_deterministic(self):
        """Same facts + intent → same decision (pure function)."""
        facts, intent = _found_facts(), _intent()
        one = apply_decision_table(facts, intent)
        two = apply_decision_table(facts, intent)
        self.assertEqual(one.flags, two.flags)
        self.assertEqual(one.status, two.status)


# ---------------------------------------------------------------------------
# route() with a mocked LLM
# ---------------------------------------------------------------------------

class RouteFlowTest(unittest.TestCase):
    """Full router flow with the LLM client mocked."""

    def test_proceed_path(self):
        router = _router_with_llm(
            ['{"valid": true, "document": "meow.pdf"}']
        )
        with _patch_fact_gathering():
            decision = router.route("ingest meow.pdf")
        self.assertEqual(decision.status, ROUTER_PROCEED)
        self.assertEqual(
            decision.flags,
            {"force_extraction": False, "force_summarization": False},
        )

    def test_document_not_found_uses_llm_wording(self):
        router = _router_with_llm([
            '{"valid": true, "document": "meow.pdf"}',
            "I could not find that document; ask me to list the documents.",
        ])
        facts = _facts(found=False)
        with mock.patch.object(ir, "gather_facts", return_value=facts):
            decision = router.route("ingest meow.pdf")
        self.assertEqual(decision.status, ROUTER_NEEDS_CLARIFICATION)
        self.assertEqual(
            decision.clarification, ClarificationKind.DOCUMENT_NOT_FOUND
        )
        self.assertIn("could not find", decision.question)

    def test_unclear_request_uses_fallback_wording_when_llm_fails(self):
        """First LLM call fails → keyword fallback; no document → fallback text."""
        router = IngestionRouter()
        client = mock.MagicMock()
        client.complete.side_effect = RuntimeError("Ollama down")
        with mock.patch.object(router, "_llm", return_value=client):
            decision = router.route("do the thing")
        self.assertEqual(decision.status, ROUTER_NEEDS_CLARIFICATION)
        self.assertEqual(
            decision.clarification, ClarificationKind.REQUEST_UNCLEAR
        )
        self.assertIn("rephrase", decision.question.lower())

    def test_force_request_sets_both_flags(self):
        router = _router_with_llm(
            ['{"valid": true, "document": "meow.pdf", "force": true}']
        )
        with _patch_fact_gathering():
            decision = router.route("re-ingest meow.pdf")
        self.assertEqual(decision.status, ROUTER_PROCEED)
        self.assertTrue(decision.flags["force_extraction"])
        self.assertTrue(decision.flags["force_summarization"])

    def test_malformed_llm_answer_falls_back_to_keywords(self):
        """A syntactically broken answer is a retryable LLM failure."""
        router = _router_with_llm(["this is not json"])
        with _patch_fact_gathering():
            decision = router.route("please reingest meow.pdf")
        self.assertEqual(decision.status, ROUTER_PROCEED)
        self.assertTrue(decision.flags["force_extraction"])

    def test_invalid_intent_despite_document_clarifies(self):
        """The LLM said 'not an ingestion order': the user is asked, even
        though a filename was found in the raw text."""
        router = _router_with_llm([
            '{"valid": false, "clarification": "request_unclear", '
            '"reason": "this reads as a search request"}',
        ])
        with _patch_fact_gathering():
            decision = router.route("what does meow.pdf say about dragons?")
        self.assertEqual(decision.status, ROUTER_NEEDS_CLARIFICATION)
        self.assertEqual(
            decision.clarification, ClarificationKind.REQUEST_UNCLEAR
        )


# ---------------------------------------------------------------------------
# Routing graph
# ---------------------------------------------------------------------------

class RoutingGraphTest(unittest.TestCase):
    """The graph traces every step of the router flow."""

    def _graph(self, answers):
        from src.graphs.ingestion_routing_graph import IngestionRoutingGraph

        router = IngestionRouter()
        client = mock.MagicMock()
        client.complete.side_effect = answers
        with mock.patch.object(router, "_llm", return_value=client):
            return IngestionRoutingGraph(router=router)

    @staticmethod
    def _collector():
        events: list = []

        def on_event(source, kind, message, **data):
            events.append(kind)

        return events, on_event

    def test_proceed_traces(self):
        graph = self._graph(['{"valid": true, "document": "meow.pdf"}'])
        events, on_event = self._collector()
        with _patch_fact_gathering():
            outcome = graph.run("ingest meow.pdf", on_event=on_event)
        self.assertEqual(outcome.status, ROUTER_PROCEED)
        self.assertIn("intent_identified", events)
        self.assertIn("facts_gathered", events)
        self.assertIn("decision", events)
        self.assertTrue(outcome.decision.flags["force_extraction"] is False)

    def test_clarification_traces(self):
        graph = self._graph([
            '{"valid": false, "clarification": "request_unclear", '
            '"reason": "unclear"}',
        ])
        events, on_event = self._collector()
        outcome = graph.run("do the thing", on_event=on_event)
        self.assertEqual(outcome.status, ROUTER_NEEDS_CLARIFICATION)
        self.assertIn("intent_identified", events)
        self.assertIn("clarification", events)
        self.assertTrue(outcome.decision.question)

    def test_observer_errors_never_break_the_flow(self):
        graph = self._graph(['{"valid": true, "document": "meow.pdf"}'])

        def broken(*args, **kwargs):
            raise RuntimeError("observer blew up")

        with _patch_fact_gathering():
            outcome = graph.run("ingest meow.pdf", on_event=broken)
        self.assertEqual(outcome.status, ROUTER_PROCEED)


if __name__ == "__main__":
    unittest.main()
