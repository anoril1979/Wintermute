"""Tests for the grouped routing request models and the LLM-answer parser.

Post-paradigm change: there is NO ingestion scope. The analyzer cannot
emit ingestion requests — an ``ingestion`` key in the LLM answer is
dropped with a warning at the parse boundary (the structural successor of
the old phantom-ingestion guard).
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

from pydantic import ValidationError

from src.routing.models import (
    AnalysisResult,
    GeneralRequest,
    RetrievalLookupKind,
    RetrievalRequest,
    RequestContextEntry,
    max_requests_per_prompt,
    parse_analysis,
)


def _payload(**scopes) -> str:
    base = {
        "retrieval": [],
        "general": [],
    }
    base.update(scopes)
    return json.dumps(base)


class ParseAnalysisTest(unittest.TestCase):
    def test_valid_grouped_answer(self):
        raw = _payload(
            retrieval=[{"question": "Who marries Edmond?", "utterance": "who marries Edmond?"}],
            general=[{"question": "Hello", "utterance": "Hello"}],
        )
        result = parse_analysis(raw)
        self.assertEqual(len(result.retrieval), 1)
        self.assertEqual(len(result.general), 1)
        self.assertEqual(result.request_count, 2)

    def test_fenced_json_is_extracted(self):
        raw = "```json\n" + _payload(general=[{"question": "hi", "utterance": "hi"}]) + "\n```"
        self.assertEqual(parse_analysis(raw).request_count, 1)

    def test_prose_wrapped_json_is_extracted(self):
        raw = "Voici l'analyse: " + _payload(general=[{"question": "hi", "utterance": "hi"}]) + " (fin)"
        self.assertEqual(parse_analysis(raw).request_count, 1)

    def test_all_scopes_empty(self):
        result = parse_analysis(_payload())
        self.assertEqual(result.request_count, 0)

    def test_missing_scope_keys_default_to_empty(self):
        raw = json.dumps({"general": [{"question": "hi"}]})
        result = parse_analysis(raw)
        self.assertEqual(result.request_count, 1)
        self.assertEqual(result.retrieval, [])

    def test_empty_answer_raises(self):
        with self.assertRaises(ValueError):
            parse_analysis("   ")

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            parse_analysis("no structure here at all")

    def test_top_level_array_raises(self):
        with self.assertRaises(ValueError):
            parse_analysis(json.dumps([{"general": []}]))

    def test_unknown_field_is_rejected(self):
        with self.assertRaises(ValidationError):
            parse_analysis(_payload(general=[{"question": "hi", "hallucinated": True}]))

    def test_flattened_is_grouped_scope_order(self):
        result = parse_analysis(_payload(
            general=[{"question": "g1", "utterance": "g1"}],
            retrieval=[{"question": "r1", "utterance": "r1"}],
        ))
        kinds = [type(r).__name__ for r in result.flattened()]
        self.assertEqual(kinds, ["RetrievalRequest", "GeneralRequest"])


class IngestionScopeRemovedTest(unittest.TestCase):
    """The paradigm change at the parse boundary: no ingestion scope."""

    def test_ingestion_scope_is_dropped_with_warning(self):
        # The old model reflex (or an older prompt still live somewhere):
        # whatever the analyzer puts under "ingestion" is not dispatchable.
        raw = _payload(
            ingestion=[{"document": "vif-argent.pdf", "utterance": "ingest vif-argent.pdf"}],
            retrieval=[{"question": "que sait-on du vif-argent ?", "utterance": "vif-argent ?"}],
        )
        with self.assertLogs("src.routing.models", level="WARNING"):
            result = parse_analysis(raw)
        self.assertEqual(result.request_count, 1)
        self.assertEqual(len(result.retrieval), 1)

    def test_documentless_ingestion_also_dropped(self):
        raw = _payload(
            ingestion=[{"utterance": "ingest some documents"}],
            general=[{"question": "hello", "utterance": "hello"}],
        )
        with self.assertLogs("src.routing.models", level="WARNING"):
            result = parse_analysis(raw)
        self.assertEqual(len(result.general), 1)
        self.assertEqual(result.retrieval, [])

    def test_shorthand_flags_are_gone(self):
        # force/redo_summaries/origin were ingestion shorthands: no longer
        # part of the schema (extra="forbid" rejects them).
        raw = json.dumps({
            "retrieval": [], "general": [],
            "force": True, "redo_summaries": False, "origin": "canon",
        })
        with self.assertRaises(ValidationError):
            parse_analysis(raw)


class RetrievalRequestValidationTest(unittest.TestCase):
    def test_question_required(self):
        with self.assertRaises(ValidationError):
            RetrievalRequest(question="   ")

    def test_lookup_kinds(self):
        for kind in ("semantic", "lookup", "relationship"):
            request = RetrievalRequest(question="q", lookup_kind=kind)
            self.assertEqual(request.lookup_kind.value, kind)

    def test_lookup_kind_carries_the_entity(self):
        request = RetrievalRequest(
            question="who is Marcus", lookup_kind="lookup", entity="Marcus"
        )
        self.assertEqual(request.entity, "Marcus")
        blank = RetrievalRequest(question="q", lookup_kind="lookup", entity="  ")
        self.assertIsNone(blank.entity, "a blank entity normalizes to None")

    def test_unknown_lookup_kind_rejected(self):
        with self.assertRaises(ValidationError):
            RetrievalRequest(question="q", lookup_kind="telepathy")

    def test_top_k_bounds(self):
        with self.assertRaises(ValidationError):
            RetrievalRequest(question="q", top_k=0)
        request = RetrievalRequest(question="q", top_k=10)
        self.assertEqual(request.top_k, 10)

    def test_path_like_document_rejected(self):
        with self.assertRaises(ValidationError):
            RetrievalRequest(question="q", document="sub/dir.pdf")

    def test_blank_document_normalized_to_none(self):
        request = RetrievalRequest(question="q", document="   ")
        self.assertIsNone(request.document)


class GeneralRequestTest(unittest.TestCase):
    def test_question_required(self):
        with self.assertRaises(ValidationError):
            GeneralRequest(question="  ")

    def test_summary(self):
        request = GeneralRequest(question="Who are you?")
        self.assertEqual(request.summary()["question"], "Who are you?")


class PromptLocalMemoryTest(unittest.TestCase):
    """``preceding`` — the dispatcher-owned prompt-local memory."""

    def test_preceding_defaults_to_empty(self):
        request = GeneralRequest(question="hi")
        self.assertEqual(request.preceding, [])

    def test_preceding_entries_survive_construction(self):
        request = GeneralRequest(
            question="so, is it indexed?",
            preceding=[{
                "kind": "retrieval", "utterance": "index it?",
                "status": "done",
            }],
        )
        self.assertEqual(len(request.preceding), 1)
        self.assertEqual(request.preceding[0].status, "done")

    def test_analyzer_emitted_preceding_is_dropped(self):
        raw = _payload(general=[{
            "question": "and now?", "utterance": "and now?",
            "preceding": [{"kind": "retrieval", "utterance": "invented"}],
        }])
        with self.assertLogs("src.routing.models", level="WARNING"):
            result = parse_analysis(raw)
        self.assertEqual(result.general[0].preceding, [])

    def test_entry_rejects_unknown_fields(self):
        with self.assertRaises(ValidationError):
            RequestContextEntry(kind="GeneralRequest", utterance="u", bogus=1)


class CapTest(unittest.TestCase):
    """The cap comes from setup.yaml routing.max_requests_per_prompt."""

    def test_cap_helper_reads_config(self):
        with mock.patch("src.tools.config_loader.load_routing_config",
                        return_value={"max_requests_per_prompt": 5}):
            self.assertEqual(max_requests_per_prompt(), 5)

    def test_cap_helper_fails_open(self):
        with mock.patch("src.tools.config_loader.load_routing_config",
                        side_effect=RuntimeError("broken yaml")):
            self.assertEqual(max_requests_per_prompt(), 8)

    def test_over_cap_analysis_rejected(self):
        # Default cap is 8: a 9-request answer is a malformed analysis,
        # not an order.
        many = [{"question": f"q{i}", "utterance": f"u{i}"} for i in range(9)]
        with mock.patch("src.tools.config_loader.load_routing_config",
                        return_value={"max_requests_per_prompt": 8}):
            with self.assertRaises(ValidationError):
                AnalysisResult.model_validate({"general": many})

    def test_cap_three_requests_with_cap_of_two(self):
        with mock.patch("src.tools.config_loader.load_routing_config",
                        return_value={"max_requests_per_prompt": 2}):
            with self.assertRaises(ValidationError):
                AnalysisResult(
                    retrieval=[RetrievalRequest(question="r1"),
                               RetrievalRequest(question="r2")],
                    general=[GeneralRequest(question="g")],
                )


if __name__ == "__main__":
    unittest.main()
