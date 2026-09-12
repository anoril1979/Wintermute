"""Tests for the grouped routing request models and the LLM-answer parser."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from pydantic import ValidationError

from src.routing.models import (
    AnalysisResult,
    GeneralRequest,
    IngestionRequest,
    RetrievalLookupKind,
    RetrievalRequest,
    RequestContextEntry,
    max_requests_per_prompt,
    parse_analysis,
)


def _payload(**scopes) -> str:
    base = {
        "ingestion": [],
        "retrieval": [],
        "general": [],
        "force": False,
        "redo_summaries": False,
        "origin": None,
    }
    base.update(scopes)
    return json.dumps(base)


class ParseAnalysisTest(unittest.TestCase):
    def test_valid_grouped_answer(self):
        raw = _payload(
            ingestion=[{"document": "Dumas.pdf", "utterance": "Ingest Dumas.pdf"}],
            retrieval=[{"question": "Who marries Edmond?", "utterance": "who marries Edmond?"}],
            general=[{"question": "Hello", "utterance": "Hello"}],
        )
        result = parse_analysis(raw)
        self.assertEqual(len(result.ingestion), 1)
        self.assertEqual(len(result.retrieval), 1)
        self.assertEqual(len(result.general), 1)
        self.assertEqual(result.request_count, 3)

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
        self.assertEqual(result.ingestion, [])

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
            ingestion=[{"document": "a.pdf", "utterance": "ingest a.pdf"}],
        ))
        kinds = [type(r).__name__ for r in result.flattened()]
        self.assertEqual(
            kinds, ["IngestionRequest", "RetrievalRequest", "GeneralRequest"]
        )


class ShorthandTest(unittest.TestCase):
    """Prompt-level force/redo_summaries/origin apply to ingestion requests."""

    def test_prompt_level_flags_propagate(self):
        result = parse_analysis(_payload(
            ingestion=[
                {"document": "a.pdf", "utterance": "ingest a.pdf"},
                {"document": "b.pdf", "force": True, "utterance": "re-extract b.pdf"},
            ],
            force=True,
            origin="community",
        ))
        a, b = result.ingestion
        # prompt-level shorthand fills what the request left unstated...
        self.assertTrue(a.force)
        self.assertEqual(a.origin, "community")
        # ...but a per-request value wins
        self.assertTrue(b.force)
        self.assertEqual(b.origin, "community")

    def test_per_request_origin_wins(self):
        result = parse_analysis(_payload(
            ingestion=[{"document": "a.pdf", "origin": "rpg", "utterance": "ingest a.pdf"}],
            origin="canon",
        ))
        self.assertEqual(result.ingestion[0].origin, "rpg")

    def test_unknown_origin_rejected(self):
        with self.assertRaises(ValidationError):
            parse_analysis(_payload(origin="official"))

    def test_unknown_request_origin_rejected(self):
        with self.assertRaises(ValidationError):
            parse_analysis(_payload(
                ingestion=[{"document": "a.pdf", "origin": "official", "utterance": "ingest a.pdf"}]
            ))


class IngestionRequestValidationTest(unittest.TestCase):
    def test_document_required(self):
        with self.assertRaises(ValidationError):
            IngestionRequest(document="")

    def test_path_like_document_rejected(self):
        for bad in ("../etc/passwd.pdf", "C:/tmp/x.pdf", "sub/dir.pdf", "a\\b.pdf"):
            with self.assertRaises(ValidationError):
                IngestionRequest(document=bad)

    def test_document_is_stripped(self):
        request = IngestionRequest(document="  a.pdf ")
        self.assertEqual(request.document, "a.pdf")

    def test_valid_origins_accepted(self):
        for origin in ("canon", "community", "rpg", None):
            request = IngestionRequest(document="a.pdf", origin=origin)
            self.assertEqual(request.origin, origin)

    def test_summary(self):
        request = IngestionRequest(document="a.pdf", force=True, origin="rpg")
        summary = request.summary()
        self.assertEqual(summary["document"], "a.pdf")
        self.assertTrue(summary["force"])
        self.assertEqual(summary["origin"], "rpg")


class RetrievalRequestValidationTest(unittest.TestCase):
    def test_question_required(self):
        with self.assertRaises(ValidationError):
            RetrievalRequest(question="   ")

    def test_lookup_kinds(self):
        for kind in ("semantic", "index", "relation", "summary", "listing"):
            request = RetrievalRequest(question="q", lookup_kind=kind)
            self.assertEqual(request.lookup_kind.value, kind)

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
                "kind": "IngestionRequest", "utterance": "ingest meow.pdf",
                "status": "done",
            }],
        )
        self.assertEqual(len(request.preceding), 1)
        self.assertEqual(request.preceding[0].status, "done")

    def test_analyzer_emitted_preceding_is_dropped(self):
        raw = _payload(general=[{
            "question": "and now?", "utterance": "and now?",
            "preceding": [{"kind": "IngestionRequest", "utterance": "invented"}],
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
                AnalysisResult.model_validate({
                    "general": many,
                    "force": False, "redo_summaries": False, "origin": None,
                })

    def test_cap_three_requests_with_cap_of_two(self):
        with mock.patch("src.tools.config_loader.load_routing_config",
                        return_value={"max_requests_per_prompt": 2}):
            with self.assertRaises(ValidationError):
                AnalysisResult(
                    ingestion=[IngestionRequest(document="a.pdf")],
                    retrieval=[RetrievalRequest(question="r")],
                    general=[GeneralRequest(question="g")],
                )


class PhantomIngestionGuardTest(unittest.TestCase):
    """The parse boundary drops hallucinated ingestion orders (vif-argent)."""

    def test_phantom_from_content_question_dropped(self):
        # The original incident: a French content question turned into
        # an ingestion of "vif-argent.pdf"; the retrieval survived.
        raw = _payload(
            ingestion=[{
                "document": "vif-argent.pdf",
                "force": False, "redo_summaries": False, "origin": None,
                "utterance": "OK, bien. Dis-moi ce que tu sais d'une épée de vif-argent ?",
            }],
            retrieval=[{
                "lookup_kind": "semantic",
                "question": "tout savoir sur l'épée de vif-argent",
                "document": None, "chapter_title": None, "top_k": None,
                "reason": "open content question",
                "utterance": "Dis-moi ce que tu sais d'une épée de vif-argent ?",
            }],
        )
        result = parse_analysis(raw)
        self.assertEqual(result.ingestion, [])
        self.assertEqual(len(result.retrieval), 1)

    def test_real_orders_survive(self):
        raw = _payload(ingestion=[
            {"document": "a.pdf", "utterance": "ingest a.pdf please"},
            {"document": "b.pdf", "origin": "canon", "utterance": "re-extract b.pdf, the file changed"},
            {"document": "c.pdf", "utterance": "ajoute c.pdf à la bibliothèque"},
            {"document": "d.pdf", "utterance": "recharge d.pdf avec le nouveau contenu"},
        ])
        result = parse_analysis(raw)
        self.assertEqual([r.document for r in result.ingestion], ["a.pdf", "b.pdf", "c.pdf", "d.pdf"])

    def test_mixed_batch_phantom_and_real(self):
        raw = _payload(ingestion=[
            {"document": "vif.pdf", "utterance": "Dis-moi ce que tu sais d'une épée de vif-argent ?"},
            {"document": "a.pdf", "utterance": "ajoute a.pdf à la bibliothèque"},
        ])
        result = parse_analysis(raw)
        self.assertEqual(len(result.ingestion), 1)
        self.assertEqual(result.ingestion[0].document, "a.pdf")

    def test_empty_utterance_kept_fail_open(self):
        # CLI/tests build items without an utterance: nothing to check,
        # the guard must not eat them.
        raw = _payload(ingestion=[{"document": "a.pdf"}])
        result = parse_analysis(raw)
        self.assertEqual(len(result.ingestion), 1)

    def test_no_storage_intent_dropped(self):
        raw = _payload(ingestion=[
            {"document": "rules.pdf", "utterance": "What are the rules of this document?"},
        ])
        result = parse_analysis(raw)
        self.assertEqual(result.ingestion, [])


if __name__ == "__main__":
    unittest.main()
