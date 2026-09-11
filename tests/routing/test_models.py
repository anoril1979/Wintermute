"""Tests for the routing request models and the LLM-answer parser."""

from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from src.routing.models import (
    AnalysisResult,
    RequestContextEntry,
    RequestKind,
    UserRequest,
    parse_analysis,
)


def _payload(requests: list) -> str:
    return json.dumps({"requests": requests})


class ParseAnalysisTest(unittest.TestCase):
    def test_valid_batch(self):
        raw = _payload([
            {"kind": "ingestion", "utterance": "Ingest Dumas.pdf", "document": "Dumas.pdf"},
            {"kind": "retrieval", "utterance": "who marries Edmond?", "question": "Who marries Edmond?"},
            {"kind": "general", "utterance": "Hello"},
        ])
        result = parse_analysis(raw)
        self.assertEqual([r.kind for r in result.requests],
                         [RequestKind.INGESTION, RequestKind.RETRIEVAL, RequestKind.GENERAL])

    def test_fenced_json_is_extracted(self):
        raw = "```json\n" + _payload([{"kind": "general", "utterance": "hi"}]) + "\n```"
        self.assertEqual(len(parse_analysis(raw).requests), 1)

    def test_prose_wrapped_json_is_extracted(self):
        raw = "Voici l'analyse: " + _payload([{"kind": "general", "utterance": "hi"}]) + " (fin)"
        self.assertEqual(len(parse_analysis(raw).requests), 1)

    def test_empty_batch(self):
        self.assertEqual(parse_analysis(_payload([])).requests, [])

    def test_empty_answer_raises(self):
        with self.assertRaises(ValueError):
            parse_analysis("   ")

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            parse_analysis("no structure here at all")

    def test_top_level_array_raises(self):
        with self.assertRaises(ValueError):
            parse_analysis(json.dumps([{"kind": "general", "utterance": "x"}]))

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValidationError):
            parse_analysis(_payload([{"kind": "summary", "utterance": "x"}]))

    def test_unknown_field_is_rejected(self):
        with self.assertRaises(ValidationError):
            parse_analysis(_payload([{"kind": "general", "utterance": "x", "hallucinated": True}]))


class UserRequestValidationTest(unittest.TestCase):
    def test_ingestion_without_document_is_valid_but_incomplete(self):
        # Deliberate per-request degradation: the analyzer must not invent
        # file names, so an ingestion request may legitimately lack one; the
        # dispatcher degrades it instead of destroying the whole batch.
        request = UserRequest(kind=RequestKind.INGESTION, utterance="ingest it")
        self.assertIsNone(request.document)

    def test_retrieval_requires_question(self):
        with self.assertRaises(ValidationError):
            UserRequest(kind=RequestKind.RETRIEVAL, utterance="what?")

    def test_blank_utterance_rejected(self):
        with self.assertRaises(ValidationError):
            UserRequest(kind=RequestKind.GENERAL, utterance="   ")

    def test_path_like_document_rejected(self):
        for bad in ("../etc/passwd.pdf", "C:/tmp/x.pdf", "sub/dir.pdf", "a\\b.pdf"):
            with self.assertRaises(ValidationError):
                UserRequest(kind=RequestKind.INGESTION, utterance="u", document=bad)

    def test_general_must_not_carry_payload(self):
        with self.assertRaises(ValidationError):
            UserRequest(kind=RequestKind.GENERAL, utterance="u", document="x.pdf")
        with self.assertRaises(ValidationError):
            UserRequest(kind=RequestKind.GENERAL, utterance="u", question="q?")

    def test_option_and_summary(self):
        request = UserRequest(
            kind=RequestKind.INGESTION, utterance="u", document="a.pdf",
            options={"force_reingest": True},
        )
        self.assertTrue(request.option("force_reingest"))
        self.assertIsNone(request.option("section_scope"))
        self.assertEqual(request.summary()["kind"], "ingestion")


class PromptLocalMemoryTest(unittest.TestCase):
    """UserRequest.preceding — the dispatcher-owned prompt-local memory."""

    def test_preceding_defaults_to_empty(self):
        request = UserRequest(kind=RequestKind.GENERAL, utterance="hi")
        self.assertEqual(request.preceding, [])

    def test_preceding_entries_survive_construction(self):
        # The routing graph builds requests' context programmatically:
        # entries set in code must be kept (only the *analyzer* is blocked).
        request = UserRequest(
            kind=RequestKind.GENERAL,
            utterance="so, is it indexed?",
            preceding=[{
                "kind": "ingestion", "utterance": "ingest meow.pdf",
                "document": "meow.pdf", "status": "done",
            }],
        )
        self.assertEqual(len(request.preceding), 1)
        entry = request.preceding[0]
        self.assertEqual(entry.kind, "ingestion")
        self.assertEqual(entry.document, "meow.pdf")
        self.assertEqual(entry.status, "done")

    def test_analyzer_emitted_preceding_is_dropped(self):
        # The analyzer must never emit 'preceding' (dispatcher-owned); a
        # hallucinated value is dropped with a warning instead of killing
        # the whole batch.
        raw = _payload([{
            "kind": "general", "utterance": "and now?",
            "preceding": [{"kind": "ingestion", "utterance": "invented"}],
        }])
        with self.assertLogs("src.routing.models", level="WARNING"):
            result = parse_analysis(raw)
        self.assertEqual(result.requests[0].preceding, [])

    def test_summary_includes_preceding(self):
        request = UserRequest(
            kind=RequestKind.GENERAL, utterance="and the file?",
            preceding=[RequestContextEntry(
                kind="ingestion", utterance="ingest a.pdf",
                document="a.pdf", status="done", detail="handled",
            )],
        )
        summary = request.summary()
        self.assertEqual(summary["preceding"][0]["document"], "a.pdf")
        self.assertEqual(summary["preceding"][0]["status"], "done")

    def test_entry_rejects_unknown_fields(self):
        with self.assertRaises(ValidationError):
            RequestContextEntry(kind="general", utterance="u", bogus=1)


if __name__ == "__main__":
    unittest.main()
