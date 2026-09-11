"""Tests for the LLM-backed request analyzer (mocked LLM client)."""

from __future__ import annotations

import json
import unittest

from src.routing import models as routing_models
from src.routing.models import AnalysisResult, RequestKind
from src.routing.request_analyzer import (
    ANALYSIS_PROMPT_PATH,
    RequestAnalysisError,
    RequestAnalyzer,
)


class FakeLLM:
    def __init__(self, answer: str, *, fail: Exception | None = None) -> None:
        self.answer = answer
        self.fail = fail
        self.calls: list = []

    def complete(self, prompt: str, max_tokens: int | None = None) -> str:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        if self.fail is not None:
            raise self.fail
        return self.answer


class AnalyzerTest(unittest.TestCase):
    def test_analyzes_prompt_into_requests(self):
        answer = json.dumps({"requests": [
            {"kind": "ingestion", "utterance": "ingest a.pdf", "document": "a.pdf"},
            {"kind": "retrieval", "utterance": "who?", "question": "Who?"},
        ]})
        llm = FakeLLM(answer)
        analyzer = RequestAnalyzer()
        analyzer._llm = lambda: llm

        result = analyzer.analyze("ingest a.pdf then who?")
        self.assertEqual([r.kind for r in result.requests],
                         [RequestKind.INGESTION, RequestKind.RETRIEVAL])
        # The user prompt must travel inside the LLM prompt...
        self.assertIn("ingest a.pdf then who?", llm.calls[0]["prompt"])
        # ...after the analysis instructions...
        self.assertIn(ANALYSIS_PROMPT_PATH.read_text(encoding="utf-8")[:80].strip(),
                      llm.calls[0]["prompt"])
        # ...with no explicit budget: the role's max_response_tokens
        # (config/llm.yaml) is the delegated ceiling.
        self.assertIsNone(llm.calls[0]["max_tokens"])

    def test_blank_prompt_yields_empty_batch_without_llm_call(self):
        llm = FakeLLM("should not be called")
        analyzer = RequestAnalyzer()
        analyzer._llm = lambda: llm
        result = analyzer.analyze("   ")
        self.assertEqual(result.requests, [])
        self.assertEqual(llm.calls, [])

    def test_llm_failure_maps_to_llm_request_cause(self):
        analyzer = RequestAnalyzer()
        analyzer._llm = lambda: FakeLLM("", fail=RuntimeError("connection refused"))
        with self.assertRaises(RequestAnalysisError) as ctx:
            analyzer.analyze("hello")
        self.assertEqual(ctx.exception.cause, "llm_request")

    def test_malformed_answer_maps_to_llm_response_cause(self):
        analyzer = RequestAnalyzer()
        analyzer._llm = lambda: FakeLLM("this is not json at all")
        with self.assertRaises(RequestAnalysisError) as ctx:
            analyzer.analyze("hello")
        self.assertEqual(ctx.exception.cause, "llm_response")

    def test_documentless_ingestion_answer_is_valid_incomplete(self):
        # Fix A: an ingestion request without a document is valid-but-
        # incomplete (the analyzer must not invent file names); the routing
        # graph degrades it per request instead of failing the batch.
        analyzer = RequestAnalyzer()
        analyzer._llm = lambda: FakeLLM(json.dumps(
            {"requests": [{"kind": "ingestion", "utterance": "x"}]}  # no document
        ))
        result = analyzer.analyze("hello")
        self.assertEqual(len(result.requests), 1)
        self.assertIsNone(result.requests[0].document)


if __name__ == "__main__":
    unittest.main()
