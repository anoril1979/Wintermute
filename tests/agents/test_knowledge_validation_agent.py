"""Tests for the KnowledgeValidatorAgent (knowledge_validation step).

Hermetic: no LLM, no stores — pure validation. Covers the OK path,
per-field shape problems (LLM_RESPONSE → retryable), provenance-format
checking, model-consistency checking, warnings, and status mapping.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.agents.agents.knowledge_validation_agent import KnowledgeValidatorAgent
from src.agents.contexts import IngestionContext
from src.agents.protocols import AgentStatus

VALID_ID = "doc:36a911e2::chp:1::pg:1::sec:2"


def make_context(entries):
    context = IngestionContext(document_path=Path("gazette.pdf"))
    context.outputs["knowledge_characters"] = {"characters": entries}
    return context


def valid_entry(**overrides):
    entry = {
        "full_name": "Joe le Clodo",
        "short_name": "Joe",
        "aliases": ["Bobby"],
        "source_ids": [VALID_ID],
    }
    entry.update(overrides)
    return entry


class ValidatorTest(unittest.TestCase):
    def setUp(self):
        self.agent = KnowledgeValidatorAgent()

    def test_no_output_is_skipped(self):
        context = IngestionContext(document_path=Path("gazette.pdf"))
        result = self.agent.run(context)
        self.assertEqual(result.status, AgentStatus.SKIPPED)

    def test_valid_entries_pass(self):
        context = make_context([valid_entry(), valid_entry(full_name="Mercedes")])
        result = self.agent.run(context)
        self.assertEqual(result.status, AgentStatus.OK)
        self.assertEqual(result.payload["entries"], 2)
        self.assertEqual(result.payload["warnings"], [])

    def test_malformed_entry_is_retryable_llm_response(self):
        context = make_context([valid_entry(), {"aliases": ["no name"]}])
        result = self.agent.run(context)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.failure_domain.value, "llm_response")
        self.assertIn("full_name", result.detail)

    def test_non_object_entry_fails(self):
        context = make_context(["not a dict"])
        result = self.agent.run(context)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.failure_domain.value, "llm_response")

    def test_missing_characters_list_fails(self):
        context = IngestionContext(document_path=Path("gazette.pdf"))
        context.outputs["knowledge_characters"] = {"nope": []}
        result = self.agent.run(context)
        self.assertEqual(result.status, AgentStatus.FAILED)

    def test_bad_source_id_format_fails(self):
        context = make_context([valid_entry(source_ids=["doc:xyz::sec:0"])])
        result = self.agent.run(context)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("source_ids", result.detail)

    def test_non_unit_provenance_fails(self):
        # A bare doc id is NOT a unit id: the resolver writes unit-level
        # provenance, the removal purges by the doc prefix — a bare id
        # would silently lose the purge.
        context = make_context([valid_entry(source_ids=["doc:36a911e2"])])
        result = self.agent.run(context)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("source_ids", result.detail)

    def test_empty_source_ids_warn_but_pass(self):
        context = make_context([valid_entry(source_ids=[])])
        result = self.agent.run(context)
        self.assertEqual(result.status, AgentStatus.OK)
        self.assertEqual(len(result.payload["warnings"]), 1)

    def test_model_inconsistency_fails(self):
        # 'char:...' style ids are internal: a Character model built from
        # this entry must still derive cleanly — an entry that cannot
        # instantiate the model is a problem.
        context = make_context([valid_entry(full_name="   ", aliases=None)])
        result = self.agent.run(context)
        self.assertEqual(result.status, AgentStatus.FAILED)

    def test_duplicate_full_names_warn(self):
        context = make_context([
            valid_entry(short_name="Joe"),
            valid_entry(short_name=None),
        ])
        result = self.agent.run(context)
        self.assertEqual(result.status, AgentStatus.OK)
        self.assertEqual(len(result.payload["warnings"]), 1)
        self.assertIn("share the full name", result.payload["warnings"][0])

    def test_document_level_provenance_accepted(self):
        # doc::<level> chains of any depth are legitimate (chapter- or
        # page-granularity extractions).
        context = make_context([valid_entry(
            source_ids=["doc:36a911e2::chp:1::pg:2",
                        "doc:36a911e2::chp:1"])])
        result = self.agent.run(context)
        self.assertEqual(result.status, AgentStatus.OK)

    def test_trace_kinds_emitted(self):
        context = make_context([valid_entry()])
        self.agent.run(context)
        kinds = [e["kind"] for e in context.events]
        self.assertIn("knowledge_check", kinds)
        self.assertIn("knowledge_validated", kinds)


if __name__ == "__main__":
    unittest.main()
