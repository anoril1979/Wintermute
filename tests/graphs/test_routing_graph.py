"""Tests for the routing graph dispatch loop (grouped requests, origin gate)."""

from __future__ import annotations

import unittest
from unittest import mock

from src.agents.contexts import RoutingContext
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
from src.graphs import RoutingGraph
from src.routing.models import (
    GeneralRequest,
    IngestionRequest,
    RetrievalRequest,
)


class StubAgent:
    name = "stub"

    def __init__(self, *, fail: bool = False, detail: str = "ok") -> None:
        self.fail = fail
        self.detail = detail
        self.calls: list = []

    def run(self, context, request):
        self.calls.append(request)
        if self.fail:
            return AgentResult(agent_name=self.name, status=AgentStatus.FAILED,
                               failure_domain=FailureDomain.INPUT_DATA, detail=self.detail)
        return AgentResult(agent_name=self.name, status=AgentStatus.OK,
                           detail=self.detail, payload={"answer": "42"})

    def validate(self, context, request):
        return None


class RoutingGraphTest(unittest.TestCase):
    def test_dispatch_per_request_type(self):
        ingestion, general = StubAgent(), StubAgent()
        graph = RoutingGraph(agents={"ingestion_task": ingestion, "general_task": general})
        outcome = graph.run(RoutingContext(), [
            IngestionRequest(document="a.pdf"),
            GeneralRequest(question="hello"),
        ])
        self.assertEqual(len(ingestion.calls), 1)
        self.assertEqual(len(general.calls), 1)
        self.assertEqual([o.status for o in outcome.outcomes], ["done", "done"])
        self.assertTrue(outcome.handled)

    def test_missing_agent_is_not_implemented(self):
        graph = RoutingGraph(agents={})
        outcome = graph.run(RoutingContext(), [RetrievalRequest(question="q?")])
        self.assertEqual(outcome.outcomes[0].status, "not_implemented")
        self.assertFalse(outcome.handled)

    def test_agent_failure_does_not_abort_batch(self):
        graph = RoutingGraph(agents={
            "ingestion_task": StubAgent(fail=True, detail="nope"),
            "general_task": StubAgent(detail="fine"),
        })
        outcome = graph.run(RoutingContext(), [
            IngestionRequest(document="a.pdf"),
            GeneralRequest(question="hello"),
        ])
        self.assertEqual([o.status for o in outcome.outcomes], ["rejected", "done"])
        self.assertEqual(outcome.outcomes[0].detail, "nope")

    def test_crashing_agent_does_not_abort_batch(self):
        class Boom(StubAgent):
            def run(self, context, request):
                raise RuntimeError("boom")

        graph = RoutingGraph(agents={"general_task": Boom()})
        outcome = graph.run(RoutingContext(), [GeneralRequest(question="hi")])
        self.assertEqual(outcome.outcomes[0].status, "rejected")
        self.assertIn("boom", outcome.outcomes[0].detail)

    def test_results_appended_to_context_in_order(self):
        graph = RoutingGraph(agents={"general_task": StubAgent()})
        context = RoutingContext(request="p")
        outcome = graph.run(context, [GeneralRequest(question="hi")])
        self.assertEqual(len(context.results), 1)
        self.assertEqual(context.results[0]["status"], "done")
        self.assertEqual(outcome.as_list(), context.results)

    def test_kind_is_the_request_scope_label(self):
        graph = RoutingGraph(agents={"general_task": StubAgent()})
        outcome = graph.run(RoutingContext(), [GeneralRequest(question="hi")])
        self.assertEqual(outcome.outcomes[0].kind, "general")


class OriginGateTest(unittest.TestCase):
    """The deterministic origin gate: undecidable origin → set aside."""

    def _run_one(self, request):
        graph = RoutingGraph(agents={"ingestion_task": StubAgent()})
        context = RoutingContext(request="p")
        outcome = graph.run(context, [request])
        return outcome.outcomes[0], context

    def test_stated_origin_dispatches(self):
        outcome, _ = self._run_one(
            IngestionRequest(document="a.pdf", origin="rpg")
        )
        self.assertEqual(outcome.status, "done")

    def test_stored_origin_dispatches(self):
        with mock.patch(
            "src.tools.ingest_tool.ingest_document",
            return_value={"status": "ready", "path": "data/sources/pdf/a.pdf"},
        ), mock.patch(
            "src.helpers.document_extract_json_store.canonical_path_for",
            return_value=mock.Mock(exists=lambda: True),
        ), mock.patch(
            "src.helpers.document_extract_json_store.load_extract",
            return_value=mock.Mock(origin=mock.Mock(value="canon")),
        ):
            outcome, _ = self._run_one(IngestionRequest(document="a.pdf"))
        self.assertEqual(outcome.status, "done")

    def test_inferred_origin_dispatches(self):
        with mock.patch(
            "src.tools.ingest_tool.ingest_document",
            return_value={"status": "ready", "path": "data/sources/pdf/gazette #1.pdf"},
        ), mock.patch(
            "src.helpers.document_extract_json_store.canonical_path_for",
            return_value=mock.Mock(exists=lambda: False),
        ):
            outcome, _ = self._run_one(IngestionRequest(document="gazette #1.pdf"))
        self.assertEqual(outcome.status, "done")

    def test_undecidable_origin_is_set_aside(self):
        with mock.patch(
            "src.tools.ingest_tool.ingest_document",
            return_value={"status": "ready", "path": "data/sources/pdf/meow.pdf"},
        ), mock.patch(
            "src.helpers.document_extract_json_store.canonical_path_for",
            return_value=mock.Mock(exists=lambda: False),
        ), mock.patch(
            "src.ingestion.ingestion_router.infer_origin",
            return_value=None,
        ):
            outcome, context = self._run_one(IngestionRequest(document="meow.pdf"))
        self.assertEqual(outcome.status, "set_aside")
        self.assertIn("origin", outcome.detail)
        self.assertIn("question", outcome.payload)
        # The agent was never reached — nothing was ingested.
        self.assertNotIn("agent", outcome.payload)
        set_aside = [e for e in context.events if e.get("kind") == "origin_set_aside"]
        self.assertEqual(len(set_aside), 1)

    def test_set_aside_does_not_abort_batch(self):
        graph = RoutingGraph(agents={"general_task": StubAgent(detail="fine")})
        with mock.patch(
            "src.tools.ingest_tool.ingest_document",
            return_value={"status": "ready", "path": "data/sources/pdf/meow.pdf"},
        ), mock.patch(
            "src.helpers.document_extract_json_store.canonical_path_for",
            return_value=mock.Mock(exists=lambda: False),
        ), mock.patch(
            "src.ingestion.ingestion_router.infer_origin",
            return_value=None,
        ):
            outcome = graph.run(RoutingContext(), [
                IngestionRequest(document="meow.pdf"),
                GeneralRequest(question="still there?"),
            ])
        self.assertEqual(
            [o.status for o in outcome.outcomes], ["set_aside", "done"]
        )

    def test_document_not_found_falls_through_to_agent(self):
        # Resolution failure is NOT the gate's business: the task agent
        # reports candidates itself.
        with mock.patch(
            "src.tools.ingest_tool.ingest_document",
            return_value={"status": "refused", "message": "nope",
                          "candidates": ["a.pdf"]},
        ):
            outcome, _ = self._run_one(IngestionRequest(document="missing.pdf"))
        self.assertEqual(outcome.status, "done")

    def test_facts_failure_set_aside(self):
        # Fail-open: cannot check the state → set aside (never a guessed
        # origin).
        with mock.patch(
            "src.tools.ingest_tool.ingest_document",
            side_effect=RuntimeError("disk gone"),
        ):
            outcome, _ = self._run_one(IngestionRequest(document="a.pdf"))
        self.assertEqual(outcome.status, "set_aside")


class PromptLocalMemoryTest(unittest.TestCase):
    """Sequential dispatch attaches the preceding requests to each one."""

    def test_preceding_attached_with_outcomes(self):
        graph = RoutingGraph(agents={"general_task": StubAgent(detail="fine")})
        context = RoutingContext(request="p")
        outcome = graph.run(context, [
            GeneralRequest(question="first"),
            GeneralRequest(question="second — and the first?"),
        ])
        second = outcome.outcomes[1]
        self.assertEqual(second.status, "done")
        # The second request's memory holds the first's outcome.
        # (The agent received it via request.preceding; verify via the trace.)
        local = [e for e in context.events if e.get("kind") == "local_context"]
        self.assertEqual(len(local), 1)
        self.assertEqual(local[0]["data"]["index"], 1)

    def test_first_request_has_no_memory(self):
        graph = RoutingGraph(agents={"general_task": StubAgent()})
        context = RoutingContext(request="p")
        graph.run(context, [GeneralRequest(question="first")])
        local = [e for e in context.events if e.get("kind") == "local_context"]
        self.assertEqual(local, [])


if __name__ == "__main__":
    unittest.main()
