"""Tests for the two graceful-degradation fixes.

Fix A — an ingestion request without a document no longer invalidates the
whole analyzer batch (per-request degradation): the model keeps validating,
the routing graph reports the request as ``incomplete`` and the API turns
that into a question for the user.

Fix B — an unanalyzable prompt produces an honest answer text instead of an
HTTPException(503), which chat clients treated as a retryable failure and
silently re-sent (re-running accepted routing under the hood).
"""

from __future__ import annotations

import unittest
import unittest.mock

import app.api as api
from src.graphs.routing_graph import (
    STATUS_INCOMPLETE,
    RoutingGraph,
)
from src.agents.contexts import RoutingContext
from src.routing.models import AnalysisResult, RequestKind, UserRequest, parse_analysis


class _StubTaskAgent:
    """Minimal UserTaskAgent stand-in recording it was called."""

    name = "stub"

    def __init__(self) -> None:
        self.calls: list = []

    def run(self, context, request):
        self.calls.append(request)
        from src.agents.protocols import AgentResult, AgentStatus

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.OK,
            detail="handled",
            payload={"answer": "stub answer"},
        )


class FixAIngestionWithoutDocumentTest(unittest.TestCase):
    """Fix A: valid-but-incomplete ingestion requests degrade per request."""

    def test_documentless_ingestion_batch_still_parses_with_siblings(self):
        """The regression from the incident: one documentless ingestion must
        not destroy the general request that accompanied it."""
        raw = (
            '{"requests": ['
            '{"kind": "ingestion", "utterance": "ingest some documents",'
            ' "document": null, "question": null,'
            ' "options": {"force_reingest": false, "section_scope": null}},'
            '{"kind": "general", "utterance": "Hi Winter!"}'
            "]}"
        )
        result = parse_analysis(raw)
        self.assertEqual(len(result.requests), 2)
        self.assertIsNone(result.requests[0].document)
        self.assertEqual(result.requests[1].kind, RequestKind.GENERAL)

    def test_graph_reports_incomplete_without_calling_the_agent(self):
        agent = _StubTaskAgent()
        graph = RoutingGraph(agents={"ingestion_task": agent})
        context = RoutingContext()

        outcome = graph.run(
            context,
            [UserRequest(kind=RequestKind.INGESTION, utterance="ingest something")],
        )

        self.assertEqual(outcome.outcomes[0].status, STATUS_INCOMPLETE)
        self.assertEqual(agent.calls, [])  # nothing was dispatched
        self.assertIn("which document", outcome.outcomes[0].detail)

    def test_complete_ingestion_still_dispatches_normally(self):
        agent = _StubTaskAgent()
        graph = RoutingGraph(agents={"ingestion_task": agent})
        context = RoutingContext()

        outcome = graph.run(
            context,
            [UserRequest(
                kind=RequestKind.INGESTION,
                utterance="ingest Dumas.pdf",
                document="Dumas.pdf",
            )],
        )

        self.assertEqual(outcome.outcomes[0].status, "done")
        self.assertEqual(len(agent.calls), 1)

    def test_compose_reply_turns_incomplete_into_a_question(self):
        text, needs_rag = api._compose_reply([
            {"kind": "ingestion", "status": "incomplete",
             "detail": "which document should be ingested?"},
        ])
        self.assertIn("More information needed", text)
        self.assertIn("which document", text)
        self.assertFalse(needs_rag)

    def test_incomplete_does_not_poison_sibling_outcomes(self):
        agent = _StubTaskAgent()
        graph = RoutingGraph(agents={"general_task": agent})
        context = RoutingContext()

        outcome = graph.run(
            context,
            [
                UserRequest(kind=RequestKind.INGESTION, utterance="ingest some documents"),
                UserRequest(kind=RequestKind.GENERAL, utterance="Hi!"),
            ],
        )

        statuses = [o.status for o in outcome.outcomes]
        self.assertEqual(statuses, [STATUS_INCOMPLETE, "done"])


class FixBAnalysisErrorAnswerTest(unittest.TestCase):
    """Fix B: analysis errors answer the user instead of raising a 503."""

    def _route_with_analysis_error(self, cause: str):
        """_route_or_answer imports run_routing at call time, so patching the
        module attribute intercepts it."""
        with unittest.mock.patch(
            "src.routing.routing_orchestrator.run_routing",
            return_value={
                "status": "analysis_error",
                "cause": cause,
                "message": "request analysis failed: unusable LLM answer (boom)",
                "results": [],
                "traces": [],
            },
        ):
            return api._route_or_answer("some prompt")

    def test_analysis_error_becomes_an_answer_not_an_exception(self):
        text, results = self._route_with_analysis_error("llm_response")
        self.assertIn("could not analyze", text)
        self.assertIn("rephrase", text)
        self.assertEqual(results, [])

    def test_llm_request_cause_names_ollama(self):
        text, _ = self._route_with_analysis_error("llm_request")
        self.assertIn("Ollama", text)

    def test_config_cause_carries_the_message(self):
        text, _ = self._route_with_analysis_error("config")
        self.assertIn("configuration", text)

    def test_streaming_path_yields_the_answer_instead_of_the_error(self):
        """The streaming path must not blow up either: the worker converts
        HTTPException to an in-stream final answer; with fix B there is no
        exception at all — the honest reply flows through as content."""
        with unittest.mock.patch(
            "src.routing.routing_orchestrator.run_routing",
            return_value={
                "status": "analysis_error",
                "cause": "llm_response",
                "message": "request analysis failed: unusable LLM answer (boom)",
                "results": [],
                "traces": [],
            },
        ):
            items = list(api._routing_stream("some prompt"))

        final = [item for item in items if item[0] == "final"]
        self.assertEqual(len(final), 1)
        self.assertIn("could not analyze", final[0][1])


if __name__ == "__main__":
    unittest.main()
