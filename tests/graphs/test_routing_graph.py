"""Tests for the routing graph dispatch loop (grouped requests).

Post-paradigm change: no ingestion requests, no origin gate — the graph
dispatches retrieval and general requests only. Ingestion is a CLI
operation (scripts/ingest.py), unreachable from the chat.
"""

from __future__ import annotations

import unittest

from src.agents.contexts import RoutingContext
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
from src.graphs import RoutingGraph
from src.routing.models import (
    GeneralRequest,
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
        retrieval, general = StubAgent(), StubAgent()
        graph = RoutingGraph(agents={"retrieval_task": retrieval, "general_task": general})
        outcome = graph.run(RoutingContext(), [
            RetrievalRequest(question="what is stored?"),
            GeneralRequest(question="hello"),
        ])
        self.assertEqual(len(retrieval.calls), 1)
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
            "retrieval_task": StubAgent(fail=True, detail="nope"),
            "general_task": StubAgent(detail="fine"),
        })
        outcome = graph.run(RoutingContext(), [
            RetrievalRequest(question="what is stored?"),
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

    def test_dispatch_grouped_scope_order_retrieval_first(self):
        """Dispatch order: all retrievals, then generals (the grouped
        scope order :meth:`AnalysisResult.flattened` produces)."""
        seen = []

        class Recorder(StubAgent):
            def run(self, context, request):
                seen.append(type(request).__name__)
                return super().run(context, request)

        graph = RoutingGraph(agents={
            "retrieval_task": Recorder(),
            "general_task": Recorder(),
        })
        graph.run(RoutingContext(), [
            RetrievalRequest(question="r"),
            GeneralRequest(question="g"),
        ])
        self.assertEqual(seen, ["RetrievalRequest", "GeneralRequest"])

    def test_dispatch_preserves_input_order(self):
        """The graph runs the list it receives as-is: the grouped scope
        order is :meth:`AnalysisResult.flattened`'s job (analyzer side),
        not the dispatcher's."""
        seen = []

        class Recorder(StubAgent):
            def run(self, context, request):
                seen.append(type(request).__name__)
                return super().run(context, request)

        graph = RoutingGraph(agents={
            "retrieval_task": Recorder(),
            "general_task": Recorder(),
        })
        graph.run(RoutingContext(), [
            GeneralRequest(question="g"),
            RetrievalRequest(question="r"),
        ])
        self.assertEqual(seen, ["GeneralRequest", "RetrievalRequest"])


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
