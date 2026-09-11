"""Tests for the routing graph dispatch loop."""

from __future__ import annotations

import unittest

from src.agents.contexts import RoutingContext
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
from src.graphs import RoutingGraph
from src.routing.models import RequestKind, UserRequest


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


def _request(kind: RequestKind, **kwargs) -> UserRequest:
    base = {"utterance": "u"}
    base.update(kwargs)
    return UserRequest(kind=kind, **base)


class RoutingGraphTest(unittest.TestCase):
    def test_dispatch_per_kind(self):
        ingestion, general = StubAgent(), StubAgent()
        graph = RoutingGraph(agents={"ingestion_task": ingestion, "general_task": general})
        outcome = graph.run(RoutingContext(), [
            _request(RequestKind.INGESTION, document="a.pdf"),
            _request(RequestKind.GENERAL),
        ])
        self.assertEqual(len(ingestion.calls), 1)
        self.assertEqual(len(general.calls), 1)
        self.assertEqual([o.status for o in outcome.outcomes], ["done", "done"])
        self.assertTrue(outcome.handled)

    def test_missing_agent_is_not_implemented(self):
        graph = RoutingGraph(agents={})
        outcome = graph.run(RoutingContext(), [_request(RequestKind.RETRIEVAL, question="q?")])
        self.assertEqual(outcome.outcomes[0].status, "not_implemented")
        self.assertFalse(outcome.handled)

    def test_agent_failure_does_not_abort_batch(self):
        graph = RoutingGraph(agents={
            "ingestion_task": StubAgent(fail=True, detail="nope"),
            "general_task": StubAgent(detail="fine"),
        })
        outcome = graph.run(RoutingContext(), [
            _request(RequestKind.INGESTION, document="a.pdf"),
            _request(RequestKind.GENERAL),
        ])
        self.assertEqual([o.status for o in outcome.outcomes], ["rejected", "done"])
        self.assertEqual(outcome.outcomes[0].detail, "nope")

    def test_crashing_agent_does_not_abort_batch(self):
        class Boom(StubAgent):
            def run(self, context, request):
                raise RuntimeError("boom")

        graph = RoutingGraph(agents={"general_task": Boom()})
        outcome = graph.run(RoutingContext(), [_request(RequestKind.GENERAL)])
        self.assertEqual(outcome.outcomes[0].status, "rejected")
        self.assertIn("boom", outcome.outcomes[0].detail)

    def test_results_appended_to_context_in_order(self):
        graph = RoutingGraph(agents={"general_task": StubAgent()})
        context = RoutingContext(request="p")
        outcome = graph.run(context, [_request(RequestKind.GENERAL)])
        self.assertEqual(len(context.results), 1)
        self.assertEqual(context.results[0]["status"], "done")
        self.assertEqual(outcome.as_list(), context.results)


if __name__ == "__main__":
    unittest.main()
