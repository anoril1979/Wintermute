"""End-to-end: prompt-local memory across one user prompt's requests.

The user asks several things in one prompt ("ingest meow.pdf, then tell
me: is it indexed?"). The analyzer splits them into grouped scopes; the
routing graph dispatches them in grouped order, attaching each request's
predecessors; the task agent renders that local context into its LLM
prompt. These tests run the whole flow with stubbed components — no live
Ollama.
"""

from __future__ import annotations

import unittest

from src.agents.contexts import RoutingContext
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
from src.graphs import RoutingGraph
from src.routing.models import (
    AnalysisResult,
    GeneralRequest,
    IngestionRequest,
)


class FakeAnalyzer:
    def __init__(self, requests):
        self.requests = requests

    def analyze(self, prompt):
        result = AnalysisResult()
        for request in self.requests:
            if isinstance(request, IngestionRequest):
                result.ingestion.append(request)
            else:
                result.general.append(request)
        return result


class RecordingGeneralAgent:
    """Captures the request it receives, answers OK."""

    name = "recording_general"

    def __init__(self):
        self.seen = []

    def run(self, context, request):
        self.seen.append(request)
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.OK,
            detail="answered",
            payload={"answer": "Cold and clear."},
        )

    def validate(self, context, request):
        return None


class FailingIngestionAgent:
    name = "failing_ingestion"

    def run(self, context, request):
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.FAILED,
            failure_domain=FailureDomain.LLM_RESPONSE,
            detail="ollama down",
        )

    def validate(self, context, request):
        return None


def _ingestion(document):
    return IngestionRequest(utterance=f"ingest {document}", document=document)


class PromptLocalMemoryEndToEndTest(unittest.TestCase):
    def test_general_request_sees_preceding_ingestion(self):
        """ingest meow.pdf + 'is it indexed?' — the second request resolves 'it'."""
        agent = RecordingGeneralAgent()
        graph = RoutingGraph(agents={
            "ingestion_task": FailingIngestionAgent(),  # failure must not block
            "general_task": agent,
        })
        requests = [
            _ingestion("meow.pdf"),
            GeneralRequest(utterance="is it indexed?", question="is it indexed?"),
        ]
        outcome = graph.run(RoutingContext(request="p"), requests)

        self.assertEqual(len(agent.seen), 1)
        seen = agent.seen[0]
        self.assertEqual(len(seen.preceding), 1)
        entry = seen.preceding[0]
        self.assertEqual(entry.kind, "ingestion")
        self.assertEqual(entry.utterance, "ingest meow.pdf")
        self.assertEqual(entry.status, "rejected")  # the real outcome
        self.assertEqual(entry.detail, "ollama down")

        self.assertEqual(outcome.outcomes[1].status, "done")

    def test_three_request_chain_accumulates(self):
        """Request D sees C, B and A — the full user example."""
        agent = RecordingGeneralAgent()
        graph = RoutingGraph(agents={
            "ingestion_task": RecordingGeneralAgent(),  # placeholder, recorded not used
            "general_task": agent,
        })
        requests = [
            _ingestion("a.pdf"),                                        # A
            _ingestion("b.pdf"),                                        # B
            GeneralRequest(utterance="do C", question="do C"),          # C
            GeneralRequest(utterance="is D?", question="is D?"),        # D
        ]
        graph.run(RoutingContext(request="p"), requests)

        seen = agent.seen[-1]
        self.assertEqual([p.utterance for p in seen.preceding],
                         ["ingest a.pdf", "ingest b.pdf", "do C"])
        # statuses reflect what actually happened
        self.assertEqual([p.status for p in seen.preceding],
                         ["done", "done", "done"])
        # accumulation: C saw A+B; D saw A+B+C
        self.assertEqual(len(requests[2].preceding), 2)
        self.assertEqual(len(requests[3].preceding), 3)

    def test_analyzer_never_sees_or_invents_context(self):
        """The analyzer prompt contract: it only splits; the graph owns memory."""
        analyzer = FakeAnalyzer(requests=[
            _ingestion("meow.pdf"),
            GeneralRequest(utterance="is it indexed?", question="is it indexed?"),
        ])
        result = analyzer.analyze("anything")
        for request in result.flattened():
            self.assertEqual(request.preceding, [])


if __name__ == "__main__":
    unittest.main()
