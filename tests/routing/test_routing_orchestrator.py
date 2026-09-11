"""Tests for the routing orchestrator (analysis + dispatch + statuses)."""

from __future__ import annotations

import unittest

from src.agents.contexts import RoutingContext
from src.agents.agents.ingestion_task_agent import IngestionTaskAgent
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
from src.routing import RequestKind
from src.routing.models import AnalysisResult, UserRequest
from src.routing.request_analyzer import RequestAnalysisError, RequestAnalyzer
from src.routing.routing_orchestrator import (
    STATUS_ANALYSIS_ERROR,
    STATUS_HANDLED,
    STATUS_PARTIAL,
    run_routing,
)
from src.tools.config_loader import ConfigError


class FakeAnalyzer:
    """Deterministic analyzer for orchestrator tests."""

    def __init__(self, requests=None, error: RequestAnalysisError | None = None):
        self.requests = requests or []
        self.error = error

    def analyze(self, prompt):
        if self.error is not None:
            raise self.error
        return AnalysisResult(requests=self.requests)


def _ingestion(document="a.pdf"):
    return UserRequest(kind=RequestKind.INGESTION, utterance="ingest a.pdf", document=document)


def _general():
    return UserRequest(kind=RequestKind.GENERAL, utterance="hello")


class OkTaskAgent:
    name = "ok_agent"

    def run(self, context, request):
        return AgentResult(agent_name=self.name, status=AgentStatus.OK, detail="done")

    def validate(self, context, request):
        return None


class RunRoutingTest(unittest.TestCase):
    def test_handled_when_an_agent_answers(self):
        result = run_routing(
            "anything", analyzer=FakeAnalyzer([_general()]),
            agents={"general_task": OkTaskAgent()},
        )
        self.assertEqual(result["status"], STATUS_HANDLED)
        self.assertEqual(result["results"][0]["status"], "done")

    def test_partial_when_nothing_reached_an_agent(self):
        result = run_routing("anything", analyzer=FakeAnalyzer([_general()]), agents={})
        self.assertEqual(result["status"], STATUS_PARTIAL)
        self.assertEqual(result["results"][0]["status"], "not_implemented")

    def test_empty_analysis_is_partial(self):
        result = run_routing("anything", analyzer=FakeAnalyzer([]), agents={})
        self.assertEqual(result["status"], STATUS_PARTIAL)
        self.assertEqual(result["results"], [])

    def test_analysis_error_llm_request(self):
        result = run_routing(
            "anything",
            analyzer=FakeAnalyzer(error=RequestAnalysisError("down", cause="llm_request")),
            agents={},
        )
        self.assertEqual(result["status"], STATUS_ANALYSIS_ERROR)
        self.assertEqual(result["cause"], "llm_request")

    def test_analysis_error_llm_response(self):
        result = run_routing(
            "anything",
            analyzer=FakeAnalyzer(error=RequestAnalysisError("bad", cause="llm_response")),
            agents={},
        )
        self.assertEqual(result["status"], STATUS_ANALYSIS_ERROR)
        self.assertEqual(result["cause"], "llm_response")

    def test_agent_wiring_config_error_is_graceful(self):
        from unittest import mock

        with mock.patch(
            "src.routing.routing_orchestrator.build_default_task_agents",
            side_effect=ConfigError("missing role"),
        ):
            result = run_routing("anything", analyzer=FakeAnalyzer([_ingestion()]), agents=None)
        self.assertEqual(result["status"], STATUS_ANALYSIS_ERROR)
        self.assertEqual(result["cause"], "config")


if __name__ == "__main__":
    unittest.main()
