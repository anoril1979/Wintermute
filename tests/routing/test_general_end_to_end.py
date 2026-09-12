"""End-to-end routing with the real GeneralTaskAgent (LLM stubbed, API composed).

Covers the full production path minus the LLM itself:

    run_routing -> RoutingGraph -> GeneralTaskAgent.run -> payload["answer"]
    -> app.api._compose_reply -> the user-facing text.
"""

from __future__ import annotations

import unittest
import unittest.mock

from src.agents.agents.general_task_agent import GeneralTaskAgent
from src.routing.models import (
    AnalysisResult,
    GeneralRequest,
    IngestionRequest,
)
from src.routing.routing_orchestrator import run_routing

import app.api as api


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


def _general(utterance="What color is the sky?"):
    return GeneralRequest(utterance=utterance, question=utterance)


class _FakeLLM:
    def __init__(self, answer="...the color of television, tuned to a dead channel."):
        self.answer = answer

    def complete(self, prompt, max_tokens=None):
        return self.answer


class GeneralEndToEndTest(unittest.TestCase):
    def test_general_request_reaches_the_agent_and_the_reply_is_composed(self):
        agent = GeneralTaskAgent(allow_missing_role=True, llm=_FakeLLM())
        result = run_routing(
            "What color is the sky?",
            analyzer=FakeAnalyzer(requests=[_general()]),
            agents={"general_task": agent},
        )
        self.assertEqual(result["status"], "handled")
        entry = result["results"][0]
        self.assertEqual(entry["kind"], "general")
        self.assertEqual(entry["status"], "done")
        self.assertEqual(entry["agent"], "general_task")
        # The API composes the user-facing text from the answer payload.
        text, _needs_rag = api._compose_reply(result["results"])
        self.assertIn("color of television", text)

    def test_batch_keeps_general_answer_after_an_ingestion_failure(self):
        """One broken request must not poison the general answer."""

        class _FailingIngestion:
            name = "failing_ingestion"

            def run(self, context, request):
                from src.agents.protocols import AgentResult, AgentStatus, FailureDomain

                return AgentResult(
                    agent_name=self.name, status=AgentStatus.FAILED,
                    failure_domain=FailureDomain.INPUT_DATA, detail="nope",
                )

            def validate(self, context, request):
                return None

        agent = GeneralTaskAgent(allow_missing_role=True, llm=_FakeLLM("the answer"))
        result = run_routing(
            "ingest a.pdf then talk to me",
            analyzer=FakeAnalyzer(requests=[
                IngestionRequest(utterance="ingest a.pdf", document="a.pdf"),
                _general(),
            ]),
            agents={"ingestion_task": _FailingIngestion(), "general_task": agent},
        )
        statuses = {r["kind"]: r["status"] for r in result["results"]}
        self.assertEqual(statuses["ingestion"], "rejected")
        self.assertEqual(statuses["general"], "done")
        text, _ = api._compose_reply(result["results"])
        self.assertIn("the answer", text)
        self.assertIn("Could not do it", text)

    def test_default_registry_general_request_answers_without_touching_ingest_tool(self):
        """The default wiring routes general requests; nothing file-related runs."""

        with unittest.mock.patch("src.tools.ingest_tool.ingest_document") as ingest, \
                unittest.mock.patch(
                    "src.llm.llm_client_ollama.get_llm_client",
                    return_value=_FakeLLM("Case was a console cowboy."),
                ):
            result = run_routing(
                "Who is Case?",
                analyzer=FakeAnalyzer(requests=[_general("Who is Case?")]),
            )
        ingest.assert_not_called()
        self.assertEqual(result["status"], "handled")
        self.assertEqual(result["results"][0]["agent"], "general_task")
        self.assertTrue(result["results"][0]["answer"])  # agent payload spread at top level


if __name__ == "__main__":
    unittest.main()
