"""End-to-end routing with the real GeneralTaskAgent (LLM stubbed, API composed).

Covers the full production path minus the LLM itself:

    run_routing -> RoutingGraph -> GeneralTaskAgent.run -> payload["answer"]
    -> app.api._compose_reply -> the user-facing text.

Post-paradigm change: an ingestion ask in conversation lands on the
general agent (ingestion is a CLI operation, never routed).
"""

from __future__ import annotations

import unittest
import unittest.mock

from src.agents.agents.general_task_agent import GeneralTaskAgent
from src.routing.models import (
    AnalysisResult,
    GeneralRequest,
    RetrievalRequest,
)
from src.routing.routing_orchestrator import run_routing

import app.api as api


class FakeAnalyzer:
    def __init__(self, requests):
        self.requests = requests

    def analyze(self, prompt):
        result = AnalysisResult()
        for request in self.requests:
            if isinstance(request, RetrievalRequest):
                result.retrieval.append(request)
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

    def test_batch_keeps_general_answer_after_a_retrieval_failure(self):
        """One broken request must not poison the general answer."""

        class _FailingRetrieval:
            name = "failing_retrieval"

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
            "ask the corpus then talk to me",
            analyzer=FakeAnalyzer(requests=[
                RetrievalRequest(utterance="what is stored?", question="what is stored?"),
                _general(),
            ]),
            agents={"retrieval_task": _FailingRetrieval(), "general_task": agent},
        )
        statuses = {r["kind"]: r["status"] for r in result["results"]}
        self.assertEqual(statuses["retrieval"], "rejected")
        self.assertEqual(statuses["general"], "done")
        text, _ = api._compose_reply(result["results"])
        self.assertIn("the answer", text)
        self.assertIn("Could not do it", text)

    def test_ingestion_ask_lands_on_the_general_agent(self):
        """The paradigm change end to end: an ingest ask in conversation
        reaches the general agent, which explains the CLI workflow — and
        nothing file-related runs."""
        agent = GeneralTaskAgent(
            allow_missing_role=True,
            llm=_FakeLLM("Ingestion is a command-line operation, human."),
        )
        with unittest.mock.patch(
            "src.tools.ingest_tool.resolve_document"
        ) as resolve, unittest.mock.patch(
            "src.ingestion.ingestion_orchestrator.run_ingestion_file"
        ) as run_ingestion:
            result = run_routing(
                "Please ingest meow.pdf",
                analyzer=FakeAnalyzer(requests=[
                    _general("Please ingest meow.pdf"),
                ]),
                agents={"general_task": agent},
            )
        resolve.assert_not_called()
        run_ingestion.assert_not_called()
        self.assertEqual(result["status"], "handled")
        self.assertEqual(result["results"][0]["kind"], "general")
        text, _ = api._compose_reply(result["results"])
        self.assertIn("command-line", text)

    def test_default_registry_general_request_answers(self):
        with unittest.mock.patch(
            "src.llm.llm_client_ollama.get_llm_client",
            return_value=_FakeLLM("Case was a console cowboy."),
        ):
            result = run_routing(
                "Who is Case?",
                analyzer=FakeAnalyzer(requests=[_general("Who is Case?")]),
            )
        self.assertEqual(result["status"], "handled")
        self.assertEqual(result["results"][0]["agent"], "general_task")
        self.assertTrue(result["results"][0]["answer"])  # agent payload spread at top level


if __name__ == "__main__":
    unittest.main()
