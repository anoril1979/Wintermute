"""Tests for routing trace events (context emitter + wire mapping).

Post-paradigm change: no ingestion requests exist — the FakeAnalyzer
group helper and the memory tests use retrieval + general requests only.
"""

from __future__ import annotations

import json
import unittest
import unittest.mock

from fastapi.testclient import TestClient

import app.api as api
import src.routing.routing_orchestrator as routing_orchestrator_module
from src.agents.contexts import RoutingContext
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
from src.graphs import RoutingGraph
from src.routing.models import (
    AnalysisResult,
    GeneralRequest,
    RetrievalRequest,
)


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


def _retrieval(question="what is in the corpus?"):
    return RetrievalRequest(utterance=question, question=question)


def _general(utterance="hello"):
    return GeneralRequest(utterance=utterance, question=utterance)


class OkAgent:
    name = "ok_agent"

    def run(self, context, request):
        return AgentResult(agent_name=self.name, status=AgentStatus.OK, detail="done")

    def validate(self, context, request):
        return None


class EmitTest(unittest.TestCase):
    def test_emit_appends_and_calls_observer(self):
        seen = []
        context = RoutingContext(request="x", on_event=seen.append)
        context.emit("analysis", "understood", "2 requests", kinds=["retrieval"])
        self.assertEqual(len(context.events), 1)
        event = context.events[0]
        self.assertEqual(event["phase"], "analysis")
        self.assertEqual(event["kind"], "understood")
        self.assertEqual(event["message"], "2 requests")
        self.assertEqual(event["data"], {"kinds": ["retrieval"]})
        self.assertEqual(seen, [event])

    def test_observer_failure_is_swallowed_but_event_kept(self):
        def broken(event):
            raise RuntimeError("observer down")

        context = RoutingContext(request="x", on_event=broken)
        context.emit("dispatch", "dispatching", "going")
        self.assertEqual(len(context.events), 1)  # recorded anyway

    def test_no_observer_is_fine(self):
        context = RoutingContext(request="x")
        context.emit("task", "retrieval_start", "go")
        self.assertEqual(len(context.events), 1)


class OrchestratorTracesTest(unittest.TestCase):
    def test_traces_returned_and_ordered(self):
        result = routing_orchestrator_module.run_routing(
            "anything",
            analyzer=FakeAnalyzer(requests=[_retrieval(), _general()]),
            agents={"retrieval_task": OkAgent()},
        )
        self.assertEqual(result["status"], "handled")
        kinds = [(t["phase"], t["kind"]) for t in result["traces"]]
        self.assertIn(("analysis", "understood"), kinds)
        self.assertIn(("dispatch", "dispatching"), kinds)
        self.assertIn(("dispatch", "dispatched"), kinds)
        self.assertIn(("dispatch", "not_implemented"), kinds)
        # analysis trace comes first
        self.assertEqual(kinds[0], ("analysis", "understood"))

    def test_traces_on_analysis_error(self):
        from src.routing.request_analyzer import RequestAnalysisError

        class BoomAnalyzer:
            def analyze(self, prompt):
                raise RequestAnalysisError("down", cause="llm_request")

        result = routing_orchestrator_module.run_routing(
            "anything", analyzer=BoomAnalyzer(), agents={}
        )
        self.assertEqual(result["status"], "analysis_error")
        self.assertEqual([t["kind"] for t in result["traces"]], ["analysis_failed"])

    def test_live_callback_receives_events(self):
        live = []
        result = routing_orchestrator_module.run_routing(
            "anything",
            analyzer=FakeAnalyzer(requests=[_general()]),
            agents={"general_task": OkAgent()},
            on_event=live.append,
        )
        self.assertEqual(len(live), len(result["traces"]))


class GraphTraceTest(unittest.TestCase):
    def test_dispatch_traces_carry_index_and_agent(self):
        graph = RoutingGraph(agents={"general_task": OkAgent()})
        context = RoutingContext(request="x")
        graph.run(context, [_general()])
        dispatching = [t for t in context.events if t["kind"] == "dispatching"]
        self.assertEqual(len(dispatching), 1)
        self.assertEqual(dispatching[0]["data"]["index"], 0)
        self.assertEqual(dispatching[0]["data"]["agent"], "ok_agent")


class PromptLocalMemoryTest(unittest.TestCase):
    """The graph attaches same-prompt predecessors to each request."""

    def test_first_request_has_no_preceding(self):
        requests = [_retrieval("what is stored?"), _general()]
        graph = RoutingGraph(agents={"general_task": OkAgent()})
        graph.run(RoutingContext(request="x"), requests)
        self.assertEqual(requests[0].preceding, [])

    def test_preceding_carries_utterance_and_outcome(self):
        # "ask the corpus, then <general about it>": the general request
        # must see the retrieval's utterance AND its dispatch outcome.
        requests = [_retrieval("what is stored?"), _general("is it indexed?")]
        graph = RoutingGraph(agents={
            "retrieval_task": OkAgent(),
            "general_task": OkAgent(),
        })
        graph.run(RoutingContext(request="x"), requests)
        entry = requests[1].preceding[0]
        self.assertEqual(entry.kind, "retrieval")
        self.assertEqual(entry.utterance, "what is stored?")
        self.assertEqual(entry.status, "done")
        self.assertEqual(entry.detail, "done")

    def test_preceding_reflects_actual_outcomes(self):
        # A rejected earlier request must not be reported as done: the
        # status comes from the dispatch history, not the analyzer.
        class FailAgent:
            name = "fail_agent"

            def run(self, context, request):
                return AgentResult(
                    agent_name=self.name, status=AgentStatus.FAILED,
                    failure_domain=FailureDomain.INPUT_DATA, detail="boom",
                )

            def validate(self, context, request):
                return None

        requests = [_retrieval("what is stored?"), _general("so, is it in?")]
        graph = RoutingGraph(agents={
            "retrieval_task": FailAgent(),
            "general_task": OkAgent(),
        })
        graph.run(RoutingContext(request="x"), requests)
        entry = requests[1].preceding[0]
        self.assertEqual(entry.status, "rejected")
        self.assertEqual(entry.detail, "boom")

    def test_local_context_trace_emitted(self):
        requests = [_retrieval("what is stored?"), _general()]
        context = RoutingContext(request="x")
        graph = RoutingGraph(agents={"general_task": OkAgent()})
        graph.run(context, requests)
        local = [t for t in context.events if t["kind"] == "local_context"]
        self.assertEqual(len(local), 1)
        self.assertEqual(local[0]["data"]["index"], 1)


class StreamingWireTest(unittest.TestCase):
    """Trace -> thinking-channel mapping on both chat dialects."""

    def setUp(self):
        self.client = TestClient(api.app)
        self._orig = routing_orchestrator_module.RequestAnalyzer

    def tearDown(self):
        routing_orchestrator_module.RequestAnalyzer = self._orig

    def test_sse_thinking_then_content_then_done(self):
        # run_routing builds RequestAnalyzer() itself: patch with a factory.
        routing_orchestrator_module.RequestAnalyzer = lambda: FakeAnalyzer(
            requests=[_retrieval()]
        )
        # Hermetic: a stub agent serves the retrieval request (the default
        # registry would build the real RetrievalTaskAgent).
        with unittest.mock.patch(
            "src.routing.routing_orchestrator.build_default_task_agents",
            return_value={"retrieval_task": OkAgent()},
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "what is stored?"}],
                      "stream": True},
            )
        lines = [l[6:] for l in response.text.splitlines() if l.startswith("data: ")]
        self.assertEqual(lines[-1], "[DONE]")
        deltas = [json.loads(l)["choices"][0]["delta"] for l in lines[:-1]]
        kinds = ["reasoning" if "reasoning_content" in d else "content" for d in deltas]
        self.assertIn("reasoning", kinds)
        self.assertIn("content", kinds)
        # all reasoning chunks precede all content chunks
        first_content = kinds.index("content")
        self.assertNotIn("reasoning", kinds[first_content:])
        thinking = "".join(d.get("reasoning_content", "") for d in deltas)
        self.assertIn("[analysis] understood", thinking)
        self.assertIn("[dispatch] dispatching", thinking)

    def test_ndjson_thinking_then_content_then_done(self):
        routing_orchestrator_module.RequestAnalyzer = lambda: FakeAnalyzer(
            requests=[_retrieval()]
        )
        with unittest.mock.patch(
            "src.routing.routing_orchestrator.build_default_task_agents",
            return_value={"retrieval_task": OkAgent()},
        ):
            response = self.client.post(
                "/api/chat",
                json={"messages": [{"role": "user", "content": "what is stored?"}],
                      "stream": True},
            )
        payloads = [json.loads(x) for x in response.text.splitlines() if x.strip()]
        self.assertTrue(payloads[-1]["done"])
        kinds = [
            "thinking" if p["message"].get("thinking") else "content"
            for p in payloads[:-1]
        ]
        self.assertIn("thinking", kinds)
        self.assertIn("content", kinds)
        first_content = kinds.index("content")
        self.assertNotIn("thinking", kinds[first_content:])
        thinking = "".join(p["message"].get("thinking", "") for p in payloads[:-1])
        self.assertIn("[analysis] understood", thinking)

    def test_non_streaming_has_no_thinking_field(self):
        routing_orchestrator_module.RequestAnalyzer = lambda: FakeAnalyzer(
            requests=[_general()]
        )
        # Hermetic: a stub agent answers the general request (the default
        # registry would build the real GeneralTaskAgent -> live Ollama).
        with unittest.mock.patch(
            "src.routing.routing_orchestrator.build_default_task_agents",
            return_value={"general_task": OkAgent()},
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("reasoning_content", response.text)


if __name__ == "__main__":
    unittest.main()
