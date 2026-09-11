"""Tests for the two graceful-degradation fixes.

Fix A — an ingestion request without a document no longer invalidates the
whole analyzer batch (per-request degradation): the model keeps validating,
the routing graph reports the request as ``incomplete`` and the API turns
that into a question for the user.

Fix B — an unanalyzable prompt produces an honest answer text instead of an
HTTPException(503), which chat clients treated as a retryable failure and
silently re-sent (re-running accepted routing under the hood).

Meta-prompt guard — front-end auxiliary traffic (title/tags/follow-ups)
is answered in place, never routed (see ``src/llm/guard.py``).
"""

from __future__ import annotations

import unittest
import unittest.mock

from fastapi.testclient import TestClient

import app.api as api
from src.llm import guard
from src.graphs.routing_graph import (
    STATUS_INCOMPLETE,
    RoutingGraph,
)
from src.routing.request_analyzer import RequestAnalyzer
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


class RagFallbackNeverRaisesTest(unittest.TestCase):
    """The legacy RAG fallback (retrieval requests before RetrievalTaskAgent
    exists) must also answer in-band instead of raising: any HTTPException
    there restarted the chat-client retry loop (third incident).

    ``run_routing`` is stubbed: these tests exercise the API's fallback
    branch, not the routing layer (covered elsewhere, hermetically).
    """

    def _routed_retrieval(self):
        return {
            "status": "handled",
            "results": [{
                "kind": "retrieval", "utterance": "what is in the docs?",
                "status": "not_implemented",
                "detail": "no agent implemented for kind 'retrieval' yet",
            }],
            "traces": [],
        }

    def test_unimportable_rag_stack_answers_in_band(self):
        with unittest.mock.patch(
            "src.routing.routing_orchestrator.run_routing",
            return_value=self._routed_retrieval(),
        ), unittest.mock.patch.object(api, "_rag_answer", None):
            text, results = api._route_or_answer("what is in the docs?")
        self.assertIn("retrieval memory is unavailable", text)
        self.assertTrue(results)  # the routing outcomes still flow back

    def test_failing_rag_answer_answers_in_band(self):
        with unittest.mock.patch(
            "src.routing.routing_orchestrator.run_routing",
            return_value=self._routed_retrieval(),
        ), unittest.mock.patch.object(api, "_rag_answer", side_effect=RuntimeError("boom")):
            text, results = api._route_or_answer("what is in the docs?")
        self.assertIn("retrieval memory hit an error", text)
        self.assertTrue(results)

    def test_streaming_path_always_yields_a_final_answer(self):
        """Even if the whole answer pipeline explodes inside the worker
        thread, the stream ends with one final text — never an exception."""
        with unittest.mock.patch.object(
            api, "_route_or_answer", side_effect=RuntimeError("kaboom")
        ):
            items = list(api._routing_stream("anything"))
        final = [item for item in items if item[0] == "final"]
        self.assertEqual(len(final), 1)
        self.assertIn("went wrong", final[0][1])


class RagDormantMessageTest(unittest.TestCase):
    """The dormant-chain answer must invite ingestion — not tell the user to
    run ingest.py first (the self-locking loop: ingestion itself comes
    through Wintermute)."""

    def test_dormant_message_never_says_lancez_ingest(self):
        import inspect

        from src.retrieval import rag

        source = inspect.getsource(rag)
        self.assertNotIn("Lancez ingest.py", source)

    def test_dormant_answer_invites_ingestion(self):
        import unittest.mock

        from src.retrieval import rag

        with unittest.mock.patch.object(rag, "_rag_chain", None), \
                unittest.mock.patch.object(rag, "_initialiser_chaine", return_value=False):
            answer = rag.answer("qui est Jean?")
        self.assertIn("dormant", answer)
        self.assertIn("ingest", answer)


class FixCObservabilityTest(unittest.TestCase):
    """Fix C: what the API received and what the analyzer answered must be
    provable from the log file alone (which client sent the conversation,
    whether the model invented content, why an analysis split a prompt)."""

    def test_extract_question_logs_shape_and_text(self):
        messages = [
            api.Message(role="user", content="Hello"),
            api.Message(role="assistant", content="Hi."),
            api.Message(role="user", content="  Ingest meow.pdf, please.  "),
        ]
        with self.assertLogs("wintermute", level="INFO") as captured:
            question = api._extract_question(messages)
        self.assertEqual(question, "Ingest meow.pdf, please.")
        line = " ".join(captured.output)
        self.assertIn("3 message(s) [user,assistant,user]", line)
        self.assertIn("Ingest meow.pdf, please.", line)

    def test_log_snippet_is_bounded_and_flattened(self):
        text = "line1\n\n  line2\tline3"
        snippet = api._log_snippet(text + "x" * 1000)
        self.assertTrue(snippet.startswith("line1 line2 line3"))
        self.assertLessEqual(len(snippet), 400)

    def test_analyzer_logs_input_and_raw_answer(self):
        analyzer = RequestAnalyzer()
        fake = unittest.mock.Mock()
        fake.complete.return_value = (
            '{"requests": [{"kind": "general", "utterance": "Hi",'
            ' "document": null, "question": null,'
            ' "options": {"force_reingest": false, "section_scope": null}}]}'
        )
        with unittest.mock.patch.object(analyzer, "_llm", return_value=fake), \
                self.assertLogs("src.routing.request_analyzer", level="INFO") as captured:
            analyzer.analyze("What is the color of the sky?")
        line = " ".join(captured.output)
        self.assertIn("Analyzing prompt (29 chars): What is the color of the sky?", line)
        self.assertIn("Analyzer raw answer", line)
        self.assertIn('"kind": "general"', line)


class MetaPromptGuardTest(unittest.TestCase):
    """The meta-prompt guard: front-end auxiliary traffic (Open WebUI's
    title generation, follow-up suggestions, topic tagging) must never
    reach the analyzer or the routing graph — fix C's logs proved those
    prompts each triggered a full routing run whose answer the front-end
    threw away. An explicit sentinel also lets a client probe the
    endpoint without paying for a routing run."""

    # --- the classifier -------------------------------------------------

    def test_openwebui_title_prompt_is_meta(self):
        # Verbatim shape from the fix C log session.
        self.assertTrue(guard.prompt_is_meta(
            "### Task: Generate a concise, 3-5 word title with an emoji "
            "summarizing the chat history."
        ))

    def test_openwebui_followup_prompt_is_meta(self):
        self.assertTrue(guard.prompt_is_meta(
            "### Task: Suggest 3-5 relevant follow-up questions or prompts "
            "that the user might naturally ask next in this conversation."
        ))

    def test_openwebui_tagging_prompt_is_meta(self):
        self.assertTrue(guard.prompt_is_meta(
            "### Task: Generate 1-3 broad tags categorizing the main themes "
            "of the chat history."
        ))

    def test_sentinel_is_meta(self):
        self.assertTrue(guard.prompt_is_meta(guard.META_SENTINEL))

    def test_real_user_prompts_are_not_meta(self):
        self.assertFalse(guard.prompt_is_meta(
            "OK… Errr, another roll, good? Well, I would like you to ingest "
            "some documents, then provide me with the summary of it."
        ))
        self.assertFalse(guard.prompt_is_meta("Ingest meow.pdf, please."))
        self.assertFalse(guard.prompt_is_meta("Hi!"))

    def test_meta_wording_inside_a_real_message_is_not_enough(self):
        # The match anchors on the opening instruction: a user *quoting*
        # a title task mid-message must still be routed.
        self.assertFalse(guard.prompt_is_meta(
            "Wintermute, when I say '### Task: Generate a title' I mean "
            "the front-end is talking, not me."
        ))

    # --- the analyzer boundary -------------------------------------------

    def test_analyzer_refuses_meta_prompt_without_llm_call(self):
        analyzer = RequestAnalyzer()
        with unittest.mock.patch.object(analyzer, "_llm") as llm_factory:
            result = analyzer.analyze(
                "### Task: Generate a concise title for the chat history."
            )
        self.assertEqual(result.requests, [])
        llm_factory.assert_not_called()  # not even constructed

    def test_analyzer_still_accepts_normal_prompts(self):
        analyzer = RequestAnalyzer()
        fake = unittest.mock.Mock()
        fake.complete.return_value = (
            '{"requests": [{"kind": "general", "utterance": "Hi",'
            ' "document": null, "question": null,'
            ' "options": {"force_reingest": false, "section_scope": null}}]}'
        )
        with unittest.mock.patch.object(analyzer, "_llm", return_value=fake):
            result = analyzer.analyze("Hello there!")
        self.assertEqual(len(result.requests), 1)

    # --- the API boundary -------------------------------------------------

    def test_api_meta_answer(self):
        text = guard.meta_answer()
        self.assertIn("background", text)
        self.assertIn("no routing was performed", text)

    def test_api_meta_short_circuit_never_reaches_routing(self):
        with unittest.mock.patch(
            "src.routing.routing_orchestrator.run_routing"
        ) as run_routing:
            answer, _ = api._route_or_answer(guard.META_SENTINEL)
        run_routing.assert_not_called()
        self.assertIn("background", answer)

    def test_openai_endpoint_answers_meta_prompt_without_routing(self):
        # The reply_to_meta_request switch is pinned OFF: this test owns
        # the guard semantics (fixed answer, routing never called) — the
        # agent-answering mode has its own tests (test_meta_request_agent).
        # Unpinned, the real agent answered here with a live LLM call.
        with unittest.mock.patch(
            "src.tools.config_loader.load_setup_config",
            return_value={"reply_to_meta_request": False},
        ), unittest.mock.patch(
            "src.routing.routing_orchestrator.run_routing"
        ) as run_routing:
            client = TestClient(api.app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{
                        "role": "user",
                        "content": "### Task: Generate a concise title with an emoji.",
                    }],
                },
            )
        self.assertEqual(response.status_code, 200)
        run_routing.assert_not_called()
        content = response.json()["choices"][0]["message"]["content"]
        self.assertIn("background", content)


if __name__ == "__main__":
    unittest.main()
