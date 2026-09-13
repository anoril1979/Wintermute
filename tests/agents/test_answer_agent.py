"""Tests for the AnswerAgent and its wiring through the retrieval flow.

Hermetic: a stub LLM client, fake VectorChunk hits, no live Ollama.
Covers: the deterministic no-hit path, the grounded phrasing (prompt
contains numbered sources + question), failure mapping, the graph's
two-step semantic flow, the skip/fail answer behavior in the
orchestrator, the task agent's answer detail, and the API reply
composer surfacing the answer.
"""

from __future__ import annotations

import tempfile
import unittest
import unittest.mock
from pathlib import Path
from unittest.mock import MagicMock

from src.indexing.chunks import VectorChunk
from src.retrieval.retrieval_router import RetrievalFacts
from src.routing.models import RetrievalRequest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class StubLLM:
    """Records the prompt; returns a canned answer (or fails)."""

    def __init__(self, answer: str = "The king fled the capital [1]."):
        self.answer = answer
        self.prompts: list = []
        self.fail = False
        self.empty = False

    def complete(self, prompt: str) -> str:
        if self.fail:
            raise RuntimeError("model unreachable")
        self.prompts.append(prompt)
        return "" if self.empty else self.answer


def _hit(text: str = "the king fled the burning capital",
         score: float = 0.82, **meta) -> VectorChunk:
    metadata = {"doc_title": "Gazette Test", "page_number": 4,
                "origin": "canon", "kind": "content", **meta}
    return VectorChunk(id="doc:x::chp:1::pg:1::sec:1::txt:1",
                       text=text, metadata=metadata, score=score)


def _prompt_path(tmp: Path) -> Path:
    """The real answering prompt (content matters for the phrasing rules)."""
    return Path("prompts/answering/retrieval_answer.md")


def _agent(llm: StubLLM, tmp: Path) -> "AnswerAgent":
    from src.agents.agents.answer_agent import AnswerAgent

    return AnswerAgent(llm=llm, prompt_path=_prompt_path(tmp))


def _context(question: str = "what happened to the king?", hits=None):
    from src.agents.contexts import RetrievalContext

    context = RetrievalContext(question=question)
    if hits is not None:
        context.outputs["hits"] = hits
    return context


# ---------------------------------------------------------------------------
# AnswerAgent unit tests
# ---------------------------------------------------------------------------

class AnswerAgentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="wm_answer_"))
        self.addCleanup(_cleanup_tmp, self.tmp)
        self.llm = StubLLM()
        self.agent = _agent(self.llm, self.tmp)

    def test_no_hits_is_deterministic_no_answer(self):
        context = _context(hits=[])
        result = self.agent.run(context)
        self.assertEqual(result.status.value, "ok")
        self.assertTrue(result.payload["no_answer"])
        self.assertEqual(context.outputs["answer"], result.payload["answer"])
        self.assertEqual(self.llm.prompts, [], "no LLM call without sources")

    def test_hits_are_phrased_through_the_llm(self):
        context = _context(hits=[_hit()])
        result = self.agent.run(context)
        self.assertEqual(result.status.value, "ok")
        self.assertEqual(result.payload["answer"], self.llm.answer)
        self.assertEqual(context.outputs["answer"], self.llm.answer)

    def test_prompt_carries_numbered_sources_and_question(self):
        self.agent.run(_context(question="qui est le roi ?",
                                hits=[_hit(), _hit("the queen defended",
                                                   score=0.61,
                                                   doc_title="Autre.pdf",
                                                   page_number=9)]))
        self.assertEqual(len(self.llm.prompts), 1)
        prompt = self.llm.prompts[0]
        self.assertIn("[1]", prompt)
        self.assertIn("[2]", prompt)
        self.assertIn("Gazette Test", prompt)
        self.assertIn("Autre.pdf", prompt)
        self.assertIn("page 4", prompt)
        self.assertIn("origin: canon", prompt)
        self.assertIn("qui est le roi ?", prompt)
        self.assertIn("<<<<PROMPT>>>>", prompt)

    def test_llm_failure_maps_to_llm_response(self):
        self.llm.fail = True
        result = self.agent.run(_context(hits=[_hit()]))
        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.failure_domain.value, "llm_response")

    def test_empty_answer_maps_to_llm_response(self):
        self.llm.empty = True
        result = self.agent.run(_context(hits=[_hit()]))
        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.failure_domain.value, "llm_response")

    def test_missing_role_fails_config_when_hits_exist(self):
        from src.agents.agents.answer_agent import AnswerAgent
        from src.agents.llm_roles import MissingLLMRoleError

        with unittest.mock.patch(
            "src.agents.llm_roles.require_llm_role",
            side_effect=MissingLLMRoleError("no 'answerer' role"),
        ):
            agent = AnswerAgent(allow_missing_role=True, llm=StubLLM())
        context = _context(hits=[_hit()])
        result = agent.run(context)
        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.failure_domain.value, "config")

    def test_validate_flags_missing_answer_with_hits(self):
        context = _context(hits=[_hit()])
        self.agent.run(context)
        self.assertIsNone(self.agent.validate(context))
        context.outputs.pop("answer")
        validation = self.agent.validate(context)
        self.assertIsNotNone(validation)


def _cleanup_tmp(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Retrieval graph: the two-step semantic flow
# ---------------------------------------------------------------------------

class RetrievalGraphAnswerStepTest(unittest.TestCase):
    def _graph(self, llm: StubLLM):
        import tempfile as _tf
        import shutil as _sh

        from src.agents.agents.answer_agent import AnswerAgent
        from src.agents.agents.semantic_retrieval_agent import (
            SemanticRetrievalAgent,
        )
        from src.graphs.retrieval_graph import RetrievalGraph

        tmp = Path(_tf.mkdtemp(prefix="wm_ansgraph_"))
        self.addCleanup(_sh.rmtree, tmp, ignore_errors=True)
        from tests.retrieval.test_retrieval import StubEmbedder, _fill_store

        return RetrievalGraph(agents={
            "semantic_retriever": SemanticRetrievalAgent(
                embedder=StubEmbedder(), store=_fill_store(tmp),
                instruction=""),
            "answerer": AnswerAgent(llm=llm, prompt_path=_prompt_path(tmp)),
        }), tmp

    def _context(self, **metadata):
        from src.agents.contexts import RetrievalContext

        context = RetrievalContext(question="the king fled the burning capital")
        context.metadata.update(metadata)
        return context

    def test_semantic_runs_search_then_answer(self):
        llm = StubLLM()
        graph, _ = self._graph(llm)
        context = self._context(top_k=5)
        outcome = graph.run(context, kind="semantic")
        self.assertTrue(outcome.ok)
        self.assertEqual([s.step for s in outcome.steps],
                         ["semantic_search", "answer"])
        self.assertEqual(context.outputs["answer"], llm.answer)

    def test_missing_answerer_skips_but_search_stands(self):
        from src.graphs.retrieval_graph import RetrievalGraph

        import tempfile as _tf
        import shutil as _sh

        from src.agents.agents.semantic_retrieval_agent import (
            SemanticRetrievalAgent,
        )
        from tests.retrieval.test_retrieval import StubEmbedder, _fill_store

        tmp = Path(_tf.mkdtemp(prefix="wm_ansgraph2_"))
        self.addCleanup(_sh.rmtree, tmp, ignore_errors=True)
        graph = RetrievalGraph(agents={
            "semantic_retriever": SemanticRetrievalAgent(
                embedder=StubEmbedder(), store=_fill_store(tmp),
                instruction=""),
        })
        context = self._context(top_k=5)
        outcome = graph.run(context, kind="semantic")
        self.assertTrue(outcome.ok, "the search results are real work")
        self.assertEqual(outcome.steps[-1].step, "answer")
        self.assertEqual(outcome.steps[-1].status, "skipped")
        self.assertTrue(context.outputs["hits"])

    def test_failing_answerer_marks_step_failed(self):
        from src.graphs.retrieval_graph import RetrievalGraph

        import tempfile as _tf
        import shutil as _sh

        from src.agents.agents.answer_agent import AnswerAgent
        from src.agents.agents.semantic_retrieval_agent import (
            SemanticRetrievalAgent,
        )
        from tests.retrieval.test_retrieval import StubEmbedder, _fill_store

        tmp = Path(_tf.mkdtemp(prefix="wm_ansgraph3_"))
        self.addCleanup(_sh.rmtree, tmp, ignore_errors=True)
        failing_llm = StubLLM()
        failing_llm.fail = True
        graph = RetrievalGraph(agents={
            "semantic_retriever": SemanticRetrievalAgent(
                embedder=StubEmbedder(), store=_fill_store(tmp),
                instruction=""),
            "answerer": AnswerAgent(llm=failing_llm,
                                    prompt_path=_prompt_path(tmp)),
        })
        context = self._context(top_k=5)
        outcome = graph.run(context, kind="semantic")
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.steps[-1].status, "failed")
        # The hits are still there for the caller to degrade on.
        self.assertTrue(context.outputs["hits"])


# ---------------------------------------------------------------------------
# Orchestrator + task agent surfacing
# ---------------------------------------------------------------------------

class AnswerSurfacingTest(unittest.TestCase):
    """The answer flows: graph → orchestrator sub-result → task agent → API."""

    def _run_pipeline(self, agents: dict):
        import shutil
        import tempfile

        from src.agents.agents.retrieval_task_agent import RetrievalTaskAgent
        from src.agents.contexts import RoutingContext
        from src.graphs.retrieval_graph import RetrievalGraph
        from src.retrieval import retrieval_orchestrator

        tmp = Path(tempfile.mkdtemp(prefix="wm_anssurf_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        from tests.retrieval.test_retrieval import StubEmbedder, _fill_store

        graph = RetrievalGraph(agents=agents)
        with unittest.mock.patch.object(
            retrieval_orchestrator, "gather_facts",
            return_value=RetrievalFacts(chunk_count=10),
        ):
            result = retrieval_orchestrator.run_retrieval(
                [RetrievalRequest(question="what happened to the king?",
                                  lookup_kind="semantic")],
                graph=graph,
            )
        return result, graph

    def test_orchestrator_carries_the_answer(self):
        from src.agents.agents.answer_agent import AnswerAgent
        from src.agents.agents.semantic_retrieval_agent import (
            SemanticRetrievalAgent,
        )

        llm = StubLLM()
        import shutil
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="wm_anssurf2_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        from tests.retrieval.test_retrieval import StubEmbedder, _fill_store

        result, _ = self._run_pipeline({
            "semantic_retriever": SemanticRetrievalAgent(
                embedder=StubEmbedder(), store=_fill_store(tmp),
                instruction=""),
            "answerer": AnswerAgent(llm=llm, prompt_path=_prompt_path(tmp)),
        })
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["requests"][0]["answer"], llm.answer)
        self.assertEqual(result["requests"][0]["hits"][0]["text"],
                         "the king fled the burning capital")

    def test_task_agent_detail_is_the_answer(self):
        from src.agents.agents.retrieval_task_agent import RetrievalTaskAgent
        from src.agents.contexts import RoutingContext

        result = {
            "status": "ok",
            "requests": [{"status": "ok", "answer": "The king fled [1].",
                          "hits": [{}]}],
            "hits": [{}],
            "traces": [],
        }
        runner = MagicMock(return_value=result)
        agent = RetrievalTaskAgent(runner=runner)
        context = RoutingContext()
        agent_result = agent.run(
            context, RetrievalRequest(question="what happened to the king?"),
        )
        self.assertEqual(agent_result.status.value, "ok")
        self.assertEqual(agent_result.detail, "The king fled [1].")
        self.assertEqual(agent_result.payload["retrieval"], result)

    def test_task_agent_detail_falls_back_to_chunk_count(self):
        from src.agents.agents.retrieval_task_agent import RetrievalTaskAgent
        from src.agents.contexts import RoutingContext

        agent = RetrievalTaskAgent(runner=MagicMock(return_value={
            "status": "ok",
            "requests": [{"status": "ok", "answer": "", "hits": [{}]}],
            "hits": [{}],
            "traces": [],
        }))
        agent_result = agent.run(
            RoutingContext(),
            RetrievalRequest(question="q"),
        )
        self.assertEqual(agent_result.detail, "1 chunk(s) retrieved")


# ---------------------------------------------------------------------------
# API reply composer
# ---------------------------------------------------------------------------

class ComposeReplyAnswerTest(unittest.TestCase):
    def test_retrieval_done_uses_the_answer_detail(self):
        from app.api import _compose_reply

        text, needs_rag = _compose_reply([
            {"kind": "retrieval", "status": "done",
             "detail": "The king fled the capital [1]."},
        ])
        self.assertFalse(needs_rag)
        self.assertIn("The king fled the capital", text)
        self.assertNotIn("chunk", text.lower())

    def test_retrieval_done_without_answer_keeps_detail(self):
        from app.api import _compose_reply

        text, _ = _compose_reply([
            {"kind": "retrieval", "status": "done",
             "detail": "3 chunk(s) retrieved"},
        ])
        self.assertIn("3 chunk(s) retrieved", text)


if __name__ == "__main__":
    unittest.main()
