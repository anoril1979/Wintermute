"""Tests for the KnowledgeLookupAgent and its wiring through the
retrieval flow (graph step, answer-kept-verbatim path, orchestrator).

Hermetic: a temp markdown knowledge base written through the real store
primitives; the answerer's LLM is a stub; no live Ollama.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from src.agents.agents.knowledge_lookup_agent import KnowledgeLookupAgent
from src.agents.contexts import RetrievalContext
from src.knowledge.character_markdown_store import (
    character_path_for,
    rebuild_index,
    write_character,
)


def _entry(alias: str, *source_ids: str) -> dict:
    return {"alias": alias, "source_ids": list(source_ids)}


class _KnowledgeBaseMixin:
    """A temp knowledge base with two characters (Joe + a sword)."""

    def make_base(self) -> Path:
        base = Path(tempfile.mkdtemp(prefix="wm_klookup_"))
        self.addCleanup(shutil.rmtree, base, ignore_errors=True)
        write_character(
            "Joe le Clodo",
            [
                _entry("Joe le Clodo", "doc:aaaaaaaa::chp:1::pg:1::sec:1"),
                _entry("Bobby", "doc:aaaaaaaa::chp:1::pg:1::sec:2"),
            ],
            character_path_for("Joe le Clodo", base),
        )
        write_character(
            "Épée de vif-argent",
            [_entry("Épée de vif-argent",
                    "doc:bbbbbbbb::chp:1::pg:2::sec:1")],
            character_path_for("Épée de vif-argent", base),
        )
        # Third character: makes "le" an ambiguous fragment (two full
        # names contain it) for the candidate-listing tests.
        write_character(
            "Bobby le Frelon",
            [_entry("Bobby le Frelon", "doc:cccccccc::chp:1::pg:1::sec:1")],
            character_path_for("Bobby le Frelon", base),
        )
        rebuild_index(base)
        return base


# ---------------------------------------------------------------------------
# Agent unit tests
# ---------------------------------------------------------------------------

class KnowledgeLookupAgentTest(_KnowledgeBaseMixin, unittest.TestCase):
    def setUp(self):
        self.base = self.make_base()
        self.agent = KnowledgeLookupAgent(base_dir=self.base)

    def _context(self, entity=None, language="en", question="q"):
        context = RetrievalContext(question=question)
        if entity is not None:
            context.metadata["entity"] = entity
        context.metadata["language"] = language
        return context

    def test_found_entity_emits_one_identity_source(self):
        context = self._context(entity="Joe le Clodo")
        result = self.agent.run(context)
        self.assertEqual(result.status.value, "ok")
        hits = context.outputs["hits"]
        self.assertEqual(len(hits), 1)
        chunk = hits[0]
        self.assertEqual(chunk.score, 1.0)
        self.assertIn("Joe le Clodo", chunk.text)
        self.assertIn("Bobby", chunk.text)
        self.assertIn("doc:aaaaaaaa::chp:1::pg:1::sec:2", chunk.text)
        self.assertEqual(chunk.metadata["kind"], "entity")
        self.assertEqual(
            context.metadata["resolved_entity"]["full_name"], "Joe le Clodo")

    def test_alias_resolution_finds_the_file(self):
        context = self._context(entity="bobby")
        result = self.agent.run(context)
        self.assertEqual(result.status.value, "ok")
        self.assertEqual(
            context.metadata["resolved_entity"]["full_name"], "Joe le Clodo")

    def test_fragment_of_one_full_name_auto_resolves(self):
        """Strategy 3: 'Vif-argent' spans exactly ONE known full name —
        the sword — so it resolves (the user's usual shorthand)."""
        context = self._context(entity="Vif-argent")
        result = self.agent.run(context)
        self.assertEqual(result.status.value, "ok")
        self.assertEqual(
            context.metadata["resolved_entity"]["full_name"],
            "Épée de vif-argent")

    def test_unknown_entity_writes_a_deterministic_reply(self):
        context = self._context(entity="Fantôme Inexistant", language="en")
        result = self.agent.run(context)
        self.assertEqual(result.status.value, "ok")
        self.assertTrue(result.payload["unknown"])
        self.assertEqual(context.outputs["hits"], [])
        reply = context.outputs["answer"]
        self.assertIn("Fantôme Inexistant", reply)
        self.assertTrue(context.metadata["entity_unknown"])
        self.assertEqual(context.metadata["entity_candidates"], [])

    def test_ambiguous_fragment_misses_with_candidates(self):
        """A fragment spanning SEVERAL full names ('le') cannot
        auto-resolve: the reply lists the close candidates."""
        context = self._context(entity="le", language="en")
        result = self.agent.run(context)
        self.assertEqual(result.status.value, "ok")
        self.assertTrue(result.payload["unknown"])
        candidates = context.metadata["entity_candidates"]
        self.assertIn("Joe le Clodo", candidates)
        self.assertIn("Bobby le Frelon", candidates)
        self.assertIn("Did you mean one of these?", context.outputs["answer"])

    def test_missing_entity_falls_back_to_the_question(self):
        """A degraded analyzer output (lookup without a name): the
        self-contained question is the last-resort lookup key."""
        context = self._context(question="Joe le Clodo")
        result = self.agent.run(context)
        self.assertEqual(result.status.value, "ok")
        self.assertEqual(
            context.metadata["resolved_entity"]["full_name"], "Joe le Clodo")

    def test_no_entity_no_question_fails_input_data(self):
        context = self._context(question="   ")
        result = self.agent.run(context)
        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.failure_domain.value, "input_data")

    def test_validate_flags_a_resolution_without_source(self):
        context = self._context(entity="Joe le Clodo")
        self.agent.run(context)
        self.assertIsNone(self.agent.validate(context))
        context.outputs.pop("hits")
        validation = self.agent.validate(context)
        self.assertIsNotNone(validation)


# ---------------------------------------------------------------------------
# Graph wiring: lookup runs, then the answer step
# ---------------------------------------------------------------------------

class _StubLLM:
    """Records the prompts it receives; answers a canned stub text."""

    def __init__(self):
        self.prompts: list = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "stub answer"


class _StubAnswerer:
    """Placeholder answerer (never run: its test kind is not implemented)."""

    name = "answerer"

    def run(self, context: RetrievalContext):
        from src.agents.protocols import AgentResult, AgentStatus

        return AgentResult(agent_name=self.name, status=AgentStatus.OK,
                           detail="unused")


class RetrievalGraphLookupWiringTest(_KnowledgeBaseMixin, unittest.TestCase):
    def setUp(self):
        self.base = self.make_base()

    def _graph(self, answerer):
        from src.graphs.retrieval_graph import RetrievalGraph

        return RetrievalGraph(agents={
            "knowledge_lookup": KnowledgeLookupAgent(base_dir=self.base),
            "answerer": answerer,
        })

    def test_lookup_kind_runs_lookup_then_answer(self):
        from src.agents.agents.answer_agent import AnswerAgent

        llm = _StubLLM()
        graph = self._graph(AnswerAgent(llm=llm))
        context = RetrievalContext(question="who is Joe le Clodo?")
        context.metadata["entity"] = "Joe le Clodo"
        context.metadata["language"] = "en"
        outcome = graph.run(context, kind="lookup")
        self.assertTrue(outcome.ok)
        self.assertEqual([s.step for s in outcome.steps],
                         ["knowledge_lookup", "answer"])
        # The real answer agent called its LLM with the identity source.
        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("Joe le Clodo", llm.prompts[0])
        self.assertEqual(outcome.steps[-1].agent, "answerer")

    def test_lookup_miss_keeps_the_deterministic_reply_verbatim(self):
        from src.agents.agents.answer_agent import AnswerAgent

        llm = _StubLLM()
        graph = self._graph(AnswerAgent(llm=llm))
        context = RetrievalContext(question="qui est Fantôme ?")
        context.metadata["entity"] = "Fantôme"
        context.metadata["language"] = "en"
        outcome = graph.run(context, kind="lookup")
        self.assertTrue(outcome.ok, "an unknown entity is served, not failed")
        self.assertEqual(outcome.steps[-1].agent, "answerer")
        self.assertEqual(outcome.steps[-1].status, "ok")
        # The miss reply survives the answer step verbatim (no LLM call).
        self.assertEqual(len(llm.prompts), 0)
        self.assertIn("No entity named 'Fantôme'", context.outputs["answer"])

    def test_relationship_kind_is_still_not_implemented(self):
        graph = self._graph(_StubAnswerer())
        outcome = graph.run(RetrievalContext(question="who is his wife?"),
                            kind="relationship")
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.steps[0].status, "not_implemented")


# ---------------------------------------------------------------------------
# Orchestrator: the entity flows in, the miss answer flows out
# ---------------------------------------------------------------------------

class RetrievalOrchestratorLookupFlowTest(_KnowledgeBaseMixin, unittest.TestCase):
    def setUp(self):
        self.base = self.make_base()

    def _run(self, requests, **kwargs):
        from src.graphs.retrieval_graph import RetrievalGraph
        from src.retrieval.retrieval_orchestrator import run_retrieval
        from src.retrieval.retrieval_router import RetrievalFacts

        graph = RetrievalGraph(agents={
            "knowledge_lookup": KnowledgeLookupAgent(base_dir=self.base),
        })
        import unittest.mock

        with unittest.mock.patch(
            "src.retrieval.retrieval_orchestrator.gather_facts",
            return_value=RetrievalFacts(chunk_count=10,
                                        knowledge_entities=2),
        ):
            return run_retrieval(requests, graph=graph, **kwargs)

    def test_lookup_flow_serves_the_identity(self):
        from src.routing.models import RetrievalRequest

        result = self._run([
            RetrievalRequest(question="who is Joe le Clodo?",
                             lookup_kind="lookup", entity="Joe le Clodo"),
        ])
        self.assertEqual(result["status"], "ok")
        sub = result["requests"][0]
        self.assertEqual(sub["status"], "ok")
        self.assertEqual(len(sub["hits"]), 1)
        # The identity source IS the hit text; no answerer is registered
        # in this graph, so the reply is empty and the raw hits stand.
        self.assertIn("Joe le Clodo", sub["hits"][0]["text"])
        self.assertEqual(sub["intent"]["entity"], "Joe le Clodo")

    def test_lookup_miss_is_served_with_the_unknown_reply(self):
        from src.routing.models import RetrievalRequest

        result = self._run([
            RetrievalRequest(question="who is Fantôme?",
                             lookup_kind="lookup", entity="Fantôme"),
        ], language="en")
        self.assertEqual(result["status"], "ok")
        sub = result["requests"][0]
        self.assertEqual(sub["status"], "ok")
        self.assertEqual(sub["hits"], [])
        self.assertIn("No entity named 'Fantôme'", sub["answer"])

    def test_lookup_needs_no_vector_corpus(self):
        """The decision-table exemption, end to end: with an EMPTY vector
        store a lookup is still served (it reads the knowledge base)."""
        from src.routing.models import RetrievalRequest
        from src.retrieval.retrieval_router import RetrievalFacts

        from src.graphs.retrieval_graph import RetrievalGraph
        from src.retrieval.retrieval_orchestrator import run_retrieval

        graph = RetrievalGraph(agents={
            "knowledge_lookup": KnowledgeLookupAgent(base_dir=self.base),
        })
        import unittest.mock

        with unittest.mock.patch(
            "src.retrieval.retrieval_orchestrator.gather_facts",
            return_value=RetrievalFacts(store_exists=False, chunk_count=0),
        ):
            result = run_retrieval(
                [RetrievalRequest(question="who is Joe?",
                                  lookup_kind="lookup", entity="Joe")],
                graph=graph,
            )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["requests"][0]["hits"])

    def test_registry_includes_the_lookup_agent(self):
        from src.agents.registry import build_retrieval_agents

        registry = build_retrieval_agents()
        self.assertIn("knowledge_lookup", registry)
        self.assertEqual(registry["knowledge_lookup"].name, "knowledge_lookup")


if __name__ == "__main__":
    unittest.main()
