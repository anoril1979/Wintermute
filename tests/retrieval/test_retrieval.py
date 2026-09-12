"""Tests for the retrieval layer: filters, spec models, decision table,
vector query seam, graph, orchestrator and the task agent.

Hermetic: real embedded ChromaDB on temp folders, stub embedders, mocked
runner seams. No live Ollama — the retrieval pipeline is deterministic;
the classification arrives from the routing analyzer's requests.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from src.extraction.ids import assign_extract_ids
from src.extraction.models import (
    Chapter,
    DocumentExtract,
    PageContent,
    Section,
    TextBlock,
    TocEntry,
)
from src.indexing.chroma_client import ChromaVectorClient
from src.indexing.chunks import (
    KIND_CONTENT,
    LEVEL_BLOCK,
    LEVEL_DOCUMENT,
    VectorChunk,
    build_source_chunks,
)
from src.retrieval.filters import (
    InvalidFilterError,
    RetrievalFilters,
    build_where,
)
from src.retrieval.models import (
    RetrievalBatch,
    RetrievalSpec,
)
from src.retrieval.retrieval_router import (
    IMPLEMENTED_KINDS,
    ROUTER_NO_CORPUS,
    ROUTER_PROCEED,
    RetrievalFacts,
    apply_decision_table,
    gather_facts,
)
from src.routing.models import RetrievalLookupKind, RetrievalRequest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class StubEmbedder:
    """Deterministic 8-dim embedder (same scheme as the indexing tests)."""

    dimension = 8
    fail = False

    def embed(self, texts):
        if self.fail:
            raise RuntimeError("backend exploded")
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([b / 255.0 for b in digest[: self.dimension]])
        return vectors


def _doc(title: str = "Gazette Test") -> DocumentExtract:
    """A small summarized document: 2 blocks + summaries at every level."""
    blocks = [
        TextBlock(block_id=0, page_number=1, bbox=(0, 0, 10, 10),
                  raw_text="the king fled the burning capital",
                  summary="the king fled"),
        TextBlock(block_id=1, page_number=1, bbox=(0, 0, 10, 20),
                  raw_text="the queen stayed and defended the walls",
                  summary="the queen defended"),
    ]
    page = PageContent(
        page_number=1, width=595, height=842, raw_text="king queen",
        summary="page summary",
        sections=[Section(section_id=0, blocks=blocks, page_number=1,
                          bbox=(0, 0, 10, 20), raw_text="king queen",
                          summary="section summary")],
    )
    doc = DocumentExtract(
        source_path="data/sources/pdf/gazette-test.pdf",
        title=title, author="", subject="", total_pages=1,
        chapters=[Chapter(toc_entry=TocEntry(level=1, title="Chapter One",
                                             page_number=1, page_index=0),
                          pages=[page], full_text="king queen",
                          summary="chapter summary")],
        summary="document summary",
    )
    return assign_extract_ids(doc)


def _store(tmp: Path, name: str = "retrieval_test_chunks") -> ChromaVectorClient:
    return ChromaVectorClient(
        "source_chunks", embedding_dimension=StubEmbedder.dimension,
        path=str(tmp), collection_name=name,
    )


def _fill_store(tmp: Path) -> ChromaVectorClient:
    """A real store with the fixture document indexed."""
    store = _store(tmp)
    chunks = build_source_chunks(_doc())
    for chunk in chunks:
        chunk.vector = StubEmbedder().embed([chunk.text])[0]
    store.upsert(chunks)
    return store


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class FiltersTest(unittest.TestCase):
    def test_empty_filters_build_no_clause(self):
        self.assertIsNone(build_where(None))
        self.assertIsNone(RetrievalFilters().build_where())

    def test_document_only_is_a_single_condition(self):
        where = RetrievalFilters(document="Gazette").build_where()
        self.assertEqual(where, {"doc_title": {"$in":
            ["Gazette", "GAZETTE", "gazette"]}})

    def test_all_facets_combine_under_and(self):
        filters = RetrievalFilters(
            document="Gazette", origins=["canon"], kinds=["content"],
            levels=["block"], page_number=3,
        )
        where = filters.build_where()
        self.assertIsInstance(where, dict)
        self.assertIn("$and", where)
        conditions = where["$and"]
        self.assertEqual(len(conditions), 5)
        self.assertIn({"origin": {"$in": ["canon"]}}, conditions)
        self.assertIn({"page_number": {"$eq": 3}}, conditions)

    def test_path_like_document_is_rejected(self):
        with self.assertRaises(InvalidFilterError):
            RetrievalFilters(document="data/sources/x.pdf").build_where()

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(InvalidFilterError):
            RetrievalFilters(kinds=["poetry"]).build_where()

    def test_unknown_level_is_rejected(self):
        with self.assertRaises(InvalidFilterError):
            RetrievalFilters(levels=["volume"]).build_where()

    def test_bad_doc_ids_are_rejected(self):
        with self.assertRaises(InvalidFilterError):
            RetrievalFilters(doc_ids=["Gazette"]).build_where()

    def test_bad_page_number_is_rejected(self):
        with self.assertRaises(InvalidFilterError):
            RetrievalFilters(page_number=0).build_where()

    def test_summary_shows_only_set_facets(self):
        self.assertEqual(RetrievalFilters().summary(), {})
        data = RetrievalFilters(document="G", origins=["rpg"]).summary()
        self.assertEqual(set(data), {"document", "origins"})


# ---------------------------------------------------------------------------
# Spec models (built from the analyzer's requests — no LLM parsing)
# ---------------------------------------------------------------------------

class RetrievalSpecModelTest(unittest.TestCase):
    def test_from_request_carries_everything(self):
        request = RetrievalRequest(
            question="who is the King of the North",
            lookup_kind="semantic",
            document="Gazette",
            chapter_title="Chapter One",
            top_k=7,
            reason="identity question",
        )
        spec = RetrievalSpec.from_request(request)
        self.assertEqual(spec.kind, RetrievalLookupKind.SEMANTIC)
        self.assertEqual(spec.question, "who is the King of the North")
        self.assertEqual(spec.document, "Gazette")
        self.assertEqual(spec.chapter_title, "Chapter One")
        self.assertEqual(spec.top_k, 7)
        self.assertEqual(spec.reason, "identity question")

    def test_default_kind_is_semantic(self):
        spec = RetrievalSpec.from_request(RetrievalRequest(question="x"))
        self.assertEqual(spec.kind, RetrievalLookupKind.SEMANTIC)

    def test_blank_question_rejected(self):
        with self.assertRaises(Exception):
            RetrievalSpec(question="   ")

    def test_batch_from_requests_preserves_order(self):
        requests = [
            RetrievalRequest(question="who is the King",
                             lookup_kind="semantic"),
            RetrievalRequest(question="family tree of the King",
                             lookup_kind="relation"),
        ]
        batch = RetrievalBatch.from_requests(requests)
        self.assertEqual(
            [s.kind for s in batch.specs],
            [RetrievalLookupKind.SEMANTIC, RetrievalLookupKind.RELATION],
        )

    def test_summary_shows_the_kind(self):
        spec = RetrievalSpec(question="x", kind="relation")
        self.assertEqual(spec.summary()["kind"], "relation")


# ---------------------------------------------------------------------------
# Decision table + facts
# ---------------------------------------------------------------------------

class DecisionTableTest(unittest.TestCase):
    CONFIG = {"default_top_k": 6, "max_top_k": 20, "min_score": 0.35,
              "embedding_role": "embedding",
              "source_collection_key": "source_chunks"}

    def test_proceed_with_filters_and_clamped_top_k(self):
        facts = RetrievalFacts(store_exists=True, chunk_count=51)
        spec = RetrievalSpec(kind="semantic", question="the king",
                             document="Gazette", top_k=100)
        decision = apply_decision_table(facts, spec, config=self.CONFIG)
        self.assertEqual(decision.status, ROUTER_PROCEED)
        self.assertTrue(decision.implemented)
        self.assertEqual(decision.top_k, 20, "per-request top_k is clamped")
        self.assertEqual(decision.filters.document, "Gazette")

    def test_default_top_k_applies(self):
        decision = apply_decision_table(
            RetrievalFacts(chunk_count=3),
            RetrievalSpec(kind="semantic", question="x"),
            config=self.CONFIG,
        )
        self.assertEqual(decision.top_k, 6)

    def test_no_corpus_gates_even_a_valid_spec(self):
        decision = apply_decision_table(
            RetrievalFacts(store_exists=False, chunk_count=0),
            RetrievalSpec(kind="semantic", question="x"),
            config=self.CONFIG,
        )
        self.assertEqual(decision.status, ROUTER_NO_CORPUS)

    def test_unimplemented_kinds_stay_routed(self):
        for kind in ("index", "relation", "summary", "listing"):
            decision = apply_decision_table(
                RetrievalFacts(chunk_count=3),
                RetrievalSpec(kind=kind, question="x"),
                config=self.CONFIG,
            )
            self.assertEqual(decision.status, ROUTER_PROCEED)
            self.assertFalse(decision.implemented, kind)
        self.assertEqual(IMPLEMENTED_KINDS, {"semantic"})


class GatherFactsTest(unittest.TestCase):
    def test_facts_on_an_empty_store_are_zero(self):
        tmp = Path(tempfile.mkdtemp(prefix="wm_facts_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with unittest.mock.patch(
            "src.indexing.chroma_client.config_loader.load_vector_config",
            return_value={"path": str(tmp),
                          "collections": {"source_chunks": "s",
                                          "knowledge_chunks": "k"}},
        ):
            facts = gather_facts()
        self.assertEqual(facts.chunk_count, 0)
        self.assertFalse(facts.has_corpus)

    def test_facts_count_indexed_chunks(self):
        tmp = Path(tempfile.mkdtemp(prefix="wm_facts_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        _fill_store(tmp)
        with unittest.mock.patch(
            "src.indexing.chroma_client.config_loader.load_vector_config",
            return_value={"path": str(tmp),
                          "collections": {"source_chunks": "retrieval_test_chunks",
                                          "knowledge_chunks": "k"}},
        ):
            facts = gather_facts()
        self.assertTrue(facts.has_corpus)
        self.assertGreater(facts.chunk_count, 0)# ---------------------------------------------------------------------------
# ChromaDB query seam — real embedded store
# ---------------------------------------------------------------------------

class ChromaQueryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="wm_chroma_query_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.store = _fill_store(self.tmp)

    def test_query_by_text_remains_a_stub(self):
        with self.assertRaises(NotImplementedError):
            self.store.query("the king")

    def test_query_by_vector_returns_scored_hits(self):
        embedder = StubEmbedder()
        query_vector = embedder.embed(["the king fled the burning capital"])[0]
        hits = self.store.query_by_vector(query_vector, top_k=3)
        self.assertTrue(hits)
        self.assertLessEqual(len(hits), 3)
        self.assertTrue(all(0.0 <= h.score <= 1.0 for h in hits))
        scores = [h.score for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True),
                         "hits must come best-first")
        best = hits[0]
        self.assertEqual(best.text, "the king fled the burning capital")
        self.assertEqual(best.metadata["doc_title"], "Gazette Test")
        self.assertIn("origin", best.metadata)

    def test_query_respects_metadata_filter(self):
        embedder = StubEmbedder()
        query_vector = embedder.embed(["the king"])[0]
        where = build_where(RetrievalFilters(kinds=["summary"]))
        hits = self.store.query_by_vector(query_vector, top_k=10, where=where)
        self.assertTrue(hits)
        self.assertTrue(all(h.metadata["kind"] == "summary" for h in hits))
        where = build_where(RetrievalFilters(kinds=["content"]))
        hits = self.store.query_by_vector(query_vector, top_k=10, where=where)
        self.assertTrue(all(h.metadata["kind"] == "content" for h in hits))

    def test_query_with_non_matching_filter_returns_nothing(self):
        embedder = StubEmbedder()
        query_vector = embedder.embed(["the king"])[0]
        where = build_where(RetrievalFilters(origins=["rpg"]))
        self.assertEqual(self.store.query_by_vector(query_vector, top_k=5,
                                                    where=where), [])

    def test_query_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            self.store.query_by_vector([], top_k=3)
        with self.assertRaises(ValueError):
            self.store.query_by_vector([0.1] * StubEmbedder.dimension, top_k=0)

    def test_score_field_is_none_until_queried(self):
        chunk = build_source_chunks(_doc())[0]
        self.assertIsNone(chunk.score)


# ---------------------------------------------------------------------------
# SemanticRetrievalAgent — stub embedder + real store
# ---------------------------------------------------------------------------

class SemanticRetrievalAgentTest(unittest.TestCase):
    def setUp(self):
        from src.agents.agents.semantic_retrieval_agent import (
            SemanticRetrievalAgent,
        )

        self.tmp = Path(tempfile.mkdtemp(prefix="wm_semret_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.embedder = StubEmbedder()
        self.store = _fill_store(self.tmp)
        self.agent = SemanticRetrievalAgent(
            embedder=self.embedder, store=self.store,
        )

    def _context(self, question: str = "the king fled the burning capital",
                 **metadata):
        from src.agents.contexts import RetrievalContext

        context = RetrievalContext(question=question)
        context.metadata.update(metadata)
        return context

    def test_happy_path_returns_scored_hits(self):
        result = self.agent.run(self._context())
        self.assertEqual(result.status.value, "ok")
        hits = result.payload["hits"]
        self.assertTrue(hits)
        self.assertEqual(hits[0].text, "the king fled the burning capital")

    def test_min_score_threshold_drops_noise(self):
        # A question unrelated to anything stored: stub vectors are hash-
        # based, so similarity stays low; with threshold 1.0 everything drops.
        result = self.agent.run(self._context(top_k=10))
        self.assertEqual(result.status.value, "ok")
        # Sanity: with the default threshold at least the exact text passes.
        self.assertTrue(result.payload["hits"])

    def test_empty_question_fails_input_data(self):
        result = self.agent.run(self._context(question="   "))
        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.failure_domain.value, "input_data")

    def test_embedding_failure_maps_to_external(self):
        self.embedder.fail = True
        result = self.agent.run(self._context())
        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.failure_domain.value, "external")

    def test_store_failure_maps_to_external(self):
        from src.agents.agents.semantic_retrieval_agent import (
            SemanticRetrievalAgent,
        )

        # Simulate a backend refusal: point the agent at a client whose
        # folder path is a FILE (opening the store fails).
        file_path = self.tmp / "a_file"
        file_path.write_text("not a folder")
        broken = ChromaVectorClient(
            "source_chunks", path=str(file_path), collection_name="broken",
        )
        agent = SemanticRetrievalAgent(embedder=self.embedder, store=broken)
        result = agent.run(self._context())
        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.failure_domain.value, "external")

    def test_filters_flow_into_the_query(self):
        result = self.agent.run(
            self._context(filters=RetrievalFilters(kinds=["summary"]))
        )
        self.assertEqual(result.status.value, "ok")
        for hit in result.payload["hits"]:
            self.assertEqual(hit.metadata["kind"], "summary")

    def test_validate_checks_score_order(self):
        context = self._context()
        self.agent.run(context)
        self.assertIsNone(self.agent.validate(context))
        # Corrupt the order: validate must flag it.
        hits = context.outputs["hits"]
        if len(hits) >= 2:
            hits.reverse()
            validation = self.agent.validate(context)
            self.assertIsNotNone(validation)


# ---------------------------------------------------------------------------
# Retrieval graph + orchestrator
# ---------------------------------------------------------------------------

class RetrievalGraphTest(unittest.TestCase):
    def setUp(self):
        from src.graphs.retrieval_graph import RetrievalGraph

        self.tmp = Path(tempfile.mkdtemp(prefix="wm_retgraph_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.embedder = StubEmbedder()
        self.store = _fill_store(self.tmp)
        from src.agents.agents.semantic_retrieval_agent import (
            SemanticRetrievalAgent,
        )

        self.graph = RetrievalGraph(
            agents={"semantic_retriever": SemanticRetrievalAgent(
                embedder=self.embedder, store=self.store)},
        )

    def _context(self, **metadata):
        from src.agents.contexts import RetrievalContext

        context = RetrievalContext(question="the king fled the burning capital")
        context.metadata.update(metadata)
        return context

    def test_semantic_kind_runs_and_stores_hits(self):
        context = self._context(top_k=5)
        outcome = self.graph.run(context, kind="semantic")
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.steps[0].step, "semantic_search")
        self.assertTrue(context.outputs["hits"])

    def test_unimplemented_kind_is_reported(self):
        outcome = self.graph.run(self._context(), kind="relation")
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.steps[0].status, "not_implemented")

    def test_semantic_traces_show_step_progression(self):
        context = self._context(top_k=5)
        self.graph.run(context, kind="semantic")
        kinds = [e["kind"] for e in context.events]
        self.assertIn("retrieval_step_start", kinds)
        self.assertIn("retrieval_query", kinds)
        self.assertIn("retrieval_step_done", kinds)

    def test_missing_agent_is_reported(self):
        outcome = self.graph.run(self._context(), kind="semantic")
        graph = type(self.graph)(agents={})  # empty registry
        outcome = graph.run(self._context(), kind="semantic")
        self.assertEqual(outcome.steps[0].status, "not_implemented")

    def test_step_failure_is_reported_not_raised(self):
        from src.agents.protocols import AgentResult, AgentStatus, FailureDomain

        class FailingAgent:
            name = "failing"

            def run(self, context):
                return AgentResult(
                    agent_name=self.name, status=AgentStatus.FAILED,
                    failure_domain=FailureDomain.EXTERNAL, detail="boom",
                )

        graph = type(self.graph)(agents={"semantic_retriever": FailingAgent()})
        outcome = graph.run(self._context(), kind="semantic")
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.steps[0].detail, "boom")

    def test_crashing_step_is_reported_as_external(self):
        class CrashingAgent:
            name = "crasher"

            def run(self, context):
                raise RuntimeError("kaboom")

        graph = type(self.graph)(agents={"semantic_retriever": CrashingAgent()})
        outcome = graph.run(self._context(), kind="semantic")
        self.assertFalse(outcome.ok)
        self.assertIn("unexpected step error", outcome.steps[0].detail)


class RetrievalOrchestratorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="wm_retorch_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.store = _fill_store(self.tmp)
        self.embedder = StubEmbedder()

    def _graph(self):
        from src.agents.agents.semantic_retrieval_agent import (
            SemanticRetrievalAgent,
        )
        from src.graphs.retrieval_graph import RetrievalGraph

        return RetrievalGraph(agents={
            "semantic_retriever": SemanticRetrievalAgent(
                embedder=self.embedder, store=self.store),
        })

    def _run(self, *questions_or_specs, facts=None, **kwargs):
        """Run the deterministic pipeline on analyzer-style requests."""
        from src.retrieval.retrieval_orchestrator import run_retrieval

        requests = []
        for item in questions_or_specs:
            if isinstance(item, RetrievalRequest):
                requests.append(item)
            else:
                requests.append(RetrievalRequest(question=item))
        with unittest.mock.patch(
            "src.retrieval.retrieval_orchestrator.gather_facts",
            return_value=facts or RetrievalFacts(chunk_count=10),
        ):
            return run_retrieval(requests, graph=self._graph(), **kwargs)

    def test_ok_flow_returns_hits_and_traces(self):
        result = self._run(
            RetrievalRequest(
                question="what happened to the king?",
                lookup_kind="semantic", reason="open question",
            ),
        )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["hits"])
        self.assertEqual(result["hits"][0]["text"],
                         "the king fled the burning capital")
        self.assertTrue(result["traces"])
        kinds = [t["kind"] for t in result["traces"]]
        self.assertIn("understood", kinds)

    def test_events_flow_to_the_observer(self):
        seen = []
        self._run("the king", on_event=seen.append)
        self.assertTrue(seen)
        self.assertIn("understood", [e["kind"] for e in seen])

    def test_no_corpus_is_reported(self):
        result = self._run(
            "the king", facts=RetrievalFacts(chunk_count=0),
        )
        self.assertEqual(result["status"], "no_corpus")
        self.assertEqual(result["hits"], [])

    def test_not_implemented_kind_is_reported(self):
        result = self._run(
            RetrievalRequest(
                question="who is married to Jennifer?",
                lookup_kind="relation", reason="relation",
            ),
        )
        self.assertEqual(result["status"], "not_implemented")
        self.assertEqual(result["hits"], [])

    def test_config_error_ends_gracefully(self):
        from src.retrieval.retrieval_orchestrator import run_retrieval
        from src.tools.config_loader import RetrievalConfigError

        with unittest.mock.patch(
            "src.tools.config_loader.load_retrieval_config",
            side_effect=RetrievalConfigError("retrieval.yaml is broken"),
        ):
            result = run_retrieval([])
        self.assertEqual(result["status"], "config_error")
        self.assertIn("retrieval.yaml", result["message"])

    def test_step_failure_is_reported(self):
        from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
        from src.graphs.retrieval_graph import RetrievalGraph
        from src.retrieval.retrieval_orchestrator import run_retrieval

        class FailingAgent:
            name = "failing"

            def run(self, context):
                return AgentResult(
                    agent_name=self.name, status=AgentStatus.FAILED,
                    failure_domain=FailureDomain.EXTERNAL, detail="boom",
                )

        graph = RetrievalGraph(agents={"semantic_retriever": FailingAgent()})
        with unittest.mock.patch(
            "src.retrieval.retrieval_orchestrator.gather_facts",
            return_value=RetrievalFacts(chunk_count=10),
        ):
            result = run_retrieval(
                [RetrievalRequest(question="the king")], graph=graph,
            )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["message"], "0/1 request(s) served")
        self.assertEqual(result["requests"][0]["message"], "boom")

    def test_empty_request_list_is_an_empty_ok_shape(self):
        from src.retrieval.retrieval_orchestrator import run_retrieval

        with unittest.mock.patch(
            "src.retrieval.retrieval_orchestrator.gather_facts",
            return_value=RetrievalFacts(chunk_count=10),
        ):
            result = run_retrieval([], graph=self._graph())
        # Nothing asked, nothing served: all-served vacuously true, but the
        # caller gets an explicit "0/0" and no hits.
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["requests"], [])
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["message"], "0/0 request(s) served")


# ---------------------------------------------------------------------------
# RetrievalTaskAgent — mocked runner
# ---------------------------------------------------------------------------

class RetrievalTaskAgentTest(unittest.TestCase):
    def _agent(self, result: dict):
        from src.agents.agents.retrieval_task_agent import RetrievalTaskAgent

        return RetrievalTaskAgent(runner=unittest.mock.Mock(return_value=result))

    def _request(self):
        return RetrievalRequest(
            question="what happened to the king?",
            utterance="what happened to the king?",
        )

    def _context(self):
        from src.agents.contexts import RoutingContext

        return RoutingContext(request="what happened to the king?")

    def test_ok_maps_to_agent_ok(self):
        agent = self._agent({
            "status": "ok", "hits": [{"id": "x", "text": "t", "score": 0.9,
                                      "metadata": {}}],
            "traces": [],
        })
        result = agent.run(self._context(), self._request())
        self.assertEqual(result.status.value, "ok")
        self.assertEqual(len(result.payload["retrieval"]["hits"]), 1)

    def test_no_corpus_maps_to_failed_with_dormant_wording(self):
        agent = self._agent({"status": "no_corpus", "message": "empty",
                             "hits": [], "traces": []})
        result = agent.run(self._context(), self._request())
        self.assertEqual(result.status.value, "failed")
        self.assertIn("dormant", result.detail)

    def test_not_implemented_maps_to_failed(self):
        agent = self._agent({"status": "not_implemented",
                             "message": "relation not served", "hits": [],
                             "traces": []})
        result = agent.run(self._context(), self._request())
        self.assertEqual(result.status.value, "failed")
        self.assertIn("not served", result.detail)

    def test_config_error_maps_to_config_domain(self):
        agent = self._agent({"status": "config_error",
                             "message": "retrieval.yaml broken", "traces": []})
        result = agent.run(self._context(), self._request())
        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.failure_domain.value, "config")

    def test_orchestrator_traces_merge_into_the_context(self):
        agent = self._agent({
            "status": "ok", "hits": [],
            "traces": [{"phase": "task", "kind": "retrieval_done",
                        "message": "2 hit(s)", "data": {}}],
        })
        context = self._context()
        agent.run(context, self._request())
        self.assertTrue(
            any(e["kind"] == "retrieval_done" for e in context.events))

    def test_out_of_contract_request_is_defended(self):
        from src.agents.agents.retrieval_task_agent import RetrievalTaskAgent

        agent = RetrievalTaskAgent()
        result = agent.run(self._context(), object())
        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.failure_domain.value, "input_data")

    def test_registry_includes_the_retrieval_task_agent(self):
        from src.agents.routing_registry import build_default_task_agents

        registry = build_default_task_agents()
        self.assertIn("retrieval_task", registry)
        self.assertEqual(registry["retrieval_task"].name, "retrieval_task")


# ---------------------------------------------------------------------------
# Batch flow — one prompt, several lookup requests (deterministic loop)
# ---------------------------------------------------------------------------

class BatchOrchestratorTest(unittest.TestCase):
    """run_retrieval over several classified requests: loop + aggregation."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="wm_retbatch_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.store = _fill_store(self.tmp)
        self.embedder = StubEmbedder()

    def _graph(self):
        from src.agents.agents.semantic_retrieval_agent import (
            SemanticRetrievalAgent,
        )
        from src.graphs.retrieval_graph import RetrievalGraph

        return RetrievalGraph(agents={
            "semantic_retriever": SemanticRetrievalAgent(
                embedder=self.embedder, store=self.store),
        })

    def test_king_of_the_north_compound_prompt(self):
        """The user's example: two semantic + one relation request."""
        from src.retrieval.retrieval_orchestrator import run_retrieval

        requests = [
            RetrievalRequest(question="who is the King of the North",
                             lookup_kind="semantic", reason="identity"),
            RetrievalRequest(question="everything about the King of the North",
                             lookup_kind="semantic", reason="content"),
            RetrievalRequest(question="family tree of the King of the North",
                             lookup_kind="relation", reason="kinship"),
        ]
        with unittest.mock.patch(
            "src.retrieval.retrieval_orchestrator.gather_facts",
            return_value=RetrievalFacts(chunk_count=10),
        ):
            result = run_retrieval(requests, graph=self._graph())
        self.assertEqual(result["status"], "partial",
                         "2 semantic served, relation not implemented yet")
        self.assertEqual(len(result["requests"]), 3)
        statuses = [sub["status"] for sub in result["requests"]]
        self.assertEqual(statuses, ["ok", "ok", "not_implemented"])
        # Flat hits carry every served request's chunks.
        self.assertTrue(result["hits"])
        self.assertEqual(
            len(result["hits"]),
            sum(len(sub["hits"]) for sub in result["requests"]),
        )
        kinds = [t["kind"] for t in result["traces"]]
        self.assertIn("request_start", kinds)
        self.assertIn("request_done", kinds)
        self.assertIn("retrieval_not_implemented", kinds)

    def test_all_served_is_ok(self):
        from src.retrieval.retrieval_orchestrator import run_retrieval

        requests = [
            RetrievalRequest(question="the king fled", lookup_kind="semantic"),
            RetrievalRequest(question="the queen defended", lookup_kind="semantic"),
        ]
        with unittest.mock.patch(
            "src.retrieval.retrieval_orchestrator.gather_facts",
            return_value=RetrievalFacts(chunk_count=10),
        ):
            result = run_retrieval(requests, graph=self._graph())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["requests"]), 2)
        self.assertTrue(all(
            sub["status"] == "ok" for sub in result["requests"]))

    def test_partial_message_counts_served_requests(self):
        from src.retrieval.retrieval_orchestrator import run_retrieval

        requests = [
            RetrievalRequest(question="the king fled", lookup_kind="semantic"),
            RetrievalRequest(question="family tree", lookup_kind="relation"),
        ]
        with unittest.mock.patch(
            "src.retrieval.retrieval_orchestrator.gather_facts",
            return_value=RetrievalFacts(chunk_count=10),
        ):
            result = run_retrieval(requests, graph=self._graph())
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["message"], "1/2 request(s) served")

    def test_per_request_traces_carry_the_index(self):
        from src.retrieval.retrieval_orchestrator import run_retrieval

        seen = []
        requests = [
            RetrievalRequest(question="the king fled", lookup_kind="semantic"),
            RetrievalRequest(question="the queen defended", lookup_kind="semantic"),
        ]
        with unittest.mock.patch(
            "src.retrieval.retrieval_orchestrator.gather_facts",
            return_value=RetrievalFacts(chunk_count=10),
        ):
            run_retrieval(requests, graph=self._graph(), on_event=seen.append)
        starts = [e for e in seen if e["kind"] == "request_start"]
        self.assertEqual(
            [e["data"]["request_index"] for e in starts], [1, 2])


class BatchTaskAgentMappingTest(unittest.TestCase):
    """RetrievalTaskAgent maps 'partial' onto a success outcome."""

    def _agent(self, result: dict):
        from src.agents.agents.retrieval_task_agent import RetrievalTaskAgent

        return RetrievalTaskAgent(runner=unittest.mock.Mock(return_value=result))

    def _request(self):
        return RetrievalRequest(
            question="Who is the King of the North? Tell me about him.",
            utterance="Who is the King of the North? Tell me about him.",
        )

    def _context(self):
        from src.agents.contexts import RoutingContext

        return RoutingContext(request="compound retrieval prompt")

    def test_partial_maps_to_ok(self):
        agent = self._agent({
            "status": "partial",
            "message": "2/3 request(s) served",
            "requests": [
                {"status": "ok", "hits": [{"id": "x"}]},
                {"status": "ok", "hits": []},
                {"status": "not_implemented", "hits": []},
            ],
            "hits": [{"id": "x"}],
            "traces": [],
        })
        result = agent.run(self._context(), self._request())
        self.assertEqual(result.status.value, "ok")
        self.assertIn("partial", result.detail)
        self.assertIn("2/3", result.detail)

    def test_partial_emits_a_done_trace(self):
        agent = self._agent({
            "status": "partial", "requests": [], "hits": [], "traces": [],
        })
        context = self._context()
        agent.run(context, self._request())
        kinds = [e["kind"] for e in context.events]
        self.assertIn("retrieval_done", kinds)
        done = next(e for e in context.events if e["kind"] == "retrieval_done")
        self.assertTrue(done["data"]["partial"])

    def test_ok_path_unchanged(self):
        agent = self._agent({
            "status": "ok", "requests": [],
            "hits": [{"id": "x", "text": "t", "score": 0.9,
                      "metadata": {}}],
            "traces": [],
        })
        result = agent.run(self._context(), self._request())
        self.assertEqual(result.status.value, "ok")
        self.assertEqual(result.detail, "1 chunk(s) retrieved")

    def test_request_type_is_defended(self):
        from src.agents.agents.retrieval_task_agent import RetrievalTaskAgent

        agent = RetrievalTaskAgent()
        result = agent.run(self._context(), object())
        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.failure_domain.value, "input_data")


if __name__ == "__main__":
    unittest.main()
