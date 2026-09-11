"""Tests for the SourceIndexingAgent — chunk → embed → store, hermetically.

Real embedded ChromaDB on a temp folder; a deterministic stub embedder;
no Ollama, no live anything.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from src.agents.agents import source_indexing_agent as sia_module
from src.agents.agents.source_indexing_agent import SourceIndexingAgent
from src.agents.contexts import IngestionContext
from src.agents.protocols import AgentStatus, FailureDomain
from src.extraction.ids import assign_extract_ids
from src.extraction.models import (
    Chapter,
    DocumentExtract,
    PageContent,
    Section,
    TextBlock,
    TocEntry,
)
from src.indexing.chroma_client import ChromaVectorClient, VectorStoreError


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


def _doc() -> DocumentExtract:
    """A small summarized document: 2 blocks, page + chapter + doc summaries."""
    blocks = [
        TextBlock(block_id=0, page_number=1, bbox=(0, 0, 10, 10),
                  raw_text="alpha content", summary="s-alpha"),
        TextBlock(block_id=1, page_number=1, bbox=(0, 0, 10, 20),
                  raw_text="beta content", summary="s-beta"),
    ]
    page = PageContent(
        page_number=1, width=595, height=842, raw_text="alpha beta",
        summary="page summary",
        sections=[Section(section_id=0, blocks=blocks, page_number=1,
                          bbox=(0, 0, 10, 20), raw_text="alpha beta",
                          summary="section summary")],
    )
    doc = DocumentExtract(
        source_path="data/sources/pdf/gazette-test.pdf",
        title="Gazette Test", author="", subject="", total_pages=1,
        chapters=[Chapter(toc_entry=TocEntry(level=1, title="Ch1",
                                             page_number=1, page_index=0),
                          pages=[page], full_text="alpha beta",
                          summary="chapter summary")],
        summary="document summary",
    )
    return assign_extract_ids(doc)


class SourceIndexingAgentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="wm_idx_agent_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.embedder = StubEmbedder()
        self.store = ChromaVectorClient(
            "source_chunks", embedding_dimension=StubEmbedder.dimension,
            path=str(self.tmp), collection_name="agent_test_chunks",
        )
        self.agent = SourceIndexingAgent(
            embedder=self.embedder, store=self.store, batch_size=2
        )

    def _context(self, doc=None) -> IngestionContext:
        context = IngestionContext(request="[test]")
        if doc is not None:
            context.outputs["content_extraction"] = doc
        return context

    def test_happy_path_stores_everything(self):
        result = self.agent.run(self._context(_doc()))
        self.assertEqual(result.status, AgentStatus.OK)
        # 2 blocks + 1 section + 1 page + 1 chapter + 1 doc summary = 6
        self.assertEqual(result.payload["chunks"], 6)
        self.assertEqual(result.payload["stored"], 6)
        self.assertEqual(self.store.count(), 6)

    def test_metadata_carries_origin_and_ids(self):
        self.agent.run(self._context(_doc()))
        # Spot-check one chunk through a fresh client on the same store.
        reopened = ChromaVectorClient("source_chunks", path=str(self.tmp),
                                      collection_name="agent_test_chunks")
        self.assertEqual(reopened.count(), 6)

    def test_reindex_is_idempotent(self):
        doc = _doc()
        self.agent.run(self._context(doc))
        self.agent.run(self._context(_doc()))  # same ids → update, not duplicate
        self.assertEqual(self.store.count(), 6)

    def test_no_document_in_context_fails_input_data(self):
        result = self.agent.run(self._context(None))
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.failure_domain, FailureDomain.INPUT_DATA)

    def test_unassigned_ids_fail_input_data(self):
        doc = _doc()
        doc.id = ""
        result = self.agent.run(self._context(doc))
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.failure_domain, FailureDomain.INPUT_DATA)
        self.assertIn("assign_extract_ids", result.detail)

    def test_embedding_failure_maps_to_external(self):
        self.embedder.fail = True
        result = self.agent.run(self._context(_doc()))
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.failure_domain, FailureDomain.EXTERNAL)

    def test_contentless_document_is_a_noop(self):
        doc = _doc()
        doc.chapters[0].summary = None
        page = doc.chapters[0].pages[0]
        page.summary = None
        page.sections[0].summary = None
        for block in page.sections[0].blocks:
            block.raw_text = ""
        doc.summary = None
        result = self.agent.run(self._context(doc))
        self.assertEqual(result.status, AgentStatus.OK)
        self.assertEqual(result.payload["chunks"], 0)
        self.assertEqual(self.store.count(), 0)

    def test_validate_passes_after_real_run(self):
        self.agent.run(self._context(_doc()))
        self.assertIsNone(self.agent.validate(self._context(_doc())))

    def test_validate_allows_the_contentless_noop(self):
        doc = _doc()
        doc.summary = None
        doc.chapters[0].summary = None
        page = doc.chapters[0].pages[0]
        page.summary = None
        page.sections[0].summary = None
        for block in page.sections[0].blocks:
            block.raw_text = ""
        context = self._context(doc)
        self.agent.run(context)
        self.assertIsNone(self.agent.validate(context))

    def test_validate_flags_an_empty_collection_after_real_run(self):
        context = self._context(_doc())
        self.agent.run(context)
        # Simulate an external wipe: point the agent's store at a fresh,
        # never-written folder (no sqlite file → count() == 0).
        empty_dir = self.tmp / "wiped"
        empty_dir.mkdir()
        self.agent._store = ChromaVectorClient(
            "source_chunks", path=str(empty_dir),
            collection_name="agent_test_chunks",
        )
        validation = self.agent.validate(context)
        self.assertIsNotNone(validation)
        self.assertEqual(validation.status, AgentStatus.FAILED)
        self.assertEqual(validation.failure_domain, FailureDomain.EXTERNAL)

    def test_batching_calls_embedder_in_batches(self):
        calls = []
        original = self.embedder.embed

        def spy(texts):
            calls.append(len(texts))
            return original(texts)

        self.embedder.embed = spy
        self.agent.run(self._context(_doc()))
        # 6 chunks, batch_size 2 → 3 batches (2, 2, 2).
        self.assertEqual(calls, [2, 2, 2])

    def test_batch_size_defaults_to_setup_yaml(self):
        """No explicit batch_size → the vector_db.embedding_batch_size knob."""
        original = sia_module.load_vector_config

        def fake_config():
            return {"path": "data/vector",
                    "collections": {"source_chunks": "source_chunks",
                                    "knowledge_chunks": "knowledge_chunks"},
                    "embedding_batch_size": 3}

        try:
            sia_module.load_vector_config = fake_config
            agent = SourceIndexingAgent(embedder=self.embedder, store=self.store)
            self.assertEqual(agent._batch_size, 3)
        finally:
            sia_module.load_vector_config = original

    def test_batch_size_fails_open_to_default_on_broken_config(self):
        """An unreadable config must not block indexing: default 32 applies."""
        original = sia_module.load_vector_config

        def broken_config():
            raise RuntimeError("setup.yaml unreadable")

        try:
            sia_module.load_vector_config = broken_config
            agent = SourceIndexingAgent(embedder=self.embedder, store=self.store)
            self.assertEqual(agent._batch_size, 32)
        finally:
            sia_module.load_vector_config = original

    def test_traces_narrate_the_steps(self):
        context = self._context(_doc())
        self.agent.run(context)
        kinds = [e["kind"] for e in context.events if e["phase"] == "task"]
        self.assertIn("indexing_chunks_built", kinds)
        self.assertIn("indexing_embedding", kinds)
        self.assertIn("indexed", kinds)


if __name__ == "__main__":
    unittest.main()
