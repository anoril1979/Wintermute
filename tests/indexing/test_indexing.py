"""Tests for the vector indexing client layer.

Hermetic by design:

* ChromaDB embedded mode needs no network — tests exercise the real store
  on temporary folders;
* the embedder is a deterministic stub (fake vectors derived from the text
  via a seeded hash) — no Ollama anywhere;
* the Ollama embedding client itself is tested with a mock httpx transport,
  same convention as tests/llm/test_llm_client.py.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from src.extraction.models import (
    BlockType,
    Chapter,
    DocumentExtract,
    PageContent,
    Section,
    TextBlock,
    TocEntry,
)
from src.indexing import (
    ChromaVectorClient,
    DimensionMismatchError,
    EmbeddingRequestError,
    LEVEL_BLOCK,
    LEVEL_DOCUMENT,
    LEVEL_PAGE,
    KIND_CONTENT,
    KIND_SUMMARY,
    OllamaEmbeddingClient,
    VectorChunk,
    build_source_chunks,
    doc_id_of,
    get_embedding_client,
)
from src.extraction.ids import assign_extract_ids
from src.indexing.chunks import build_knowledge_chunks
from src.indexing.protocols import (
    EmbeddingClientProtocol,
    VectorStoreProtocol,
)
from src.indexing.chroma_client import (
    vector_db_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _doc(**overrides) -> DocumentExtract:
    """A tiny two-chapter, three-page summarized DocumentExtract."""
    def block(block_id: int, page: int, text: str) -> TextBlock:
        return TextBlock(
            block_id=block_id,
            page_number=page,
            bbox=(0.0, 0.0, 10.0, 10.0),
            raw_text=text,
            summary=f"summary of {text}",
        )

    def page(number: int, blocks: list[TextBlock], summary: str) -> PageContent:
        return PageContent(
            page_number=number,
            width=595.0,
            height=842.0,
            raw_text=" ".join(b.raw_text for b in blocks),
            summary=summary,
            sections=[Section(
                section_id=0,
                blocks=blocks,
                page_number=number,
                bbox=(0.0, 0.0, 10.0, 10.0),
                raw_text=" ".join(b.raw_text for b in blocks),
                summary=f"section summary p{number}",
            )],
        )

    chapters = [
        Chapter(
            toc_entry=TocEntry(level=1, title="Chapter One", page_number=1, page_index=0),
            pages=[page(1, [block(0, 1, "alpha text"), block(1, 1, "beta text")],
                        "page one summary"),
                   page(2, [block(0, 2, "gamma text")], "page two summary")],
            full_text="alpha beta gamma",
            summary="chapter one summary",
        ),
        Chapter(
            toc_entry=TocEntry(level=1, title="Chapter Two", page_number=3, page_index=2),
            pages=[page(3, [block(0, 3, "delta text")], "page three summary")],
            full_text="delta",
            summary="chapter two summary",
        ),
    ]
    doc = DocumentExtract(
        source_path="data/sources/pdf/Gazette.pdf",
        title="Gazette",
        author="",
        subject="",
        total_pages=3,
        chapters=chapters,
        summary="whole document summary",
    )
    for key, value in overrides.items():
        setattr(doc, key, value)
    return assign_extract_ids(doc)


class _StubEmbedder:
    """Deterministic embedder: fake 8-dim vectors derived from the text."""

    dimension = 8

    def embed(self, texts):
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([b / 255.0 for b in digest[: self.dimension]])
        return vectors


# ---------------------------------------------------------------------------
# Chunk builders
# ---------------------------------------------------------------------------

class SourceChunksTest(unittest.TestCase):
    def test_levels_and_kinds(self):
        chunks = build_source_chunks(_doc())
        kinds = {(c.metadata["level"], c.metadata["kind"]) for c in chunks}

        self.assertIn((LEVEL_BLOCK, KIND_CONTENT), kinds)      # 4 blocks
        self.assertIn(("section", KIND_SUMMARY), kinds)        # 3 sections
        self.assertIn((LEVEL_PAGE, KIND_SUMMARY), kinds)       # 3 pages
        self.assertIn(("chapter", KIND_SUMMARY), kinds)        # 2 chapters
        self.assertIn((LEVEL_DOCUMENT, KIND_SUMMARY), kinds)   # 1 doc summary

        self.assertEqual(
            sum(1 for c in chunks if c.metadata["level"] == LEVEL_BLOCK), 4)
        self.assertEqual(len(chunks), 4 + 3 + 3 + 2 + 1)

    def test_ids_are_deterministic_and_use_unified_scheme(self):
        doc = _doc()
        self.assertEqual(build_source_chunks(doc), build_source_chunks(_doc()))
        ids = [c.id for c in build_source_chunks(doc)]
        self.assertEqual(len(ids), len(set(ids)), "chunk ids must be unique")
        self.assertTrue(all(i.startswith(doc.id + "::") for i in ids),
                        "chunk ids must chain from the document's unified id")
        # Flat per-parent element ids chained hierarchically:
        self.assertIn(f"{doc.id}::chp:1::pg:1::sec:1::txt:1", ids)
        self.assertIn(f"{doc.id}::sum", ids)
        self.assertIn(f"{doc.id}::chp:1::sum", ids)

    def test_block_summary_is_not_indexed(self):
        chunks = build_source_chunks(_doc())
        self.assertNotIn(
            "summary of alpha text", [c.text for c in chunks],
            "a block's own summary must not become a chunk",
        )

    def test_blank_blocks_are_skipped(self):
        doc = _doc()
        doc.chapters[0].pages[0].sections[0].blocks[0].raw_text = "   \t "
        chunks = build_source_chunks(doc)
        self.assertEqual(
            sum(1 for c in chunks if c.metadata["level"] == LEVEL_BLOCK), 3)

    def test_no_summary_no_chunk(self):
        doc = _doc()
        doc.chapters[0].summary = None
        chunks = build_source_chunks(doc)
        self.assertNotIn(
            "chapter one summary", [c.text for c in chunks])

    def test_metadata_is_chroma_scalar_only(self):
        chunks = build_source_chunks(_doc())
        for chunk in chunks:
            for value in chunk.metadata.values():
                self.assertIsInstance(value, (str, int, float, bool))

    def test_unassigned_document_id_raises(self):
        doc = _doc()
        doc.id = ""  # indexing must refuse to invent an identity
        with self.assertRaises(ValueError):
            build_source_chunks(doc)

    def test_doc_id_of_returns_model_id(self):
        doc = _doc()
        self.assertEqual(doc_id_of(doc), doc.id)

    def test_knowledge_chunks_stub(self):
        with self.assertRaises(NotImplementedError):
            build_knowledge_chunks("whatever.md")


# ---------------------------------------------------------------------------
# ChromaDB client — real embedded store on a temp folder
# ---------------------------------------------------------------------------

class ChromaClientTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="wintermute_chroma_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.client = ChromaVectorClient(
            "source_chunks", path=str(self.tmp), collection_name="test_chunks"
        )
        self.addCleanup(self.client.close if hasattr(self.client, "close") else lambda: None)

    def _embedded_chunks(self, n: int = 2) -> list[VectorChunk]:
        embedder = _StubEmbedder()
        built = build_source_chunks(_doc())[:n]
        for chunk in built:
            chunk.vector = embedder.embed([chunk.text])[0]
        return built

    def test_implements_protocol(self):
        self.assertIsInstance(self.client, VectorStoreProtocol)

    def test_upsert_then_count(self):
        chunks = self._embedded_chunks(3)
        self.assertEqual(self.client.upsert(chunks), 3)
        self.assertEqual(self.client.count(), 3)

    def test_count_is_zero_and_creates_nothing(self):
        self.assertEqual(self.client.count(), 0)
        self.assertFalse((self.tmp / "chroma.sqlite3").exists(),
                         "count() must never materialize the store")

    def test_upsert_is_idempotent(self):
        chunks = self._embedded_chunks(2)
        self.client.upsert(chunks)
        self.client.upsert(chunks)
        self.assertEqual(self.client.count(), 2)

    def test_upsert_updates_existing_id(self):
        chunks = self._embedded_chunks(1)
        self.client.upsert(chunks)
        updated = [VectorChunk(id=chunks[0].id, text="rewritten text",
                               metadata=chunks[0].metadata, vector=chunks[0].vector)]
        self.client.upsert(updated)
        self.assertEqual(self.client.count(), 1)

    def test_requires_vectors(self):
        built = build_source_chunks(_doc())[:1]
        with self.assertRaises(ValueError):
            self.client.upsert(built)

    def test_rejects_empty_batch(self):
        with self.assertRaises(ValueError):
            self.client.upsert([])

    def test_rejects_mixed_dimensions(self):
        chunks = self._embedded_chunks(2)
        chunks[1].vector = [0.5] * (len(chunks[0].vector) + 1)
        with self.assertRaises(ValueError):
            self.client.upsert(chunks)

    def test_dimension_mismatch_against_client(self):
        client = ChromaVectorClient("source_chunks", embedding_dimension=4096,
                                    path=str(self.tmp), collection_name="test_dims")
        chunks = self._embedded_chunks(1)  # stub is 8-dim
        with self.assertRaises(DimensionMismatchError):
            client.upsert(chunks)

    def test_collection_persists_across_clients(self):
        self.client.upsert(self._embedded_chunks(2))
        reopened = ChromaVectorClient("source_chunks", path=str(self.tmp),
                                      collection_name="test_chunks")
        self.assertEqual(reopened.count(), 2)

    def test_query_is_a_stub(self):
        with self.assertRaises(NotImplementedError):
            self.client.query("hello")

    def test_delete_document_is_a_stub(self):
        with self.assertRaises(NotImplementedError):
            self.client.delete_document("Gazette")

    def test_unknown_collection_key(self):
        with self.assertRaises(Exception):
            ChromaVectorClient("no_such_key", path=str(self.tmp))


# ---------------------------------------------------------------------------
# Ollama embedding client — mock transport, no live Ollama
# ---------------------------------------------------------------------------

def _embed_answer(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content.decode("utf-8"))
    return httpx.Response(
        200,
        json={"embeddings": [[0.1, 0.2] for _ in payload["input"]]},
        request=request,
    )


class EmbeddingClientTest(unittest.TestCase):
    def _client(self, handler) -> OllamaEmbeddingClient:
        return OllamaEmbeddingClient(
            model="test-embed", transport=httpx.MockTransport(handler)
        )

    def test_embed_batch_order_preserved(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content.decode("utf-8"))
            return _embed_answer(request)

        with self._client(handler) as client:
            vectors = client.embed(["first", "second", "third"])
        self.assertEqual(len(vectors), 3)
        self.assertEqual(seen["payload"]["input"], ["first", "second", "third"])
        self.assertEqual(seen["payload"]["model"], "test-embed")

    def test_embed_rejects_empty_batch(self):
        with self._client(_embed_answer) as client, \
                self.assertRaises(ValueError):
            client.embed([])

    def test_embed_rejects_blank_text(self):
        with self._client(_embed_answer) as client, \
                self.assertRaises(ValueError):
            client.embed(["ok", "   "])

    def test_count_mismatch_is_an_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]},
                                  request=request)

        with self._client(handler) as client, \
                self.assertRaises(EmbeddingRequestError):
            client.embed(["one", "two"])

    def test_malformed_answer_is_an_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": True},
                                  request=request)

        with self._client(handler) as client, \
                self.assertRaises(EmbeddingRequestError):
            client.embed(["one"])

    def test_unreachable_backend_raises_after_retries(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "boom"}, request=request)

        with self._client(handler) as client:
            with self.assertRaises(EmbeddingRequestError):
                client.embed(["one"])

    def test_implements_protocol(self):
        with self._client(_embed_answer) as client:
            self.assertIsInstance(client, EmbeddingClientProtocol)

    def test_for_role_builds_from_config(self):
        role_cfg = {"model_name": "qwen3-embedding", "timeout_seconds": 120,
                    "keep_alive": "10m", "max_retries": 3}
        with mock.patch("src.tools.config_loader.get_model_config",
                        return_value=dict(role_cfg)), \
             mock.patch("src.tools.config_loader.load_setup_config",
                        return_value={"ollama": {"base_url": "http://localhost:11434"}}):
            client = OllamaEmbeddingClient.for_role("embedding")
        self.assertEqual(client.model, "qwen3-embedding")
        self.assertEqual(client.max_retries, 3)
        self.assertEqual(client.timeout, 120.0)
        client.close()

    def test_get_embedding_client_is_cached(self):
        with mock.patch.object(OllamaEmbeddingClient, "for_role",
                               return_value=OllamaEmbeddingClient(
                                   model="x",
                                   transport=httpx.MockTransport(_embed_answer))) as for_role:
            get_embedding_client.cache_clear()
            first = get_embedding_client()
            second = get_embedding_client()
            self.assertIs(first, second)
            for_role.assert_called_once()
        get_embedding_client.cache_clear()


# ---------------------------------------------------------------------------
# setup.yaml vector_db validation
# ---------------------------------------------------------------------------

VALID_VECTOR = {
    "path": "data/vector",
    "collections": {"source_chunks": "source_chunks",
                    "knowledge_chunks": "knowledge_chunks"},
}


class VectorConfigValidationTest(unittest.TestCase):
    def test_valid_config_roundtrip(self):
        from src.tools.config_loader import validate_vector_config

        config = {"vector_db": dict(VALID_VECTOR)}
        self.assertIs(validate_vector_config(config), config)

    def test_missing_section(self):
        from src.tools.config_loader import validate_vector_config

        with self.assertRaises(Exception):
            validate_vector_config({})

    def test_missing_path(self):
        from src.tools.config_loader import validate_vector_config, VectorConfigError

        with self.assertRaises(VectorConfigError):
            validate_vector_config({"vector_db": {"collections": VALID_VECTOR["collections"]}})

    def test_missing_required_collection(self):
        from src.tools.config_loader import validate_vector_config, VectorConfigError

        with self.assertRaises(VectorConfigError):
            validate_vector_config({"vector_db": {
                "path": "data/vector",
                "collections": {"source_chunks": "source_chunks"}}})

    def test_duplicate_collection_names(self):
        from src.tools.config_loader import validate_vector_config, VectorConfigError

        with self.assertRaises(VectorConfigError):
            validate_vector_config({"vector_db": {
                "path": "data/vector",
                "collections": {"source_chunks": "same",
                                "knowledge_chunks": "same"}}})

    def test_valid_embedding_batch_size_roundtrip(self):
        from src.tools.config_loader import validate_vector_config

        config = {"vector_db": {**VALID_VECTOR, "embedding_batch_size": 16}}
        self.assertIs(validate_vector_config(config), config)
        self.assertEqual(config["vector_db"]["embedding_batch_size"], 16)

    def test_embedding_batch_size_rejects_non_positive(self):
        from src.tools.config_loader import validate_vector_config, VectorConfigError

        for bad in (0, -4, 1.5, True, "32"):
            with self.assertRaises(VectorConfigError, msg=repr(bad)):
                validate_vector_config({"vector_db": {
                    **VALID_VECTOR, "embedding_batch_size": bad}})

    def test_traversal_path_rejected(self):
        from src.tools.config_loader import validate_vector_config, VectorConfigError

        with self.assertRaises(VectorConfigError):
            validate_vector_config({"vector_db": {
                "path": "../outside",
                "collections": VALID_VECTOR["collections"]}})

    def test_real_setup_yaml_loads(self):
        """The shipped setup.yaml must pass its own validation."""
        self.assertTrue(vector_db_config()["path"])


if __name__ == "__main__":
    unittest.main()
