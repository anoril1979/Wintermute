"""SourceIndexingAgent — chunking + embedding + vector storage of a document.

The first *source collection* node of the ingestion graph: it takes the
summarized ``DocumentExtract`` produced by the extraction/summarization
steps and persists it into the ``source_chunks`` vector collection:

    build_source_chunks (unified ids, src/indexing/chunks.py)
        → OllamaEmbeddingClient (``embedding`` role, batched)
        → ChromaVectorClient.upsert (idempotent: deterministic ids mean a
          re-ingestion UPDATES the document's chunks, never duplicates)

The store never embeds and the embedder never stores (protocols in
src/indexing/protocols.py); this agent is the orchestrating glue, plus
traces for the thinking panel (chunk counts, embedding batches, final
collection count).

Idempotence note: the vector store needs no job file of its own — chunk
ids are derived from the document id and element positions, so re-running
this step (re-ingestion, resume after a later failure) simply overwrites
the same ids. A stale index can only exist when the *content* changed,
which is a re-ingestion's normal flow.

Failure mapping (agents never raise for expected failures):

* no summarized document in the context  → ``INPUT_DATA`` (pipeline bug or misuse)
* no document id assigned                → ``INPUT_DATA`` (extraction layer must assign ids)
* Ollama unreachable / embedding failure → ``EXTERNAL`` (not retryable here: the
  embedding client already retries internally)
* store cannot open / ChromaDB failure   → ``EXTERNAL``
"""

from __future__ import annotations

import logging
from typing import Optional

from src.agents.contexts import IngestionContext
from src.agents.protocols import (
    AgentResult,
    AgentStatus,
    FailureDomain,
)
from src.indexing.chroma_client import (
    ChromaVectorClient,
    VectorStoreError,
)
from src.indexing.embedding_client import (
    EmbeddingClientError,
    OllamaEmbeddingClient,
)
from src.indexing.chunks import (
    VectorChunk,
    build_source_chunks,
    doc_id_of,
)
from src.tools.config_loader import load_vector_config

logger = logging.getLogger(__name__)

#: Context keys (graph step names).
INPUT_KEY = "content_extraction"
COLLECTION_KEY = "source_chunks"   # setup.yaml vector_db.collections key

#: Fallback when setup.yaml does not set ``vector_db.embedding_batch_size``
#: (must stay in sync with config/setup.yaml and the config loader's
#: validation) — qwen3-embedding handles this comfortably on a workstation;
#: one HTTP round-trip per batch.
DEFAULT_EMBED_BATCH_SIZE = 32


def embedding_batch_size() -> int:
    """Batch size from setup.yaml (``vector_db.embedding_batch_size``),

    fail-open to :data:`DEFAULT_EMBED_BATCH_SIZE`. The yaml is validated at
    load time by the config loader (strictly positive int), so this only
    covers a broken or unreadable config — indexing must keep working.
    """
    try:
        vector = load_vector_config()
        value = int(vector.get("embedding_batch_size", DEFAULT_EMBED_BATCH_SIZE))
    except Exception as exc:  # noqa: BLE001 — fail-open to the default
        logger.warning(
            "Could not read setup.yaml for embedding_batch_size; using "
            "default %d: %s", DEFAULT_EMBED_BATCH_SIZE, exc,
        )
        return DEFAULT_EMBED_BATCH_SIZE
    return value if value > 0 else DEFAULT_EMBED_BATCH_SIZE


class SourceIndexingAgent:
    """Indexes the summarized document into the source_chunks collection.

    Args:
        embedder: the embedding client to use; defaults to the shared
            ``embedding``-role client. Injectable for tests.
        store: the vector store client; defaults to a ``source_chunks``
            ChromaVectorClient. Injectable for tests.
        input_key: context key holding the summarized DocumentExtract.
        batch_size: texts per embedding call; ``None`` (the default) reads
            setup.yaml's ``vector_db.embedding_batch_size``. Injectable for
            tests.
    """

    name = "source_indexer"

    def __init__(
        self,
        embedder: Optional[OllamaEmbeddingClient] = None,
        store: Optional[ChromaVectorClient] = None,
        input_key: str = INPUT_KEY,
        batch_size: Optional[int] = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._input_key = input_key
        self._batch_size = max(1, int(
            batch_size if batch_size is not None else embedding_batch_size()
        ))

    @property
    def embedder(self) -> OllamaEmbeddingClient:
        """Lazily-built default embedder (the ``embedding`` llm.yaml role).

        Config layering: the agent never reads llm.yaml itself — model,
        options and retries are the embedder client's own settings, read
        when the client is built (``OllamaEmbeddingClient.for_role``).
        Only the batch size (this agent's own tuning knob) is read here,
        from setup.yaml's ``vector_db`` section.
        """
        if self._embedder is None:
            self._embedder = OllamaEmbeddingClient.for_role("embedding")
        return self._embedder

    @property
    def store(self) -> ChromaVectorClient:
        """Lazily-built default store (the ``source_chunks`` collection).

        Config layering: store path and collection names are the store
        client's own settings, read by ``ChromaVectorClient`` when it is
        built (``load_vector_config``).
        """
        if self._store is None:
            self._store = ChromaVectorClient(COLLECTION_KEY)
        return self._store

    # -- IngestionAgent contract ---------------------------------------------

    def run(self, context: IngestionContext) -> AgentResult:
        """Chunk, embed and store the document found in the context."""
        document = context.outputs.get(self._input_key)
        if document is None:
            detail = f"no document in context.outputs[{self._input_key!r}]"
            context.emit("task", "indexing_failed", detail)
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail=detail,
            )

        # -- chunks -------------------------------------------------------------
        try:
            chunks = build_source_chunks(document)
        except ValueError as exc:
            # No assigned document id: the extraction layer owns identity.
            detail = f"cannot build chunks: {exc}"
            context.emit("task", "indexing_failed", detail)
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail=detail,
            )

        if not chunks:
            # A structurally valid document can still be contentless after
            # pruning: nothing to index is a successful no-op, not a failure.
            context.metadata["indexed_chunk_count"] = 0
            context.emit(
                "task", "indexing_skipped",
                "no indexable content (no blocks, no summaries); "
                "nothing stored",
            )
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.OK,
                payload={"chunks": 0, "stored": 0},
            )

        doc_id = doc_id_of(document)
        context.emit(
            "task", "indexing_chunks_built",
            f"built {len(chunks)} chunk(s) for '{doc_id}' "
            f"({sum(1 for c in chunks if c.metadata['kind'] == 'content')} content, "
            f"{sum(1 for c in chunks if c.metadata['kind'] == 'summary')} summaries)",
            doc_id=doc_id, chunk_count=len(chunks),
        )

        # -- embedding (batched) ---------------------------------------------------
        vectors: list[list[float]] = []
        try:
            for start in range(0, len(chunks), self._batch_size):
                batch = chunks[start:start + self._batch_size]
                batch_vectors = self.embedder.embed([c.text for c in batch])
                if len(batch_vectors) != len(batch):
                    detail = (
                        f"embedding backend returned {len(batch_vectors)} "
                        f"vector(s) for {len(batch)} text(s)"
                    )
                    context.emit("task", "indexing_failed", detail)
                    return AgentResult(
                        agent_name=self.name,
                        status=AgentStatus.FAILED,
                        failure_domain=FailureDomain.EXTERNAL,
                        detail=detail,
                    )
                vectors.extend(batch_vectors)
                context.emit(
                    "task", "indexing_embedding",
                    f"embedded {min(start + self._batch_size, len(chunks))}"
                    f"/{len(chunks)} chunk(s)",
                    doc_id=doc_id,
                    done=min(start + self._batch_size, len(chunks)),
                    total=len(chunks),
                )
        except EmbeddingClientError as exc:
            detail = f"embedding backend failure: {exc}"
            context.emit("task", "indexing_failed", detail)
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.EXTERNAL,
                detail=detail,
            )
        except Exception as exc:  # noqa: BLE001 — agents never raise; an
            # unexpected embedder crash is an external-dependency failure.
            detail = f"unexpected embedding failure: {exc}"
            context.emit("task", "indexing_failed", detail)
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.EXTERNAL,
                detail=detail,
            )

        for chunk, vector in zip(chunks, vectors):
            chunk.vector = vector

        # -- storage -----------------------------------------------------------------
        try:
            stored = self.store.upsert(chunks)
        except (VectorStoreError, ValueError) as exc:
            detail = f"vector storage failure: {exc}"
            context.emit("task", "indexing_failed", detail)
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.EXTERNAL,
                detail=detail,
            )

        total = self.store.count()
        context.metadata.update(
            {
                "indexed_doc_id": doc_id,
                "indexed_chunk_count": stored,
            }
        )
        context.emit(
            "task", "indexed",
            f"indexed '{doc_id}': {stored} chunk(s) stored "
            f"(collection now holds {total} chunk(s))",
            doc_id=doc_id, stored=stored, collection_count=total,
        )
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.OK,
            payload={
                "doc_id": doc_id,
                "chunks": len(chunks),
                "stored": stored,
                "collection_count": total,
            },
        )

    def validate(self, context: IngestionContext) -> Optional[AgentResult]:
        """Post-run check: the collection must hold at least one chunk.

        Skipped when this agent stored nothing (no-content no-op — an
        empty collection is legitimate there) or had nothing to check.
        """
        if self._input_key not in context.outputs:
            return None
        if context.metadata.get("indexed_chunk_count", 1) == 0:
            return None  # deliberate no-op: nothing was meant to be stored
        try:
            if self.store.count() == 0:
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.FAILED,
                    failure_domain=FailureDomain.EXTERNAL,
                    detail="source_chunks collection is empty after indexing",
                )
        except VectorStoreError as exc:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.EXTERNAL,
                detail=f"could not count the collection after indexing: {exc}",
            )
        return None
