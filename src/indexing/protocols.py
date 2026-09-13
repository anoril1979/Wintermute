"""Protocols (structural contracts) for the vector indexing layer.

Mirrors ``src/llm/protocols.py``: the contract is the single interface the
rest of the system codes against — the future IndexingAgent, the retrieval
orchestrator, test doubles — never on a concrete client. Concrete
implementations live beside it (``chroma_client.py`` for the store,
``embedding_client.py`` for embeddings) and must satisfy these interfaces.

Scope today: **storage, vector retrieval and index maintenance**.
``query_by_vector`` is the real query seam (the retrieval layer embeds the
question, the store never does); text queries remain an honest stub.
``delete_document`` is real: corpus maintenance (dedup, out-of-corpus
removal) deletes a document's projection by its unified ``doc_id``.

The ``runtime_checkable`` decorators allow ``isinstance(...)`` sanity checks
(method presence only — they do not verify signatures).
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

from src.indexing.chunks import VectorChunk


@runtime_checkable
class EmbeddingClientProtocol(Protocol):
    """Minimal interface expected from any embedding client.

    Implementations: an Ollama-backed client (``embedding_client.py``), a
    future remote endpoint, or a test double returning deterministic
    vectors. Batch-oriented by design — embedding is always cheaper in
    batches than one request per chunk.
    """

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, order-preserving.

        Args:
            texts: non-empty texts to embed. The protocol does not decide
                the batching policy — that is an implementation concern.

        Returns:
            One vector per input text, in the same order. All vectors of a
            given client have the same dimensionality (the model's).

        Raises:
            ValueError: on an empty batch or an empty/blank text.
            EmbeddingClientError: when the backend cannot embed the batch.
        """
        ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Contract for a vector store: storage now, retrieval as stubs.

    Implementations: the embedded ChromaDB client (``chroma_client.py``),
    a future server-backed client, or an in-memory test double.
    """

    def upsert(self, chunks: Sequence[VectorChunk]) -> int:
        """Insert or update chunks (idempotent by deterministic id).

        Args:
            chunks: chunks to store. Their ``vector`` field must already be
                filled (embedding is the caller's concern — keep the store
                protocol decoupled from the embedding backend).

        Returns:
            The number of chunks written.

        Raises:
            ValueError: on an empty batch, dimension mismatch against the
                collection's first write, or a chunk with no vector.
            VectorStoreError: when the backend cannot store the batch.
        """
        ...

    # -- Retrieval --------------------------------------------------------------

    def query_by_vector(
        self,
        vector: Sequence[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[VectorChunk]:
        """Nearest-neighbor search with a pre-computed query vector.

        The vector-seam variant: the caller (SemanticRetrievalAgent) embeds
        the question itself, so the store never embeds — same decoupling as
        ``upsert``. Scores are filled on the returned chunks (cosine
        similarity, higher is better).

        Args:
            vector: the query embedding (same model/dimension as the
                collection's chunks).
            top_k: number of neighbors to fetch (1..n).
            where: optional ChromaDB metadata filter (see
                ``src/retrieval/filters.py`` for building one).

        Returns:
            Chunks in descending similarity order, each with ``score``
            filled. Fewer than ``top_k`` when the collection holds less.

        Raises:
            ValueError: empty vector or non-positive top_k.
            VectorStoreUnavailableError: the store could not be opened.
            VectorStoreError: the backend refused the query.
        """
        ...

    def query(
        self,
        text: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[VectorChunk]:
        """Semantic search by text (STUB — NotImplementedError).

        Text queries would force the store to embed (a store-embedded
        coupling we deliberately avoid); the retrieval layer embeds the
        question and calls :meth:`query_by_vector` instead.
        """
        ...

    def count(self) -> int:
        """Number of chunks currently stored (trivial today, kept for symmetry)."""
        ...

    def delete_document(self, doc_id: str) -> int:
        """Delete every chunk of one document, by its unified ``doc_id``.

        Identity-based (metadata filter), never similarity-based: the
        store is a mirror of the corpus, and this removes one document's
        whole projection. Returns the number of chunks deleted; 0 for an
        unknown document (deleting from nothing is not an error).
        """
        ...
