"""Protocols (structural contracts) for the vector indexing layer.

Mirrors ``src/llm/protocols.py``: the contract is the single interface the
rest of the system codes against — the future IndexingAgent, the retrieval
orchestrator, test doubles — never on a concrete client. Concrete
implementations live beside it (``chroma_client.py`` for the store,
``embedding_client.py`` for embeddings) and must satisfy these interfaces.

Scope today: **storage**. Retrieval is stubbed on purpose (``NotImplementedError``
with an actionable message) so the contract is visible and the future
retrieval orchestrator has a shape to grow into — without pretending a
query path exists before it does.

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

    # -- Retrieval stubs (future step — see module docstring) ----------------

    def query(
        self,
        text: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[VectorChunk]:
        """Semantic search (STUB — NotImplementedError until retrieval lands)."""
        ...

    def count(self) -> int:
        """Number of chunks currently stored (trivial today, kept for symmetry)."""
        ...

    def delete_document(self, doc_id: str) -> int:
        """Delete every chunk of one document (STUB — NotImplementedError)."""
        ...
