"""ChromaDB-backed implementation of the vector store protocol.

``ChromaVectorClient`` wraps an **embedded** ChromaDB ``PersistentClient``:
the whole store is a plain folder (SQLite metadata + HNSW segments), no
server process — exactly what ``setup.yaml``'s ``vector_db.path`` points
at. One client instance per collection.

Design points:

* **The store never embeds.** ``upsert`` receives pre-computed vectors —
  embedding belongs to the embedding client, storage to this class. The
  protocol keeps the two decoupled, so a future swap of either never
  touches the other.
* **Deterministic ids** (``src/indexing/chunks.py``) make ``upsert``
  idempotent: re-indexing a document *updates* its chunks, never
  duplicates. ``get_or_create_collection`` is used for the same reason —
  the store is a mirror of the corpus, rebuilt by upserts, not an
  append-only ledger.
* **Cosine space** — the right metric for these embedding models (Chroma's
  default is L2, wrong for text similarity here).
* **Dimension guard**: Chroma rejects a write with the wrong dimension,
  but only with a cryptic generic message; the client raises its own
  explicit ``DimensionMismatchError`` naming the mismatch *before* the
  write.
* **Telemetry off** — the embedded client's posthog/analytics is noise in
  an offline, local-only system.

Retrieval is split by seam: ``query_by_vector`` is real (the retrieval
layer embeds the question and hands the vector over — the store never
embeds); ``query`` by text and ``delete_document`` remain honest stubs:
text queries would couple the store to an embedding backend, and
per-document deletion lands with index maintenance.
``count`` is real and read-only: it never creates the collection.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional, Sequence

from src.indexing.chunks import VectorChunk
from src.indexing.protocols import VectorStoreProtocol
from src.tools import config_loader
from src.tools.config_loader import VectorConfigError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class VectorStoreError(Exception):
    """Base class for vector store failures."""


class DimensionMismatchError(VectorStoreError):
    """A write's vector dimension differs from the collection's first write."""


class VectorStoreUnavailableError(VectorStoreError):
    """The store could not be opened (folder not usable, corrupted DB...)."""


# ---------------------------------------------------------------------------
# Config access
# ---------------------------------------------------------------------------

def vector_db_config() -> dict:
    """The validated ``vector_db`` section of setup.yaml (cached loader)."""
    return config_loader.load_vector_config()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class ChromaVectorClient:
    """Vector store client over embedded ChromaDB, one instance per collection.

    Implements :class:`src.indexing.protocols.VectorStoreProtocol` (upsert
    and query_by_vector now; text query / delete_document are stubs, count
    is real).
    """

    def __init__(
        self,
        collection_key: str,
        embedding_dimension: Optional[int] = None,
        path: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """Create a client for one collection of the configured vector DB.

        Args:
            collection_key: the setup.yaml key under ``vector_db.collections``
                (``source_chunks`` or ``knowledge_chunks``).
            embedding_dimension: expected vector dimension. ``None`` (the
                default) means "accept whatever the first write carries" —
                pass the embedding model's dimension to get the early,
                explicit ``DimensionMismatchError`` instead of Chroma's
                cryptic one at write time.
            path: store folder override (tests); the setup.yaml value otherwise.
            collection_name: collection name override (tests); the setup.yaml
                mapping otherwise.

        Raises:
            VectorConfigError: malformed ``vector_db`` section or unknown
                ``collection_key``.
        """
        self._config = vector_db_config()
        collections: dict = self._config.get("collections", {})
        if collection_key not in collections:
            raise VectorConfigError(
                f"Unknown vector_db collection key '{collection_key}' "
                f"(known keys: {', '.join(sorted(collections))})."
            )
        self.collection_key = collection_key
        self.embedding_dimension = embedding_dimension
        self._path = Path(path) if path else Path(self._config["path"])
        self._collection_name = collection_name or collections[collection_key]
        self._client = None      # embedded ChromaDB client, built lazily
        self._collection = None  # the actual collection handle
        self._lock = threading.Lock()  # lazy init must not race

    # -- Lifecycle -------------------------------------------------------------

    def _ensure_collection(self):
        """Lazily open the store and get/create the collection (thread-safe).

        The whole init is lazy on purpose: importing this module (or building
        a client) must never create ``data/vector`` — only the first real
        storage operation does.
        """
        if self._collection is not None:
            return self._collection

        with self._lock:
            if self._collection is not None:
                return self._collection
            try:
                # Imported lazily: heavy import kept out of module load.
                import chromadb
                from chromadb.config import Settings

                self._client = chromadb.PersistentClient(
                    path=str(self._path),
                    settings=Settings(anonymized_telemetry=False, allow_reset=False),
                )
                self._collection = self._client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=None,
                )
            except Exception as exc:
                raise VectorStoreUnavailableError(
                    f"Cannot open the vector store at '{self._path}' "
                    f"(collection '{self._collection_name}'): {exc}"
                ) from exc
        return self._collection

    # -- Public API (VectorStoreProtocol) --------------------------------------

    def upsert(self, chunks: Sequence[VectorChunk]) -> int:
        """Insert or update chunks — idempotent thanks to deterministic ids.

        Args:
            chunks: chunks with ``vector`` already filled by the caller.

        Returns:
            The number of chunks written.

        Raises:
            ValueError: empty batch, a chunk without a vector, or a
                non-homogeneous batch (mixed dimensions).
            DimensionMismatchError: a vector's dimension differs from the
                client's ``embedding_dimension``.
            VectorStoreUnavailableError: the store could not be opened.
            VectorStoreError: ChromaDB refused the write.
        """
        if not chunks:
            raise ValueError("upsert() requires a non-empty batch of chunks.")
        for index, chunk in enumerate(chunks):
            if chunk.vector is None:
                raise ValueError(
                    f"Chunk '{chunk.id}' (index {index}) has no vector: "
                    "embed the batch before storing it."
                )

        dimension = len(chunks[0].vector)
        for index, chunk in enumerate(chunks):
            if len(chunk.vector) != dimension:
                raise ValueError(
                    f"Chunk '{chunk.id}' (index {index}) has dimension "
                    f"{len(chunk.vector)}, but the batch's dimension is "
                    f"{dimension} — a batch must be homogeneous (one model)."
                )
        if self.embedding_dimension is not None and dimension != self.embedding_dimension:
            raise DimensionMismatchError(
                f"Chunk batch dimension {dimension} differs from the "
                f"client's embedding_dimension {self.embedding_dimension} "
                f"(collection '{self._collection_name}'). If you switched "
                "embedding model, the collection must be re-embedded."
            )

        collection = self._ensure_collection()

        try:
            collection.upsert(
                ids=[c.id for c in chunks],
                embeddings=[[float(v) for v in c.vector] for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[dict(c.metadata) for c in chunks],
            )
        except Exception as exc:
            raise VectorStoreError(
                f"ChromaDB upsert failed for {len(chunks)} chunk(s) into "
                f"'{self._collection_name}': {exc}"
            ) from exc

        logger.info(
            "Upserted %d chunk(s) into collection '%s' (dimension %d)",
            len(chunks), self._collection_name, dimension,
        )
        return len(chunks)

    # -- Retrieval ----------------------------------------------------------------

    def query_by_vector(
        self,
        vector: Sequence[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[VectorChunk]:
        """Nearest-neighbor search with a pre-computed query vector.

        Args mirror the protocol (see ``src/indexing/protocols.py`` for the
        full contract). The store NEVER embeds: the caller computes the
        query vector (same model as the collection's chunks).
        """
        if not vector:
            raise ValueError("query_by_vector() requires a non-empty vector.")
        if int(top_k) <= 0:
            raise ValueError(f"top_k must be a positive int, got {top_k!r}.")

        collection = self._ensure_collection()
        try:
            result = collection.query(
                query_embeddings=[[float(v) for v in vector]],
                n_results=int(top_k),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(
                f"ChromaDB query failed on '{self._collection_name}': {exc}"
            ) from exc

        ids: list = result.get("ids", [[]])[0]
        documents: list = result.get("documents", [[]])[0]
        metadatas: list = result.get("metadatas", [[]])[0]
        distances: list = result.get("distances", [[]])[0]

        chunks: list[VectorChunk] = []
        for chunk_id, chunk_text, metadata, distance in zip(ids, documents, metadatas, distances):
            chunks.append(
                VectorChunk(
                    id=str(chunk_id),
                    text=str(chunk_text or ""),
                    metadata=dict(metadata or {}),
                    # Cosine distance -> similarity: 1 - distance (higher
                    # is better), clamped to [0, 1].
                    score=max(0.0, min(1.0, 1.0 - float(distance or 0.0))),
                )
            )
        logger.info(
            "Queried '%s': %d hit(s) (top_k=%d, where=%s)",
            self._collection_name, len(chunks), top_k, where,
        )
        return chunks

    def query(self, text: str, top_k: int = 5, where: Optional[dict] = None) -> list[VectorChunk]:
        """Semantic search by text — STUB by design.

        A text query would couple the store to an embedding backend (the
        store must never embed). Embed the question and call
        :meth:`query_by_vector` instead.
        """
        raise NotImplementedError(
            "Text queries are not supported on the store: embed the question "
            "(EmbeddingClientProtocol) and call query_by_vector() instead."
        )

    def count(self) -> int:
        """Number of chunks currently stored; 0 when the collection doesn't exist.

        Read-only by contract: unlike ``upsert``, this never creates the
        collection — counting a store that was never written must not
        materialize an empty one.
        """
        if not self._store_exists():
            return 0
        collection = self._ensure_collection()
        try:
            return int(collection.count())
        except Exception as exc:
            raise VectorStoreError(
                f"ChromaDB count failed on '{self._collection_name}': {exc}"
            ) from exc

    def delete_document(self, doc_id: str) -> int:
        """Delete every chunk of one document — STUB until retrieval lands."""
        raise NotImplementedError(
            "Per-document deletion is not implemented yet: it lands with the "
            "retrieval / index-maintenance step."
        )

    # -- Internals ---------------------------------------------------------------

    def _store_exists(self) -> bool:
        """True when a ChromaDB store was already materialized at ``self._path``.

        Probes the filesystem (ChromaDB's canonical ``chroma.sqlite3``
        marker) instead of opening a client: merely *opening* a
        PersistentClient creates the sqlite file, and count() must stay
        read-only — probing a store that was never written must not
        materialize one.
        """
        return (self._path / "chroma.sqlite3").is_file()
