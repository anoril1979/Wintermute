"""Vector indexing layer: embedding clients, vector stores, chunk builders.

Public API (client layer only for now — the future IndexingAgent will be
the first consumer, then the retrieval orchestrator):
"""

from src.indexing.chunks import (
    LEVEL_BLOCK,
    LEVEL_CHAPTER,
    LEVEL_DOCUMENT,
    LEVEL_PAGE,
    LEVEL_SECTION,
    KIND_CONTENT,
    KIND_SUMMARY,
    VectorChunk,
    build_knowledge_chunks,
    build_source_chunks,
    doc_id_from_source_path,
)
from src.indexing.chroma_client import (
    ChromaVectorClient,
    DimensionMismatchError,
    VectorStoreError,
    VectorStoreUnavailableError,
)
from src.indexing.embedding_client import (
    EmbeddingClientError,
    EmbeddingRequestError,
    OllamaEmbeddingClient,
    get_embedding_client,
)
from src.indexing.protocols import (
    EmbeddingClientProtocol,
    VectorStoreProtocol,
)

__all__ = [
    # chunks
    "VectorChunk",
    "build_source_chunks",
    "build_knowledge_chunks",
    "doc_id_from_source_path",
    "LEVEL_BLOCK",
    "LEVEL_SECTION",
    "LEVEL_PAGE",
    "LEVEL_CHAPTER",
    "LEVEL_DOCUMENT",
    "KIND_CONTENT",
    "KIND_SUMMARY",
    # chroma client
    "ChromaVectorClient",
    "VectorStoreError",
    "DimensionMismatchError",
    "VectorStoreUnavailableError",
    # embedding client
    "OllamaEmbeddingClient",
    "EmbeddingClientError",
    "EmbeddingRequestError",
    "get_embedding_client",
    # protocols
    "EmbeddingClientProtocol",
    "VectorStoreProtocol",
]
