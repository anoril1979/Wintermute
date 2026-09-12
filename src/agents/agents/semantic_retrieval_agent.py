"""SemanticRetrievalAgent — the vector-search worker of the retrieval graph.

The first retrieval agent, serving ``semantic`` request types: it embeds
the (router-rephrased) question with the SAME model that embedded the
corpus (retrieval.yaml's ``embedding_role``), then asks the vector store
for the nearest chunks — pre-computed vector seam, the store never embeds
— and returns the hits with their scores and full citation metadata.

Config layering (same as the SourceIndexingAgent): the agent never reads
llm.yaml or the store settings itself —

* the embedder client is built from the ``embedding`` role (llm.yaml);
* the store client reads setup.yaml's ``vector_db`` (path, collections);
* the agent's OWN knobs come from retrieval.yaml: ``default_top_k`` /
  ``max_top_k`` (the router applies them), ``min_score`` (applied here),
  ``embedding_role``, ``source_collection_key``.

Failure mapping (agents never raise for expected failures):

* no question in the context          → ``INPUT_DATA`` (orchestrator bug);
* embedding backend failure           → ``EXTERNAL`` (client already retries);
* vector store cannot open / query    → ``EXTERNAL``;
* empty collection                    → ``EXTERNAL`` (the router gates the
  dormant case upstream; an emptied store mid-flight is a real failure).
"""

from __future__ import annotations

import logging
from typing import Optional

from src.agents.contexts import RetrievalContext
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
from src.indexing.chroma_client import (
    ChromaVectorClient,
    VectorStoreError,
)
from src.indexing.embedding_client import (
    EmbeddingClientError,
    OllamaEmbeddingClient,
)
from src.retrieval.filters import RetrievalFilters

logger = logging.getLogger(__name__)

#: Context keys (graph step names).
INPUT_KEY = "question"
OUTPUT_KEY = "hits"

#: Fail-open default for the score threshold when retrieval.yaml cannot be
#: read (must stay in sync with config/retrieval.yaml).
DEFAULT_MIN_SCORE = 0.35


def min_score() -> float:
    """``min_score`` from retrieval.yaml, fail-open to the default.

    The yaml is validated at load time by the config loader, so a present
    value is a number in [0, 1]; this only covers a broken or unreadable
    config — retrieval must keep working.
    """
    try:
        from src.tools.config_loader import load_retrieval_config

        value = float(load_retrieval_config().get("min_score", DEFAULT_MIN_SCORE))
    except Exception as exc:  # noqa: BLE001 — fail-open to the default
        logger.warning(
            "Could not read retrieval.yaml for min_score; using default %.2f: %s",
            DEFAULT_MIN_SCORE, exc,
        )
        return DEFAULT_MIN_SCORE
    return value if 0.0 <= value <= 1.0 else DEFAULT_MIN_SCORE


class SemanticRetrievalAgent:
    """Answers a semantic request: embed → query → scored hits."""

    name = "semantic_retriever"

    def __init__(
        self,
        embedder: Optional[OllamaEmbeddingClient] = None,
        store: Optional[ChromaVectorClient] = None,
        *,
        input_key: str = INPUT_KEY,
        output_key: str = OUTPUT_KEY,
        collection_key: Optional[str] = None,
        embedding_role: Optional[str] = None,
    ) -> None:
        """Args:
        embedder: the embedding client; defaults to the retrieval.yaml
            ``embedding_role`` (injectable for tests).
        store: the vector store client; defaults to the
            ``source_collection_key`` collection (injectable for tests).
        collection_key: override of the retrieval.yaml collection key.
        embedding_role: override of the retrieval.yaml embedding role.
        """
        self._embedder = embedder
        self._store = store
        self._input_key = input_key
        self._output_key = output_key
        self._collection_key = collection_key
        self._embedding_role = embedding_role

    # -- lazily-built clients (config read at build time, not per query) ----

    @staticmethod
    def _retrieval_config() -> dict:
        try:
            from src.tools.config_loader import load_retrieval_config

            return load_retrieval_config()
        except Exception as exc:  # noqa: BLE001 — fail-open: keys stay unset
            logger.warning(
                "Could not read retrieval.yaml; using built-in defaults: %s", exc
            )
            return {}

    @property
    def embedder(self) -> OllamaEmbeddingClient:
        """Lazily-built default embedder (retrieval.yaml's embedding_role).

        Config layering: model/options live in llm.yaml and are read by
        ``OllamaEmbeddingClient.for_role`` when the client is built.
        """
        if self._embedder is None:
            role = self._embedding_role or (
                self._retrieval_config().get("embedding_role") or "embedding"
            )
            self._embedder = OllamaEmbeddingClient.for_role(role)
        return self._embedder

    @property
    def store(self) -> ChromaVectorClient:
        """Lazily-built default store (retrieval.yaml's collection key).

        Config layering: path/collections live in setup.yaml and are read
        by ``ChromaVectorClient`` when the client is built.
        """
        if self._store is None:
            key = self._collection_key or (
                self._retrieval_config().get("source_collection_key")
                or "source_chunks"
            )
            self._store = ChromaVectorClient(key)
        return self._store

    # -- IngestionAgent-style contract ----------------------------------------

    def run(self, context: RetrievalContext) -> AgentResult:
        """Embed the question and fetch the nearest chunks."""
        question = (context.question or "").strip()
        if not question:
            detail = f"no question in context ({self._input_key!r} empty)"
            context.emit("task", "retrieval_failed", detail)
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail=detail,
            )

        filters = context.metadata.get("filters")
        top_k = int(context.metadata.get("top_k", 6))

        # -- embedding (one text: the question) --------------------------------
        try:
            vector = self.embedder.embed([question])[0]
        except EmbeddingClientError as exc:
            detail = f"embedding backend failure: {exc}"
            context.emit("task", "retrieval_failed", detail)
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.EXTERNAL,
                detail=detail,
            )
        except Exception as exc:  # noqa: BLE001 — agents never raise
            detail = f"unexpected embedding failure: {exc}"
            context.emit("task", "retrieval_failed", detail)
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.EXTERNAL,
                detail=detail,
            )

        where = filters.build_where() if isinstance(filters, RetrievalFilters) else None
        context.emit(
            "task", "retrieval_query",
            f"searching {top_k} chunk(s)"
            + (f" with filters {filters.summary()}" if where else ""),
            top_k=top_k,
            filters=filters.summary() if isinstance(filters, RetrievalFilters) else {},
        )

        # -- vector query --------------------------------------------------------
        try:
            hits = self.store.query_by_vector(vector, top_k=top_k, where=where)
        except VectorStoreError as exc:
            detail = f"vector store failure: {exc}"
            context.emit("task", "retrieval_failed", detail)
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.EXTERNAL,
                detail=detail,
            )
        except Exception as exc:  # noqa: BLE001 — agents never raise
            detail = f"unexpected vector store failure: {exc}"
            context.emit("task", "retrieval_failed", detail)
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.EXTERNAL,
                detail=detail,
            )

        threshold = min_score()
        kept = [hit for hit in hits if hit.score is not None and hit.score >= threshold]
        dropped = len(hits) - len(kept)
        if dropped:
            context.metadata["low_score_dropped"] = dropped

        context.metadata["retrieved_count"] = len(hits)
        context.metadata["kept_count"] = len(kept)
        context.outputs[self._output_key] = kept
        context.emit(
            "task", "retrieval_done",
            f"{len(kept)} hit(s) above {threshold:.2f}"
            + (f" ({dropped} dropped below threshold)" if dropped else ""),
            retrieved=len(hits), kept=len(kept),
        )
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.OK,
            detail=f"{len(kept)} chunk(s) retrieved",
            payload={"hits": kept},
        )

    def validate(self, context: RetrievalContext) -> Optional[AgentResult]:
        """Post-run check: the hits must be score-ordered (best first)."""
        hits = context.outputs.get(self._output_key)
        if not isinstance(hits, list) or len(hits) < 2:
            return None
        scores = [h.score for h in hits if h.score is not None]
        if scores != sorted(scores, reverse=True):
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.EXTERNAL,
                detail="vector store returned hits out of score order",
            )
        return None
