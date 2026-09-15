"""Retrieval decision layer — facts and the decision table, no LLM.

Since the routing rework, the retrieval pipeline is deterministic Python —
the strict mirror of the ingestion tunnel:

* **Python owns the facts** — the stores' state (vector chunk count,
  knowledge-base entity count) is read here, never guessed;
* **the analyzer owns the intent** — the request_analyzer LLM already
  classified the lookup (semantic / lookup / relationship) and produced a
  self-contained query (``src/routing/models.RetrievalRequest``);
* **Python owns the decision table** — filter assembly, top-k clamping
  against retrieval.yaml, store-availability gates.

Request types the pipeline cannot serve yet (``relationship`` — the SQL
claims layer) are carried faithfully — the orchestrator reports them
``not_implemented`` — instead of being silently misread as semantic
searches. There is no keyword fallback anymore: with a single analysis
there is no second reading of the user's words, and a failed analysis is
the routing layer's failure, not a retrieval one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.retrieval.filters import RetrievalFilters
from src.retrieval.models import RetrievalSpec
from src.routing.models import RetrievalLookupKind
from src.tools.config_loader import load_retrieval_config

logger = logging.getLogger(__name__)

# Statuses the decision table can return (stable strings for callers/traces).
ROUTER_PROCEED = "proceed"          # run the retrieval graph
ROUTER_NO_CORPUS = "no_corpus"      # nothing indexed yet: memory is dormant

#: Which request types the pipeline can actually serve today. Anything
#: else stays carried (validated) but is reported not-implemented downstream.
#: ``semantic`` reads the vector store; ``lookup`` reads the markdown
#: knowledge base (its emptiness is answered deterministically by the
#: agent — "no such entity" — not gated as a failure).
IMPLEMENTED_KINDS = frozenset({"semantic", "lookup"})


# ---------------------------------------------------------------------------
# Facts — everything Python knows, gathered before any decision
# ---------------------------------------------------------------------------

@dataclass
class RetrievalFacts:
    """Store state for one retrieval request, gathered by Python."""

    store_exists: bool = False     # a ChromaDB store was materialized
    chunk_count: int = 0           # chunks in the source collection
    knowledge_entities: int = 0    # entities listed in the knowledge base

    @property
    def has_corpus(self) -> bool:
        return self.chunk_count > 0

    @property
    def has_knowledge(self) -> bool:
        return self.knowledge_entities > 0

    def summary(self) -> Dict[str, Any]:
        """Compact dict for payloads and traces."""
        return {
            "store_exists": self.store_exists,
            "chunk_count": self.chunk_count,
            "has_corpus": self.has_corpus,
            "knowledge_entities": self.knowledge_entities,
        }


@dataclass
class RetrievalDecision:
    """Outcome of the retrieval decision table for one lookup."""

    status: str                                  # proceed / no_corpus
    spec: RetrievalSpec
    filters: Optional[RetrievalFilters] = None   # assembled by Python
    top_k: int = 5                               # clamped to retrieval.yaml
    implemented: bool = True                     # kind servable today?
    explanation: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "spec": self.spec.summary(),
            "filters": self.filters.summary() if self.filters else {},
            "top_k": self.top_k,
            "implemented": self.implemented,
            "explanation": self.explanation,
        }


def apply_decision_table(
    facts: RetrievalFacts, spec: RetrievalSpec, *, config: Optional[dict] = None
) -> RetrievalDecision:
    """Turn (facts, spec) into filters, top-k and a proceed/gate decision.

    Pure and deterministic:
    * empty corpus → ``no_corpus`` for the kinds that read it (the graph
      answers "memory dormant" — no pointless embedding round-trip);
      ``lookup`` is exempt: it reads the knowledge base, whose emptiness
      the agent answers deterministically ("no such entity");
    * filters assembled from the spec's scopes (Python's job);
    * ``top_k`` clamped to ``max_top_k`` (retrieval.yaml) — a per-request
      value may lower the default, never raise the ceiling.
    """
    try:
        config = config or load_retrieval_config()
        default_top_k = int(config.get("default_top_k", 6))
        max_top_k = int(config.get("max_top_k", 20))
    except Exception:  # noqa: BLE001 — config problems surface at the orchestrator gate
        default_top_k, max_top_k = 6, 20

    filters = RetrievalFilters(
        document=spec.document,
        chapter_title=spec.chapter_title,
    )

    top_k = spec.top_k or default_top_k
    if top_k > max_top_k:
        top_k = max_top_k

    if not facts.has_corpus and spec.kind != RetrievalLookupKind.LOOKUP:
        return RetrievalDecision(
            status=ROUTER_NO_CORPUS,
            spec=spec,
            filters=filters,
            top_k=top_k,
            implemented=spec.kind.value in IMPLEMENTED_KINDS,
            explanation=(
                "the vector store holds no indexed document yet — "
                "ingest a document first"
            ),
        )

    return RetrievalDecision(
        status=ROUTER_PROCEED,
        spec=spec,
        filters=filters,
        top_k=top_k,
        implemented=spec.kind.value in IMPLEMENTED_KINDS,
        explanation=f"lookup={spec.kind.value}, question='{spec.question}'",
    )


def gather_facts(collection_key: str = "source_chunks") -> RetrievalFacts:
    """Read the stores' state (read-only, fail-open).

    ``count()`` is read-only by contract (it never materializes a store),
    so probing before any ingestion is safe; the knowledge-base probe
    reads the sidecar index only (one small file). A broken store reports
    zeros — the retrieval graph answers "memory dormant" / "no such
    entity" instead of crashing.
    """
    facts = RetrievalFacts()
    try:
        from src.indexing.chroma_client import ChromaVectorClient

        store = ChromaVectorClient(collection_key)
        facts.store_exists = store._store_exists()
        facts.chunk_count = int(store.count())
    except Exception as exc:  # noqa: BLE001 — fail-open: the fact is "no corpus"
        logger.warning("Retrieval facts gathering failed: %s", exc)
    try:
        from src.knowledge.character_markdown_store import (
            index_path_for,
            load_index,
        )

        facts.knowledge_entities = len(load_index(index_path_for()))
    except Exception as exc:  # noqa: BLE001 — fail-open: the fact is "no knowledge"
        logger.warning("Knowledge-base facts gathering failed: %s", exc)
    return facts
