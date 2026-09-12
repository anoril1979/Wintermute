"""Retrieval layer — the read side of the ingested corpus.

Public API (deterministic pipeline; the classification comes from the
single routing analyzer):

* :class:`RetrievalSpec` / :class:`RetrievalBatch` — the validated
  specification of the lookups, built from the analyzer's requests;
* :class:`RetrievalFilters` / :func:`build_where` — the fine-grained
  metadata filter (ChromaDB where-clause builder);
* facts + :func:`apply_decision_table` — corpus state and the pure
  per-request decision (filters, top-k clamp, dormant gate);
* :func:`run_retrieval` — the orchestrator entry point (config gate,
  facts, per-request graph solving).
"""

from src.retrieval.filters import (
    ALLOWED_FIELDS,
    InvalidFilterError,
    RetrievalFilters,
    build_where,
)
from src.retrieval.models import (
    RetrievalBatch,
    RetrievalSpec,
)
from src.retrieval.retrieval_orchestrator import (
    STATUS_PARTIAL,
    run_retrieval,
)
from src.retrieval.retrieval_router import (
    IMPLEMENTED_KINDS,
    ROUTER_NO_CORPUS,
    ROUTER_PROCEED,
    RetrievalDecision,
    RetrievalFacts,
    apply_decision_table,
    gather_facts,
)

__all__ = [
    "ALLOWED_FIELDS",
    "IMPLEMENTED_KINDS",
    "InvalidFilterError",
    "RetrievalBatch",
    "RetrievalDecision",
    "RetrievalFilters",
    "RetrievalFacts",
    "RetrievalSpec",
    "ROUTER_NO_CORPUS",
    "ROUTER_PROCEED",
    "STATUS_PARTIAL",
    "apply_decision_table",
    "build_where",
    "gather_facts",
    "run_retrieval",
]
