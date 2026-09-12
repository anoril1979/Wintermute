"""Retrieval request specification — deterministic, no LLM parsing.

Since the routing rework, the retrieval pipeline reads the analyzer's
classification directly (``src/routing/models.RetrievalRequest``: lookup
kind, self-contained question, scopes) and never calls an LLM itself.
This module defines:

* :class:`RetrievalSpec` — the validated specification of ONE lookup,
  built from the analyzer's request (:meth:`RetrievalSpec.from_request`);
* :class:`RetrievalBatch` — the specs of one user prompt, in dispatch
  order.

The old LLM-answer parsers (``parse_intent``/``parse_intents``) are gone:
there is no retrieval-side LLM anymore — the request types map onto the
storage layers, and the ones that do not exist yet are carried faithfully
so the orchestrator reports them ``not_implemented`` instead of misreading
them as semantic queries.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.routing.models import RetrievalLookupKind, RetrievalRequest


class _StrictModel(BaseModel):
    """Base model: reject unknown keys so wiring bugs surface as errors."""

    model_config = ConfigDict(extra="forbid")


class RetrievalSpec(_StrictModel):
    """The deterministic specification of one lookup request.

    Attributes:
        kind:       the lookup type (see :class:`RetrievalLookupKind`).
        question:   self-contained search query, in the user's language —
                    what is embedded and matched against the corpus. The
                    analyzer produced it (pronouns already resolved); no
                    downstream LLM rephrases it.
        document:   optional bare document name the question is scoped to.
        chapter_title: optional chapter title the question is scoped to.
        top_k:      per-request result count; None = config default. The
                    decision table clamps it to ``max_top_k``.
        reason:     short explanation carried from the analysis (traces).
    """

    kind: RetrievalLookupKind = RetrievalLookupKind.SEMANTIC
    question: str = Field(min_length=1)
    document: Optional[str] = None
    chapter_title: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    reason: str = ""

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()

    @classmethod
    def from_request(cls, request: RetrievalRequest) -> "RetrievalSpec":
        """Build the spec from the analyzer's validated request."""
        return cls(
            kind=request.lookup_kind,
            question=request.question,
            document=request.document,
            chapter_title=request.chapter_title,
            top_k=request.top_k,
            reason=request.reason,
        )

    # -- convenience ---------------------------------------------------------

    def summary(self) -> dict:
        """Compact dict for payloads and traces."""
        return {
            "kind": self.kind.value,
            "question": self.question,
            "document": self.document,
            "chapter_title": self.chapter_title,
            "top_k": self.top_k,
            "reason": self.reason,
        }


class RetrievalBatch(_StrictModel):
    """The lookup requests of one user prompt, in dispatch order."""

    specs: List[RetrievalSpec] = Field(default_factory=list)

    @classmethod
    def from_requests(cls, requests: List[RetrievalRequest]) -> "RetrievalBatch":
        """Build the batch from the analyzer's retrieval requests."""
        return cls(specs=[RetrievalSpec.from_request(r) for r in requests])
