"""Structured ingestion intent, validated from the router LLM's answer.

The ingestion router (``ingestion_router`` role) reads the user's request
plus the **facts** the orchestrator gathered from its stores (extraction
job, canonical JSON, summarization job) and returns one JSON object.
:class:`IngestionIntent` is the validated shape of that object.

The LLM's job is *intent identification*: classifying the utterance
(first-time ingestion / re-ingest with force / re-ingest with summaries
redone) and, when asked, writing the clarification wording. It NEVER
decides filesystem state — the decision table in ``ingestion_router.py``
(plain Python) applies the facts and turns the intent into the
orchestrator's flags.
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class ClarificationKind(str, Enum):
    """What kind of clarification the router needs from the user."""

    REQUEST_UNCLEAR = "request_unclear"      # not recognizable as an ingestion order
    DOCUMENT_NOT_FOUND = "document_not_found"
    STATE_CONFLICT = "state_conflict"        # stores disagree with the request


class _StrictModel(BaseModel):
    """Base model: reject unknown keys so LLM typos surface as errors."""

    model_config = ConfigDict(extra="forbid")


class IngestionIntent(_StrictModel):
    """The router LLM's validated answer for one ingestion request.

    Attributes:
        valid:      False when the utterance is not an ingestion order.
        document:   bare file name (the LLM only strips quotes/paths; the
                    existence check is a Python fact, never an LLM guess).
        force:      the user asks to redo the document content
                    ("re-extract", "reload", "the file changed"...).
        redo_summaries: the user explicitly asks to redo the summaries.
        clarification:  kind of clarification needed when ``valid`` is False.
        question:   free-text question for the user (LLM-written when asked;
                    a deterministic fallback exists for the no-LLM path).
        reason:     short router explanation (payload/trace only).
    """

    valid: bool = False
    document: Optional[str] = None
    force: bool = False
    redo_summaries: bool = False
    clarification: Optional[ClarificationKind] = None
    question: Optional[str] = None
    reason: str = ""

    @field_validator("document")
    @classmethod
    def _document_is_bare_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if "/" in value or "\\" in value or ".." in value or ":" in value:
            raise ValueError(f"document must be a bare file name, got {value!r}")
        return value

    @model_validator(mode="after")
    def _consistency(self) -> "IngestionIntent":
        """Cross-field rules the decision table can rely on."""
        if self.valid and not (self.document or "").strip():
            raise ValueError("a valid ingestion intent must name a document")
        if not self.valid and not self.clarification:
            # An invalid intent must say why it needs the user back.
            raise ValueError("an invalid intent must carry a clarification kind")
        return self

    # -- convenience ---------------------------------------------------------

    def summary(self) -> dict:
        """Compact dict for payloads and traces."""
        return {
            "valid": self.valid,
            "document": self.document,
            "force": self.force,
            "redo_summaries": self.redo_summaries,
            "clarification": self.clarification.value if self.clarification else None,
            "question": self.question,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# LLM text -> validated model
# ---------------------------------------------------------------------------

# A ```json ... ``` (or bare ```) fence around the payload, or any curly
# brace block, in case the model ignores the "no prose" instruction.
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_payload(text: str) -> object:
    """Pull the first JSON object out of an LLM answer (fenced or not)."""
    fenced = _FENCE_RE.search(text)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        braces = _OBJECT_RE.search(text)
        candidate = braces.group(0) if braces else text
    return json.loads(candidate)


def parse_intent(raw: str) -> IngestionIntent:
    """Parse and validate the router's raw answer into an IngestionIntent.

    Raises:
        ValueError: the answer contains no JSON object, is syntactically
            broken, or violates the model's rules (pydantic
            ValidationError is a ValueError subclass) — callers treat all
            of these as a retryable malformed-LLM-response.
    """
    if not raw or not raw.strip():
        raise ValueError("empty router response")

    payload = _extract_json_payload(raw)
    if not isinstance(payload, dict):
        raise ValueError("router response is not a JSON object")

    return IngestionIntent.model_validate(payload)
