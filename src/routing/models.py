"""Structured requests extracted from a user prompt.

The request analyzer (``request_analyzer`` role, prompts/routing/
request_analysis.md) reads the raw user prompt and explodes it into a
list of distinct requests. This module defines the pydantic models that
**validate** the analyzer's answer before anything is dispatched:

* :class:`UserRequest` — one structured request (kind, utterance, and the
  kind-specific payload: ``document`` for ingestion, ``question`` for
  retrieval);
* :class:`AnalysisResult` — the whole analyzer answer (``requests`` list).

Validators enforce cross-field consistency — with one deliberate exception
(per-request degradation): an ingestion request **without a document is
valid but incomplete**. The analyzer is instructed to never invent a file
name, so "ingest some documents" legitimately yields
``kind=ingestion, document=None``; rejecting it would destroy the whole
batch (``AnalysisResult`` validates every request at once) instead of just
that request. The routing graph reports it as an ``incomplete`` outcome and
the user is asked to name the file. A retrieval request without a question
stays invalid — its utterance alone is never answerable.

**Prompt-local memory**: the requests of one user prompt are not isolated
islands — "ingest meow.pdf, then summarize it, then is it indexed?" has
three requests whose pronouns all point at the same file. The routing
graph attaches, to every dispatched request, the entries of the requests
that *preceded it in the same prompt* (kind, utterance, dispatch status):
:class:`UserRequest.preceding`. The task-agent LLMs read that block to
resolve pronouns and understand the local context; the analyzer itself
never emits it — the field is dispatcher-owned, and an analyzer that
nonetheless emits one has its value silently dropped (a hard reject would
re-create the batch-fatal failure mode described above).

From LLM text to pydantic: :func:`parse_analysis` takes the raw string the
model returned, extracts the JSON payload (models love to wrap it in
markdown fences or prose despite the instructions), and hands the decoded
structure to pydantic so validation errors surface as data, not crashes —
the orchestrator turns them into a retryable LLM failure.
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class RequestKind(str, Enum):
    """What a user request is asking the system to do."""

    INGESTION = "ingestion"
    RETRIEVAL = "retrieval"
    GENERAL = "general"


class _StrictModel(BaseModel):
    """Base model: reject unknown keys so LLM typos surface as errors."""

    model_config = ConfigDict(extra="forbid")


class RequestContextEntry(_StrictModel):
    """One earlier request of the same prompt, as later requests see it.

    Built by the routing graph, not by the analyzer. ``status``/``detail``
    describe the *dispatch outcome* of that earlier request (``None``/
    ``""`` when the entry is being described before its own dispatch —
    which only happens in tests; in the graph, dispatch is sequential so
    every preceding entry has already run).
    """

    kind: str
    utterance: str
    document: Optional[str] = None
    question: Optional[str] = None
    status: Optional[str] = None
    detail: str = ""


class UserRequest(_StrictModel):
    """One structured request extracted from the user prompt.

    Attributes:
        kind:      what to do (ingestion / retrieval / general).
        utterance: the exact user text this request comes from (kept for
                   traceability and for agents that re-read the wording).
        document:  bare file name, ingestion requests only.
        question:  self-contained question, retrieval requests only.
        options:   request modifiers (``force_reingest``, ``section_scope``).
        preceding: prompt-local memory — the requests of the same prompt
                   that were dispatched before this one (dispatcher-built;
                   the analyzer must never emit this field).
    """

    kind: RequestKind
    utterance: str = Field(min_length=1)
    document: Optional[str] = None
    question: Optional[str] = None
    options: dict = Field(default_factory=dict)
    preceding: List[RequestContextEntry] = Field(default_factory=list)
    # NOTE: 'preceding' is dispatcher-owned — the routing graph fills it
    # from the actual dispatch history. The analyzer must never emit it;
    # parse_analysis() strips (and warns about) any hallucinated value at
    # the LLM-answer boundary, so in-code construction is unaffected.

    @field_validator("utterance")
    @classmethod
    def _utterance_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("utterance must not be blank")
        return value

    @field_validator("document")
    @classmethod
    def _document_is_bare_name(cls, value: Optional[str]) -> Optional[str]:
        """The analyzer must strip paths/quotes; anything path-like is a bug."""
        if value is None:
            return None
        if "/" in value or "\\" in value or ".." in value or ":" in value:
            raise ValueError(
                f"document must be a bare file name, got path-like {value!r}"
            )
        return value

    @model_validator(mode="after")
    def _kind_consistency(self) -> "UserRequest":
        """Cross-field consistency between the kind and its payload.

        An ingestion request without a ``document`` is deliberately valid
        (the dispatcher degrades it to an ``incomplete`` outcome and the
        user is asked to name the file): the analyzer must not invent file
        names, so "ingest some documents" cannot be honored any other way.
        """
        if self.kind is RequestKind.RETRIEVAL and not (self.question or "").strip():
            raise ValueError(
                "retrieval request must carry a question "
                f"(utterance: {self.utterance!r})"
            )
        if self.kind is RequestKind.GENERAL:
            if self.document is not None:
                raise ValueError("general request must not carry a document")
            if self.question is not None:
                raise ValueError("general request must not carry a question")
        return self

    # -- convenience ---------------------------------------------------------

    def option(self, name: str, default: object = None) -> object:
        """Read an option (``force_reingest``, ``section_scope``, ...)."""
        return self.options.get(name, default)

    def summary(self) -> dict:
        """Compact dict for result payloads and logs."""
        return {
            "kind": self.kind.value,
            "utterance": self.utterance,
            "document": self.document,
            "question": self.question,
            "options": dict(self.options),
            "preceding": [p.model_dump() for p in self.preceding],
        }


class AnalysisResult(_StrictModel):
    """The whole analyzer answer: the ordered list of user requests."""

    requests: List[UserRequest] = Field(default_factory=list)


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


def parse_analysis(raw: str) -> AnalysisResult:
    """Parse and validate the analyzer's raw answer into an AnalysisResult.

    Raises:
        ValueError: the answer contains no JSON object at all, or the JSON
            is syntactically broken. Structural (pydantic) validation
            errors raise pydantic.ValidationError (a ValueError subclass),
            which callers treat the same way: a malformed LLM response,
            typically retryable.
    """
    if not raw or not raw.strip():
        raise ValueError("empty analyzer response")

    payload = _extract_json_payload(raw)
    if not isinstance(payload, dict):
        raise ValueError("analyzer response is not a JSON object")

    payload = _strip_dispatcher_owned_fields(payload)
    return AnalysisResult.model_validate(payload)


def _strip_dispatcher_owned_fields(payload: dict) -> dict:
    """Drop fields the analyzer must never emit (``preceding``).

    The analyzer's contract is to *split* the prompt only: context between
    the requests of one prompt is the routing graph's job (it overwrites
    ``preceding`` from the actual dispatch history anyway). A hallucinated
    value is dropped here — at the LLM-answer boundary — with a warning,
    instead of either reaching the strict pydantic model (``extra=forbid``
    would reject the whole batch) or poisoning the prompt-local memory.
    Only in-code construction of ``UserRequest`` may set ``preceding``.
    """
    requests = payload.get("requests")
    if not isinstance(requests, list):
        return payload
    for index, item in enumerate(requests):
        if isinstance(item, dict) and "preceding" in item:
            logger.warning(
                "analyzer emitted 'preceding' for request %d — field is "
                "dispatcher-owned; value dropped: %r",
                index, item["preceding"],
            )
            requests[index] = {
                key: value for key, value in item.items() if key != "preceding"
            }
    return payload
