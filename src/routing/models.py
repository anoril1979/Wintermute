"""Structured requests extracted from a user prompt, grouped by scope.

The request analyzer (``request_analyzer`` role, prompts/routing/
request_analysis.md) reads the raw user prompt ONCE and extracts **every**
request it holds, grouped by scope. The output shape is::

    {
      "retrieval": [
        {"lookup_kind": "semantic|index|relation|summary|listing",
         "question": "...", "document": null, "chapter_title": null,
         "top_k": null}
      ],
      "general": [
        {"question": "..."}
      ]
    }

**No ingestion scope.** Since the paradigm change, document ingestion is a
CLI operation (scripts/ingest.py) and is deliberately unreachable from
the chat: no LLM analysis, no routing, no task agent ever ingests. When a
user asks Wintermute to ingest in conversation, the analyzer emits a
``general`` request — the general agent explains how ingestion actually
works. This keeps the router deterministic about storage: it can never
again hallucinate a file name into a phantom ingestion order (the
vif-argent incident) because the ingestion shape does not exist.

**One analysis, no re-routing.** The analyzer has the full prompt context:
pronouns are resolved HERE — a retrieval ``question`` must be
self-contained ("tell me more about the King of the North", not "tell me
more about him"). Nothing downstream re-reads the user's words: the
retrieval pipeline is deterministic Python (decision table → graph); the
only LLM-based post-routing worker is the GeneralTaskAgent (and later the
AnswerAgent).

The dispatcher (routing graph) runs the flattened requests in **grouped
scope order** — all retrievals, then all generals.
:meth:`AnalysisResult.flattened` implements that order.

**Prompt-local memory** is dispatcher-owned (``preceding``): the routing
graph attaches, to every dispatched request, the entries of the requests
that preceded it (kind, utterance, outcome) so an LLM-based agent
(GeneralTaskAgent, later AnswerAgent) can compose its reply against what
actually happened. The analyzer must never emit it — hallucinated values
are stripped at the parse boundary.

From LLM text to pydantic: :func:`parse_analysis` takes the raw string,
extracts the JSON payload, validates the whole grouped structure and
enforces the :data:`max_requests_per_prompt` cap from setup.yaml (a
confused analysis looping on the split is a malformed answer, not an
order).
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.routing.language import DEFAULT_LANGUAGE

logger = logging.getLogger(__name__)


class RequestScope(str, Enum):
    """The two dispatch scopes (execution order: retrieval > general)."""

    RETRIEVAL = "retrieval"
    GENERAL = "general"


class RetrievalLookupKind(str, Enum):
    """What kind of lookup a retrieval request calls for (mirrors the storages)."""

    SEMANTIC = "semantic"      # vector search over source chunks — ready
    INDEX = "index"            # counting/aggregation — SQL layer (later)
    RELATION = "relation"      # claims traversal — SQL layer (later)
    SUMMARY = "summary"        # document/section summaries — summarized store (later)
    LISTING = "listing"        # what is ingested — canonical store (later)


class _StrictModel(BaseModel):
    """Base model: reject unknown keys so LLM typos surface as errors."""

    model_config = ConfigDict(extra="forbid")


class RequestContextEntry(_StrictModel):
    """One earlier request of the same prompt, as later requests see it.

    Built by the routing graph, not by the analyzer. ``status``/``detail``
    describe the *dispatch outcome* of that earlier request (``None``/``""``
    when the entry is described before its own dispatch — tests only; in
    the graph, dispatch is sequential so every preceding entry has run).
    """

    kind: str
    utterance: str
    status: Optional[str] = None
    detail: str = ""


class RetrievalRequest(_StrictModel):
    """One corpus lookup, fully classified by the analyzer.

    ``question`` is the self-contained search query in the user's language
    (pronouns resolved against the prompt context — no downstream LLM will
    rephrase it). ``lookup_kind`` maps onto the storage layers; kinds the
    retrieval pipeline cannot serve yet are reported ``not_implemented``
    instead of being misread as semantic searches.
    """

    question: str = Field(min_length=1)
    lookup_kind: RetrievalLookupKind = RetrievalLookupKind.SEMANTIC
    document: Optional[str] = None
    chapter_title: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    reason: str = ""
    utterance: str = ""
    preceding: List[RequestContextEntry] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()

    @field_validator("document")
    @classmethod
    def _document_is_bare_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if "/" in value or "\\" in value or ".." in value or ":" in value:
            raise ValueError(f"document must be a bare file name, got {value!r}")
        return value

    @field_validator("chapter_title")
    @classmethod
    def _chapter_title_clean(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    def summary(self) -> dict:
        return {
            "lookup_kind": self.lookup_kind.value,
            "question": self.question,
            "document": self.document,
            "chapter_title": self.chapter_title,
            "top_k": self.top_k,
            "reason": self.reason,
            "utterance": self.utterance,
        }


class GeneralRequest(_StrictModel):
    """One out-of-scope request: the text the GeneralTaskAgent must answer."""

    question: str = Field(min_length=1)
    utterance: str = ""
    preceding: List[RequestContextEntry] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()

    def summary(self) -> dict:
        return {"question": self.question, "utterance": self.utterance}


class AnalysisResult(_StrictModel):
    """The whole analyzer answer: requests grouped by scope + the reply language.

    Ingestion is deliberately absent (see module docstring): a prompt that
    asks for ingestion in conversation yields a ``general`` request — the
    general agent explains the CLI workflow. :meth:`flattened` produces
    the dispatch order.

    ``language`` is the reply-language key the analyzer extracts from the
    prompt (prompts/routing/request_analysis.md): detected ONCE here,
    carried unchanged to the last LLM of the flow (AnswerAgent /
    GeneralTaskAgent) as an authoritative "Reply language" prompt key —
    the structural fix for prompts answered in the wrong language.
    Normalized fail-open to ``en`` by :func:`src.routing.language.normalize_language`
    (a detection problem must never break routing).
    """

    retrieval: List[RetrievalRequest] = Field(default_factory=list)
    general: List[GeneralRequest] = Field(default_factory=list)
    language: str = Field(default=DEFAULT_LANGUAGE)

    @field_validator("language")
    @classmethod
    def _normalize_language(cls, value: str) -> str:
        """Fail-open normalization: recognized label -> canonical code,
        anything else -> the English default (never a validation error)."""
        from src.routing.language import normalize_language

        return normalize_language(value)

    @model_validator(mode="after")
    def _cap_requests(self) -> "AnalysisResult":
        cap = max_requests_per_prompt()
        total = len(self.retrieval) + len(self.general)
        if total > cap:
            raise ValueError(
                f"the prompt yielded {total} requests (max {cap} per "
                "setup.yaml routing.max_requests_per_prompt) — a split this "
                "large is a malformed analysis, not an order"
            )
        return self

    @property
    def request_count(self) -> int:
        return len(self.retrieval) + len(self.general)

    def flattened(self) -> List[object]:
        """The dispatch order: all retrievals, then generals.

        Deterministic grouped-scope tunnel (the user-confirmed design);
        within a scope the analyzer's order (the user's reading order) is
        preserved.
        """
        requests: List[object] = []
        requests.extend(self.retrieval)
        requests.extend(self.general)
        return requests


# ---------------------------------------------------------------------------
# Cap from setup.yaml (fail-open like every tuning knob)
# ---------------------------------------------------------------------------

_DEFAULT_MAX_REQUESTS = 8


def max_requests_per_prompt() -> int:
    """The cap from setup.yaml ``routing.max_requests_per_prompt``.

    Fail-open on a broken/unreadable config: the routing layer must not
    die because setup.yaml is malformed — the loader's load-time
    validation covers the present-but-invalid cases; here a problem
    yields the documented default (8).
    """
    try:
        from src.tools.config_loader import load_routing_config

        return int(load_routing_config().get("max_requests_per_prompt", _DEFAULT_MAX_REQUESTS))
    except Exception as exc:  # noqa: BLE001 — tuning knob, fail open
        logger.warning("max_requests_per_prompt fallback to default (%s)", exc)
        return _DEFAULT_MAX_REQUESTS


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


def _strip_dispatcher_owned_fields(payload: dict) -> dict:
    """Drop fields the analyzer must never emit (``preceding``), with a warning.

    An ``ingestion`` key in the payload is also dropped, with a warning:
    ingestion is not a dispatchable scope anymore — whatever the model
    put there cannot be honored, and leaving it would make the validation
    fail with a less comprehensible unknown-key error.
    """
    if "ingestion" in payload:
        logger.warning(
            "analyzer emitted an 'ingestion' scope — ingestion is a CLI "
            "operation, not a dispatchable request; the scope is dropped"
        )
        payload.pop("ingestion", None)
    for scope in ("retrieval", "general"):
        items = payload.get(scope)
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            if isinstance(item, dict) and "preceding" in item:
                logger.warning(
                    "analyzer emitted 'preceding' for %s[%d] — field is "
                    "dispatcher-owned; value dropped: %r",
                    scope, index, item["preceding"],
                )
                items[index] = {
                    key: value for key, value in item.items() if key != "preceding"
                }
    return payload


def parse_analysis(raw: str) -> AnalysisResult:
    """Parse and validate the analyzer's raw answer into an AnalysisResult.

    Raises:
        ValueError: the answer contains no JSON object at all, the JSON is
            syntactically broken, or the grouped structure violates the
            models' rules (pydantic ValidationError is a ValueError
            subclass). Callers treat all of these the same way: a
            malformed LLM response, typically retryable.
    """
    if not raw or not raw.strip():
        raise ValueError("empty analyzer response")

    if raw.lstrip().startswith("["):
        raise ValueError("analyzer response is not a JSON object")

    payload = _extract_json_payload(raw)
    if not isinstance(payload, dict):
        raise ValueError("analyzer response is not a JSON object")

    payload = _strip_dispatcher_owned_fields(payload)
    return AnalysisResult.model_validate(payload)
