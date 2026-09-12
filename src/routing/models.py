"""Structured requests extracted from a user prompt, grouped by scope.

The request analyzer (``request_analyzer`` role, prompts/routing/
request_analysis.md) reads the raw user prompt ONCE and extracts **every**
request it holds, grouped by scope. The output shape is::

    {
      "ingestion": [
        {"document": "meow.pdf", "force": false, "redo_summaries": false,
         "origin": null}
      ],
      "retrieval": [
        {"lookup_kind": "semantic|index|relation|summary|listing",
         "question": "...", "document": null, "chapter_title": null,
         "top_k": null}
      ],
      "general": [
        {"question": "..."}
      ],
      "origin": null,             # prompt-level origin shorthand
      "force": false,             # prompt-level force shorthand
      "redo_summaries": false     # prompt-level shorthand
    }

**One analysis, no re-routing.** The analyzer has the full prompt context:
pronouns are resolved HERE — a retrieval ``question`` must be
self-contained ("tell me more about the King of the North", not "tell me
more about him"), an ingestion ``document`` must be the bare file name.
Nothing downstream re-reads the user's words: the ingestion and retrieval
pipelines are deterministic Python (facts → decision table → graph); the
only LLM-based post-routing worker is the GeneralTaskAgent (and later the
AnswerAgent).

The dispatcher (routing graph) runs the flattened requests in **grouped
scope order** — all ingestions, then all retrievals, then generals — so
"ingest X, then ask about it" works without the analyzer doing anything
special. :meth:`AnalysisResult.flattened` implements that order.

An ingestion request **without a document is invalid at the source**: the
analyzer must either name the file or leave the request out — "ingest some
documents" is not dispatchable. (The old per-request degradation is gone:
with a single analysis there is no second LLM pass to ask the user
anything; an unusable request is rejected with its utterance, and the
batch continues.)

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

logger = logging.getLogger(__name__)


class RequestScope(str, Enum):
    """The three dispatch scopes (execution order: ingestion > retrieval > general)."""

    INGESTION = "ingestion"
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


class IngestionRequest(_StrictModel):
    """One ingestion order: a document plus its intent modifiers.

    ``origin`` is echoed ONLY when the user stated it ("it is a canon
    document"); ``None`` means unstated — the deterministic origin
    decision (stored > filename inference > ask) happens in the task
    agent, never a guess from the LLM.
    """

    document: str = Field(min_length=1)
    force: bool = False
    redo_summaries: bool = False
    origin: Optional[str] = None
    utterance: str = ""
    # Dispatcher-owned (the routing graph fills it from the dispatch
    # history); the analyzer must never emit it — parse_analysis strips
    # hallucinated values at the boundary.
    preceding: List[RequestContextEntry] = Field(default_factory=list)

    @field_validator("document")
    @classmethod
    def _document_is_bare_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("document must not be blank")
        if "/" in value or "\\" in value or ".." in value or ":" in value:
            raise ValueError(
                f"document must be a bare file name, got path-like {value!r}"
            )
        return value

    @field_validator("origin")
    @classmethod
    def _origin_known_kind(cls, value: Optional[str]) -> Optional[str]:
        """Only the three canonical origins are acceptable; None = unstated."""
        if value is None:
            return None
        cleaned = value.strip().lower()
        if cleaned not in {"canon", "community", "rpg"}:
            raise ValueError(
                "origin must be one of 'canon', 'community', 'rpg' "
                f"(or null), got {value!r}"
            )
        return cleaned

    def summary(self) -> dict:
        return {
            "document": self.document,
            "force": self.force,
            "redo_summaries": self.redo_summaries,
            "origin": self.origin,
            "utterance": self.utterance,
        }


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
    """The whole analyzer answer: requests grouped by scope.

    ``force``/``redo_summaries``/``origin`` are prompt-level shorthands:
    they apply to every ingestion request that does not override them
    ("ingest both files again, they're community docs" — stated once).
    :meth:`flattened` produces the dispatch order.
    """

    ingestion: List[IngestionRequest] = Field(default_factory=list)
    retrieval: List[RetrievalRequest] = Field(default_factory=list)
    general: List[GeneralRequest] = Field(default_factory=list)
    force: bool = False
    redo_summaries: bool = False
    origin: Optional[str] = None

    @field_validator("origin")
    @classmethod
    def _origin_known_kind(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip().lower()
        if cleaned not in {"canon", "community", "rpg"}:
            raise ValueError(
                "origin must be one of 'canon', 'community', 'rpg' "
                f"(or null), got {value!r}"
            )
        return cleaned

    @model_validator(mode="after")
    def _cap_requests(self) -> "AnalysisResult":
        cap = max_requests_per_prompt()
        total = len(self.ingestion) + len(self.retrieval) + len(self.general)
        if total > cap:
            raise ValueError(
                f"the prompt yielded {total} requests (max {cap} per "
                "setup.yaml routing.max_requests_per_prompt) — a split this "
                "large is a malformed analysis, not an order"
            )
        return self

    @property
    def request_count(self) -> int:
        return len(self.ingestion) + len(self.retrieval) + len(self.general)

    def flattened(self) -> List[object]:
        """The dispatch order: all ingestions, then retrievals, then generals.

        Deterministic grouped-scope tunnel (the user-confirmed design):
        "ingest X, then ask about it" works because ingestion requests run
        first; within a scope the analyzer's order (the user's reading
        order) is preserved.
        """
        requests: List[object] = []
        requests.extend(self.ingestion)
        requests.extend(self.retrieval)
        requests.extend(self.general)
        return requests

    def apply_shorthands(self) -> "AnalysisResult":
        """Push prompt-level flags down onto ingestion requests.

        A per-request value always wins over the prompt-level shorthand
        (the shorthand only fills what the request left unstated). Returns
        ``self`` for chaining.
        """
        for request in self.ingestion:
            if not request.force:
                request.force = self.force
            if not request.redo_summaries:
                request.redo_summaries = self.redo_summaries
            if request.origin is None:
                request.origin = self.origin
        return self


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


# Ingestion-intent markers: an ingestion request's own ``utterance`` must
# carry one of these (any language) for the order to be self-consistent.
# This is NOT routing-by-keywords — routing stays LLM-decided — it is a
# self-consistency check on the LLM's own output at the parse boundary:
# a 8B model sometimes fills the rich ``ingestion`` shape "by reflex"
# (the vif-argent incident: a content question became "vif-argent.pdf").
_INGESTION_INTENT_RE = re.compile(
    r"\b(?:"
    # English
    r"ingest|re-?ingest|re-?extract|re-?index|import|add file|add the file"
    r"|reload|re-?load|store|index(?!ing) this|force extraction"
    # French
    r"|ingest|ingere|ingère|ajoute|ajouter|charge|charger|recharge"
    r"|re-?charger|re-?extraire|extraire|indexe|indexer|remplace"
    r"|mets a jour|mets à jour|met a jour|met à jour"
    r")\b",
    re.IGNORECASE,
)


def _has_ingestion_intent(utterance: str) -> bool:
    """True when an utterance plausibly asks for document storage/rework."""
    return bool(_INGESTION_INTENT_RE.search(utterance or ""))


def _drop_phantom_ingestions(items: list) -> tuple:
    """Drop ingestion requests whose own utterance shows no storage intent.

    Second half of the anti-hallucination defense (the first half is the
    prompt's "never invent an ingestion request" section): when the model
    hallucinates a file to ingest — typically from an in-world noun in a
    content question — the phantom's ``utterance`` betrays it: no storage
    verb anywhere. The item is dropped with a warning and the sibling
    requests survive, exactly like the Fix-A degradation.

    An empty/missing ``utterance`` cannot be checked and is kept (fail
    open — tests and CLI callers build items without one).
    """
    kept: list = []
    dropped: list = []
    for index, item in enumerate(items):
        if isinstance(item, dict):
            utterance = str(item.get("utterance") or "")
            if utterance.strip() and not _has_ingestion_intent(utterance):
                logger.warning(
                    "analyzer emitted ingestion[%d] with document %r but no "
                    "storage intent in the utterance — dropped as a phantom "
                    "(utterance: %r)",
                    index, item.get("document"), utterance,
                )
                dropped.append(item)
                continue
        kept.append(item)
    return kept, dropped


def _degrade_incomplete_ingestion(items: list) -> list:
    """Fix A defense-in-depth: keep one bad item from killing the batch.

    The prompt forbids unresolvable ingestion orders ("ingest some
    documents", no document). A disobedient model must not invalidate the
    sibling requests: an item whose ``document`` is missing/blank/null is
    converted to a ``general`` request (the general agent asks the user to
    clarify) instead of failing the whole answer's validation.
    """
    kept: list = []
    degraded: list = []
    for index, item in enumerate(items):
        if (
            isinstance(item, dict)
            and not str(item.get("document") or "").strip()
        ):
            logger.warning(
                "analyzer emitted ingestion[%d] without a document — "
                "degraded to a general clarification request: %r",
                index, item.get("utterance"),
            )
            utterance = str(item.get("utterance") or "ingest some documents")
            degraded.append({
                "question": utterance,
                "utterance": utterance,
            })
        else:
            kept.append(item)
    return kept, degraded


def _strip_dispatcher_owned_fields(payload: dict) -> dict:
    """Drop fields the analyzer must never emit (``preceding``), with a warning."""
    for scope in ("ingestion", "retrieval", "general"):
        items = payload.get(scope)
        if not isinstance(items, list):
            continue
        if scope == "ingestion":
            items, dropped = _drop_phantom_ingestions(items)
            for phantom in dropped:
                logger.info(
                    "phantom ingestion dropped (document %r) — the user's "
                    "words carry no storage order",
                    phantom.get("document") if isinstance(phantom, dict) else phantom,
                )
            kept, degraded = _degrade_incomplete_ingestion(items)
            payload["ingestion"] = kept
            if degraded:
                general = payload.get("general")
                if not isinstance(general, list):
                    general = []
                    payload["general"] = general
                general.extend(degraded)
                items = payload["ingestion"]
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
    result = AnalysisResult.model_validate(payload)
    # Prompt-level shorthands (force / redo_summaries / origin) resolve
    # HERE, at the LLM-answer boundary: everything downstream sees final,
    # per-request values.
    return result.apply_shorthands()
