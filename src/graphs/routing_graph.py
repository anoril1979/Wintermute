"""Routing graph: dispatch the analyzed requests to their task agents.

The routing orchestrator analyzes the raw user prompt ONCE (request
analyzer, ``request_analyzer`` LLM role) into a grouped
:class:`AnalysisResult`. This graph runs the flattened requests in
**grouped scope order** — all ingestions, then all retrievals, then
generals (``AnalysisResult.flattened()``) — and hands every request to
the task agent registered under its kind's key:

    ingestion -> "ingestion_task"  (IngestionTaskAgent)
    retrieval -> "retrieval_task"  (RetrievalTaskAgent)
    general   -> "general_task"    (GeneralTaskAgent)

Ingestion and retrieval pipelines are deterministic Python (facts ->
decision table -> graph); the only LLM-based worker is the general one.

Per-request outcome, not per-graph outcome: each dispatched request
produces one result entry in ``RoutingContext.results`` — the graph never
aborts the whole batch because one request failed (a broken ingestion
order must not prevent the question that follows it from being answered).
Missing agents surface as ``not_implemented`` results, so the graph runs
end-to-end while the task agents land one by one.

**Deterministic gates, before any agent runs:**

* an ingestion request without a document is REJECTED (the analyzer must
  not invent file names; with a single analysis there is no second pass
  to ask the user inside the loop — the user-facing reply says so);
* an ingestion request whose origin cannot be decided deterministically
  (user-stated, stored, or confidently inferred from the file name) is
  SET ASIDE — not ingested, batch continues — and reported at the end:
  an unverified origin must never reach storage silently.

**Prompt-local memory**: dispatch is strictly sequential, so when request
*i* runs, every earlier request of the same prompt has already been
dispatched — and its outcome is known. The graph attaches those preceding
entries (kind, utterance, outcome) to each request (``preceding``) so an
LLM-based agent (GeneralTaskAgent, later AnswerAgent) can compose its
reply against what actually happened: "ingest meow.pdf, then summarize
it" — the general/retrieval answer knows whether the ingestion worked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.agents.contexts import RoutingContext
from src.agents.task_protocols import TASK_AGENT_KEYS, UserTaskAgent
from src.routing.models import (
    GeneralRequest,
    IngestionRequest,
    RetrievalRequest,
    RequestContextEntry,
)

logger = logging.getLogger(__name__)

#: Stable status strings used in the per-request result entries.
STATUS_DONE = "done"                        # agent handled the request
STATUS_NOT_IMPLEMENTED = "not_implemented"  # no agent for this kind yet
STATUS_REJECTED = "rejected"                # request unusable (validation)
STATUS_SET_ASIDE = "set_aside"              # origin undecidable: NOT ingested

#: Wire label for a request kind: the stable scope vocabulary the API
#: composes replies with ("ingestion"/"retrieval"/"general"), not the
#: Python class name.
_KIND_LABELS = {
    IngestionRequest: "ingestion",
    RetrievalRequest: "retrieval",
    GeneralRequest: "general",
}


def _kind_label(request: object) -> str:
    """Scope label for a request (class name as defensive fallback)."""
    return _KIND_LABELS.get(type(request), type(request).__name__)


@dataclass
class RequestOutcome:
    """Result of dispatching one structured user request."""

    kind: str                                  # request scope/type name
    utterance: str
    status: str                                # STATUS_* constant
    detail: str = ""                           # human-readable explanation
    payload: Dict[str, object] = field(default_factory=dict)  # agent outcome

    def as_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "utterance": self.utterance,
            "status": self.status,
            "detail": self.detail,
            **self.payload,
        }


@dataclass
class RoutingOutcome:
    """Result of a full routing-graph run (one user prompt)."""

    outcomes: List[RequestOutcome] = field(default_factory=list)

    @property
    def handled(self) -> bool:
        """True when at least one request reached a real task agent."""
        return any(o.status == STATUS_DONE for o in self.outcomes)

    def as_list(self) -> List[Dict[str, object]]:
        return [o.as_dict() for o in self.outcomes]


class RoutingGraph:
    """Dispatches structured user requests to their task agents.

    The registry maps the per-kind agent keys (``TASK_AGENT_KEYS``) to
    agent instances satisfying :class:`UserTaskAgent`. A kind missing from
    the registry yields a ``not_implemented`` result for its requests —
    never a crash — so the graph can run before every agent exists.
    """

    def __init__(self, agents: Optional[Dict[str, UserTaskAgent]] = None) -> None:
        self._agents: Dict[str, UserTaskAgent] = dict(agents or {})

    def run(
        self,
        context: RoutingContext,
        requests: List[object],
    ) -> RoutingOutcome:
        """Handle every request in order, collecting per-request outcomes.

        Args:
            context:  the shared routing context (agents may stash state
                      there — retrieved documents, ingestion references...).
            requests: the flattened request list produced by the analyzer
                      (grouped-scope order: ingestions, retrievals,
                      generals).
                      analyzer.
        """
        outcome = RoutingOutcome()

        for index, request in enumerate(requests):
            # Prompt-local memory: every request sees the ones dispatched
            # before it in the same prompt (their kind, utterance and — for
            # the ones already run — dispatch outcome). Sequential dispatch
            # makes this exact; LLM-based agents use it to compose their
            # reply against what actually happened.
            self._attach_preceding(context, index, request, requests, outcome)
            try:
                entry = self._dispatch(context, index, request)
            except Exception as exc:  # noqa: BLE001 — one bad request must
                # never abort the batch; report and keep going.
                logger.exception("Routing dispatch crashed on request %d", index)
                kind = _kind_label(request)
                entry = RequestOutcome(
                    kind=kind,
                    utterance=str(getattr(request, "utterance", "")),
                    status=STATUS_REJECTED,
                    detail=f"unexpected dispatch error: {exc}",
                )
            outcome.outcomes.append(entry)
            context.results.append(entry.as_dict())

        return outcome

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _attach_preceding(
        context: RoutingContext,
        index: int,
        request: object,
        requests: List[object],
        outcome: "RoutingOutcome",
    ) -> None:
        """Give ``request`` the entries of its same-prompt predecessors.

        Utterances come from the analyzed requests (that is where "it"
        resolves); status/detail come from ``outcome.outcomes`` (the
        authoritative dispatch history) — the analyzer cannot fabricate
        either side of the merge. Unknown request types (defensive) are
        skipped. Traced only when the batch actually has a predecessor.
        """
        if not isinstance(request, (IngestionRequest, RetrievalRequest, GeneralRequest)):
            return
        if index <= 0:
            return
        preceding: List[RequestContextEntry] = []
        for earlier, done in zip(requests[:index], outcome.outcomes[:index]):
            if not isinstance(earlier, (IngestionRequest, RetrievalRequest, GeneralRequest)):
                continue
            preceding.append(
                RequestContextEntry(
                    kind=_kind_label(earlier),
                    utterance=earlier.utterance or _request_display(earlier),
                    status=done.status,
                    detail=done.detail,
                )
            )
        request.preceding = preceding
        context.emit(
            "dispatch", "local_context",
            f"{len(preceding)} preceding request(s) of this prompt attached",
            index=index,
        )

    def _dispatch(self, context: RoutingContext, index: int, request: object) -> RequestOutcome:
        """Hand one request to its kind's agent (or gate/reject it)."""
        kind_value = _kind_label(request)
        utterance = str(getattr(request, "utterance", ""))

        if isinstance(request, IngestionRequest):
            # Deterministic gate BEFORE any agent runs: an ingestion request
            # whose origin cannot be decided is set aside — not ingested —
            # and the batch continues. An unverified origin must never
            # reach storage silently.
            gate = _decide_origin_gate(request)
            if gate is not None:
                context.emit("dispatch", "origin_set_aside", gate[1], index=index,
                             document=request.document)
                return RequestOutcome(
                    kind=kind_value,
                    utterance=utterance,
                    status=STATUS_SET_ASIDE,
                    detail=gate[1],
                    payload={"document": request.document, "question": gate[0]},
                )

        agent_key = TASK_AGENT_KEYS.get(kind_value)
        if agent_key is None:
            context.emit("dispatch", "unknown_kind",
                         f"unknown request kind: {kind_value!r}", index=index)
            return RequestOutcome(
                kind=kind_value,
                utterance=utterance,
                status=STATUS_REJECTED,
                detail=f"unknown request kind: {kind_value!r}",
            )

        agent = self._agents.get(agent_key)
        if agent is None:
            context.emit("dispatch", "not_implemented",
                         f"no agent for kind '{kind_value}' yet", index=index)
            return RequestOutcome(
                kind=kind_value,
                utterance=utterance,
                status=STATUS_NOT_IMPLEMENTED,
                detail=f"no agent implemented for kind '{kind_value}' yet",
            )

        context.emit("dispatch", "dispatching",
                     f"[{kind_value}] -> {agent.name}", index=index,
                     agent=agent.name)
        result = agent.run(context, request)
        context.emit("dispatch", "dispatched",
                     f"[{kind_value}] {result.status.value}: {result.detail or 'handled'}",
                     index=index, agent=agent.name,
                     agent_status=result.status.value)
        payload = {
            "agent": result.agent_name,
            "agent_status": result.status.value,
            "detail": result.detail,
            **(result.payload or {}),
        }

        if result.status.value == "ok":
            status = STATUS_DONE
            detail = result.detail or "handled"
        else:
            # FAILED / SKIPPED: the agent could not handle it; the routing
            # layer reports the failure but keeps processing the batch.
            status = STATUS_REJECTED
            detail = result.detail or result.status.value

        return RequestOutcome(
            kind=kind_value,
            utterance=utterance,
            status=status,
            detail=detail,
            payload=payload,
        )


# ---------------------------------------------------------------------------
# Deterministic origin gate (before any agent runs)
# ---------------------------------------------------------------------------

def _request_display(request: object) -> str:
    """Short human-readable text of a request for memory entries/logs."""
    if isinstance(request, IngestionRequest):
        return f"ingest {request.document}"
    if isinstance(request, RetrievalRequest):
        return request.question
    if isinstance(request, GeneralRequest):
        return request.question
    return str(getattr(request, "utterance", type(request).__name__))


def _decide_origin_gate(request: IngestionRequest) -> Optional[tuple]:
    """Deterministic origin decision for one ingestion request.

    Precedence (the established one, now in the dispatch path):
    user-stated > stored (canonical JSON) > confident filename inference.
    Returns ``None`` when the origin is decided (or decidable) — dispatch
    proceeds — or ``(question_text, explanation)`` when the request must
    be SET ASIDE: not ingested, batch continues, user notified at the end.

    Fail-open: a store/facts problem counts as "cannot decide" — the
    document is set aside rather than ingested with a guessed origin.
    """
    if request.origin:
        return None  # user-stated wins, always
    try:
        from src.tools.ingest_tool import ingest_document

        resolution = ingest_document(request.document)
        if resolution.get("status") not in ("ready", "ingested"):
            return None  # not found: the agent reports candidates itself
        path = Path(str(resolution["path"]))
    except Exception:  # noqa: BLE001 — fail-open: cannot decide
        return (
            "which origin does this document have?",
            f"could not check the state of '{request.document}': the "
            "document was set aside rather than ingested with an "
            "unverified origin",
        )

    # Stored origin: the origin decided at the document's first ingestion
    # (canonical JSON). Re-ingestion keeps it.
    try:
        from src.helpers.document_extract_json_store import (
            canonical_path_for,
            load_extract,
        )

        canonical = canonical_path_for(path)
        if canonical.exists():
            stored = load_extract(canonical)
            stored_origin = (
                stored.origin.value if getattr(stored, "origin", None) else None
            )
            if stored_origin:
                return None  # known: re-ingestion keeps the stored origin
    except Exception:  # noqa: BLE001 — advisory fact only
        pass

    # Confident filename inference (deterministic, auditable).
    try:
        from src.ingestion.ingestion_router import infer_origin

        if infer_origin(request.document) is not None:
            return None
    except Exception:  # noqa: BLE001 — advisory only
        pass

    return (
        (
            f"which origin does '{request.document}' have? Phrase it as: "
            f"\"ingest {request.document}, it is a canon document\" / "
            f"\"... it is a community document\" / \"... it is an rpg "
            "document (my own content)\"."
        ),
        (
            f"origin of '{request.document}' is unknown (unstated, not "
            "stored, not inferable from the name): set aside — not "
            "ingested"
        ),
    )
