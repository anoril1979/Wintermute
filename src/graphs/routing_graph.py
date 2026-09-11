"""Routing graph: dispatch each structured user request to its task agent.

The routing orchestrator first analyzes the raw user prompt (request
analyzer, ``request_analyzer`` LLM role) into an ordered list of
:class:`UserRequest`. This graph then loops over that list and hands every
request to the task agent registered under its kind's key:

    retrieval  -> "retrieval_task"  (RetrievalTaskAgent — future)
    ingestion  -> "ingestion_task"  (IngestionTaskAgent)
    general    -> "general_task"    (GeneralTaskAgent)

Per-request outcome, not per-graph outcome: each dispatched request
produces one result entry in ``RoutingContext.results`` — the graph never
aborts the whole batch because one request failed (a broken ingestion
order must not prevent the question that follows it from being answered).
Missing agents surface as ``not_implemented`` results, so the graph runs
end-to-end while the task agents land one by one.

**Prompt-local memory**: dispatch is strictly sequential, so when request
*i* runs, every earlier request of the same prompt has already been
dispatched — and its outcome is known. The graph attaches those preceding
entries (kind, utterance, status...) to each request
(``UserRequest.preceding``) before handing it to the task agent, so an
agent's LLM can resolve pronouns against the local context: "ingest
meow.pdf, then summarize it" — *it* is meow.pdf for the second request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.agents.contexts import RoutingContext
from src.agents.task_protocols import TASK_AGENT_KEYS, UserTaskAgent
from src.routing.models import RequestContextEntry, UserRequest

logger = logging.getLogger(__name__)

#: Stable status strings used in the per-request result entries.
STATUS_DONE = "done"                        # agent handled the request
STATUS_NOT_IMPLEMENTED = "not_implemented"  # no agent for this kind yet
STATUS_REJECTED = "rejected"                # request unusable (validation)
STATUS_INCOMPLETE = "incomplete"            # underspecified: user must add info


@dataclass
class RequestOutcome:
    """Result of dispatching one structured user request."""

    kind: str                                  # RequestKind value
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
            requests: the ordered ``UserRequest`` list produced by the
                      analyzer.
        """
        outcome = RoutingOutcome()

        for index, request in enumerate(requests):
            # Prompt-local memory: every request sees the ones dispatched
            # before it in the same prompt (their kind, utterance, payload
            # and — for the ones already run — dispatch status). Sequential
            # dispatch makes this exact; agents' LLMs use it to resolve
            # pronouns ("ingest meow.pdf, then summarize it" — it = meow.pdf).
            self._attach_preceding(context, index, request, requests, outcome)
            try:
                entry = self._dispatch(context, index, request)
            except Exception as exc:  # noqa: BLE001 — one bad request must
                # never abort the batch; report and keep going.
                logger.exception("Routing dispatch crashed on request %d", index)
                kind = getattr(request, "kind", "?")
                kind = getattr(kind, "value", str(kind))
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

        Payload (``document``/``question``) comes from the analyzer's
        requests (that is where "it" resolves); status/detail come from
        ``outcome.outcomes`` (the authoritative dispatch history) — the
        analyzer cannot fabricate either side of the merge. Requests that
        are not ``UserRequest`` instances (defensive: the graph tolerates
        duck-typed objects) are skipped. Traced only when the batch
        actually has a predecessor.
        """
        if not isinstance(request, UserRequest):
            return
        if index <= 0:
            return
        preceding: List[RequestContextEntry] = []
        for earlier, done in zip(requests[:index], outcome.outcomes[:index]):
            if not isinstance(earlier, UserRequest):
                continue
            preceding.append(
                RequestContextEntry(
                    kind=earlier.kind.value,
                    utterance=earlier.utterance,
                    document=earlier.document,
                    question=earlier.question,
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
        """Hand one request to its kind's agent (or report not-implemented)."""
        kind = getattr(request, "kind", None)
        kind_value = getattr(kind, "value", str(kind))
        utterance = str(getattr(request, "utterance", ""))

        agent_key = TASK_AGENT_KEYS.get(str(kind_value))
        if agent_key is None:
            context.emit("dispatch", "unknown_kind",
                         f"unknown request kind: {kind_value!r}", index=index)
            return RequestOutcome(
                kind=str(kind_value),
                utterance=utterance,
                status=STATUS_REJECTED,
                detail=f"unknown request kind: {kind_value!r}",
            )

        # Per-request degradation, checked before agent availability: an
        # ingestion request without a document is valid-but-underspecified
        # (the analyzer must not invent file names). Asking the user takes
        # precedence over announcing a missing agent — the request stays
        # unresolvable either way, but the user can fix it by naming a file.
        if (
            str(kind_value) == "ingestion"
            and not str(getattr(request, "document", "") or "").strip()
        ):
            context.emit(
                "dispatch", "incomplete_request",
                "ingestion request without a document: asking the user",
                index=index,
            )
            return RequestOutcome(
                kind=str(kind_value),
                utterance=utterance,
                status=STATUS_INCOMPLETE,
                detail=(
                    "which document should be ingested? Name the file "
                    "you want ingested (e.g. 'ingest Dumas.pdf')."
                ),
            )

        agent = self._agents.get(agent_key)
        if agent is None:
            context.emit("dispatch", "not_implemented",
                         f"no agent for kind '{kind_value}' yet", index=index)
            return RequestOutcome(
                kind=str(kind_value),
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
            kind=str(kind_value),
            utterance=utterance,
            status=status,
            detail=detail,
            payload=payload,
        )
