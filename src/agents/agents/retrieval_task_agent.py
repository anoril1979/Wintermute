"""RetrievalTaskAgent — the deterministic bridge to the retrieval pipeline.

    RoutingGraph
        └─ RetrievalTaskAgent.run(context, RetrievalRequest)
             └─ src/retrieval/retrieval_orchestrator.run_retrieval([request])

Since the routing rework, retrieval requests arrive already classified by
the single analyzer (``lookup_kind``, self-contained question, scopes) —
the agent builds the retrieval spec and calls the deterministic pipeline
(facts → decision table → graph). NO retrieval-side LLM, no keyword
fallback: a malformed analysis never reaches this agent (the analyzer
rejects it at the boundary).

Failure mapping (never raises for expected failures):

* out-of-contract request              → INPUT_DATA;
* orchestrator config_error            → CONFIG (not retryable);
* no_corpus / not_implemented          → FAILED (the caller phrases
  "memory dormant" / "not served yet" for the user);
* pipeline step failure                → FAILED, orchestrator reason in
  ``detail``;
* ok / partial                         → OK, the scored hits in payload
  (``partial`` keeps the per-request statuses in the payload).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from src.agents.contexts import RoutingContext
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
from src.routing.models import RetrievalRequest

logger = logging.getLogger(__name__)

AGENT_NAME = "retrieval_task"

# Statuses the retrieval orchestrator can return (stable strings).
_RETRIEVAL_OK = "ok"
_RETRIEVAL_PARTIAL = "partial"      # some requests served, others not
_RETRIEVAL_NO_CORPUS = "no_corpus"
_RETRIEVAL_NOT_IMPLEMENTED = "not_implemented"
_RETRIEVAL_CONFIG_ERROR = "config_error"


class RetrievalTaskAgent:
    """Runs the deterministic retrieval pipeline for structured requests."""

    name = AGENT_NAME

    def __init__(
        self,
        runner: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        """Args:
        runner: optional replacement for
            ``src.retrieval.retrieval_orchestrator.run_retrieval`` (tests);
            the real orchestrator is imported lazily otherwise.
        """
        self._runner = runner

    # -- UserTaskAgent contract ------------------------------------------------

    def run(self, context: RoutingContext, request: object) -> AgentResult:
        """Answer one classified retrieval request from the ingested corpus.

        The request is dispatched one by one (the routing graph loops over
        the analyzer's retrieval requests); ``partial`` cannot happen for
        a single request but the mapping is kept for the shared contract.
        """
        request = self._extract_request(request)
        if isinstance(request, AgentResult):  # out-of-contract guard
            return request

        context.emit(
            "task", "retrieval_start",
            f"answering from the corpus: {request.question!r}",
            lookup_kind=request.lookup_kind.value,
        )
        result = self._run_retrieval([request], on_event=context.on_event)

        # Merge the pipeline's internal traces into the routing context's
        # event log: with a live observer they already streamed through;
        # without one they are collected here so the returned trace list
        # is complete.
        for event in result.get("traces", []):
            context.events.append(event)

        status = str(result.get("status", ""))

        if status in (_RETRIEVAL_OK, _RETRIEVAL_PARTIAL):
            hits = result.get("hits", [])
            answers = [
                str(sub.get("answer") or "")
                for sub in result.get("requests", [])
                if sub.get("answer")
            ]
            served = sum(
                1 for sub in result.get("requests", [])
                if sub.get("status") == "ok"
            )
            total = len(result.get("requests", [])) or 1
            if answers:
                # The answer agent phrased at least one request's hits:
                # the detail is the reply, not a chunk count.
                detail = "\n\n".join(answers)
            elif status == _RETRIEVAL_OK:
                detail = f"{len(hits)} chunk(s) retrieved"
            else:
                detail = (
                    f"partial: {served}/{total} request(s) served, "
                    f"{len(hits)} chunk(s) retrieved"
                )
            context.emit(
                "task", "retrieval_done",
                detail[:200],
                hits=len(hits),
                served=served,
                total=total,
                partial=status == _RETRIEVAL_PARTIAL,
                answered=len(answers),
            )
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.OK,
                detail=detail,
                payload={"retrieval": result},
            )

        if status == _RETRIEVAL_CONFIG_ERROR:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.CONFIG,
                detail=str(result.get("message", "retrieval configuration error")),
                payload={"retrieval": result},
            )

        if status == _RETRIEVAL_NO_CORPUS:
            detail = (
                "Wintermute's memory is dormant: nothing indexed yet. "
                "Ingest a document first."
            )
            context.emit("task", "retrieval_dormant", detail)
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail=detail,
                payload={"retrieval": result},
            )

        if status == _RETRIEVAL_NOT_IMPLEMENTED:
            detail = str(result.get("message", "retrieval type not served yet"))
            context.emit("task", "retrieval_unserved", detail)
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail=detail,
                payload={"retrieval": result},
            )

        # failed / unknown
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.FAILED,
            failure_domain=FailureDomain.EXTERNAL,
            detail=str(result.get("message", "retrieval failed")),
            payload={"retrieval": result},
        )

    def validate(
        self, context: RoutingContext, request: object
    ) -> Optional[AgentResult]:
        """Nothing specific to check beyond ``run``'s own outcome."""
        return None

    # -- internals ---------------------------------------------------------------

    @staticmethod
    def _extract_request(request: object):
        """Guard the request type; return the request or an AgentResult."""
        if not isinstance(request, RetrievalRequest):
            return AgentResult(
                agent_name=AGENT_NAME,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail="retrieval task expects a RetrievalRequest",
            )
        if not request.question.strip():
            return AgentResult(
                agent_name=AGENT_NAME,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail="retrieval request carries no question",
            )
        return request

    def _run_retrieval(
        self,
        requests: list,
        *,
        on_event: Optional[object] = None,
    ) -> Dict[str, Any]:
        """Call the retrieval orchestrator (lazily imported / injectable).

        ``on_event`` is forwarded so the pipeline's internal events
        (decision table, graph steps) reach the routing context's observer
        — the thinking panel then shows the whole retrieval flow live.
        """
        if self._runner is None:
            from src.retrieval.retrieval_orchestrator import run_retrieval

            self._runner = run_retrieval
        kwargs: Dict[str, Any] = {}
        if on_event is not None:
            kwargs["on_event"] = on_event
        return dict(self._runner(requests, **kwargs))
