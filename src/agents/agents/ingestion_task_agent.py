"""IngestionTaskAgent — handles structured ingestion requests.

The bridge between the routing layer and the ingestion pipeline:

    RoutingGraph
        └─ IngestionTaskAgent.run(context, UserRequest(kind=ingestion))
             ├─ resolve the document (sandboxed, via src/tools/ingest_tool)
             └─ src/ingestion/ingestion_orchestrator.run_ingestion_file(
                    path,
                    force=request.options.force_reingest,
                    force_summarization=request.options.force_summarization)

The agent translates the structured request into an orchestrator call and
maps the orchestrator's result dict onto the shared agent outcome
vocabulary (AgentResult/FailureDomain); the orchestrator owns everything
else (config gate, graph, extraction checkpoint). Request validation
happened upstream (the analyzer produced a validated UserRequest), so the
agent only defends against out-of-contract inputs.

Failure mapping (never raises for expected failures):

* out-of-contract request / resolution refused → INPUT_DATA;
* orchestrator config_error                    → CONFIG (not retryable);
* document not found / ingestion rejected      → INPUT_DATA, with the
  orchestrator's reason carried in ``detail`` for the user reply;
* pipeline partially wired (not_implemented)   → FAILED too, with the
  orchestrator's step report in the payload — the caller can phrase
  "accepted, extraction done, embedding not implemented yet".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.agents.contexts import RoutingContext
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain

logger = logging.getLogger(__name__)

AGENT_NAME = "ingestion_task"

# Statuses the ingestion orchestrator can return (stable strings).
_INGESTION_ACCEPTED = "accepted"
_INGESTION_NOT_IMPLEMENTED = "not_implemented"
_INGESTION_CONFIG_ERROR = "config_error"

#: Orchestrator statuses worth a FAILED result.
_REJECTED_STATUSES = {"rejected", _INGESTION_CONFIG_ERROR}


class IngestionTaskAgent:
    """Prepares and starts the ingestion orchestrator for one request."""

    name = AGENT_NAME

    def __init__(
        self,
        runner: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        """Args:
        runner: optional replacement for
            ``src.ingestion.ingestion_orchestrator.run_ingestion_file``
            (tests); the real orchestrator is imported lazily otherwise.
        """
        self._runner = runner

    # -- UserTaskAgent contract ------------------------------------------------

    def run(self, context: RoutingContext, request: object) -> AgentResult:
        """Ingest the document referenced by a structured ingestion request."""
        document = self._extract_document(request)
        if isinstance(document, AgentResult):  # out-of-contract guard
            return document

        # -- sandboxed resolution (bare file name -> documents tree) ----------
        from src.tools.ingest_tool import ingest_document

        resolution = ingest_document(document)
        if resolution.get("status") not in ("ready", "ingested"):
            context.emit("task", "ingestion_refused",
                         f"document '{document}' not found in the documents tree",
                         document=document)
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail=str(resolution.get("message", "document not found")),
                payload={"resolution": resolution},
            )

        path = Path(str(resolution["path"]))
        force = bool(request.option("force_reingest", False))  # type: ignore[union-attr]
        force_summarization = bool(request.option("force_summarization", False))  # type: ignore[union-attr]

        # -- orchestrate --------------------------------------------------------
        context.emit("task", "ingestion_start",
                     f"starting ingestion of '{path.name}'"
                     + (" (forced)" if force else "")
                     + (" (force summarization)" if force_summarization else ""),
                     document=path.name, force=force,
                     force_summarization=force_summarization)
        result = self._run_ingestion(
            path, force=force,
            force_summarization=force_summarization,
            on_event=context.on_event,
        )

        # Merge the pipeline's internal traces (graph steps, extraction
        # agent) into the routing context's event log, preserving order:
        # with a live observer they already streamed through; without one
        # they are collected here so the returned trace list is complete.
        for event in result.get("traces", []):
            context.events.append(event)

        context.emit("task", "ingestion_done",
                     f"ingestion of '{path.name}' ended with status '{result.get('status')}'",
                     document=path.name, status=str(result.get("status")))

        status = str(result.get("status", ""))

        if status == _INGESTION_ACCEPTED:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.OK,
                detail="ingestion completed",
                payload={"ingestion": result},
            )

        if status == _INGESTION_NOT_IMPLEMENTED:
            # The pipeline accepted the document but is not fully wired yet:
            # honest failure with the step report for the caller.
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.EXTERNAL,
                detail="ingestion pipeline not fully implemented yet "
                       f"({result.get('failed_step', 'unknown step')})",
                payload={"ingestion": result},
            )

        if status == _INGESTION_CONFIG_ERROR:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.CONFIG,
                detail=str(result.get("message", "ingestion configuration error")),
                payload={"ingestion": result},
            )

        # rejected / unknown
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.FAILED,
            failure_domain=FailureDomain.INPUT_DATA,
            detail=str(result.get("reason", result.get("failure_detail", "ingestion rejected"))),
            payload={"ingestion": result},
        )

    def validate(
        self, context: RoutingContext, request: object
    ) -> Optional[AgentResult]:
        """Nothing specific to check beyond ``run``'s own outcome."""
        return None

    # -- internals ---------------------------------------------------------------

    @staticmethod
    def _extract_document(request: object) -> Any:
        """Pull the bare document name from the request (guarded)."""
        if not hasattr(request, "kind") or not hasattr(request, "document"):
            return AgentResult(
                agent_name=AGENT_NAME,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail="ingestion task expects a UserRequest",
            )
        from src.routing.models import RequestKind

        if request.kind is not RequestKind.INGESTION:  # type: ignore[union-attr]
            return AgentResult(
                agent_name=AGENT_NAME,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail=f"kind '{request.kind}' is not ingestion",  # type: ignore[union-attr]
            )
        return request.document  # type: ignore[union-attr]

    def _run_ingestion(
        self,
        path: Path,
        *,
        force: bool,
        force_summarization: bool = False,
        on_event: Optional[object] = None,
    ) -> Dict[str, Any]:
        """Call the ingestion orchestrator (lazily imported / injectable).

        ``on_event`` is forwarded so the pipeline's internal events (graph
        steps, extraction traces) reach the routing context's observer —
        the thinking panel then shows the whole ingestion flow live.
        """
        if self._runner is None:
            from src.ingestion.ingestion_orchestrator import run_ingestion_file

            self._runner = run_ingestion_file
        kwargs: Dict[str, Any] = {
            "force": force,
            "force_summarization": force_summarization,
        }
        if on_event is not None:
            kwargs["on_event"] = on_event
        return dict(self._runner(path, **kwargs))
