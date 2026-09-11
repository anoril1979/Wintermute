"""Routing orchestrator — the front door of the assistant.

Called by the app API (app/api.py) for every user message, it:

    1. analyzes the raw prompt into an ordered list of structured requests
       (RequestAnalyzer, ``request_analyzer`` LLM role, prompt
       prompts/routing/request_analysis.md);
    2. hands the batch to the routing graph (src/graphs/routing_graph.py),
       which dispatches each request to its task agent:
         retrieval  -> RetrievalTaskAgent  (future retrieval orchestrator)
         ingestion  -> IngestionTaskAgent  (this repo: ingestion orchestrator)
         general    -> GeneralTaskAgent    (out-of-scope answers, in persona);
    3. returns a structured, per-request result list the caller (API or
       CLI) turns into the user-facing reply.

Task agents resolve from an injected registry; ``None`` builds the default
registry (src/agents/routing_registry.build_default_task_agents), which
currently contains the IngestionTaskAgent and the GeneralTaskAgent.
Missing kinds surface as ``not_implemented`` per-request results — the
batch never aborts.

Failure policy: an analysis failure (LLM down, malformed answer) is
reported as a top-level ``analysis_error`` with its cause so the calling
LLM can retry or apologize; a per-request failure is reported in that
request's result and never poisons the rest of the batch.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Dict, List, Optional

from src.agents.contexts import EventCallback, RoutingContext
from src.agents.routing_registry import build_default_task_agents
from src.graphs import RoutingGraph, RoutingOutcome
from src.logging_setup import configure_logging
from src.routing.models import UserRequest
from src.routing.request_analyzer import RequestAnalysisError, RequestAnalyzer
from src.tools.config_loader import ConfigError

logger = logging.getLogger(__name__)

# Top-level statuses of the orchestrator (stable strings for callers/LLM).
STATUS_HANDLED = "handled"              # at least one request reached an agent
STATUS_PARTIAL = "partial"              # analyzed, but nothing reached an agent
STATUS_ANALYSIS_ERROR = "analysis_error"  # prompt could not be analyzed


def run_routing(
    request: str,
    *,
    agents: Optional[Dict[str, object]] = None,
    graph: Optional[RoutingGraph] = None,
    analyzer: Optional[RequestAnalyzer] = None,
    on_event: Optional["EventCallback"] = None,
) -> Dict[str, object]:
    """Full routing flow for one user message.

    Args:
        request:  the raw user prompt.
        agents:   optional task-agent registry (key -> agent); ``None``
                  builds the default registry. Pass an explicit dict
                  (possibly empty) to override.
        graph:    optional pre-built routing graph (tests).
        analyzer: optional pre-built analyzer (tests); a default one is
                  built otherwise.

    Returns:
        A dict with a top-level ``status`` (``handled`` / ``partial`` /
        ``analysis_error``) and ``results``, the ordered per-request list:
        ``{"kind", "utterance", "status", "detail", ...agent payload}``.
        ``traces`` carries the ordered routing events (what the analyzer
        understood, what was dispatched, what agents did) — the API maps
        them onto the ``thinking`` channel. On ``analysis_error`` the dict
        carries ``cause`` (``llm_request`` or ``llm_response``) and
        ``message`` so the calling LLM can decide to retry or to apologize
        to the user.
    """
    context = RoutingContext(request=request, on_event=on_event)

    # -- 1. analysis ---------------------------------------------------------
    analyzer = analyzer or RequestAnalyzer()
    try:
        analysis = analyzer.analyze(request)
    except RequestAnalysisError as exc:
        context.emit("analysis", "analysis_failed", str(exc), cause=exc.cause)
        return {
            "status": STATUS_ANALYSIS_ERROR,
            "cause": exc.cause,
            "message": str(exc),
            "request": request,
            "results": [],
            "traces": list(context.events),
        }

    requests: List[UserRequest] = list(analysis.requests)
    context.metadata["request_count"] = len(requests)
    context.emit(
        "analysis",
        "understood",
        f"{len(requests)} request(s): "
        + "; ".join(f"[{r.kind.value}] {r.utterance}" for r in requests)
        if requests else "no actionable request found in the prompt",
        kinds=[r.kind.value for r in requests],
    )

    # -- 2. dispatch ---------------------------------------------------------
    try:
        task_agents = _resolve_task_agents(agents)
    except ConfigError as exc:  # missing LLM role at agent wiring time
        context.emit("dispatch", "wiring_failed", str(exc))
        return {
            "status": STATUS_ANALYSIS_ERROR,
            "cause": "config",
            "message": str(exc),
            "request": request,
            "results": [],
            "traces": list(context.events),
        }

    graph = graph or RoutingGraph(agents=task_agents)
    outcome: RoutingOutcome = graph.run(context, requests)

    # -- 3. summarize ----------------------------------------------------------
    status = STATUS_HANDLED if outcome.handled else STATUS_PARTIAL
    return {
        "status": status,
        "request": request,
        "request_count": len(requests),
        "results": outcome.as_list(),
        "traces": list(context.events),
    }


def _resolve_task_agents(
    agents: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """``None`` means "use the default registry"; an explicit dict wins."""
    if agents is not None:
        return agents
    return build_default_task_agents()


# ---------------------------------------------------------------------------
# CLI entry point (manual checks; the API is the real caller)
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    """CLI entry point; returns the process exit code."""
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Route a raw user prompt through the routing graph."
    )
    parser.add_argument("prompt", help="The raw user prompt to route")
    args = parser.parse_args(argv)

    result = run_routing(args.prompt)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in (STATUS_HANDLED, STATUS_PARTIAL) else 1


if __name__ == "__main__":
    sys.exit(main())
