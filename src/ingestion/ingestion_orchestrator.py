"""Ingestion orchestrator — entry point of the ingestion workflow.

Two ways in, per the project design:

* a request from the user, relayed through an LLM call (e.g. the
  ``ingest_document`` tool in src/tools), as a free-text utterance like
  ``"Please ingest the new 'meow.pdf'"``;
* a direct script invocation with the file to ingest as a parameter
  (modeled on the old ``src/ingestion/old/rag_pipeline.py`` CLI):
  ``python -m src.ingestion.ingestion_orchestrator -i path/to/file.pdf``.

Flow:

    0. configuration gate  — config/ingestion.yaml is validated first; a
       malformed yaml ends the run gracefully with a ``config_error``
       status whose message can be forwarded to the calling LLM/user.
    1. request reception
    2. request routing     — the LLM-backed ingestion router
       (``ingestion_router`` role, src/ingestion/ingestion_router.py +
       src/graphs/ingestion_routing_graph.py) identifies the intent, gathers
       the store facts (extraction job, canonical JSON, summarization job,
       fingerprint) and applies the decision table. It either returns the
       step flags (force_extraction / force_summarization) or asks the user
       for clarification — the orchestrator then ends the run with a
       ``needs_clarification`` status carrying the question, so the calling
       LLM can relay it.
    3. graph execution     — src/graphs/ingestion_graph.IngestionGraph runs
       the ordered agent steps with the router's flags. Agents are
       placeholders (protocols only, see src/agents); the graph reports
       them as not-implemented instead of failing.

The agents themselves are intentionally NOT implemented here.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Union

from src.agents import build_default_agents
from src.agents.contexts import EventCallback, IngestionContext
from src.graphs import IngestionGraph, GraphOutcome
from src.logging_setup import configure_logging
from src.graphs.ingestion_routing_graph import (
    ROUTER_PROCEED,
    IngestionRoutingGraph,
)
from src.tools.config_loader import (
    ConfigError,
    IngestionConfigError,
    load_ingestion_config,
)

logger = logging.getLogger(__name__)

# Statuses returned by the orchestrator (stable strings for callers/LLM).
STATUS_ACCEPTED = "accepted"          # validation ok, graph scheduled/run
STATUS_REJECTED = "rejected"          # request not understood or document not found
STATUS_NOT_IMPLEMENTED = "not_implemented"  # accepted but agents not built yet
STATUS_CONFIG_ERROR = "config_error"  # ingestion.yaml malformed -> user must fix it
STATUS_NEEDS_CLARIFICATION = "needs_clarification"  # ambiguous: question for the user


# ---------------------------------------------------------------------------
# Step 0: configuration gate — ingestion.yaml is validated first, always.
# ---------------------------------------------------------------------------

def _config_gate() -> Optional[Dict[str, object]]:
    """Validate config/ingestion.yaml before anything else runs.

    Returns ``None`` when the configuration is valid, otherwise a result
    dict shaped like the orchestrator's other returns, with:

    * ``status = "config_error"``;
    * ``message`` — the explicit ConfigError text, written so the calling
      LLM can forward it verbatim to the user ("your ingestion.yaml is
      broken at entry X because Y") and the user can fix the yaml;
    * ``fix_hint`` — a short, actionable correction hint.

    The flow never touches the filesystem with a half-validated config:
    a malformed yaml ends the run gracefully here, at the very first step.
    """
    try:
        load_ingestion_config()
    except IngestionConfigError as exc:
        return {
            "status": STATUS_CONFIG_ERROR,
            "message": str(exc),
            "fix_hint": "Fix the reported entry in config/ingestion.yaml, "
                        "then retry the ingestion request.",
        }
    except ConfigError as exc:
        # Missing file / YAML syntax error / unreadable file.
        return {
            "status": STATUS_CONFIG_ERROR,
            "message": str(exc),
            "fix_hint": "Check that config/ingestion.yaml exists and is "
                        "valid YAML, then retry the ingestion request.",
        }
    return None


# ---------------------------------------------------------------------------
# Step 2: LLM-backed routing (see src/ingestion/ingestion_router.py and
# src/graphs/ingestion_routing_graph.py) → graph execution
# ---------------------------------------------------------------------------

def run_ingestion(
    request: str,
    *,
    agents: Optional[Dict[str, object]] = None,
    graph: Optional[IngestionGraph] = None,
    routing_graph: Optional["IngestionRoutingGraph"] = None,
    on_event: Optional["EventCallback"] = None,
) -> Dict[str, object]:
    """Full orchestrator flow for a free-text request.

    Args:
        request:      The user request, e.g. ``"Please ingest the new 'meow.pdf'"``.
        agents:       Optional agent registry passed through to the graph; ``None``
                      (default) builds the registry of implemented agents
                      (``src/agents/registry.build_default_agents``). Pass an
                      explicit dict (possibly empty) to override.
        graph:        Optional pre-built ingestion graph (tests); a default one is
                      built otherwise.
        routing_graph: Optional pre-built ingestion *routing* graph (tests); a
                      default one is built otherwise. It decides whether the
                      request proceeds (and with which flags) or needs
                      clarification.

    Returns:
        A dict with ``status`` (``accepted`` / ``rejected`` /
        ``not_implemented`` / ``config_error`` / ``needs_clarification``),
        the resolved ``path`` when found, and the graph ``outcome`` summary
        when the graph ran. On ``config_error`` the dict carries ``message``
        + ``fix_hint`` for the calling LLM to relay to the user. On
        ``needs_clarification`` it carries ``question`` (the wording, LLM-
        or fallback-written), ``explanation`` and ``suggestions`` — the
        calling LLM forwards them; the user rephrases as a complete order.
    """
    # -- step 0: configuration gate ------------------------------------------
    config_failure = _config_gate()
    if config_failure is not None:
        return {**config_failure, "request": request}

    try:
        default_agents = _build_registry_gracefully(agents)
        # The routing graph resolves its LLM role strictly at construction:
        # a missing 'ingestion_router' role is a wiring-time config error.
        router = routing_graph or IngestionRoutingGraph()
    except AgentWiringError as exc:
        return {
            "status": STATUS_CONFIG_ERROR,
            "message": str(exc),
            "fix_hint": "Add or fix the required LLM role in config/llm.yaml, "
                        "then retry the ingestion request.",
            "request": request,
        }

    # -- step 2: LLM-backed routing (facts → intent → decision) ---------------
    routing = router.run(request, on_event=on_event)

    if routing.status != ROUTER_PROCEED:
        # needs_clarification (or rejected): everything the user needs to
        # rephrase is in the decision. Never a bare yes/no question — the
        # router's wording carries the explanation + suggested phrasings.
        decision = routing.decision
        return {
            "status": STATUS_NEEDS_CLARIFICATION,
            "request": request,
            "explanation": decision.explanation if decision else "",
            "question": decision.question if decision else "",
            "suggestions": list(decision.suggestions) if decision else [],
            "facts": routing.facts.summary() if routing.facts else None,
            "routing_traces": routing.traces,
            "traces": [],
        }

    # -- step 3: graph execution with the router's flags ----------------------
    path = routing.facts.source_path
    if path is None:  # defensive: proceed implies a resolved document
        return {
            "status": STATUS_REJECTED,
            "reason": "routing proceeded without a resolved document",
            "request": request,
            "traces": [],
        }

    context = IngestionContext(document_path=path, request=request, on_event=on_event)
    context.metadata.update(routing.decision.flags)
    graph = graph or IngestionGraph(agents=default_agents)
    outcome: GraphOutcome = graph.run(context)

    if outcome.accepted:
        status = STATUS_ACCEPTED
    elif outcome.not_implemented_steps:
        status = STATUS_NOT_IMPLEMENTED
    else:
        status = STATUS_REJECTED

    return {
        "status": status,
        "request": request,
        "document": routing.facts.file_name,
        "path": str(path),
        "flags": dict(routing.decision.flags),
        "completed_steps": outcome.completed_steps,
        "skipped_steps": outcome.skipped_steps,
        "not_implemented_steps": outcome.not_implemented_steps,
        "failed_step": outcome.failed_step,
        "failure_detail": outcome.failure_detail,
        "traces": list(context.events),
    }


def _resolve_agents(
    agents: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """``None`` means "use the default registry"; an explicit dict wins."""
    return agents if agents is not None else build_default_agents()


class AgentWiringError(Exception):
    """An agent could not be built (e.g. its LLM role is not configured)."""


def _build_registry_gracefully(
    agents: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """Build the agent registry, surfacing wiring failures as config errors.

    Agents that use an LLM role resolve it STRICTLY at construction (see
    src/agents/llm_roles): a missing/malformed role raises
    ``MissingLLMRoleError`` at wiring time — i.e. here. It is re-raised so
    the entry points can turn it into a graceful ``config_error`` result
    for the calling LLM/user, exactly like a malformed ingestion.yaml.
    """
    try:
        return _resolve_agents(agents)
    except ConfigError as exc:  # includes MissingLLMRoleError
        raise AgentWiringError(str(exc)) from exc


def run_ingestion_file(
    file_path: Union[str, Path],
    *,
    agents: Optional[Dict[str, object]] = None,
    graph: Optional[IngestionGraph] = None,
    force: bool = False,
    force_summarization: bool = False,
    origin: Optional[str] = None,
    on_event: Optional["EventCallback"] = None,
) -> Dict[str, object]:
    """Direct-entry variant: ingest a file given by path, skipping request
    validation (CLI mode is a trusted local invocation, like the old
    rag_pipeline CLI).

    The file must still exist; no sandbox is applied for direct calls.
    ``agents=None`` builds the default registry (implemented agents only).
    ``force=True`` bypasses the extraction checkpoint (re-extract the
    document even if it is already recorded in the job file).
    ``force_summarization=True`` re-runs the LLM summaries (bypasses the
    summarization job file and the summarized store).
    ``origin`` is the document origin ("canon" / "community" / "rpg") —
    governance metadata decided before storage; ``None`` defaults to canon
    (reported as unverified by the extraction validation).
    """
    # -- step 0: configuration gate (same contract as run_ingestion) ---------
    config_failure = _config_gate()
    if config_failure is not None:
        return {**config_failure, "document": str(file_path)}

    try:
        default_agents = _build_registry_gracefully(agents)
    except AgentWiringError as exc:
        return {
            "status": STATUS_CONFIG_ERROR,
            "message": str(exc),
            "fix_hint": "Add or fix the required LLM role in config/llm.yaml, "
                        "then retry the ingestion request.",
            "document": str(file_path),
        }

    path = Path(file_path)
    if not path.is_file():
        return {
            "status": STATUS_REJECTED,
            "reason": f"file not found: {path}",
            "document": str(path),
        }

    context = IngestionContext(
        document_path=path, request=f"[cli] {path}", on_event=on_event
    )
    context.metadata["force_extraction"] = bool(force)
    context.metadata["force_summarization"] = bool(force_summarization)
    if origin is not None:
        # The extraction agent validates the value; an invalid one is
        # reported there (origin_warning) and defaults to canon.
        context.metadata["document_origin"] = str(origin).strip().lower()
    graph = graph or IngestionGraph(agents=default_agents)
    outcome = graph.run(context)

    status = (
        STATUS_ACCEPTED
        if outcome.accepted
        else STATUS_NOT_IMPLEMENTED
        if outcome.not_implemented_steps
        else STATUS_REJECTED
    )
    return {
        "status": status,
        "document": str(path),
        "path": str(path),
        "completed_steps": outcome.completed_steps,
        "skipped_steps": outcome.skipped_steps,
        "not_implemented_steps": outcome.not_implemented_steps,
        "failed_step": outcome.failed_step,
        "failure_detail": outcome.failure_detail,
        "traces": list(context.events),
    }


# ---------------------------------------------------------------------------
# CLI entry point (modeled on the old rag_pipeline __main__)
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    """CLI entry point; returns the process exit code."""
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Ingest a document through the ingestion graph."
    )
    parser.add_argument(
        "-i", "--input", dest="input_file", required=True,
        help="The file to ingest", metavar="FILE",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-extraction (bypass the extraction job file)",
    )
    parser.add_argument(
        "--force-summarization", action="store_true",
        help="Force re-summarization (bypass the summarization checkpoints)",
    )
    parser.add_argument(
        "--origin", choices=["canon", "community", "rpg"], default=None,
        help="Document origin: canon (official), community (fan-made) or "
             "rpg (user-created). Default: canon (unverified).",
    )
    args = parser.parse_args(argv)

    result = run_ingestion_file(
        args.input_file,
        force=args.force,
        force_summarization=args.force_summarization,
        origin=args.origin,
    )
    if result["status"] == STATUS_CONFIG_ERROR:
        # Graceful, human-readable report instead of a traceback: the user
        # (or their LLM) must fix config/ingestion.yaml.
        print(f"[CONFIG ERROR] {result['message']}")
        print(f"Hint: {result['fix_hint']}")
        return 2
    print(result)
    return 0 if result["status"] in (STATUS_ACCEPTED, STATUS_NOT_IMPLEMENTED) else 1


if __name__ == "__main__":
    sys.exit(main())
