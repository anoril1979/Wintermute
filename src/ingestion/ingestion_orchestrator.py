"""Ingestion orchestrator — the deterministic ingestion entry point.

Since the paradigm change, ingestion is **not reachable from the chat**:
the routing layer no longer accepts ingestion requests, and no LLM
classifies ingestion intent anywhere. The single entry points are the CLI
script (``scripts/ingest.py``) and this module's :func:`run_ingestion_file`
— a strictly deterministic flow:

    0. configuration gate  — config/ingestion.yaml is validated first; a
       malformed yaml ends the run gracefully with a ``config_error``
       status, reported to the operator (and to data/logs/ingestion.log).
    1. file resolution     — the file must exist (the CLI scripts resolve
       inside the documents tree sandbox via src/tools/ingest_tool).
    2. graph execution     — src/graphs/ingestion_graph.IngestionGraph runs
       the ordered agent steps with the caller's flags (force_extraction /
       force_summarization / document_origin). Agents report
       not-implemented steps instead of failing when a step is not wired.

The agents themselves are intentionally NOT implemented here.

Related entry point: corpus removal (``scripts/remove.py``) runs through
src/ingestion/remove_from_corpus.remove_document — also deterministic,
also driven by the CLI, never by the chat.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Union

from src.agents import build_default_agents
from src.agents.contexts import EventCallback, IngestionContext
from src.graphs import IngestionGraph, GraphOutcome
from src.logging_setup import configure_logging
from src.tools.config_loader import (
    ConfigError,
    IngestionConfigError,
    coerce_origin,
    get_valid_origins,
    load_ingestion_config,
)

logger = logging.getLogger(__name__)

# Statuses returned by the orchestrator (stable strings for callers/CLI).
STATUS_ACCEPTED = "accepted"          # validation ok, graph ran
STATUS_REJECTED = "rejected"          # document not found / unusable input
STATUS_NOT_IMPLEMENTED = "not_implemented"  # accepted but agents not built yet
STATUS_CONFIG_ERROR = "config_error"  # ingestion.yaml malformed -> user must fix it


# ---------------------------------------------------------------------------
# Step 0: configuration gate — ingestion.yaml is validated first, always.
# ---------------------------------------------------------------------------

def _config_gate() -> Optional[Dict[str, object]]:
    """Validate config/ingestion.yaml before anything else runs.

    Returns ``None`` when the configuration is valid, otherwise a result
    dict shaped like the orchestrator's other returns, with:

    * ``status = "config_error"``;
    * ``message`` — the explicit ConfigError text, written so the operator
      can fix the yaml;
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
                        "then retry the ingestion.",
        }
    except ConfigError as exc:
        # Missing file / YAML syntax error / unreadable file.
        return {
            "status": STATUS_CONFIG_ERROR,
            "message": str(exc),
            "fix_hint": "Check that config/ingestion.yaml exists and is "
                        "valid YAML, then retry the ingestion.",
        }
    return None


class AgentWiringError(Exception):
    """An agent could not be built (e.g. its LLM role is not configured)."""


def _build_registry_gracefully(
    agents: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """Build the agent registry, surfacing wiring failures as config errors.

    Agents that use an LLM role resolve it STRICTLY at construction (see
    src/agents/llm_roles): a missing/malformed role raises
    ``MissingLLMRoleError`` at wiring time — i.e. here. It is re-raised so
    the entry points can turn it into a graceful ``config_error`` result,
    exactly like a malformed ingestion.yaml.
    """
    try:
        return agents if agents is not None else build_default_agents()
    except ConfigError as exc:  # includes MissingLLMRoleError
        raise AgentWiringError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Direct file ingestion — the one entry point (CLI scripts call this)
# ---------------------------------------------------------------------------

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
    """Ingest a file given by path — trusted local invocation (CLI mode).

    The file must exist; no sandbox is applied for direct calls (the CLI
    scripts resolve references inside the documents tree first).
    ``agents=None`` builds the default registry (implemented agents only).
    ``force=True`` bypasses the extraction checkpoint (re-extract the
    document even if it is already recorded in the job file).
    ``force_summarization=True`` re-runs the LLM summaries (bypasses the
    summarization job file and the summarized store).
    ``origin`` is the document origin — a governance label that MUST be
    one of the user-defined origins (setup.yaml ``documents.origins``);
    ``None`` defaults to the first configured entry (reported as
    unverified by the extraction validation).
    """
    # -- step 0: configuration gate ------------------------------------------
    config_failure = _config_gate()
    if config_failure is not None:
        return {**config_failure, "document": str(file_path)}

    # -- step 0 bis: origin gate ----------------------------------------------
    # The vocabulary is user-defined: an origin outside it is REJECTED here
    # with an explicit fix-hint (naming the configured origins), never
    # silently rewritten — governance metadata must be what the user said.
    if origin is not None:
        normalized = coerce_origin(origin)
        if normalized is None:
            return {
                "status": STATUS_CONFIG_ERROR,
                "message": (
                    f"unknown document origin {origin!r} — not in the "
                    "configured vocabulary (setup.yaml documents.origins: "
                    f"{', '.join(get_valid_origins())})"
                ),
                "fix_hint": (
                    "Use one of the configured origins (see above) or add "
                    "yours to setup.yaml, then retry."
                ),
                "document": str(file_path),
            }
        origin = normalized

    try:
        default_agents = _build_registry_gracefully(agents)
    except AgentWiringError as exc:
        return {
            "status": STATUS_CONFIG_ERROR,
            "message": str(exc),
            "fix_hint": "Add or fix the required LLM role in config/llm.yaml, "
                        "then retry the ingestion.",
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
        # Already normalized by the origin gate above.
        context.metadata["document_origin"] = origin
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

