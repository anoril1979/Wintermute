"""Shared contexts flowing through the orchestration graphs.

Each graph carries its own context object; they all follow the same shape:
the orchestrator builds the context when a request is accepted and hands it
to each agent in turn. Every agent reads what upstream agents produced and
writes its own outputs under a dedicated key, so steps stay decoupled and
re-runnable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Dedicated logger for trace events: every emit() is mirrored here so the
#: routing/ingestion traces survive the process in the configured log file.
trace_logger = logging.getLogger("wintermute.traces")

#: Shape of a routing trace event: ``{"phase", "kind", "message", "data"}``.
#: ``phase`` locates the emitter (``analysis`` / ``dispatch`` / ``task``),
#: ``kind`` is the free event name (``understood``, ``dispatching``, ...).
RoutingEvent = Dict[str, Any]

#: Observer of routing events. Must never raise (emit() enforces it) and
#: must not retain the event dict after returning if it mutates it.
EventCallback = Callable[[RoutingEvent], None]


@dataclass
class _EventEmitterMixin:
    """Trace-event plumbing shared by every graph context.

    Subclasses get ``events`` (the full trace log) and ``on_event`` (an
    optional live observer) plus ``emit()``: any component holding the
    context (orchestrator, graph, agents) can publish an event; the API
    maps them onto the ``thinking`` channel. The callback failure is
    swallowed (logged) — tracing must never break the pipeline — and the
    event is appended to ``events`` regardless, so the collected traces
    survive even without a live observer.
    """

    events: List[RoutingEvent] = field(default_factory=list)
    on_event: Optional[EventCallback] = None

    def emit(self, phase: str, kind: str, message: str = "", **data: Any) -> None:
        """Publish a trace event to the observer and the event log.

        Every event is also mirrored into the standard logging system
        (``wintermute.traces`` logger, INFO level), so the full trace —
        the same "thinking" content shown live in the API's panel — ends
        up in the configured log file and can be reviewed after the run.
        """
        event: RoutingEvent = {
            "phase": phase,
            "kind": kind,
            "message": message,
            "data": data,
        }
        self.events.append(event)
        trace_logger.info(
            "[%s] %s%s", phase, kind, f" — {message}" if message else ""
        )
        if data:
            trace_logger.debug("[%s] %s data=%r", phase, kind, data)
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception as exc:  # noqa: BLE001 — tracing must not break the pipeline
                logger.warning("Event observer failed: %s", exc)


@dataclass
class IngestionContext(_EventEmitterMixin):
    """State carried through the whole ingestion graph.

    Attributes:
        document_path:   Resolved path of the document being ingested.
        request:         Original request as received by the orchestrator
                         (user utterance or CLI invocation).
        outputs:         Per-step results, keyed by step name. Agents append
                         here; the orchestrator never interprets the payloads.
        errors:          Non-fatal issues collected so far (step name -> message).
        metadata:        Free-form document metadata accumulated along the way
                         (title, page count, ...).
    """

    document_path: Optional[Path] = None
    request: str = ""
    outputs: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingContext(_EventEmitterMixin):
    """State carried through the routing graph for one user message.

    Attributes:
        request:     The raw user prompt as received by the routing
                     orchestrator (from the app API, the CLI, ...).
        outputs:     Per-step results, keyed by step name — the analyzer
                     writes the structured requests here, task agents
                     write their per-request outcomes.
        errors:      Non-fatal issues collected so far (step -> message).
        metadata:    Free-form context (analysis flags, timings, ...).
        results:     Per-request outcomes, in the order the requests were
                     handled — one dict per request with the request
                     summary, the dispatch decision and the agent result;
                     the orchestrator turns this into the user-facing
                     payload.
    """

    request: str = ""
    outputs: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)
