"""Routing layer: turn a raw user prompt into dispatched, structured tasks.

* ``models.py``              — the grouped request models (AnalysisResult
                               with retrieval/general scopes) +
                               the LLM-text parser;
* ``request_analyzer.py``    — the single LLM-backed prompt analyzer;
* ``routing_orchestrator.py``— the entry point the app API calls.

Ingestion is deliberately not a scope here: document ingestion is a CLI
operation (scripts/ingest.py), unreachable from the chat.
"""

from src.routing.models import (
    AnalysisResult,
    GeneralRequest,
    RequestScope,
    RetrievalLookupKind,
    RetrievalRequest,
    max_requests_per_prompt,
    parse_analysis,
)

__all__ = [
    "AnalysisResult",
    "GeneralRequest",
    "RequestScope",
    "RetrievalLookupKind",
    "RetrievalRequest",
    "max_requests_per_prompt",
    "parse_analysis",
]
