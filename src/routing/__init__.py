"""Routing layer: turn a raw user prompt into dispatched, structured tasks.

* ``models.py``              — the validated request models (UserRequest,
                               AnalysisResult) + the LLM-text parser;
* ``request_analyzer.py``    — the LLM-backed prompt analyzer;
* ``routing_orchestrator.py``— the entry point the app API calls.
"""

from src.routing.models import (
    AnalysisResult,
    RequestKind,
    UserRequest,
    parse_analysis,
)

__all__ = [
    "AnalysisResult",
    "RequestKind",
    "UserRequest",
    "parse_analysis",
]
