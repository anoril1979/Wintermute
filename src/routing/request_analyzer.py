"""LLM-backed request analyzer — the ONE analysis of the front door.

Calls the ``request_analyzer`` role (config/llm.yaml) with the analysis
prompt (prompts/routing/request_analysis.md) plus the raw user prompt, and
validates the answer into a grouped :class:`AnalysisResult`
(src/routing/models.py): ``{"retrieval": [...], "general": [...]}`` —
with retrieval lookups already fully classified (``lookup_kind``) and
self-contained questions.

**One analysis, no re-routing**: pronouns are resolved here, at analysis
time — nothing downstream re-reads the user's words. The retrieval
pipeline is deterministic Python; the only LLM-based post-routing worker
is the GeneralTaskAgent (and later the AnswerAgent). Ingestion is not a
scope: a prompt that asks for ingestion in conversation yields a
``general`` request (ingestion is a CLI operation, scripts/ingest.py).

A failure here — Ollama unreachable, malformed JSON answer,
schema-violating requests — raises :class:`RequestAnalysisError`, which
the orchestrator reports as a retryable/agent-shaped failure instead of
crashing the API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.routing.models import AnalysisResult, parse_analysis
from src.llm.guard import prompt_is_meta

logger = logging.getLogger(__name__)

#: Where the analysis prompt lives (loaded lazily, once).
ANALYSIS_PROMPT_PATH = Path("prompts/routing/request_analysis.md")


# Bound for logged prompt/answer snippets — long enough to diagnose,
# short enough to keep one log line one line.
_LOG_SNIPPET_LIMIT = 500


def _log_snippet(text: str) -> str:
    """Flattened, bounded text for log lines (fix C observability)."""
    return " ".join(text.split())[:_LOG_SNIPPET_LIMIT]


class RequestAnalysisError(Exception):
    """The user prompt could not be analyzed into structured requests.

    ``cause`` distinguishes the two retryable families: ``llm_request``
    (Ollama unreachable / timeout) and ``llm_response`` (malformed or
    schema-violating answer).
    """

    def __init__(self, message: str, cause: str) -> None:
        super().__init__(message)
        self.cause = cause


class RequestAnalyzer:
    """Explodes a raw user prompt into validated, grouped-scope requests."""

    def __init__(self, llm_role: str = "request_analyzer") -> None:
        self._llm_role = llm_role
        self._prompt_template: Optional[str] = None

    # -- internals -------------------------------------------------------------

    def _prompt_template_text(self) -> str:
        """Load (and cache) the analysis prompt."""
        if self._prompt_template is None:
            self._prompt_template = ANALYSIS_PROMPT_PATH.read_text(encoding="utf-8")
        return self._prompt_template

    def _build_prompt(self, user_prompt: str) -> str:
        """Full prompt: instructions + the raw user prompt, delimited."""
        return (
            f"{self._prompt_template_text().strip()}\n\n"
            "---\n\n"
            "User prompt to analyze:\n"
            "<<<<PROMPT>>>>\n"
            f"{user_prompt}\n"
            "<<<<PROMPT>>>>\n"
        )

    def _llm(self):
        """Shared client for the analyzer role (built lazily)."""
        from src.llm.llm_client_ollama import get_llm_client

        return get_llm_client(self._llm_role)

    # -- public API --------------------------------------------------------------

    def analyze(self, user_prompt: str) -> AnalysisResult:
        """Analyze a raw user prompt into a grouped AnalysisResult.

        Raises:
            RequestAnalysisError: the prompt could not be analyzed —
                ``cause='llm_request'`` when the LLM could not be reached,
                ``cause='llm_response'`` when the answer was unusable.
        """
        if not user_prompt or not user_prompt.strip():
            # Nothing to analyze: an empty batch, not an error.
            return AnalysisResult()

        if prompt_is_meta(user_prompt):
            # Front-end auxiliary task (title/tags/follow-ups) that slipped
            # past the API guard: refuse it here rather than produce a
            # garbage analysis — last line of defense.
            logger.warning("Meta/background prompt refused at the analyzer boundary.")
            return AnalysisResult()

        prompt = self._build_prompt(user_prompt)
        logger.info(
            "Analyzing prompt (%d chars): %s",
            len(user_prompt), _log_snippet(user_prompt),
        )

        try:
            # No explicit budget: the generation ceiling is the role's
            # ``max_response_tokens`` in config/llm.yaml (single source of
            # truth — tune the analyzer's verbosity there).
            raw = self._llm().complete(prompt=prompt)
        except Exception as exc:  # LLMClientError / ValueError / transport
            logger.warning("Analyzer LLM call failed: %s", exc)
            raise RequestAnalysisError(
                f"request analysis failed: the LLM could not be reached ({exc})",
                cause="llm_request",
            ) from exc

        logger.info(
            "Analyzer raw answer (%d chars): %s", len(raw), _log_snippet(raw)
        )

        try:
            result = parse_analysis(raw)
        except ValueError as exc:
            logger.warning("Analyzer returned an unusable answer: %s", exc)
            raise RequestAnalysisError(
                f"request analysis failed: unusable LLM answer ({exc})",
                cause="llm_response",
            ) from exc

        logger.info(
            "Analyzed prompt into %d request(s) "
            "(retrieval=%d, general=%d).",
            result.request_count,
            len(result.retrieval),
            len(result.general),
        )
        return result
