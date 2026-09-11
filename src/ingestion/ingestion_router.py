"""LLM-backed ingestion request router.

Replaces the keyword validator (``validate_request`` / ``force_requested``)
as the identification layer of the ingestion orchestrator. The division of
labor is strict:

* **Python owns the facts** — job files and stores are read here, never
  guessed by the LLM: extraction job file, canonical extracted JSON,
  summarization job file, content fingerprint match;
* **the LLM owns the intent** — reading the *user's words* and classifying
  the utterance (first ingestion / force / redo summaries / not an
  ingestion order), via the ``ingestion_router`` role;
* **Python owns the decision table** — a pure function of (facts, intent):
  it produces the orchestrator flags (``force_extraction``,
  ``force_summarization``), a proceed decision, or a clarification need;
* **the LLM writes the clarification wording** when asked — grounded in
  the actual facts. A deterministic fallback (no LLM) exists so the
  no-history limitation never produces an unanswerable yes/no prompt: the
  user gets an explanation plus suggested phrasings for the next message.

When the LLM classifies the utterance as not-an-ingestion-order, the
router does not decide alone: it asks the user (``needs_clarification``)
— per the design, an ambiguous request goes back to the user instead of a
silent keyword rejection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.llm_roles import require_llm_role
from src.ingestion.models import ClarificationKind, IngestionIntent, parse_intent
from src.tools.extraction_job_file import (
    STATUS_ALREADY_DONE,
    STATUS_NEW,
    STATUS_STALE,
    ExtractionJobFile,
    SummarizationJobFile,
)

logger = logging.getLogger(__name__)

#: Where the prompts live (loaded lazily, once): one for intent
#: classification, one for clarification wording.
INTENT_PROMPT_PATH = Path("prompts/routing/ingestion_intent.md")
CLARIFICATION_PROMPT_PATH = Path("prompts/routing/ingestion_clarification.md")

# Statuses the router (and the orchestrator on its behalf) can return.
ROUTER_PROCEED = "proceed"                    # flags are final, run the graph
ROUTER_NEEDS_CLARIFICATION = "needs_clarification"  # ask the user
ROUTER_REJECTED = "rejected"                  # not an ingestion order, hopeless

#: Standard phrasings offered when the request itself is the problem.
_DEFAULT_SUGGESTIONS = [
    '"ingest <file>.pdf"',
    '"re-ingest <file>.pdf" (force re-extraction)',
    '"re-ingest <file>.pdf and redo the summaries"',
]


# ---------------------------------------------------------------------------
# Facts — everything Python knows, gathered before any decision
# ---------------------------------------------------------------------------

@dataclass
class IngestionFacts:
    """Store state for one document, gathered by Python (never the LLM)."""

    file_name: str
    source_path: Optional[Path] = None     # resolved inside the documents tree
    found: bool = False                    # document exists in the documents tree
    extraction_job: str = STATUS_NEW       # already_done / stale / new
    canonical_json: bool = False           # data/extracted/<stem>.json exists
    summarization_job: str = STATUS_NEW
    summarized_json: bool = False          # data/summarized/<stem>.json exists
    summaries_stale: Optional[bool] = None  # fingerprint match known (None: unknown)

    def summary(self) -> Dict[str, Any]:
        """Compact dict for payloads and traces."""
        return {
            "file_name": self.file_name,
            "found": self.found,
            "extraction_job": self.extraction_job,
            "canonical_json": self.canonical_json,
            "summarization_job": self.summarization_job,
            "summarized_json": self.summarized_json,
            "summaries_stale": self.summaries_stale,
        }


# ---------------------------------------------------------------------------
# Decision table (pure function of facts + intent)
# ---------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    """Outcome of the decision table."""

    status: str                            # proceed / needs_clarification / rejected
    flags: Dict[str, bool] = field(default_factory=dict)
    explanation: str = ""
    question: str = ""                     # clarification wording (LLM or fallback)
    suggestions: List[str] = field(default_factory=list)
    clarification: Optional[ClarificationKind] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "flags": dict(self.flags),
            "explanation": self.explanation,
            "question": self.question,
            "suggestions": list(self.suggestions),
            "clarification": self.clarification.value if self.clarification else None,
        }


def apply_decision_table(
    facts: IngestionFacts, intent: IngestionIntent
) -> RoutingDecision:
    """Turn (facts, intent) into flags, a proceed decision or a clarification.

    Pure and deterministic: same facts + same intent → same decision. The
    LLM classified the *utterance*; this table applies the *state*:

    * the LLM classified the utterance as not-an-ingestion-order →
      clarify: the router never silently guesses a document from the raw
      text when the intent itself is invalid;
    * unknown document, LLM claimed a valid intent → clarify (not found);
    * forced extraction implies redoing the summaries — the extraction is
      new, the stored summaries are obsolete (the fingerprint would flag
      them stale anyway);
    * re-ingestion with summaries that no longer match the extraction
      (fingerprint stale) → they are obsolete: redo them;
    * summarization checkpoint / summarized store present but the canonical
      extraction is gone (user cleaned data/extracted by hand) → the
      summarization state is orphaned: clarify.
    """
    flags: Dict[str, bool] = {
        "force_extraction": bool(intent.force),
        "force_summarization": False,
    }

    # -- the utterance itself is not an ingestion order -------------------------
    if not intent.valid:
        return RoutingDecision(
            status=ROUTER_NEEDS_CLARIFICATION,
            clarification=intent.clarification or ClarificationKind.REQUEST_UNCLEAR,
            explanation=(
                intent.reason
                or "the request is not a recognizable ingestion order"
            ),
            suggestions=list(_DEFAULT_SUGGESTIONS),
        )

    # -- unknown document ----------------------------------------------------
    if intent.valid and not facts.found:
        return RoutingDecision(
            status=ROUTER_NEEDS_CLARIFICATION,
            clarification=ClarificationKind.DOCUMENT_NOT_FOUND,
            explanation=(
                f"no document named '{facts.file_name}' exists in the "
                "documents tree"
            ),
            suggestions=[
                "list the available documents",
                "check the file name and ask again",
            ],
        )

    # -- forced extraction: summaries become obsolete --------------------------
    if intent.force:
        flags["force_summarization"] = True

    if intent.redo_summaries:
        flags["force_summarization"] = True

    # -- state conflicts the user must arbitrate -------------------------------
    summaries_orphaned = (
        facts.summarized_json
        and facts.summarization_job != STATUS_NEW
        and not facts.canonical_json
    )
    if summaries_orphaned and not flags["force_extraction"]:
        return RoutingDecision(
            status=ROUTER_NEEDS_CLARIFICATION,
            clarification=ClarificationKind.STATE_CONFLICT,
            explanation=(
                f"a summarization checkpoint exists for '{facts.file_name}' "
                "but its extracted content is gone — the summaries are "
                "orphaned and it is unclear whether to rebuild everything "
                "or drop the old summaries"
            ),
            suggestions=[
                f"\"re-ingest {facts.file_name} and redo the summaries\"",
                f"\"ingest {facts.file_name}\" (fresh extraction)",
            ],
        )

    # Fingerprint-stale summaries on a plain re-ingestion: the summaries no
    # longer match the (user-fixed) canonical content — redo them.
    if (
        not flags["force_extraction"]
        and flags.get("force_summarization") is False
        and facts.summaries_stale
    ):
        flags["force_summarization"] = True

    return RoutingDecision(
        status=ROUTER_PROCEED,
        flags=flags,
        explanation="state and intent are consistent; proceeding",
    )


# ---------------------------------------------------------------------------
# Facts gathering
# ---------------------------------------------------------------------------

def gather_facts(
    file_name: str,
    *,
    extraction_jobs: Optional[ExtractionJobFile] = None,
    summarization_jobs: Optional[SummarizationJobFile] = None,
) -> IngestionFacts:
    """Read every store that bears on the ingestion decision.

    Fail-open by design: an unreadable store reports its default (new /
    absent) so the router can still decide — the same philosophy as the
    stores themselves.
    """
    facts = IngestionFacts(file_name=file_name)

    # Document existence in the documents tree (sandboxed resolution).
    try:
        from src.tools.ingest_tool import ingest_document

        resolution = ingest_document(file_name)
        if resolution.get("status") in ("ready", "ingested"):
            facts.found = True
            facts.source_path = Path(str(resolution["path"]))
    except Exception as exc:  # noqa: BLE001 — fail-open: the fact is "not found"
        logger.warning("Document resolution failed for '%s': %s", file_name, exc)

    # Extraction checkpoint + canonical store.
    if facts.source_path is not None:
        stem = facts.source_path.stem
        try:
            facts.extraction_job = ExtractionJobFile().status_of(file_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Extraction job lookup failed: %s", exc)
        try:
            from src.helpers.document_extract_json_store import canonical_path_for

            facts.canonical_json = canonical_path_for(facts.source_path).exists()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Canonical JSON lookup failed: %s", exc)

        # Summarization checkpoint + store + fingerprint staleness.
        try:
            facts.summarization_job = SummarizationJobFile().status_of(file_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Summarization job lookup failed: %s", exc)
        try:
            from src.summarization.summarized_store import (
                load_summarized,
                summarized_path_for,
            )

            summarized_path = summarized_path_for(facts.source_path)
            facts.summarized_json = summarized_path.exists()
            if facts.summarized_json and facts.canonical_json:
                # Fingerprint match between the stored summaries and the
                # current canonical content (staleness oracle).
                from src.helpers.document_extract_json_store import load_extract
                from src.summarization.summarized_store import content_fingerprint

                _, stored_fingerprint = load_summarized(summarized_path)
                current = content_fingerprint(load_extract(
                    canonical_path_for(facts.source_path)
                ))
                facts.summaries_stale = stored_fingerprint != current
        except Exception as exc:  # noqa: BLE001 — advisory fact only
            logger.warning("Summarized store lookup failed: %s", exc)
            facts.summaries_stale = None

    return facts


# ---------------------------------------------------------------------------
# The router
# ---------------------------------------------------------------------------

class IngestionRouter:
    """Identifies and validates an ingestion request; decides or clarifies."""

    def __init__(
        self,
        llm_role: str = "ingestion_router",
        *,
        intent_prompt_path: Optional[Path] = None,
        clarification_prompt_path: Optional[Path] = None,
        extraction_jobs: Optional[ExtractionJobFile] = None,
        summarization_jobs: Optional[SummarizationJobFile] = None,
    ) -> None:
        self._llm_role = llm_role
        self._intent_prompt_path = intent_prompt_path or INTENT_PROMPT_PATH
        self._clarification_prompt_path = (
            clarification_prompt_path or CLARIFICATION_PROMPT_PATH
        )
        self._intent_template: Optional[str] = None
        self._clarification_template: Optional[str] = None
        self._extraction_jobs = extraction_jobs
        self._summarization_jobs = summarization_jobs
        # Strict role resolution at construction (fail fast at wiring time,
        # same philosophy as LLMRoleAgent): a missing/malformed llm.yaml role
        # is a config error. A runtime LLM failure is a different thing —
        # the intent identification falls back to keywords then.
        self.llm_config: dict = require_llm_role(self._llm_role)

    # -- prompts / llm ---------------------------------------------------------

    def _intent_template_text(self) -> str:
        if self._intent_template is None:
            self._intent_template = self._intent_prompt_path.read_text(encoding="utf-8")
        return self._intent_template

    def _clarification_template_text(self) -> str:
        if self._clarification_template is None:
            self._clarification_template = self._clarification_prompt_path.read_text(
                encoding="utf-8"
            )
        return self._clarification_template

    def _build_intent_prompt(self, request: str) -> str:
        return (
            f"{self._intent_template_text().strip()}\n"
            "\n---\n\n"
            "User request to analyze:\n"
            "<<<<PROMPT>>>>\n"
            f"{request}\n"
            "<<<<PROMPT>>>>\n"
        )

    def _build_clarification_prompt(
        self, request: str, facts: IngestionFacts, decision: RoutingDecision
    ) -> str:
        import json as _json

        return (
            f"{self._clarification_template_text().strip()}\n"
            "\n---\n\n"
            "User request:\n"
            "<<<<PROMPT>>>>\n"
            f"{request}\n"
            "<<<<PROMPT>>>>\n\n"
            "System state (facts gathered from the ingestion stores):\n"
            "<<<<FACTS>>>>\n"
            f"{_json.dumps(facts.summary(), ensure_ascii=False, indent=2)}\n"
            "<<<<FACTS>>>>\n\n"
            "What is unclear (internal analysis):\n"
            "<<<<CONFLICT>>>>\n"
            f"{decision.explanation}\n"
            "<<<<CONFLICT>>>>\n"
        )

    def _llm(self):
        from src.llm.llm_client_ollama import get_llm_client

        return get_llm_client(self._llm_role)

    # -- public API --------------------------------------------------------------

    def route(self, request: str) -> RoutingDecision:
        """Full flow: facts → intent → decision table → (maybe) clarification.

        Never raises for expected conditions: LLM problems fall back to
        keyword heuristics (documented, deterministic), and every outcome
        is a :class:`RoutingDecision`.
        """
        # -- 1. LLM intent identification ----------------------------------------
        intent, llm_ok = self._identify_intent(request)

        # -- 2. document + facts -------------------------------------------------
        document = intent.document or self._fallback_document(request)
        if document is None:
            return self._clarify_request_unclear(request, intent,
                                                 llm_used=llm_ok)
        facts = gather_facts(
            document,
            extraction_jobs=self._extraction_jobs,
            summarization_jobs=self._summarization_jobs,
        )

        # -- 3. decision table ---------------------------------------------------
        decision = apply_decision_table(facts, intent)
        if decision.status == ROUTER_PROCEED:
            decision.explanation = (
                f"intent={intent.summary()['reason'] or 'classified'}, "
                f"document='{facts.file_name}' (found={facts.found})"
            )
            return decision

        # -- 4. clarification wording (LLM-written, fallback deterministic) ------
        return self._word_clarification(request, facts, decision, llm_used=llm_ok)

    # -- internals -----------------------------------------------------------------

    def _identify_intent(self, request: str) -> tuple:
        """Classify the utterance; falls back to keywords when the LLM fails.

        Returns ``(intent, llm_ok)``. The fallback never invents an
        intent: it produces either a valid bare-document intent or a
        ``REQUEST_UNCLEAR`` clarification — the user is asked, never a
        silent guess.
        """
        try:
            raw = self._llm().complete(prompt=self._build_intent_prompt(request))
            return parse_intent(raw), True
        except Exception as exc:  # noqa: BLE001 — transport or malformed answer
            logger.warning("Ingestion router LLM call failed (%s); "
                           "falling back to keyword heuristics", exc)
            return self._keyword_intent(request), False

    def _keyword_intent(self, request: str) -> IngestionIntent:
        """Deterministic fallback classification (no LLM available)."""
        text = (request or "").strip().lower()
        force = any(k in text for k in (
            "force", "reingest", "re-ingest", "re-extract", "reextract",
            "reindex", "reload",
        ))
        redo = any(k in text for k in (
            "re-summarize", "resummarize", "redo the summar", "force summar",
            "refais le résumé", "refaire le résumé",
        ))
        document = self._fallback_document(request)
        if document is None:
            return IngestionIntent(
                valid=False,
                clarification=ClarificationKind.REQUEST_UNCLEAR,
                reason="no document reference found in the request",
            )
        return IngestionIntent(
            valid=True,
            document=document,
            force=force,
            redo_summaries=redo,
            reason="classified by keyword fallback (LLM unavailable)",
        )

    @staticmethod
    def _fallback_document(request: str) -> Optional[str]:
        """Extract a file reference from the raw request (quoted first)."""
        import re

        quoted = re.compile(r"[\"']([^\"']+?\.[A-Za-z0-9]{1,6})[\"']")
        bare = re.compile(r"([\w\-.]+?\.[A-Za-z0-9]{1,6})")
        match = quoted.search(request) or bare.search(request)
        return match.group(1).strip() if match else None

    def _clarify_request_unclear(
        self, request: str, intent: IngestionIntent, *, llm_used: bool
    ) -> RoutingDecision:
        """The request is not an ingestion order (or names no document)."""
        decision = RoutingDecision(
            status=ROUTER_NEEDS_CLARIFICATION,
            clarification=ClarificationKind.REQUEST_UNCLEAR,
            explanation=intent.reason or "the request is not a recognizable ingestion order",
            suggestions=list(_DEFAULT_SUGGESTIONS),
        )
        return self._word_clarification(request, None, decision, llm_used=llm_used)

    def _word_clarification(
        self,
        request: str,
        facts: Optional[IngestionFacts],
        decision: RoutingDecision,
        *,
        llm_used: bool,
    ) -> RoutingDecision:
        """Let the LLM phrase the clarification; fall back to deterministic text."""
        if llm_used and facts is not None:
            try:
                raw = self._llm().complete(
                    prompt=self._build_clarification_prompt(request, facts, decision)
                )
                text = (raw or "").strip()
                if text:
                    decision.question = text
                    return decision
            except Exception as exc:  # noqa: BLE001 — wording is best-effort
                logger.warning("Clarification LLM call failed (%s); "
                               "using the deterministic wording", exc)
        decision.question = (
            f"{decision.explanation}.\n\n"
            "Could you rephrase? For example:\n- "
            + "\n- ".join(decision.suggestions)
        )
        return decision
