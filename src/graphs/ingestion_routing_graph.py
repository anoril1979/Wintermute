"""Ingestion routing graph: facts → intent → decision → graph or clarify.

The graph form of the router flow (src/ingestion/ingestion_router.py), so
the ingestion orchestrator's front door is inspectable and traceable like
every other graph in the system:

    gather facts (Python: job files, stores, fingerprint)
        → identify intent (LLM: ingestion_router role)
        → apply the decision table (pure Python)
        → proceed        → hand the flags to the caller (graph execution)
        → clarify/reject → return the explanation + question to the caller

The graph carries no logic of its own beyond tracing: every step delegates
to the router module's tested functions. It exists so the flow can be
observed in the thinking panel (``facts_gathered``, ``intent_identified``,
``decision``, ``clarification`` events) and so a future interactive
clarification loop (context history) can replace the terminal clarify node
with a question node without touching the router.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.ingestion.ingestion_router import (
    ROUTER_NEEDS_CLARIFICATION,
    ROUTER_PROCEED,
    ROUTER_REJECTED,
    IngestionFacts,
    IngestionRouter,
    RoutingDecision,
    apply_decision_table,
    gather_facts,
)
from src.ingestion.models import IngestionIntent

logger = logging.getLogger(__name__)


@dataclass
class IngestionRoutingOutcome:
    """Result of one pass of the ingestion routing graph."""

    status: str                                  # proceed / needs_clarification / rejected
    flags: Dict[str, bool] = field(default_factory=dict)
    decision: Optional[RoutingDecision] = None
    facts: Optional[IngestionFacts] = None
    intent: Optional[IngestionIntent] = None
    traces: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """Compact, LLM-forwardable summary."""
        return {
            "status": self.status,
            "flags": dict(self.flags),
            "explanation": self.decision.explanation if self.decision else "",
            "question": self.decision.question if self.decision else "",
            "suggestions": list(self.decision.suggestions) if self.decision else [],
            "facts": self.facts.summary() if self.facts else None,
        }


class IngestionRoutingGraph:
    """One decision pass for an ingestion request, fully traced.

    Args:
        router: the :class:`IngestionRouter` (LLM intent + clarification);
            a default one is built when omitted. Injectable for tests.
    """

    def __init__(self, router: Optional[IngestionRouter] = None) -> None:
        self._router = router or IngestionRouter()

    def run(
        self,
        request: str,
        *,
        on_event: Optional[Any] = None,
    ) -> IngestionRoutingOutcome:
        """Route one raw ingestion request; every step is traced."""

        def emit(kind: str, message: str, **data: Any) -> None:
            if on_event is not None:
                try:
                    on_event("routing", kind, message, **data)
                except Exception:  # noqa: BLE001 — observer must never break the flow
                    logger.warning("Routing event observer raised", exc_info=True)

        outcome = IngestionRoutingOutcome(status=ROUTER_REJECTED)

        # -- 1. intent (LLM, keyword fallback) -----------------------------------
        intent, llm_ok = self._router._identify_intent(request)
        outcome.intent = intent
        emit("intent_identified",
             f"intent: valid={intent.valid}, document={intent.document!r}, "
             f"force={intent.force}, redo_summaries={intent.redo_summaries}"
             + ("" if llm_ok else " (keyword fallback, LLM unavailable)"),
             intent=intent.summary(), llm_used=llm_ok)

        # -- 2. document + facts ---------------------------------------------------
        document = intent.document or self._router._fallback_document(request)
        if document is None:
            decision = self._router._clarify_request_unclear(
                request, intent, llm_used=llm_ok
            )
            emit("clarification", decision.question,
                 clarification=decision.clarification.value
                 if decision.clarification else None)
            outcome.status = decision.status
            outcome.decision = decision
            outcome.traces.append({"kind": "clarification",
                                   "clarification": decision.clarification.value
                                   if decision.clarification else None})
            return outcome

        facts = gather_facts(
            document,
            extraction_jobs=self._router._extraction_jobs,
            summarization_jobs=self._router._summarization_jobs,
        )
        outcome.facts = facts
        emit("facts_gathered",
             f"facts for '{facts.file_name}': found={facts.found}, "
             f"extraction_job={facts.extraction_job}, "
             f"canonical_json={facts.canonical_json}, "
             f"summarized_json={facts.summarized_json}, "
             f"summaries_stale={facts.summaries_stale}",
             facts=facts.summary())
        outcome.traces.append({"kind": "facts", "facts": facts.summary()})

        # -- 3. decision table ------------------------------------------------------
        decision = apply_decision_table(facts, intent)
        outcome.decision = decision
        if decision.status == ROUTER_PROCEED:
            decision.explanation = (
                f"intent classified, state consistent; flags={decision.flags}"
            )
            emit("decision",
                 f"proceed: flags={decision.flags}",
                 flags=dict(decision.flags))
            outcome.status = ROUTER_PROCEED
            outcome.traces.append({"kind": "decision", "status": "proceed",
                                   "flags": dict(decision.flags)})
            return outcome

        # -- 4. clarification / rejection wording -------------------------------------
        decision = self._router._word_clarification(
            request, facts, decision, llm_used=llm_ok
        )
        emit("clarification", decision.question,
             clarification=decision.clarification.value
             if decision.clarification else None,
             suggestions=list(decision.suggestions))
        outcome.status = decision.status
        outcome.traces.append({
            "kind": "clarification",
            "clarification": decision.clarification.value
            if decision.clarification else None,
        })
        return outcome
