"""Retrieval graph: run the retrieval steps for one classified question.

Mirror of the ingestion graph, read side. Today two real steps for a
``semantic`` request:

    semantic_search  -> "semantic_retriever" (SemanticRetrievalAgent)
    answer           -> "answerer"           (AnswerAgent)

The routing analyzer (upstream) already classified the lookup and the
decision table assembled the filters; the graph hands the context to the
steps matching the request type — the search fetches the scored hits,
the answer agent phrases them into the user-facing reply (grounded in
the hits only, prompt-enforced). Request types without serving agents
(index, relation, summary, listing) are reported not-implemented —
never silently degraded to a semantic search.

Failure policy mirrors the ingestion graph: a step failing with a
non-retryable domain ends the run with that step's result; retryable
domains are retried up to ``max_retries``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.agents.contexts import RetrievalContext
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
from src.routing.models import RetrievalLookupKind

logger = logging.getLogger(__name__)

#: Retryable failure domains (same policy as the ingestion graph).
RETRYABLE_DOMAINS = {FailureDomain.EXTERNAL}
DEFAULT_MAX_RETRIES = 1

#: Stable step name -> agent key mapping (registry keys, like ingestion).
#: The answer step runs after the search of the kinds it serves: it
#: phrases the fetched hits into the user-facing reply.
STEPS = {
    "semantic_search": "semantic_retriever",
    "answer": "answerer",
}

#: Extra step the graph appends after the kind's lookup step succeeded:
#: the hits of these kinds are phrased into a user-facing answer.
ANSWER_AFTER = {"semantic_search"}


@dataclass
class RetrievalStepOutcome:
    """Result of one graph step."""

    step: str
    agent: str
    status: str                     # ok / failed / skipped / not_implemented
    detail: str = ""
    payload: dict = field(default_factory=dict)


@dataclass
class RetrievalGraphOutcome:
    """Result of a full retrieval-graph run (one question)."""

    steps: List[RetrievalStepOutcome] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every executed step succeeded (and at least one ran).

        A ``skipped`` step (e.g. no answerer registered) does not void
        the run: the lookup's results are real work, phrasing is extra.
        """
        return (
            bool(self.steps)
            and all(s.status in ("ok", "skipped") for s in self.steps)
            and any(s.status == "ok" for s in self.steps)
        )

    @property
    def last_step(self) -> Optional[RetrievalStepOutcome]:
        return self.steps[-1] if self.steps else None

    def as_list(self) -> List[dict]:
        return [
            {
                "step": s.step,
                "agent": s.agent,
                "status": s.status,
                "detail": s.detail,
                **s.payload,
            }
            for s in self.steps
        ]


class RetrievalGraph:
    """Sequences the retrieval agents for one classified question."""

    def __init__(
        self,
        agents: Optional[Dict[str, object]] = None,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """Args:
        agents: registry keyed by agent key (``STEPS`` values); ``None``
            builds the default registry (src/agents/registry).
        max_retries: attempts per step for retryable failure domains.
        """
        if agents is None:
            from src.agents.registry import build_retrieval_agents

            agents = build_retrieval_agents()
        self._agents: Dict[str, object] = dict(agents)
        self._max_retries = max(1, int(max_retries))

    def run(
        self,
        context: RetrievalContext,
        *,
        kind: str = RetrievalLookupKind.SEMANTIC.value,
    ) -> RetrievalGraphOutcome:
        """Run the steps serving ``kind`` against the prepared context."""
        outcome = RetrievalGraphOutcome()

        step_name = self._step_for_kind(kind)
        if step_name is None:
            context.emit(
                "task", "retrieval_not_implemented",
                f"no retrieval step for request type '{kind}' yet",
                request_kind=kind,
            )
            outcome.steps.append(
                RetrievalStepOutcome(
                    step=kind,
                    agent="",
                    status="not_implemented",
                    detail=f"no retrieval step implemented for request type '{kind}' yet",
                )
            )
            return outcome

        agent_key = STEPS[step_name]
        agent = self._agents.get(agent_key)
        if agent is None:
            context.emit(
                "task", "retrieval_not_implemented",
                f"no agent registered for step '{step_name}' yet",
                step=step_name,
                request_kind=agent_key,
            )
            outcome.steps.append(
                RetrievalStepOutcome(
                    step=step_name,
                    agent=agent_key,
                    status="not_implemented",
                    detail=f"no agent registered for '{agent_key}' yet",
                )
            )
            return outcome

        outcome.steps.append(
            self._run_step(context, step_name=step_name, agent_key=agent_key, agent=agent)
        )
        if (
            step_name in ANSWER_AFTER
            and outcome.steps[-1].status == "ok"
        ):
            # The lookup succeeded: phrase its hits into the user reply.
            answer_key = STEPS["answer"]
            answer_agent = self._agents.get(answer_key)
            if answer_agent is None:
                # A missing answerer must not void the search: the run
                # stays ok, the detail says what is missing.
                context.emit(
                    "task", "retrieval_step_skipped",
                    f"no agent registered for step 'answer' — hits left unphrased",
                    step="answer",
                )
                outcome.steps.append(
                    RetrievalStepOutcome(
                        step="answer", agent=answer_key, status="skipped",
                        detail="no agent registered for 'answerer' yet",
                    )
                )
            else:
                outcome.steps.append(
                    self._run_step(context, step_name="answer",
                                   agent_key=answer_key, agent=answer_agent)
                )
        return outcome

    # -- internals ---------------------------------------------------------------

    @staticmethod
    def _step_for_kind(kind: str) -> Optional[str]:
        """The lookup step serving a request type (None = not implemented)."""
        if kind == RetrievalLookupKind.SEMANTIC.value:
            return "semantic_search"
        return None

    def _run_step(
        self,
        context: RetrievalContext,
        *,
        step_name: str,
        agent_key: str,
        agent: object,
    ) -> RetrievalStepOutcome:
        """Run one step with the ingestion graph's retry policy."""
        context.emit("task", "retrieval_step_start",
                     f"step '{step_name}' starting", step=step_name)

        last: Optional[AgentResult] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                result = agent.run(context)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 — one step must not
                # kill the graph; report as an external failure.
                logger.exception("Retrieval step '%s' crashed", step_name)
                result = AgentResult(
                    agent_name=str(agent_key),
                    status=AgentStatus.FAILED,
                    failure_domain=FailureDomain.EXTERNAL,
                    detail=f"unexpected step error: {exc}",
                )
            if result.status == AgentStatus.OK or result.failure_domain not in RETRYABLE_DOMAINS:
                last = result
                break
            last = result
            context.emit("task", "retrieval_step_retry",
                         f"step '{step_name}' failed ({result.failure_domain.value}), "
                         f"retry {attempt}/{self._max_retries}",
                         step=step_name)

        assert last is not None
        status = "ok" if last.status == AgentStatus.OK else "failed"
        context.emit(
            "task", "retrieval_step_done",
            f"step '{step_name}' ended with status '{status}': {last.detail or ''}",
            step=step_name, status=status,
        )
        return RetrievalStepOutcome(
            step=step_name,
            agent=last.agent_name,
            status=status,
            detail=last.detail,
            payload=dict(last.payload or {}),
        )
