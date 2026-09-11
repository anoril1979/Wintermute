"""Ingestion graph: the ordered sequence of steps the orchestrator runs.

The graph wires agent *placeholders* into the pipeline order described in
the project design:

    1. content extraction          (per file type)
    2. extraction validation       (structural gate)
    3. embedding + storage         (raw content chunks)
    4. hierarchical summarization, embedding + storage
    5. knowledge extraction        (one pass per knowledge model: characters, claims, ...)
    6. knowledge validation        (ClaimValidator / SourceValidator / ...)
    7. check-n-merge               (create-or-merge characters, reconcile claims)
    8. indexation, embedding + storage (vector / sql / markdown per knowledge model)

Agents themselves are NOT implemented: this module resolves them from an
injected registry (name -> agent). Missing agents surface as a clear
"not implemented" outcome so the graph can be dry-run before any agent exists.

Failure policy: when a step (or its validation) fails, the graph analyses
the failure domain and either restarts the step (bounded retries, for
transient domains like LLM_RESPONSE / LLM_TIMEOUT) or rejects the whole
ingestion (for data domains like INPUT_DATA). A future LLM-based failure
analyzer can replace the default policy via ``failure_analyzer``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from src.agents.contexts import IngestionContext
from src.agents.protocols import (
    AgentResult,
    AgentStatus,
    FailureDomain,
    IngestionAgent,
)

# Domains worth retrying (transient) vs domains leading to rejection (data).
RETRYABLE_DOMAINS = frozenset({FailureDomain.LLM_RESPONSE, FailureDomain.LLM_TIMEOUT})
DEFAULT_MAX_RETRIES = 2


@dataclass
class GraphStep:
    """One step of the ingestion graph."""

    name: str                                  # stable step id (also the context.outputs key)
    agent_key: str                             # key looked up in the agent registry
    required: bool = True                      # False: step may be skipped without failing the run
    max_retries: int = DEFAULT_MAX_RETRIES     # restarts allowed when failure domain is retryable


@dataclass
class GraphOutcome:
    """Result of a full graph run."""

    accepted: bool                             # True when every required step succeeded
    completed_steps: List[str] = field(default_factory=list)
    skipped_steps: List[str] = field(default_factory=list)
    not_implemented_steps: List[str] = field(default_factory=list)
    failed_step: Optional[str] = None          # step that caused rejection, if any
    failure_detail: str = ""


class IngestionGraph:
    """Runs the ordered ingestion steps against a registry of agents.

    The registry maps agent keys to agent instances implementing
    ``src.agents.protocols.IngestionAgent``. Keys missing from the registry
    are reported as not-implemented steps instead of crashing, so the graph
    can be exercised end-to-end while agents land one by one.
    """

    #: The canonical ingestion order. Keep in sync with the design doc.
    DEFAULT_STEPS: List[GraphStep] = [
        GraphStep(name="content_extraction", agent_key="content_extractor"),
        GraphStep(name="extraction_validation", agent_key="extraction_validator"),
        GraphStep(name="hierarchical_summarization", agent_key="summarizer"),
        # First SOURCE COLLECTION node: chunking + embedding + vector storage
        # of the document content (blocks + summaries) into source_chunks.
        # (The former content_embedding / content_storage placeholder pair.)
        GraphStep(name="source_indexing", agent_key="source_indexer"),
        GraphStep(name="knowledge_extraction", agent_key="knowledge_extractor"),
        GraphStep(name="knowledge_validation", agent_key="knowledge_validator"),
        GraphStep(name="check_and_merge", agent_key="check_and_merge"),
        GraphStep(name="indexation", agent_key="indexer"),
    ]

    def __init__(
        self,
        agents: Optional[Dict[str, IngestionAgent]] = None,
        steps: Optional[List[GraphStep]] = None,
        failure_analyzer: Optional[Callable[[AgentResult], FailureDomain]] = None,
    ) -> None:
        """Build a graph.

        Args:
            agents:           registry of agent key -> agent instance (all missing for now).
            steps:            override the default step list (mainly for tests).
            failure_analyzer: optional hook refining the failure domain before the
                              restart/reject decision; a future LLM-based analyzer
                              plugs in here. Defaults to the raw reported domain.
        """
        self._agents: Dict[str, IngestionAgent] = dict(agents or {})
        self._steps: List[GraphStep] = list(steps if steps is not None else self.DEFAULT_STEPS)
        self._failure_analyzer = failure_analyzer

    # -- public API ---------------------------------------------------------

    def run(self, context: IngestionContext) -> GraphOutcome:
        """Execute the steps in order, stopping on the first fatal failure."""
        outcome = GraphOutcome(accepted=False)

        for step in self._steps:
            agent = self._agents.get(step.agent_key)

            if agent is None:
                context.emit("pipeline", "not_implemented",
                             f"step '{step.name}' needs agent '{step.agent_key}', "
                             "which is not implemented yet", step=step.name)
                outcome.not_implemented_steps.append(step.name)
                if step.required:
                    # A required agent that does not exist yet stops the run:
                    # running further steps would operate on missing inputs.
                    outcome.failed_step = step.name
                    outcome.failure_detail = (
                        f"step '{step.name}' requires agent '{step.agent_key}', "
                        "which is not implemented yet"
                    )
                    return outcome
                outcome.skipped_steps.append(step.name)
                continue

            context.emit("pipeline", "step_started",
                         f"step '{step.name}' started", step=step.name)
            result = self._run_with_retries(step, agent, context)

            if result.status is AgentStatus.OK:
                context.emit("pipeline", "step_done",
                             f"step '{step.name}' done", step=step.name)
                outcome.completed_steps.append(step.name)
                continue
            if result.status is AgentStatus.SKIPPED:
                context.emit("pipeline", "step_skipped",
                             f"step '{step.name}' skipped", step=step.name)
                outcome.skipped_steps.append(step.name)
                continue

            # Fatal failure after retries.
            context.emit("pipeline", "step_failed",
                         f"step '{step.name}' failed: {result.detail}",
                         step=step.name, domain=result.failure_domain.value)
            outcome.failed_step = step.name
            outcome.failure_detail = result.detail
            return outcome

        outcome.accepted = True
        return outcome

    # -- internals ------------------------------------------------------------

    def _run_with_retries(
        self, step: GraphStep, agent: IngestionAgent, context: IngestionContext
    ) -> AgentResult:
        """Run one step, restarting up to ``max_retries`` on retryable failures."""
        attempts = 0
        while True:
            result = agent.run(context)

            # Structural validation hook: a failing validate() counts as a
            # step failure carrying the same domain.
            if result.status is AgentStatus.OK:
                validation = agent.validate(context)
                if validation is not None and validation.status is not AgentStatus.OK:
                    result = validation

            if result.status is not AgentStatus.FAILED:
                return result

            domain = self._analyze_failure(result)
            if domain not in RETRYABLE_DOMAINS or attempts >= step.max_retries:
                return result

            attempts += 1
            context.metadata.setdefault("retries", {})[step.name] = attempts
            context.emit("pipeline", "step_retry",
                         f"step '{step.name}' failed ({domain.value}); "
                         f"retrying (attempt {attempts}/{step.max_retries})",
                         step=step.name, attempt=attempts, domain=domain.value)

    def _analyze_failure(self, result: AgentResult) -> FailureDomain:
        """Decide the failure domain driving restart-vs-rejection.

        Placeholder for a future LLM-based analyzer (e.g. classifying whether
        a knowledge-validation failure stems from a misformed LLM response —
        retryable — or from genuinely contradictory source data — reject).
        """
        if self._failure_analyzer is not None:
            return self._failure_analyzer(result)
        return result.failure_domain
