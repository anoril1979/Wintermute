"""Protocols for the ingestion agents.

Agents are the workers of the ingestion graph: each one performs one specific
task of the pipeline (extraction, validation, embedding, summarization,
knowledge extraction, indexing...). They are NOT implemented yet — this module
only freezes the contracts the orchestrator and the graph will rely on, so the
implementations can be written independently afterwards.

Two families of definitions live here:

* :class:`AgentResult` / :class:`AgentStatus` / :class:`FailureDomain` — the
  outcome vocabulary every agent uses to report how its run went.
* :class:`IngestionAgent` — the base contract every agent satisfies: a stable
  ``name``, a ``run(context)`` entry point, and an optional ``validate`` hook
  for post-run structural checks. One protocol per graph step then narrows it
  (extraction, validation, embedding, summarization, knowledge extraction,
  knowledge validation, check-n-merge, indexing) — mostly documented intent
  for now.

Failure handling: agents report domain-specific failures via
:class:`FailureDomain` on their result; the graph decides between
restart and rejection based on that domain (see src/graphs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from src.agents.contexts import IngestionContext


class AgentStatus(str, Enum):
    """Outcome of an agent run."""

    OK = "ok"                        # step produced its output
    FAILED = "failed"                # step could not produce its output
    SKIPPED = "skipped"              # step not applicable for this document


class FailureDomain(str, Enum):
    """Category of a failure, driving the restart-vs-rejection decision.

    Transient/LLM-shaped domains (bad LLM output, timeouts) are worth a
    retry; data-shaped domains (unreadable file, structurally invalid
    extraction) are not — they lead to rejection of the request.
    """

    NONE = "none"                    # no failure
    LLM_RESPONSE = "llm_response"    # LLM returned malformed/unusable output (bad JSON, wrong schema)
    LLM_TIMEOUT = "llm_timeout"      # LLM did not answer in time
    INPUT_DATA = "input_data"        # input to this step is unusable (bad file, invalid structure)
    EXTERNAL = "external"            # external dependency unavailable (DB down, model missing)
    CONFIG = "config"                # configuration problem (missing LLM role, malformed llm.yaml)
    UNKNOWN = "unknown"


@dataclass
class AgentResult:
    """What every agent run returns."""

    agent_name: str
    status: AgentStatus
    failure_domain: FailureDomain = FailureDomain.NONE
    detail: str = ""                       # human-readable explanation on failure
    payload: Dict[str, Any] = field(default_factory=dict)


# --- Base contract -----------------------------------------------------------


@runtime_checkable
class IngestionAgent(Protocol):
    """Base contract for every agent in the ingestion graph.

    Implementations must be callable with the shared context and must NOT
    raise for expected failures: they return an :class:`AgentResult` carrying
    the failure domain instead, so the graph can decide what to do.
    """

    name: str

    def run(self, context: IngestionContext) -> AgentResult:
        """Execute the agent's task on the context and return its outcome."""
        ...

    def validate(self, context: IngestionContext) -> Optional[AgentResult]:
        """Optional post-run structural validation of what ``run`` produced.

        Return ``None`` when the agent has nothing specific to check (the
        graph then treats the step as valid if ``run`` succeeded).
        """
        ...


# --- Per-step protocols (placeholders; implementations come later) -----------

@runtime_checkable
class ContentExtractorAgent(IngestionAgent, Protocol):
    """Extracts structured content from the document, depending on file type.

    Expected payload: the raw extraction (text blocks / sections / pages)
    stored under ``context.outputs[self.name]``.
    """


@runtime_checkable
class ExtractionValidatorAgent(IngestionAgent, Protocol):
    """Structurally validates the extraction (pages present, blocks sane...).

    Expected payload: boolean validity plus a list of issues found.
    This is the first gate: failures here are typically INPUT_DATA domain.
    """


@runtime_checkable
class EmbedderAgent(IngestionAgent, Protocol):
    """Embeds given items (chunks, summaries, knowledge) into vectors.

    Expected payload: embedded items keyed by id, ready for storage.
    """


@runtime_checkable
class StorageAgent(IngestionAgent, Protocol):
    """Persists items into one or more stores (vector, sql, markdown...).

    The concrete store selection depends on the knowledge model being
    stored; the graph passes that through the context.
    """


@runtime_checkable
class SummarizerAgent(IngestionAgent, Protocol):
    """Produces hierarchical summaries (page -> section -> document).

    LLM-based; failures expected in LLM_RESPONSE domain (malformed output).
    """


@runtime_checkable
class KnowledgeExtractorAgent(IngestionAgent, Protocol):
    """Extracts knowledge (characters, claims, future models) from content.

    Runs several passes, one per knowledge model. LLM-based; failures
    expected in LLM_RESPONSE domain (misformed JSON...).
    """


@runtime_checkable
class KnowledgeValidatorAgent(IngestionAgent, Protocol):
    """Validates extracted knowledge through src/validation validators
    (ClaimValidator, SourceValidator...).

    Failures can be INPUT_DATA (contradictory extraction) or LLM_RESPONSE
    (misformed models); the graph's restart/rejection decision uses the
    reported domain.
    """


@runtime_checkable
class CheckAndMergeAgent(IngestionAgent, Protocol):
    """Resolves duplicate/contradicting knowledge before storage
    (create-or-merge characters, reconcile claims...).
    """


@runtime_checkable
class IndexerAgent(IngestionAgent, Protocol):
    """Indexes validated knowledge and content into the search stores,
    choosing vector / sql / markdown storage depending on knowledge model.
    """
