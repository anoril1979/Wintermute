"""User-task agent protocols — the workers of the routing graph.

Where ``protocols.py`` freezes the contracts of the *ingestion* pipeline
steps, this module freezes the contracts of the agents a **user request**
is dispatched to. The routing graph (src/graphs/routing_graph.py) loops
over the structured requests produced by the request analyzer and hands
each one to the task agent matching its kind:

* ``RetrievalTaskAgent``  — answer a question from the ingested corpus
  (delegates to the retrieval orchestrator);
* ``GeneralTaskAgent``    — fallback: try to answer or cleanly reject
  requests outside the system's scope; this is also where a prompt that
  asks for ingestion lands — ingestion is a CLI operation, never routed.

The contracts share the outcome vocabulary of the ingestion agents
(``AgentResult`` / ``AgentStatus`` / ``FailureDomain``), so the routing
graph can report uniformly whatever the dispatched agent produced.

The contracts share the outcome vocabulary of the ingestion agents
(``AgentResult`` / ``AgentStatus`` / ``FailureDomain``), so the routing
graph can report uniformly whatever the dispatched agent produced.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from src.agents.contexts import RoutingContext
from src.agents.protocols import AgentResult

#: Registry keys the routing graph looks task agents up under.
#: Keyed by request scope label (the wire vocabulary the graph emits):
#: "retrieval" -> retrieval_task, "general" -> general_task.
TASK_AGENT_KEYS = {
    "retrieval": "retrieval_task",
    "general": "general_task",
}


@runtime_checkable
class UserTaskAgent(Protocol):
    """Base contract for every agent a user request can be dispatched to.

    Implementations must NOT raise for expected failures: they return an
    :class:`AgentResult` carrying the failure domain, so the routing graph
    can decide what to report to the caller (API -> user, CLI -> operator).
    """

    name: str

    def run(self, context: RoutingContext, request: object) -> AgentResult:
        """Handle one structured user request.

        Args:
            context: the routing context shared by the whole request batch.
            request: one of the request models (src/routing/models.py:
                RetrievalRequest / GeneralRequest) to handle. Typed as
                ``object`` here to avoid an import cycle — the request
                models belong to the routing layer, which depends on this
                package; implementations narrow the type.

        Returns:
            The agent's outcome; ``payload`` carries whatever the caller
            needs to phrase the user-facing reply.
        """
        ...

    def validate(self, context: RoutingContext, request: object) -> Optional[AgentResult]:
        """Optional post-run check of what ``run`` produced.

        Return ``None`` when there is nothing specific to check.
        """
        ...


@runtime_checkable
class RetrievalTaskAgent(UserTaskAgent, Protocol):
    """Answers a question from the ingested corpus.

    Will delegate to the retrieval orchestrator (retrieval_graph) once it
    exists; until then implementations may report not-implemented.
    """


@runtime_checkable
class GeneralTaskAgent(UserTaskAgent, Protocol):
    """Fallback for out-of-scope requests: try to answer, or reject cleanly."""
