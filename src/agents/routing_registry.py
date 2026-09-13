"""Default task-agent registry for the routing graph.

``build_default_task_agents()`` is the single wiring point for the agents
that handle structured user requests, mirroring
``registry.build_default_agents()`` for the ingestion graph. As new task
agents land, add them here — the routing graph and orchestrator need no
change.

The GeneralTaskAgent is built with ``allow_missing_role=True``: a missing
``general_task`` role in llm.yaml must not break the whole routing front
door, only general requests (they fail with a clean CONFIG-domain result
the API phrases for the user). The retrieval task agent keeps strict
wiring: a broken retrieval configuration is a real outage.

Ingestion has no task agent: document ingestion is a CLI operation
(scripts/ingest.py), deliberately unreachable from the chat.
"""

from __future__ import annotations

from typing import Dict

from src.agents.agents.general_task_agent import GeneralTaskAgent
from src.agents.agents.retrieval_task_agent import RetrievalTaskAgent
from src.agents.task_protocols import TASK_AGENT_KEYS, UserTaskAgent


def build_default_task_agents() -> Dict[str, UserTaskAgent]:
    """Build the registry of currently implemented task agents.

    Only agents that actually exist are returned; request kinds whose key
    is absent are reported as ``not_implemented`` by the routing graph.

    The GeneralTaskAgent is built with ``allow_missing_role=True`` (a
    missing ``general_task`` role must not break the front door); the
    retrieval task agent keeps strict wiring — a broken retrieval
    configuration is a real outage, surfaced as a clean CONFIG-domain
    result the API phrases for the user.
    """
    return {
        TASK_AGENT_KEYS["retrieval"]: RetrievalTaskAgent(),
        TASK_AGENT_KEYS["general"]: GeneralTaskAgent(allow_missing_role=True),
    }
