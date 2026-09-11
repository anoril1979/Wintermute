"""Agent contracts, contexts, registries and implementations.

Layout (contracts above, workers below):

* ``protocols.py``      — ingestion-graph agent protocols + the Agent result
                          vocabulary (AgentStatus / FailureDomain / AgentResult);
* ``task_protocols.py`` — user-task agent protocols for the routing graph
                          (UserTaskAgent: Retrieval / Ingestion / General);
* ``contexts.py``       — the context objects flowing through the graphs
                          (IngestionContext, RoutingContext);
* ``llm_roles.py``      — strict LLM-role resolution for LLM-backed agents;
* ``registry.py``       — the single wiring point for implemented agents;
* ``agents/``           — the concrete agent implementations.
"""

from src.agents.agents import PDFExtractionAgent, SummarizerAgent
from src.agents.agents.extraction_validation_agent import ExtractionValidationAgent
from src.agents.agents.ingestion_task_agent import IngestionTaskAgent
from src.agents.contexts import IngestionContext, RoutingContext
from src.agents.llm_roles import LLMRoleAgent, MissingLLMRoleError, require_llm_role
from src.agents.protocols import (
    AgentResult,
    AgentStatus,
    CheckAndMergeAgent,
    ContentExtractorAgent,
    EmbedderAgent,
    ExtractionValidatorAgent,
    FailureDomain,
    IndexerAgent,
    IngestionAgent,
    KnowledgeExtractorAgent,
    KnowledgeValidatorAgent,
    StorageAgent,
    SummarizerAgent,
)
from src.agents.registry import build_default_agents
from src.agents.routing_registry import build_default_task_agents
from src.agents.task_protocols import (
    GeneralTaskAgent,
    IngestionTaskAgent,
    RetrievalTaskAgent,
    UserTaskAgent,
)

__all__ = [
    "AgentResult",
    "AgentStatus",
    "CheckAndMergeAgent",
    "ContentExtractorAgent",
    "EmbedderAgent",
    "ExtractionValidationAgent",
    "ExtractionValidatorAgent",
    "FailureDomain",
    "GeneralTaskAgent",
    "IngestionAgent",
    "IngestionContext",
    "IngestionTaskAgent",
    "IndexerAgent",
    "KnowledgeExtractorAgent",
    "KnowledgeValidatorAgent",
    "LLMRoleAgent",
    "MissingLLMRoleError",
    "PDFExtractionAgent",
    "RetrievalTaskAgent",
    "RoutingContext",
    "StorageAgent",
    "SummarizerAgent",
    "UserTaskAgent",
    "build_default_agents",
    "build_default_task_agents",
    "require_llm_role",
]
