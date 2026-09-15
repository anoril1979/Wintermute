"""Default agent registry for the ingestion graph.

``build_default_agents()`` is the single place where implemented agents are
instantiated and wired to the graph's agent keys. As new agents land
(extraction validator, embedder, summarizer, ...), add them here — the
orchestrator and the graph need no change.
"""

from __future__ import annotations

from typing import Dict

from src.agents.agents.answer_agent import AnswerAgent
from src.agents.agents.character_extraction_agent import CharacterExtractionAgent
from src.agents.agents.entity_resolver_agent import CharacterResolver
from src.agents.agents.extraction_validation_agent import ExtractionValidationAgent
from src.agents.agents.knowledge_lookup_agent import KnowledgeLookupAgent
from src.agents.agents.knowledge_validation_agent import KnowledgeValidatorAgent
from src.agents.agents.pdf_extraction_agent import PDFExtractionAgent
from src.agents.agents.semantic_retrieval_agent import SemanticRetrievalAgent
from src.agents.agents.source_indexing_agent import SourceIndexingAgent
from src.agents.agents.summarizer_agent import SummarizerAgent
from src.agents.protocols import IngestionAgent


def build_default_agents() -> Dict[str, IngestionAgent]:
    """Build the registry of currently implemented agents.

    Only agents that actually exist are returned; graph steps whose key is
    absent are reported as not-implemented by the graph.
    """
    return {
        "content_extractor": PDFExtractionAgent(),
        "extraction_validator": ExtractionValidationAgent(),
        "summarizer": SummarizerAgent(),
        "source_indexer": SourceIndexingAgent(),
        # First knowledge pass: LLM extraction of the characters.
        "knowledge_extractor": CharacterExtractionAgent(),
        # Semantic gate on the discovered entities (deterministic).
        "knowledge_validator": KnowledgeValidatorAgent(),
        # check_and_merge: reconcile discovered characters into the
        # markdown knowledge base (create-or-merge, no LLM).
        "entity_resolver": CharacterResolver(),
    }


def build_retrieval_agents() -> Dict[str, object]:
    """Build the registry of the retrieval graph's agents.

    Mirror of :func:`build_default_agents` for the read side: the retrieval
    graph resolves its step keys (``STEPS`` values) here. As new retrieval
    agents land (SQL relation/index readers), add them here.
    """
    return {
        "semantic_retriever": SemanticRetrievalAgent(),
        # ``lookup`` kind: entity resolution against the markdown base
        # (deterministic, no LLM).
        "knowledge_lookup": KnowledgeLookupAgent(),
        "answerer": AnswerAgent(allow_missing_role=True),
    }
