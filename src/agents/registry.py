"""Default agent registry for the ingestion graph.

``build_default_agents()`` is the single place where implemented agents are
instantiated and wired to the graph's agent keys. As new agents land
(extraction validator, embedder, summarizer, ...), add them here — the
orchestrator and the graph need no change.
"""

from __future__ import annotations

from typing import Dict

from src.agents.agents.extraction_validation_agent import ExtractionValidationAgent
from src.agents.agents.pdf_extraction_agent import PDFExtractionAgent
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
    }
