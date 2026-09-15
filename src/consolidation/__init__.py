"""Ingestion-consolidation tools: in-memory merge of too-small text blocks."""

from src.consolidation.consolidator import (
    ConsolidationError,
    ConsolidationStats,
    consolidate_document,
)

__all__ = [
    "ConsolidationError",
    "ConsolidationStats",
    "consolidate_document",
]
