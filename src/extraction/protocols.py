"""Protocols (structural contracts) for document extractors.

The protocol generalizes the prototype's ``PDFExtractorProtocol`` draft
(``src/extraction/rag_pdf_extractor.py``): the pipeline must be able to swap
the extraction backend (MinerU today, another OCR/layout engine tomorrow)
without touching downstream steps. Consumers code against
``DocumentExtractorProtocol``; concrete implementations derive from
``DocumentExtractor`` (``document_extractor.py``) and its PDF lineage
(``PDFExtractor`` → ``MineruPDFExtractor``).

As everywhere else in the codebase (see ``src/llm/protocols.py``), the
contract is deliberately minimal — one method — and ``runtime_checkable``
so ``isinstance`` sanity checks are possible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from src.extraction.models import DocumentExtract


@runtime_checkable
class DocumentExtractorProtocol(Protocol):
    """Interface that every document extractor of the pipeline must satisfy.

    Allows swapping the implementation without touching downstream steps
    (extraction validation, chunking, summarization...).
    """

    def extract(self, document_path: str | Path) -> DocumentExtract:
        """Read a document and return a structured DocumentExtract.

        Args:
            document_path: absolute or relative path to the source document.

        Returns:
            DocumentExtract: structured object consumable by the pipeline.

        Raises:
            FileNotFoundError: if the document is missing.
            ValueError: if the document is not a valid/expected file type.
            RuntimeError: for any unexpected extraction failure.
        """
        ...


# Backward-compatible alias: the prototype named the contract after PDFs.
PDFExtractorProtocol = DocumentExtractorProtocol
