"""Extraction layer: document models, extractor protocol and backends.

Public API:
    models.DocumentExtract / Chapter / PageContent / Section / TextBlock /
        TocEntry / BlockType          — the structured extraction models
    protocols.DocumentExtractorProtocol — the extractor contract
    document_extractor.DocumentExtractor — template-method base
    document_extractor.PDFExtractor      — PDF-specialized base
    mineru_pdf_extractor.MineruPDFExtractor — MinerU-backed implementation

Related stores moved to their natural homes:

* the canonical extracted-content JSON store lives in ``src.helpers``
  (``document_extract_json_store`` — a storage helper, not extraction
  logic);
* the LLM-summaries store lives in ``src.summarization``
  (``summarized_store`` — owned by the summarization process).
"""

from src.extraction.document_extractor import DocumentExtractor, PDFExtractor
from src.extraction.mineru_pdf_extractor import MineruPDFExtractor
from src.extraction.models import (
    BlockType,
    Chapter,
    DocumentExtract,
    PageContent,
    Section,
    TextBlock,
    TocEntry,
    document_extract_to_dict,
)
from src.extraction.protocols import DocumentExtractorProtocol, PDFExtractorProtocol

__all__ = [
    "BlockType",
    "Chapter",
    "DocumentExtract",
    "DocumentExtractor",
    "DocumentExtractorProtocol",
    "MineruPDFExtractor",
    "PageContent",
    "PDFExtractor",
    "PDFExtractorProtocol",
    "Section",
    "TextBlock",
    "TocEntry",
    "document_extract_to_dict",
]
