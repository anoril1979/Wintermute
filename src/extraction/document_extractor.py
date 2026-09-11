"""Base classes for document extractors.

``DocumentExtractor`` defines the extraction flow as a template method:
validate the source, run the backend-specific extraction, then build the
structured ``DocumentExtract``. Subclasses only implement the two hooks,
which keeps every extractor consistent with ``DocumentExtractorProtocol``
without repeating the plumbing.

``PDFExtractor`` narrows the base to PDF sources (suffix validation) and is
the direct parent of ``MineruPDFExtractor``.
"""

from __future__ import annotations

from pathlib import Path

from src.extraction.models import DocumentExtract
from src.extraction.protocols import DocumentExtractorProtocol


class DocumentExtractor(DocumentExtractorProtocol):
    """Template-method base for all extractors.

    Subclasses implement:

    * ``_run_backend(path)`` — perform the actual content extraction
      (OCR/layout engine call, parser invocation...); may be a no-op when
      the backend works on previously produced artifacts;
    * ``_build_document(path)`` — turn the backend output into a
      ``DocumentExtract``.
    """

    # -- Public API (DocumentExtractorProtocol) ------------------------------

    def extract(self, document_path: str | Path) -> DocumentExtract:
        """Extract a document following the validate → extract → build flow."""
        path = self._validate_source(Path(document_path))
        self._run_backend(path)
        return self._build_document(path)

    # -- Hooks -----------------------------------------------------------------

    def _validate_source(self, path: Path) -> Path:
        """Check the source exists; subclasses may add format checks."""
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        return path

    def _run_backend(self, path: Path) -> None:
        """Run the extraction backend for the given source (no return value)."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _run_backend()."
        )

    def _build_document(self, path: Path) -> DocumentExtract:
        """Build the DocumentExtract from the backend's output."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_document()."
        )


class PDFExtractor(DocumentExtractor):
    """Extractor specialized for PDF sources.

    Adds PDF-specific source validation on top of the base flow; the
    extraction hooks remain abstract. Concrete PDF backends (MinerU, ...)
    derive from this class.
    """

    def _validate_source(self, path: Path) -> Path:
        path = super()._validate_source(path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {path}")
        return path
