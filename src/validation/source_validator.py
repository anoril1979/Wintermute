"""Source validation against registered SourceLocators."""

from typing import Dict, Optional

from src.knowledge.models import SourceRef, SourceLocator, Section


class SourceValidationError(ValueError):
    """Base class for source validation errors."""


class UnknownLocatorError(SourceValidationError):
    """The source's locator_id is not among the registered locators."""


class UnknownSectionError(SourceValidationError):
    """The targeted section was not collected in the locator."""


class UnknownPageError(SourceValidationError):
    """The targeted page was not collected in the targeted section."""


class InvalidParagraphIndexError(SourceValidationError):
    """paragraph_index points outside the page's paragraphs."""


class SourceValidator:
    """Validates SourceRef pointers against registered SourceLocators.

    Indices are DOCUMENT-GLOBAL 0-based numbers:

    - ``section_index`` must match a collected ``Section.section_index``
      (None means document section 0, valid only when collected);
    - ``page_index`` must match a collected ``Page.page_index`` of the
      targeted section (None means document page 0, valid only when
      collected there). The document-wide ``page_count`` bound is
      enforced by the SourceLocator model itself at construction;
    - ``paragraph_index`` must be lower than the page's
      ``num_paragraphs``.

    This supports partially-collected documents: a 300-page book where
    only chapter 3 (pages 15 and 17) was parsed is registered with
    page_count=300 and one section tagged section_index=2 holding
    pages 15 and 17; a ref to page 15 or 17 validates, a ref to any
    other page or section does not (it was not collected).

    ``validate`` returns the resolved SourceLocator.
    """

    def validate(
        self,
        source: SourceRef,
        locators: Dict[str, SourceLocator],
    ) -> SourceLocator:
        """Validate ``source`` and return the resolved locator.

        Raises a SourceValidationError subclass on the first problem found.
        """
        locator = locators.get(source.locator_id)
        if locator is None:
            raise UnknownLocatorError(f"Unknown locator: {source.locator_id}")

        section = self._resolve_section(locator, source.section_index)
        page = self._resolve_page(locator, section, source.page_index)

        if source.paragraph_index is not None and source.paragraph_index >= page.num_paragraphs:
            raise InvalidParagraphIndexError(
                f"{source.locator_id}: paragraph_index {source.paragraph_index} out of range "
                f"(page {source.page_index if source.page_index is not None else 0} of section "
                f"{section.section_index} has {page.num_paragraphs} paragraph(s))"
            )

        return locator

    def _resolve_section(
        self, locator: SourceLocator, section_index: Optional[int]
    ) -> Section:
        """Return the collected section for a document-global section index.

        None means document section 0 — accepted only when section 0 was
        actually collected (on a sparse locator it was not, so an explicit
        index is required).
        """
        wanted = section_index if section_index is not None else 0
        for section in locator.sections:
            if section.section_index == wanted:
                return section
        raise UnknownSectionError(
            f"{locator.id}: section_index {wanted} was not collected "
            f"(collected section(s): {sorted(s.section_index for s in locator.sections)})"
        )

    def _resolve_page(
        self, locator: SourceLocator, section: Section, page_index: Optional[int]
    ):
        """Return the collected page for a document-global page index.

        None means document page 0 — accepted only when page 0 was
        collected in the targeted section.
        """
        wanted = page_index if page_index is not None else 0
        for page in section.pages:
            if page.page_index == wanted:
                return page
        if page_index is None:
            raise UnknownPageError(
                f"{locator.id}: page_index 0 was not collected in section "
                f"{section.section_index} — a partially-collected locator "
                f"requires an explicit page_index "
                f"(collected page(s): {sorted(p.page_index for p in section.pages)})"
            )
        raise UnknownPageError(
            f"{locator.id}: page_index {wanted} was not collected in section "
            f"{section.section_index} "
            f"(collected page(s): {sorted(p.page_index for p in section.pages)})"
        )
