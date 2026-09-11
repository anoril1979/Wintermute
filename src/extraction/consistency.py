"""Data-consistency checking and pruning of a ``DocumentExtract``.

The extraction_validation agent runs this pass after the extraction
agent's *structural* gate (``src/extraction/validation.py``). Purely
logical, mutating-by-design checks:

* **Page coherence** — with ``total_pages`` in mind, page numbers must be
  non-decreasing (equal is legitimate: several sections of one page,
  repeated pages across chapter boundaries) and never exceed
  ``total_pages``. Out-of-bounds pages are **removed** with a warning;
  merely unsorted lists are **warned about** but kept (re-ordering would
  silently destroy whatever ordering the backend produced on purpose).
* **Empty-text pruning** — a ``raw_text`` (block, section, page, chapter
  ``full_text``) containing nothing but spaces/tabs/newlines means there
  is nothing to work out: the object is **removed** and a warning is
  raised. This lightens the final ``DocumentExtract``.
* **Empty-container pruning** — after text pruning, containers are
  re-checked: sections without blocks, pages without sections and
  chapters without pages are **removed** with a warning. An entirely
  content-free document is not cleaned, it *fails* (the agent turns that
  into an INPUT_DATA error).
* **Child alignment** — a section or block whose ``page_number`` disagrees
  with the page containing it is a coherence defect: the child is
  **realigned** to the parent page (warned). TOC entries are expected
  sorted by page number (warned, never reordered).

Every action is reported in a :class:`ConsistencyReport`; nothing raises.
The pass is deterministic and idempotent: running it twice changes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.extraction.models import (
    Chapter,
    DocumentExtract,
    PageContent,
    Section,
    TextBlock,
)


def _is_blank(text: object) -> bool:
    """True when the text is absent or only spaces/tabs/newlines."""
    return not isinstance(text, str) or not text.strip()


@dataclass
class ConsistencyReport:
    """What the consistency pass did, for traces and result payloads."""

    warnings: List[str] = field(default_factory=list)
    removed_blocks: int = 0
    removed_sections: int = 0
    removed_pages: int = 0
    removed_chapters: int = 0
    realigned: int = 0
    out_of_bounds_pages: int = 0
    unsorted_lists: int = 0

    @property
    def removals(self) -> int:
        return (
            self.removed_blocks + self.removed_sections
            + self.removed_pages + self.removed_chapters
        )

    def as_payload(self) -> dict:
        return {
            "warnings": list(self.warnings),
            "warning_count": len(self.warnings),
            "removed_blocks": self.removed_blocks,
            "removed_sections": self.removed_sections,
            "removed_pages": self.removed_pages,
            "removed_chapters": self.removed_chapters,
            "removals": self.removals,
            "realigned": self.realigned,
            "out_of_bounds_pages": self.out_of_bounds_pages,
            "unsorted_lists": self.unsorted_lists,
        }


def _warn(report: ConsistencyReport, message: str) -> None:
    report.warnings.append(message)


def _prune_blocks(section: Section, path: str, report: ConsistencyReport) -> None:
    kept: List[TextBlock] = []
    for index, block in enumerate(section.blocks):
        if _is_blank(getattr(block, "raw_text", None)):
            report.removed_blocks += 1
            _warn(report, f"{path}.blocks[{index}]: blank raw_text, block removed")
            continue
        kept.append(block)
    section.blocks = kept


def _prune_sections(page: PageContent, path: str, report: ConsistencyReport) -> None:
    kept: List[Section] = []
    for index, section in enumerate(page.sections):
        section_path = f"{path}.sections[{index}]"
        # Blank-text sections go first...
        if _is_blank(section.raw_text):
            report.removed_sections += 1
            _warn(report, f"{section_path}: blank raw_text, section removed")
            continue
        # ...then empty containers (possibly created by block pruning).
        _prune_blocks(section, section_path, report)
        if not section.blocks:
            report.removed_sections += 1
            _warn(report, f"{section_path}: no content left in section, removed")
            continue
        # Child alignment: a section must claim its parent page's number.
        if section.page_number != page.page_number:
            report.realigned += 1
            _warn(
                report,
                f"{section_path}: page_number {section.page_number} realigned "
                f"to parent page {page.page_number}",
            )
            section.page_number = page.page_number
        kept.append(section)
    page.sections = kept


def _check_sorted_pages(pages: List[PageContent], path: str,
                        report: ConsistencyReport) -> None:
    """Warn on decreasing page_number (kept as-is, never reordered)."""
    for previous, current in zip(pages, pages[1:]):
        if current.page_number < previous.page_number:
            report.unsorted_lists += 1
            _warn(
                report,
                f"{path}: page numbers not sorted "
                f"({previous.page_number} -> {current.page_number}); kept as-is",
            )


def _prune_pages(
    pages: List[PageContent],
    path: str,
    total_pages: int,
    report: ConsistencyReport,
) -> List[PageContent]:
    """Bounds-check, prune and alignment-check a page list."""
    kept: List[PageContent] = []
    for index, page in enumerate(pages):
        page_path = f"{path}.pages[{index}]"
        if page.page_number > total_pages or page.page_number < 1:
            report.out_of_bounds_pages += 1
            report.removed_pages += 1
            _warn(
                report,
                f"{page_path}: page_number {page.page_number} outside "
                f"[1, {total_pages}], page removed",
            )
            continue
        _prune_sections(page, page_path, report)
        if not page.sections:
            report.removed_pages += 1
            _warn(report, f"{page_path}: no content left in page, removed")
            continue
        if _is_blank(page.raw_text):
            report.removed_pages += 1
            _warn(report, f"{page_path}: blank raw_text, page removed")
            continue
        kept.append(page)
    _check_sorted_pages(kept, path, report)
    return kept


def check_and_prune(document: DocumentExtract) -> ConsistencyReport:
    """Run the full data-consistency pass **in place** on ``document``.

    See the module docstring for the rules. Deterministic and idempotent:
    a second run yields no further changes and no new warnings.
    """
    report = ConsistencyReport()
    total_pages = document.total_pages

    # TOC sanity: sorted by page number (warn only).
    for previous, current in zip(document.toc, document.toc[1:]):
        if current.page_number < previous.page_number:
            report.unsorted_lists += 1
            _warn(report, "toc: entries not sorted by page number; kept as-is")

    kept_chapters: List[Chapter] = []
    for chapter_index, chapter in enumerate(document.chapters):
        chapter_path = f"chapters[{chapter_index}]"
        if _is_blank(chapter.full_text) and not chapter.pages:
            # Nothing at all in this chapter.
            report.removed_chapters += 1
            _warn(report, f"{chapter_path}: empty chapter (no pages, blank text), removed")
            continue
        chapter.pages = _prune_pages(
            chapter.pages, chapter_path, total_pages, report
        )
        if not chapter.pages:
            report.removed_chapters += 1
            _warn(report, f"{chapter_path}: no content left in chapter, removed")
            continue
        if _is_blank(chapter.full_text):
            _warn(
                report,
                f"{chapter_path}: blank full_text while pages have content "
                "(chapter kept; the summary step will rebuild the text)",
            )
        kept_chapters.append(chapter)
    document.chapters = kept_chapters

    document.orphan_pages = _prune_pages(
        document.orphan_pages, "orphan_pages", total_pages, report
    )

    return report


def has_content(document: DocumentExtract) -> bool:
    """True when the document still carries at least one page of content."""
    return bool(document.chapters or document.orphan_pages)
