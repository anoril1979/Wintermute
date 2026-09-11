"""Pedantic structural validation of a ``DocumentExtract``.

The **extraction agent** runs this right after building (or resuming) a
``DocumentExtract``, before returning its status: it is the "shape" gate —
types, hierarchy membership, positive counts. Pure logic on the object
graph, no cleanup: it reports, it never mutates.

Issues carry a ``severity``: "error" (a genuine shape defect — fails the
extraction) or "warning" (metadata the pipeline can live without — reported,
never fatal). Warnings exist for extraction backends that legitimately omit
optional data: MinerU's JSON carries no page geometry (width/height stay
``None``) and its fallback TOC uses 0-based levels. Coherence *against*
``total_pages`` (bounds, ordering) deliberately belongs to the consistency
pass (``src/extraction/consistency.py``), whose job is to remove or warn —
not to fail the extraction.

The **data-consistency** pass (ordering vs ``total_pages``, empty-text and
empty-container pruning) belongs to the extraction_validation agent
(``src.agents.agents.extraction_validation_agent``) — the next graph step —
so a broken shape fails *extraction*, while inconsistent-but-shaped data is
cleaned and warned during *validation*.

Every issue is a :class:`StructuralIssue`: a stable ``code``, a ``path``
locating the offending object (``chapters[0].pages[2].sections[1]``), and a
human-readable ``message``. The issue list is what the agent's ``validate()``
turns into a FAILED result (INPUT_DATA domain — not retryable), with the
issues riding the result payload for the traces/fix_hint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.extraction.models import (
    Chapter,
    DocumentExtract,
    PageContent,
    Section,
    TocEntry,
)


SEVERITY_ERROR = "error"      # fails the extraction (agent validate() → FAILED)
SEVERITY_WARNING = "warning"  # reported only — the pipeline can live without it


@dataclass(frozen=True)
class StructuralIssue:
    """One structural defect found in a DocumentExtract."""

    code: str        # stable machine-readable id (e.g. "page_number_out_of_bounds")
    path: str        # locator in the object graph (e.g. "chapters[0].pages[2]")
    message: str     # human-readable explanation
    severity: str = SEVERITY_ERROR  # "error" (fatal) or "warning" (advisory)

    def __str__(self) -> str:  # pragma: no cover — display convenience
        return f"[{self.severity}] {self.code} {self.path}: {self.message}"


def _bad_int(value: object) -> bool:
    return isinstance(value, bool) or not isinstance(value, int)


def _bad_bbox(bbox: object) -> bool:
    return (
        not isinstance(bbox, tuple)
        or len(bbox) != 4
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in bbox)
    )


def _check_toc_entry(entry: TocEntry, path: str,
                     issues: List[StructuralIssue]) -> None:
    # Level 0 is legitimate: 0-based levels, and a document without a real
    # TOC gets a fallback entry at level 0 (MinerU/PyMuPDF backends do this).
    if _bad_int(entry.level) or entry.level < 0:
        issues.append(StructuralIssue(
            "toc_level_invalid", path,
            f"TOC level must be a non-negative int (0-based), got {entry.level!r}",
        ))
    if not isinstance(entry.title, str):
        issues.append(StructuralIssue(
            "toc_title_invalid", path, f"TOC title must be a string, got {type(entry.title).__name__}"
        ))
    if _bad_int(entry.page_number) or entry.page_number < 1:
        issues.append(StructuralIssue(
            "toc_page_number_invalid", path,
            f"TOC page_number (1-based) must be a positive int, got {entry.page_number!r}"
        ))
    if _bad_int(entry.page_index) or entry.page_index < 0:
        issues.append(StructuralIssue(
            "toc_page_index_invalid", path,
            f"TOC page_index (0-based) must be a non-negative int, got {entry.page_index!r}",
        ))


def _check_section(section: Section, path: str,
                   issues: List[StructuralIssue]) -> None:
    if _bad_int(section.section_id):
        issues.append(StructuralIssue(
            "section_id_invalid", path, f"section_id must be an int, got {section.section_id!r}"
        ))
    if not isinstance(section.page_number, int) or isinstance(section.page_number, bool) \
            or section.page_number < 1:
        issues.append(StructuralIssue(
            "section_page_number_invalid", path,
            f"section page_number (1-based) must be a positive int, got {section.page_number!r}",
        ))
    if _bad_bbox(section.bbox):
        issues.append(StructuralIssue(
            "section_bbox_invalid", path, f"bbox must be a 4-number tuple, got {section.bbox!r}"
        ))
    if not isinstance(section.raw_text, str):
        issues.append(StructuralIssue(
            "section_raw_text_invalid", path,
            f"raw_text must be a string, got {type(section.raw_text).__name__}",
        ))
    for index, block in enumerate(section.blocks):
        block_path = f"{path}.blocks[{index}]"
        if not hasattr(block, "raw_text") or not isinstance(block.raw_text, str):
            issues.append(StructuralIssue(
                "block_raw_text_invalid", block_path, "block raw_text must be a string",
            ))
        if _bad_bbox(getattr(block, "bbox", None)):
            issues.append(StructuralIssue(
                "block_bbox_invalid", block_path, "block bbox must be a 4-number tuple",
            ))


def _check_page(page: PageContent, path: str,
                issues: List[StructuralIssue]) -> None:
    if _bad_int(page.page_number) or page.page_number < 1:
        issues.append(StructuralIssue(
            "page_number_invalid", path,
            f"page page_number (1-based) must be a positive int, got {page.page_number!r}",
        ))
    # Geometry is advisory: MinerU's JSON carries no page width/height (they
    # stay None) and nothing downstream depends on them. Missing or bogus
    # geometry is a warning, never a failure.
    for attr in ("width", "height"):
        value = getattr(page, attr, None)
        if value is None:
            issues.append(StructuralIssue(
                f"page_{attr}_invalid", path,
                f"page {attr} is not set (geometry unknown) — non-fatal",
                severity=SEVERITY_WARNING,
            ))
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            issues.append(StructuralIssue(
                f"page_{attr}_invalid", path,
                f"page {attr} should be a positive number, got {value!r} — non-fatal",
                severity=SEVERITY_WARNING,
            ))
    if not isinstance(page.raw_text, str):
        issues.append(StructuralIssue(
            "page_raw_text_invalid", path,
            f"raw_text must be a string, got {type(page.raw_text).__name__}",
        ))
    for index, section in enumerate(page.sections):
        _check_section(section, f"{path}.sections[{index}]", issues)


def structural_issues(document: DocumentExtract) -> List[StructuralIssue]:
    """Full structural audit of a ``DocumentExtract``.

    Checks (pure logic, no mutation):

    * root scalars: non-empty string ``title``, positive ``total_pages``;
    * every TOC entry: non-negative level (0-based / no-TOC fallback),
      string title, 1-based positive page_number, 0-based ``page_index`` >= 0;
    * every chapter's pages and every orphan page: 1-based positive
      page_number, string raw_text — width/height are warnings (geometry
      is optional metadata);
    * every section: positive 1-based page_number, 4-number bbox,
      string raw_text, and its blocks' raw_text/bbox types;
    * hierarchy: chapters own ``PageContent`` pages, pages own ``Section``
      sections, sections own text blocks (anything else is a defect).
    """
    issues: List[StructuralIssue] = []

    if not isinstance(document.title, str) or not document.title.strip():
        issues.append(StructuralIssue(
            "title_invalid", "root", f"title must be a non-empty string, got {document.title!r}"
        ))
    total_pages = document.total_pages
    if _bad_int(total_pages) or total_pages < 1:
        issues.append(StructuralIssue(
            "total_pages_invalid", "root",
            f"total_pages must be a positive int, got {total_pages!r}",
        ))

    for index, entry in enumerate(document.toc):
        if not isinstance(entry, TocEntry):
            issues.append(StructuralIssue(
                "toc_entry_type_invalid", f"toc[{index}]",
                f"expected TocEntry, got {type(entry).__name__}",
            ))
            continue
        _check_toc_entry(entry, f"toc[{index}]", issues)

    for chapter_index, chapter in enumerate(document.chapters):
        chapter_path = f"chapters[{chapter_index}]"
        if not isinstance(chapter, Chapter):
            issues.append(StructuralIssue(
                "chapter_type_invalid", chapter_path,
                f"expected Chapter, got {type(chapter).__name__}",
            ))
            continue
        if not isinstance(chapter.toc_entry, TocEntry):
            issues.append(StructuralIssue(
                "chapter_toc_entry_invalid", chapter_path,
                f"chapter must reference a TocEntry, got {type(chapter.toc_entry).__name__}",
            ))
        else:
            _check_toc_entry(chapter.toc_entry, f"{chapter_path}.toc_entry", issues)
        if not isinstance(chapter.full_text, str):
            issues.append(StructuralIssue(
                "chapter_full_text_invalid", chapter_path,
                f"full_text must be a string, got {type(chapter.full_text).__name__}",
            ))
        for page_index, page in enumerate(chapter.pages):
            page_path = f"{chapter_path}.pages[{page_index}]"
            if not isinstance(page, PageContent):
                issues.append(StructuralIssue(
                    "page_type_invalid", page_path,
                    f"expected PageContent, got {type(page).__name__}",
                ))
                continue
            _check_page(page, page_path, issues)

    for index, page in enumerate(document.orphan_pages):
        page_path = f"orphan_pages[{index}]"
        if not isinstance(page, PageContent):
            issues.append(StructuralIssue(
                "page_type_invalid", page_path,
                f"expected PageContent, got {type(page).__name__}",
            ))
            continue
        _check_page(page, page_path, issues)

    return issues


def structural_errors(document: DocumentExtract) -> List[StructuralIssue]:
    """Only the *fatal* issues (``severity == "error"``).

    What the extraction agent's ``validate()`` fails on; warnings are
    surfaced as traces/payload but never stop the pipeline.
    """
    return [i for i in structural_issues(document) if i.severity == SEVERITY_ERROR]
