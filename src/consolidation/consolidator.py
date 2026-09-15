"""Consolidation — merge too-small text blocks before summarization.

The pass between extraction validation and summarization. Wintermute's own
words when he sent the SI units after Kline's text: *they* were the
narrative — and a three-word chunk like ``Armement : Néant`` is no
narrative at all. Summarized, embedded or fed to knowledge extraction
as-is it is context-free noise. So before any of that runs, the pass
rebuilds every section's block list:

* a block whose ``raw_text`` is shorter than ``paragraph_min_length``
  (the **hard** low limit) is merged with the NEXT block — repeatedly,
  until the grown block reaches about ``paragraph_max_length`` (the soft
  high limit) or the section runs out of followers;
* a trailing block still under the hard limit falls BACK onto the
  previous block (same section) rather than staying alone;
* a section reduced to a single still-too-small merged block is left as
  is — sections are never merged, boundaries are narrative structure.

The merged block keeps the FIRST merged block's identity (``id``, bbox,
page) and concatenates raw texts in reading order. ``summary`` is dropped
(None): the merged text is new material for the summarizer, not the union
of two stale summaries — and running before summarization, no summary
exists yet anyway.

In-memory by design: the extraction store is untouched. The result is
serialized once, to ``<cache>/consolidation/<stem>.json``, by the
dedicated agent (storage tools, not this module).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

from src.extraction.models import (
    Chapter,
    DocumentExtract,
    PageContent,
    Section,
    TextBlock,
)

logger = logging.getLogger("wintermute.consolidation")

__all__ = [
    "ConsolidationError",
    "ConsolidationStats",
    "consolidate_document",
]


class ConsolidationError(ValueError):
    """The extraction cannot be consolidated (bad parameters or shape)."""


@dataclass
class ConsolidationStats:
    """What the pass did, for traces and the consolidation report."""

    sections_touched: int = 0      # sections that actually changed
    blocks_before: int = 0         # text blocks seen across the document
    blocks_after: int = 0          # text blocks left after merging
    merged_blocks: int = 0         # blocks absorbed into a survivor
    tail_merges: int = 0           # trailing-block fallback merges

    @property
    def merged_away(self) -> int:
        """Net block reduction (``blocks_before - blocks_after``)."""
        return self.blocks_before - self.blocks_after

    def as_dict(self) -> dict:
        return {
            "sections_touched": self.sections_touched,
            "blocks_before": self.blocks_before,
            "blocks_after": self.blocks_after,
            "merged_blocks": self.merged_blocks,
            "tail_merges": self.tail_merges,
        }


def consolidate_document(
    document: DocumentExtract,
    min_length: int,
    max_length: int,
) -> Tuple[DocumentExtract, ConsolidationStats]:
    """Return a consolidated *copy* of ``document`` (the original is intact).

    Args:
        document:   The validated, summarized ``DocumentExtract``.
        min_length: Hard low limit — a block below this many characters is
                    merged with the next one.
        max_length: Soft high limit — merging stops once the grown block
                    reaches about this size.

    Raises:
        ConsolidationError: ``min_length`` is not a positive int,
            ``max_length`` is below ``min_length``, or the document
            carries no content to consolidate.
    """
    if not isinstance(min_length, int) or isinstance(min_length, bool) \
            or min_length <= 0:
        raise ConsolidationError(
            f"paragraph_min_length doit être un entier strictement positif "
            f"(reçu : {min_length!r})."
        )
    if not isinstance(max_length, int) or isinstance(max_length, bool) \
            or max_length < min_length:
        raise ConsolidationError(
            f"paragraph_max_length doit être un entier >= "
            f"paragraph_min_length ({min_length}) (reçu : {max_length!r})."
        )
    if not document.chapters and not document.orphan_pages:
        raise ConsolidationError(
            "Rien à consolider : le document n'a ni chapitre ni page orpheline."
        )

    stats = ConsolidationStats()
    stats.blocks_before = _count_blocks(document)

    consolidated = DocumentExtract(
        id=document.id,
        source_path=document.source_path,
        title=document.title,
        author=document.author,
        subject=document.subject,
        total_pages=document.total_pages,
        origin=document.origin,
        chapters=[
            _consolidate_chapter(ch, min_length, max_length, stats)
            for ch in document.chapters
        ],
        summary=document.summary,
        orphan_pages=[
            _consolidate_page(pg, min_length, max_length, stats)
            for pg in document.orphan_pages
        ],
        toc=document.toc,
        metadata=dict(document.metadata),
    )

    stats.blocks_after = _count_blocks(consolidated)
    return consolidated, stats


# --------------------------------------------------------------------------
# Per-container rebuilds — pure copies, one list transformation each.
# --------------------------------------------------------------------------

def _consolidate_chapter(
    chapter: Chapter,
    min_length: int,
    max_length: int,
    stats: ConsolidationStats,
) -> Chapter:
    """Rebuild a chapter; TOC metadata and aggregate fields carry over."""
    return Chapter(
        toc_entry=chapter.toc_entry,
        pages=[
            _consolidate_page(pg, min_length, max_length, stats)
            for pg in chapter.pages
        ],
        full_text=chapter.full_text,
        summary=chapter.summary,
        metadata=dict(chapter.metadata),
        id=chapter.id,
    )


def _consolidate_page(
    page: PageContent,
    min_length: int,
    max_length: int,
    stats: ConsolidationStats,
) -> PageContent:
    """Rebuild a page by consolidating each of its sections in turn."""
    return PageContent(
        page_number=page.page_number,
        width=page.width,
        height=page.height,
        raw_text=page.raw_text,
        summary=page.summary,
        sections=[
            _consolidate_section(sec, min_length, max_length, stats)
            for sec in page.sections
        ],
        chapter_title=page.chapter_title,
        id=page.id,
    )


def _consolidate_section(
    section: Section,
    min_length: int,
    max_length: int,
    stats: ConsolidationStats,
) -> Section:
    """Merge too-small blocks inside ONE section (never across sections)."""
    merged = _merge_section_blocks(section.blocks, min_length, max_length, stats)
    return Section(
        section_id=section.section_id,
        blocks=merged,
        page_number=section.page_number,
        bbox=section.bbox,
        raw_text=section.raw_text,
        summary=section.summary,
        id=section.id,
    )


def _merge_section_blocks(
    blocks: List[TextBlock],
    min_length: int,
    max_length: int,
    stats: ConsolidationStats,
) -> List[TextBlock]:
    """The merge algorithm over one block list.

    Left-to-right: a too-small block grows by absorbing followers until it
    is ``about`` ``max_length`` (i.e. the NEXT block would overshoot past
    max + min — no point leaving a next block that is doomed to be merged
    anyway). Then the tail check: a last block still under the hard limit
    falls onto the previous survivor.
    """
    if not blocks:
        return []

    merged: List[TextBlock] = []
    i = 0
    while i < len(blocks):
        current = blocks[i]
        if not _is_too_small(current, min_length):
            merged.append(current)
            i += 1
            continue

        # Absorb followers while the result is still UNDER the soft max:
        # merging stops once the grown block reaches about/above
        # max_length (the spec's "about or above"). max is soft — a
        # follower that overshoots it is accepted, never split.
        while i + 1 < len(blocks) and len(current.raw_text) < max_length:
            current = _absorb(current, blocks[i + 1], stats)
            i += 1
        merged.append(current)
        i += 1

    # Tail fix: a last block under the hard limit falls onto the previous
    # one (same section). If it ends up alone, the spec says: leave it.
    if len(merged) > 1 and _is_too_small(merged[-1], min_length):
        tail = merged.pop()
        merged[-1] = _absorb(merged[-1], tail, stats, tail=True)

    if len(merged) != len(blocks):
        stats.sections_touched += 1
    return merged


def _is_too_small(block: TextBlock, min_length: int) -> bool:
    """Hard low limit check on the block's raw text."""
    return len(block.raw_text) < min_length


def _absorb(
    keep: TextBlock,
    gone: TextBlock,
    stats: ConsolidationStats,
    tail: bool = False,
) -> TextBlock:
    """Merge ``gone`` into ``keep``: keep's identity, concatenated texts.

    The survivor carries the FIRST block's id/bbox/page — the id scheme
    stays a valid citation anchor. ``summary`` is dropped (None): the
    merged text is new material for the summarizer, not the union of two
    stale summaries.
    """
    if tail:
        stats.tail_merges += 1
    stats.merged_blocks += 1
    logger.debug(
        "Consolidation : bloc %s absorbé dans %s (%d + %d caractères).",
        gone.id or f"block_id={gone.block_id}",
        keep.id or f"block_id={keep.block_id}",
        len(keep.raw_text),
        len(gone.raw_text),
    )
    return TextBlock(
        block_id=keep.block_id,
        page_number=keep.page_number,
        bbox=keep.bbox,
        raw_text=keep.raw_text + " " + gone.raw_text,
        summary=None,
        block_type=keep.block_type,
        text_level=keep.text_level,
        id=keep.id,
    )


def _count_blocks(document: DocumentExtract) -> int:
    """Total text blocks across chapters and orphan pages."""
    total = 0
    for chapter in document.chapters:
        for page in chapter.pages:
            for section in page.sections:
                total += len(section.blocks)
    for page in document.orphan_pages:
        for section in page.sections:
            total += len(section.blocks)
    return total
