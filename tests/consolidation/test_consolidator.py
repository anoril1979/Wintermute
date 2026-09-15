"""Tests for the consolidation pass (src/consolidation/consolidator.py).

Pure-model tests: build DocumentExtract fixtures, run the deterministic
merge, assert the spec's rules — hard low limit, soft high limit, first-id
identity, tail fallback, never cross sections, original untouched.
"""

from __future__ import annotations

import unittest

from src.consolidation.consolidator import (
    ConsolidationError,
    ConsolidationStats,
    consolidate_document,
)
from src.extraction.models import (
    BlockType,
    Chapter,
    DocumentExtract,
    PageContent,
    Section,
    TextBlock,
    TocEntry,
)


def _block(text: str, block_id: int = 0, flat_id: str = "") -> TextBlock:
    return TextBlock(
        block_id=block_id,
        page_number=1,
        bbox=(0.0, 0.0, 100.0, 20.0),
        raw_text=text,
        block_type=BlockType.TEXT,
        id=flat_id,
    )


def _section(blocks: list, section_id: int = 1, flat_id: str = "sec:1") -> Section:
    return Section(
        section_id=section_id,
        blocks=blocks,
        page_number=1,
        bbox=(0.0, 0.0, 595.0, 842.0),
        raw_text=" ".join(b.raw_text for b in blocks),
        id=flat_id,
    )


def _page(sections: list, page_number: int = 1, flat_id: str = "pg:1") -> PageContent:
    return PageContent(
        page_number=page_number,
        width=595.0,
        height=842.0,
        raw_text=" ".join(s.raw_text for s in sections),
        sections=sections,
        id=flat_id,
    )


def _document(sections_per_page: list) -> DocumentExtract:
    """One chapter whose single page holds the given section list."""
    page = _page(sections_per_page)
    chapter = Chapter(
        toc_entry=TocEntry(level=1, title="Ch1", page_number=1, page_index=0),
        pages=[page],
        id="chp:1",
    )
    return DocumentExtract(
        id="doc:00000001",
        title="Test Doc",
        total_pages=1,
        chapters=[chapter],
    )


class ParameterValidationTest(unittest.TestCase):
    def test_min_length_must_be_positive_int(self):
        doc = _document([_section([_block("x" * 200, flat_id="txt:1")])])
        for bad in (0, -5, 10.5, "100", True):
            with self.assertRaises(ConsolidationError):
                consolidate_document(doc, bad, 1000)

    def test_max_length_must_be_at_least_min(self):
        doc = _document([_section([_block("x" * 200, flat_id="txt:1")])])
        for bad in (99, 0, -1):
            with self.assertRaises(ConsolidationError):
                consolidate_document(doc, 100, bad)

    def test_empty_document_is_rejected(self):
        with self.assertRaises(ConsolidationError):
            consolidate_document(DocumentExtract(id="doc:00000001"), 100, 1000)


class MergeRuleTest(unittest.TestCase):
    """The spec's core example: merge the too-small heads, stop at max."""

    MIN = 100
    MAX = 1000

    def test_small_heads_merge_into_one_block_keeping_first_id(self):
        # txt:1 (5c) + txt:2 (4c) + txt:3 (900c) → one block (909c, id txt:1);
        # txt:4 (6c) merges too only if the result is still under max.
        doc = _document([_section([
            _block("miaou", block_id=0, flat_id="txt:1"),
            _block("meow", block_id=1, flat_id="txt:2"),
            _block("x" * 900, block_id=2, flat_id="txt:3"),
            _block("text...", block_id=3, flat_id="txt:4"),
        ])])
        consolidated, stats = consolidate_document(doc, self.MIN, self.MAX)
        blocks = consolidated.chapters[0].pages[0].sections[0].blocks

        # 909 < 1000 → txt:4 is absorbed as well (the letter of the rule).
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].id, "txt:1")
        self.assertEqual(blocks[0].block_id, 0)
        self.assertTrue(blocks[0].raw_text.startswith("miaou meow"))
        self.assertIn("x" * 900, blocks[0].raw_text)
        self.assertEqual(stats.merged_away, 3)

    def test_merge_stops_once_reached_max_and_tail_falls_back(self):
        # txt:3 long enough that the merged head crosses max; the forward
        # loop stops, but txt:4 is a too-small LAST block → tail rule
        # merges it onto the survivor anyway.
        doc = _document([_section([
            _block("miaou", block_id=0, flat_id="txt:1"),
            _block("meow", block_id=1, flat_id="txt:2"),
            _block("x" * 995, block_id=2, flat_id="txt:3"),
            _block("text...", block_id=3, flat_id="txt:4"),
        ])])
        consolidated, stats = consolidate_document(doc, self.MIN, self.MAX)
        blocks = consolidated.chapters[0].pages[0].sections[0].blocks

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].id, "txt:1")
        self.assertGreaterEqual(len(blocks[0].raw_text), self.MAX)
        self.assertTrue(blocks[0].raw_text.endswith("text..."))
        self.assertEqual(stats.tail_merges, 1)

    def test_forward_merge_stops_leaving_a_big_enough_tail(self):
        # Same head, but the tail block is big enough to stand alone:
        # it survives with its own id.
        doc = _document([_section([
            _block("miaou", block_id=0, flat_id="txt:1"),
            _block("meow", block_id=1, flat_id="txt:2"),
            _block("x" * 995, block_id=2, flat_id="txt:3"),
            _block("t" * 150, block_id=3, flat_id="txt:4"),
        ])])
        consolidated, _ = consolidate_document(doc, self.MIN, self.MAX)
        blocks = consolidated.chapters[0].pages[0].sections[0].blocks

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].id, "txt:1")
        self.assertEqual(blocks[1].id, "txt:4")
        self.assertEqual(blocks[1].raw_text, "t" * 150)

    def test_big_blocks_pass_through_untouched(self):
        doc = _document([_section([
            _block("y" * 500, block_id=0, flat_id="txt:1"),
            _block("z" * 600, block_id=1, flat_id="txt:2"),
        ])])
        consolidated, _ = consolidate_document(doc, self.MIN, self.MAX)
        blocks = consolidated.chapters[0].pages[0].sections[0].blocks

        self.assertEqual(len(blocks), 2)
        self.assertEqual([b.id for b in blocks], ["txt:1", "txt:2"])
        self.assertEqual([b.raw_text for b in blocks], ["y" * 500, "z" * 600])

    def test_trailing_small_block_merges_backwards(self):
        # txt:1 big, txt:2 too small and last → falls onto txt:1.
        doc = _document([_section([
            _block("y" * 500, block_id=0, flat_id="txt:1"),
            _block("tail", block_id=1, flat_id="txt:2"),
        ])])
        consolidated, stats = consolidate_document(doc, self.MIN, self.MAX)
        blocks = consolidated.chapters[0].pages[0].sections[0].blocks

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].id, "txt:1")
        self.assertTrue(blocks[0].raw_text.endswith("tail"))
        self.assertEqual(stats.tail_merges, 1)

    def test_single_small_block_in_section_is_left_alone(self):
        doc = _document([_section([_block("tiny", block_id=0, flat_id="txt:1")])])
        consolidated, _ = consolidate_document(doc, self.MIN, self.MAX)
        blocks = consolidated.chapters[0].pages[0].sections[0].blocks

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].raw_text, "tiny")

    def test_sections_are_never_merged(self):
        # Both sections hold one too-small block: each stays its own.
        doc = _document([
            _section([_block("tiny one", block_id=0, flat_id="txt:1")],
                     section_id=1, flat_id="sec:1"),
            _section([_block("tiny two", block_id=0, flat_id="txt:1")],
                     section_id=2, flat_id="sec:2"),
        ])
        consolidated, stats = consolidate_document(doc, self.MIN, self.MAX)
        sections = consolidated.chapters[0].pages[0].sections

        self.assertEqual(len(sections), 2)
        self.assertEqual([s.id for s in sections], ["sec:1", "sec:2"])
        self.assertEqual(sections[0].blocks[0].raw_text, "tiny one")
        self.assertEqual(sections[1].blocks[0].raw_text, "tiny two")
        self.assertEqual(stats.merged_away, 0)


class IntegrityTest(unittest.TestCase):
    MIN = 100
    MAX = 1000

    def test_original_document_is_untouched(self):
        doc = _document([_section([
            _block("miaou", block_id=0, flat_id="txt:1"),
            _block("meow", block_id=1, flat_id="txt:2"),
        ])])
        before = len(doc.chapters[0].pages[0].sections[0].blocks)
        consolidate_document(doc, self.MIN, self.MAX)
        after = len(doc.chapters[0].pages[0].sections[0].blocks)
        self.assertEqual(before, after)
        self.assertEqual(
            doc.chapters[0].pages[0].sections[0].blocks[1].raw_text, "meow",
        )

    def test_merged_block_carries_first_identity_and_drops_summary(self):
        doc = _document([_section([
            _block("miaou", block_id=0, flat_id="txt:1"),
            _block("meow", block_id=1, flat_id="txt:2"),
        ])])
        doc.chapters[0].pages[0].sections[0].blocks[0].summary = "old summary"
        consolidated, _ = consolidate_document(doc, self.MIN, self.MAX)
        merged = consolidated.chapters[0].pages[0].sections[0].blocks[0]

        self.assertEqual(merged.id, "txt:1")
        self.assertEqual(merged.block_id, 0)
        self.assertIsNone(merged.summary)

    def test_orphan_pages_are_consolidated_too(self):
        orphan = _page(
            [_section([_block("miaou", block_id=0, flat_id="txt:1"),
                       _block("meow", block_id=1, flat_id="txt:2")])],
            page_number=3, flat_id="pg:1",
        )
        doc = DocumentExtract(
            id="doc:00000001", title="Test Doc", total_pages=3,
            orphan_pages=[orphan],
        )
        consolidated, stats = consolidate_document(doc, self.MIN, self.MAX)
        blocks = consolidated.orphan_pages[0].sections[0].blocks

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].id, "txt:1")
        self.assertGreater(stats.merged_away, 0)

    def test_document_metadata_and_summaries_carry_over(self):
        doc = _document([_section([_block("y" * 500, block_id=0, flat_id="txt:1")])])
        doc.title = "Gazette"
        doc.origin = "community"
        doc.summary = "doc summary"
        doc.chapters[0].summary = "chapter summary"
        doc.chapters[0].pages[0].summary = "page summary"
        consolidated, _ = consolidate_document(doc, self.MIN, self.MAX)

        self.assertEqual(consolidated.title, "Gazette")
        self.assertEqual(consolidated.origin, "community")
        self.assertEqual(consolidated.summary, "doc summary")
        self.assertEqual(consolidated.chapters[0].summary, "chapter summary")
        self.assertEqual(consolidated.chapters[0].pages[0].summary, "page summary")

    def test_stats_counts_are_consistent(self):
        doc = _document([_section([
            _block("a", block_id=0, flat_id="txt:1"),
            _block("b", block_id=1, flat_id="txt:2"),
            _block("c", block_id=2, flat_id="txt:3"),
            _block("y" * 500, block_id=3, flat_id="txt:4"),
        ])])
        _, stats = consolidate_document(doc, self.MIN, self.MAX)

        # The head grows a→ab→abc→abc·y500 (5 chars, still under max when
        # txt:4 arrives): everything lands in ONE block.
        self.assertEqual(stats.blocks_before, 4)
        self.assertEqual(stats.blocks_after, 1)
        self.assertEqual(stats.merged_away, 3)
        self.assertEqual(stats.merged_blocks, 3)   # three absorptions
        self.assertEqual(stats.sections_touched, 1)


if __name__ == "__main__":
    unittest.main()
