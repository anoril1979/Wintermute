"""Tests for the unified id scheme (src/extraction/ids.py) and its
persistence: JSON roundtrip and summarization-fingerprint stability."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.extraction.ids import (
    CHP_PREFIX,
    DOC_PREFIX,
    PG_PREFIX,
    SEC_PREFIX,
    TXT_PREFIX,
    assign_extract_ids,
    doc_id_from_filename,
    flat_id,
    full_id,
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
from src.helpers.document_extract_json_store import (
    load_extract,
    save_extract,
)
from src.summarization.summarized_store import content_fingerprint


def _doc(**overrides) -> DocumentExtract:
    def block(block_id: int, page: int, text: str) -> TextBlock:
        return TextBlock(block_id=block_id, page_number=page,
                         bbox=(0.0, 0.0, 10.0, 10.0), raw_text=text)

    def page(number: int, text: str) -> PageContent:
        return PageContent(
            page_number=number, width=595.0, height=842.0, raw_text=text,
            sections=[Section(section_id=0, blocks=[block(0, number, text)],
                              page_number=number, bbox=(0.0, 0.0, 10.0, 10.0),
                              raw_text=text)],
        )

    doc = DocumentExtract(
        source_path="data/sources/pdf/Dark Earth - Gazette #1.pdf",
        title="Dark Earth - Gazette #1",
        author="", subject="", total_pages=2,
        toc=[TocEntry(level=1, title="One", page_number=1, page_index=0),
             TocEntry(level=1, title="Two", page_number=2, page_index=1)],
        chapters=[
            Chapter(toc_entry=TocEntry(level=1, title="One", page_number=1, page_index=0),
                    pages=[page(1, "alpha")], full_text="alpha", summary=None),
            Chapter(toc_entry=TocEntry(level=1, title="Two", page_number=2, page_index=1),
                    pages=[page(2, "beta")], full_text="beta", summary=None),
        ],
        summary=None,
        orphan_pages=[page(3, "gamma")],
    )
    for key, value in overrides.items():
        setattr(doc, key, value)
    return doc


class DocIdTest(unittest.TestCase):
    def test_deterministic_and_normalized(self):
        self.assertEqual(doc_id_from_filename("Gazette.PDF"),
                         doc_id_from_filename("gazette.pdf"))
        self.assertTrue(doc_id_from_filename("Gazette.pdf").startswith(f"{DOC_PREFIX}:"))

    def test_extension_matters(self):
        self.assertNotEqual(doc_id_from_filename("Gazette.pdf"),
                            doc_id_from_filename("Gazette.md"))

    def test_path_ignored_id_comes_from_filename(self):
        # Same filename in a different folder → same id.
        self.assertEqual(
            doc_id_from_filename("data/sources/pdf/Gazette.pdf".rsplit("/", 1)[-1]),
            doc_id_from_filename("Gazette.pdf"),
        )


class FullIdTest(unittest.TestCase):
    def test_chain_is_one_based_per_parent(self):
        self.assertEqual(
            full_id("doc:abcd1234", chapter_index=0, page_index=1,
                    section_index=0, block_index=2),
            "doc:abcd1234::chp:1::pg:2::sec:1::txt:3",
        )

    def test_orphan_page_has_no_chapter_segment(self):
        self.assertEqual(full_id("doc:abcd1234", page_index=0),
                         "doc:abcd1234::pg:1")

    def test_flat_id_is_one_based(self):
        self.assertEqual(flat_id(CHP_PREFIX, 0), "chp:1")
        self.assertEqual(flat_id(PG_PREFIX, 1), "pg:2")
        self.assertEqual(flat_id(SEC_PREFIX, 2), "sec:3")
        self.assertEqual(flat_id(TXT_PREFIX, 9), "txt:10")


class AssignIdsTest(unittest.TestCase):
    def test_assigns_every_level(self):
        doc = assign_extract_ids(_doc())
        self.assertTrue(doc.id.startswith("doc:"))
        self.assertEqual(doc.chapters[0].id, "chp:1")
        self.assertEqual(doc.chapters[1].id, "chp:2")
        self.assertEqual(doc.chapters[0].pages[0].id, "pg:1")
        self.assertEqual(doc.chapters[0].pages[0].sections[0].id, "sec:1")
        self.assertEqual(doc.chapters[0].pages[0].sections[0].blocks[0].id, "txt:1")
        self.assertEqual(doc.orphan_pages[0].id, "pg:1")

    def test_document_id_from_filename_not_path(self):
        doc = assign_extract_ids(_doc())
        self.assertEqual(doc.id, doc_id_from_filename("dark earth - gazette #1.pdf"))

    def test_falls_back_to_title(self):
        doc = assign_extract_ids(_doc(source_path=""))
        self.assertEqual(doc.id, doc_id_from_filename("dark earth - gazette #1"))

    def test_no_name_raises(self):
        with self.assertRaises(ValueError):
            assign_extract_ids(_doc(source_path="", title=""))

    def test_idempotent_existing_ids_are_kept(self):
        doc = _doc()
        doc.id = "doc:cafe1234"
        doc.chapters[0].id = "chp:1"
        doc.chapters[0].pages[0].id = "pg:7"  # unusual but explicit
        assign_extract_ids(doc)
        self.assertEqual(doc.id, "doc:cafe1234")
        self.assertEqual(doc.chapters[0].id, "chp:1")
        self.assertEqual(doc.chapters[0].pages[0].id, "pg:7")
        # The rest is still assigned.
        self.assertEqual(doc.chapters[1].id, "chp:2")


class PersistenceTest(unittest.TestCase):
    def test_ids_survive_json_roundtrip(self):
        doc = assign_extract_ids(_doc())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gazette.json"
            save_extract(doc, path)
            loaded = load_extract(path)
        self.assertEqual(loaded.id, doc.id)
        self.assertEqual(loaded.chapters[0].pages[0].sections[0].blocks[0].id,
                         doc.chapters[0].pages[0].sections[0].blocks[0].id)

    def test_legacy_file_without_ids_loads(self):
        doc = _doc()  # no ids assigned
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gazette.json"
            save_extract(doc, path)
            loaded = load_extract(path)
        self.assertEqual(loaded.id, "")  # tolerated; agent assigns later


class FingerprintStabilityTest(unittest.TestCase):
    def test_id_assignment_does_not_invalidate_summaries(self):
        """Identity changes must never look like content changes."""
        without_ids = _doc()
        with_ids = assign_extract_ids(_doc())
        self.assertEqual(content_fingerprint(without_ids),
                         content_fingerprint(with_ids))

    def test_real_content_change_still_invalidates(self):
        doc = assign_extract_ids(_doc())
        other = assign_extract_ids(_doc())
        other.chapters[0].pages[0].sections[0].blocks[0].raw_text = "changed"
        self.assertNotEqual(content_fingerprint(doc), content_fingerprint(other))


if __name__ == "__main__":
    unittest.main()
