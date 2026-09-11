"""Tests for the summarized-content store (envelope, fingerprint, I/O)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.extraction.models import (
    Chapter,
    DocumentExtract,
    PageContent,
    Section,
    TextBlock,
    TocEntry,
)
from src.summarization.summarized_store import (
    SummarizedJsonError,
    content_fingerprint,
    load_summarized,
    save_summarized,
    summarized_document_from_json_dict,
    summarized_document_to_json_dict,
)


def _document() -> DocumentExtract:
    block = TextBlock(block_id=0, page_number=1, bbox=(0, 0, 10, 10), raw_text="body")
    section = Section(
        section_id=0, blocks=[block], page_number=1, bbox=(0, 0, 100, 20),
        raw_text="body",
    )
    page = PageContent(page_number=1, width=612, height=792, raw_text="body",
                       sections=[section])
    chapter = Chapter(toc_entry=TocEntry(1, "Ch1", 1, 0), pages=[page],
                      full_text="body")
    return DocumentExtract(
        source_path="doc.pdf", title="Doc", author="", subject="",
        total_pages=1, chapters=[chapter],
    )


class FingerprintTest(unittest.TestCase):
    def test_fingerprint_ignores_summaries(self):
        doc = _document()
        raw = content_fingerprint(doc)
        doc.summary = "document summary"
        doc.chapters[0].summary = "chapter summary"
        doc.chapters[0].pages[0].summary = "page summary"
        doc.chapters[0].pages[0].sections[0].summary = "section summary"
        doc.chapters[0].pages[0].sections[0].blocks[0].summary = "block summary"
        self.assertEqual(content_fingerprint(doc), raw)

    def test_fingerprint_changes_with_content(self):
        doc_a = _document()
        doc_b = _document()
        doc_b.chapters[0].pages[0].raw_text += " changed"
        self.assertNotEqual(content_fingerprint(doc_a), content_fingerprint(doc_b))

    def test_fingerprint_is_hex_sha256(self):
        fingerprint = content_fingerprint(_document())
        self.assertEqual(len(fingerprint), 64)
        int(fingerprint, 16)  # raises if not hex


class EnvelopeTest(unittest.TestCase):
    def test_round_trip(self):
        doc = _document()
        doc.summary = "the summary"
        payload = summarized_document_to_json_dict(doc, "abc123")
        restored, fingerprint = summarized_document_from_json_dict(payload)
        self.assertEqual(fingerprint, "abc123")
        self.assertEqual(restored, doc)
        self.assertEqual(restored.summary, "the summary")

    def test_wrong_schema_marker_rejected(self):
        doc = _document()
        payload = summarized_document_to_json_dict(doc, "abc")
        payload["schema"] = "something-else/9"
        with self.assertRaises(SummarizedJsonError):
            summarized_document_from_json_dict(payload)

    def test_missing_fingerprint_rejected(self):
        doc = _document()
        payload = summarized_document_to_json_dict(doc, "abc")
        del payload["content_fingerprint"]
        with self.assertRaises(SummarizedJsonError):
            summarized_document_from_json_dict(payload)

    def test_missing_document_rejected(self):
        payload = {"schema": "wintermute-summarized/1", "content_fingerprint": "abc"}
        with self.assertRaises(SummarizedJsonError):
            summarized_document_from_json_dict(payload)

    def test_malformed_document_body_reports_locator(self):
        payload = {"content_fingerprint": "abc", "document": {"chapters": "nope"}}
        with self.assertRaises(SummarizedJsonError) as ctx:
            summarized_document_from_json_dict(payload)
        self.assertIn("root.document", str(ctx.exception))


class FileIOTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_then_load_round_trip(self):
        doc = _document()
        doc.summary = "the summary"
        path = self.tmp / "doc.json"
        save_summarized(doc, path)
        restored, _ = load_summarized(path)
        self.assertEqual(restored, doc)

    def test_saved_file_is_pretty_json_with_marker(self):
        path = self.tmp / "doc.json"
        save_summarized(_document(), path)
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(raw["schema"], "wintermute-summarized/1")
        self.assertEqual(len(raw["content_fingerprint"]), 64)

    def test_load_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_summarized(self.tmp / "nowhere.json")

    def test_load_invalid_json_raises_summarized_error(self):
        path = self.tmp / "doc.json"
        path.write_text("{ broken", encoding="utf-8")
        with self.assertRaises(SummarizedJsonError):
            load_summarized(path)

    def test_save_creates_parent_folders(self):
        path = self.tmp / "a" / "b" / "doc.json"
        save_summarized(_document(), path)
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
