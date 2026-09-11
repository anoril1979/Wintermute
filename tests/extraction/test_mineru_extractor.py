"""Tests for the extraction layer: models, base classes, MinerU extractor."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fitz  # PyMuPDF — used to build tiny fixture PDFs

from src.tools.config_loader import PROJECT_ROOT
from src.extraction import (
    BlockType,
    DocumentExtractor,
    DocumentExtractorProtocol,
    MineruPDFExtractor,
    PDFExtractor,
    PDFExtractorProtocol,
    document_extract_to_dict,
)

# MinerU-format fixture: flat list of blocks (text/table) with page_idx,
# bbox and optional text_level (a block with text_level starts a section).
MINERU_JSON = [
    {"type": "text", "page_idx": 0, "text": "Cover page intro", "bbox": [0, 0, 100, 20]},
    {"type": "text", "page_idx": 1, "text": "Chapter One", "text_level": 1, "bbox": [0, 0, 100, 20]},
    {"type": "text", "page_idx": 1, "text": "Body of chapter one", "bbox": [0, 30, 100, 50]},
    {"type": "text", "page_idx": 2, "text": "Chapter Two", "text_level": 1, "bbox": [0, 0, 100, 20]},
    {"type": "text", "page_idx": 2, "text": "Body of chapter two", "bbox": [0, 30, 100, 50]},
]


def build_tiny_pdf(path: Path) -> None:
    """Create a real 3-page PDF with a TOC (chapters start at pages 2 and 3)."""
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"page {i + 1} body")
    doc.set_toc([[1, "Chapter One", 2], [1, "Chapter Two", 3]])  # 1-based pages
    doc.save(str(path))
    doc.close()


class ExtractorBasicsTest(unittest.TestCase):
    def test_hierarchy(self):
        self.assertTrue(issubclass(PDFExtractor, DocumentExtractor))
        self.assertTrue(issubclass(MineruPDFExtractor, PDFExtractor))

    def test_protocol_conformance(self):
        extractor = MineruPDFExtractor(bypass_ocr=True)
        self.assertIsInstance(extractor, DocumentExtractorProtocol)
        self.assertIsInstance(extractor, PDFExtractorProtocol)

    def test_base_extractor_hooks_are_abstract(self):
        # Validation runs first, so the hook failure needs an existing file.
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "doc.pdf"
            existing.write_bytes(b"%PDF-1.4 fake")
            with self.assertRaises(NotImplementedError):
                DocumentExtractor().extract(existing)

    def test_pdf_extractor_rejects_non_pdf_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            txt = Path(tmp) / "doc.txt"
            txt.write_text("hello", encoding="utf-8")
            with self.assertRaises(ValueError):
                PDFExtractorDummy().extract(txt)

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            MineruPDFExtractor(bypass_ocr=True).extract(Path("no/such/file.pdf"))


class PDFExtractorDummy(PDFExtractor):
    """Concrete PDFExtractor whose hooks are no-ops (suffix validation test)."""

    def _run_backend(self, path: Path) -> None:
        pass

    def _build_document(self, path: Path) -> "object":  # pragma: no cover
        raise AssertionError("should not be reached in suffix test")


class MineruExtractorTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.pdf_path = self.tmp / "sample.pdf"
        build_tiny_pdf(self.pdf_path)

        # MinerU output layout: <folder>/<stem>/auto/<stem>_content_list.json
        output_dir = self.tmp / "extracted" / "sample" / "auto"
        output_dir.mkdir(parents=True)
        (output_dir / "sample_content_list.json").write_text(
            json.dumps(MINERU_JSON), encoding="utf-8"
        )
        self.extractor = MineruPDFExtractor(
            mineru_folder=self.tmp / "extracted", bypass_ocr=True
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_extract_returns_document(self):
        document = self.extractor.extract(self.pdf_path)
        self.assertEqual(document.source_path, str(self.pdf_path))  # full path, not stem
        self.assertEqual(document.total_pages, 3)
        self.assertEqual(len(document.toc), 2)
        self.assertEqual(document.title, "sample")  # fixture PDF has no metadata title

    def test_pages_and_sections(self):
        document = self.extractor.extract(self.pdf_path)
        pages = {p.page_number: p for p in document.all_pages()}
        self.assertEqual(set(pages), {1, 2, 3})

        # Page 1: single untitled section, not orphan (first page overall).
        self.assertEqual(len(pages[1].sections), 1)
        self.assertIsNone(pages[1].sections[0].section_title)
        self.assertFalse(pages[1].sections[0].is_orphan)

        # Page 2: titled section ("Chapter One") holding one body block.
        self.assertEqual(len(pages[2].sections), 1)
        section = pages[2].sections[0]
        self.assertEqual(section.section_title, "Chapter One")
        self.assertEqual(len(section.blocks), 1)
        self.assertEqual(section.blocks[0].block_type, BlockType.TEXT)
        self.assertIn("Body of chapter one", section.raw_text)

    def test_page_geometry_read_from_pdf(self):
        """Page width/height come from PyMuPDF (MinerU JSON has none)."""
        document = self.extractor.extract(self.pdf_path)
        for page in document.all_pages():
            self.assertIsNotNone(page.width)
            self.assertIsNotNone(page.height)
            self.assertGreater(page.width, 0)
            self.assertGreater(page.height, 0)

    def test_extract_page_uses_zero_based_geometry_key(self):
        """geometry maps 0-based page index → (width, height); missing key → None."""
        groups = [{"type": "text", "page_idx": 0, "text": "x", "bbox": [0, 0, 1, 1]}]
        page = MineruPDFExtractor.extract_page(groups, [], geometry={0: (100.0, 200.0)})
        self.assertEqual(page.page_number, 1)
        self.assertEqual(page.width, 100.0)
        self.assertEqual(page.height, 200.0)

        unknown = MineruPDFExtractor.extract_page(groups, [], geometry={5: (1.0, 1.0)})
        self.assertIsNone(unknown.width)
        self.assertIsNone(unknown.height)

    def test_chapters_and_orphans(self):
        document = self.extractor.extract(self.pdf_path)
        # TOC chapters + the fallback "Orphans" chapter for page 1.
        self.assertEqual(len(document.chapters), 3)
        by_title = {c.toc_entry.title: c for c in document.chapters}

        self.assertEqual([p.page_number for p in by_title["Chapter One"].pages], [2])
        self.assertEqual([p.page_number for p in by_title["Chapter Two"].pages], [3])
        # Boundary pages are NOT double-assigned (cleaned off-by-one).
        self.assertNotIn(3, [p.page_number for p in by_title["Chapter One"].pages])

        orphans = by_title["Orphans"]
        self.assertEqual([p.page_number for p in orphans.pages], [1])
        self.assertEqual(document.orphan_pages, orphans.pages)
        self.assertIn("Cover page intro", orphans.full_text)

    def test_chapter_full_text(self):
        document = self.extractor.extract(self.pdf_path)
        chapter_one = next(c for c in document.chapters if c.toc_entry.title == "Chapter One")
        self.assertIn("Chapter One", chapter_one.full_text)
        self.assertIn("Body of chapter one", chapter_one.full_text)
        self.assertEqual(chapter_one.metadata["page_count"], 1)

    def test_missing_mineru_output_raises(self):
        extractor = MineruPDFExtractor(
            mineru_folder=self.tmp / "nowhere", bypass_ocr=True
        )
        with self.assertRaises(FileNotFoundError):
            extractor.extract(self.pdf_path)

    def test_to_dict_serialization_round_trip(self):
        document = self.extractor.extract(self.pdf_path)
        serialized = document_extract_to_dict(document)
        # JSON-serializable and shape-stable.
        payload = json.dumps(serialized, ensure_ascii=False)
        self.assertIn("Chapter One", payload)
        self.assertEqual(serialized["total_pages"], 3)
        self.assertEqual(serialized["orphan_page_count"], 1)
        self.assertEqual(len(serialized["chapters"]), 3)


class ConfigDrivenDefaultsTest(unittest.TestCase):
    """Output folder and JSON suffix come from config/ingestion.yaml."""

    def _extractor(self, config: dict) -> MineruPDFExtractor:
        with mock.patch(
            "src.tools.config_loader.load_ingestion_config", return_value=config
        ):
            return MineruPDFExtractor(bypass_ocr=True)

    def test_config_values_are_honored(self):
        extractor = self._extractor(
            {
                "documents_root": "data/sources",
                "extensions": {".pdf": "pdf"},
                "extraction_output_dir": "custom/out",
                "mineru_json_extension": "_x.json",
            }
        )
        self.assertEqual(extractor.mineru_folder, PROJECT_ROOT / "custom" / "out")
        self.assertEqual(extractor.json_extension, "_x.json")

    def test_absolute_output_dir_is_used_as_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            extractor = self._extractor(
                {
                    "documents_root": "data/sources",
                    "extensions": {".pdf": "pdf"},
                    "extraction_output_dir": tmp,
                }
            )
            self.assertEqual(extractor.mineru_folder, Path(tmp))

    def test_fallbacks_when_keys_missing(self):
        extractor = self._extractor(
            {"documents_root": "data/sources", "extensions": {}}
        )
        self.assertEqual(
            extractor.mineru_folder, PROJECT_ROOT / "data" / "extracted" / "mineru"
        )
        self.assertEqual(extractor.json_extension, "_content_list.json")

    def test_legacy_extraction_output_dir_is_honored_as_fallback(self):
        """Configurations written before the two-store split keep working."""
        extractor = self._extractor(
            {"documents_root": "data/sources", "extensions": {},
             "extraction_output_dir": "legacy/out"}
        )
        self.assertEqual(extractor.mineru_folder, PROJECT_ROOT / "legacy" / "out")

    def test_mineru_output_dir_key_wins_over_legacy(self):
        extractor = self._extractor(
            {"documents_root": "data/sources", "extensions": {},
             "extraction_output_dir": "legacy/out",
             "extraction_mineru_output_dir": "mineru/out"}
        )
        self.assertEqual(extractor.mineru_folder, PROJECT_ROOT / "mineru" / "out")

    def test_explicit_folder_argument_wins_over_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            extractor = self._extractor(
                {
                    "documents_root": "data/sources",
                    "extensions": {},
                    "extraction_output_dir": "custom/out",
                }
            )
            extractor = MineruPDFExtractor(mineru_folder=tmp, bypass_ocr=True)
            self.assertEqual(extractor.mineru_folder, Path(tmp))

    def test_configured_extension_used_in_output_lookup(self):
        """The extension read from config drives the JSON file lookup."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pdf_path = tmp_path / "sample.pdf"
            build_tiny_pdf(pdf_path)
            output_dir = tmp_path / "out" / "sample" / "auto"
            output_dir.mkdir(parents=True)
            # Deliberately NOT the default suffix: only a config-driven
            # json_extension makes this extraction succeed.
            (output_dir / "sample_x.json").write_text(
                json.dumps(MINERU_JSON), encoding="utf-8"
            )

            with mock.patch(
                "src.tools.config_loader.load_ingestion_config",
                return_value={
                    "documents_root": "data/sources",
                    "extensions": {},
                    "extraction_output_dir": str(tmp_path / "out"),
                    "mineru_json_extension": "_x.json",
                },
            ):
                extractor = MineruPDFExtractor(bypass_ocr=True)
                document = extractor.extract(pdf_path)

            self.assertEqual(document.total_pages, 3)


if __name__ == "__main__":
    unittest.main()
