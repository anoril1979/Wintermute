"""Tests for the canonical extracted-content JSON store."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.helpers.document_extract_json_store import (
    ExtractJsonError,
    canonical_path_for,
    document_extract_from_json_dict,
    document_extract_to_json_dict,
    extraction_output_dir,
    load_extract,
    save_extract,
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
from src.tools.config_loader import PROJECT_ROOT


def make_document(path: str = "x/doc.pdf") -> DocumentExtract:
    """A realistic DocumentExtract exercising every model field."""
    return DocumentExtract(
        source_path=path,
        title="Test Doc",
        author="Author",
        subject="Subject",
        total_pages=2,
        toc=[TocEntry(level=1, title="Ch", page_number=2, page_index=1)],
        chapters=[
            Chapter(
                toc_entry=TocEntry(level=1, title="Ch", page_number=2, page_index=1),
                pages=[
                    PageContent(
                        page_number=2, width=612.0, height=792.0, raw_text="body",
                        summary=None,
                        sections=[
                            Section(
                                section_id=0, page_number=2,
                                bbox=(0.0, 0.0, 10.0, 10.0),
                                raw_text="body", section_title="Ch", section_level=1,
                                is_orphan=False,
                                blocks=[
                                    TextBlock(
                                        block_id=0, page_number=2, bbox=(0, 0, 5, 5),
                                        raw_text="body", block_type=BlockType.TABLE,
                                        text_level=1, summary="s",
                                    )
                                ],
                            )
                        ],
                    )
                ],
                full_text="body",
                summary="chapter summary",
                metadata={"page_count": 1},
            )
        ],
        orphan_pages=[PageContent(page_number=1, width=None, height=None, raw_text="")],
        metadata={"producer": "p", "encrypted": False},
    )


class RoundTripTest(unittest.TestCase):
    def test_save_then_load_is_lossless(self):
        doc = make_document()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.json"
            save_extract(doc, path)
            restored = load_extract(path)
        self.assertEqual(restored, doc)

    def test_saved_file_shape(self):
        doc = make_document()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.json"
            save_extract(doc, path)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema"], "wintermute-extract/1")
        self.assertEqual(data["title"], "Test Doc")
        self.assertEqual(
            data["chapters"][0]["pages"][0]["sections"][0]["blocks"][0]["block_type"],
            "table",
        )
        self.assertIsNone(data["orphan_pages"][0]["width"])
        self.assertEqual(data["chapters"][0]["metadata"], {"page_count": 1})

    def test_dict_level_round_trip(self):
        doc = make_document()
        restored = document_extract_from_json_dict(document_extract_to_json_dict(doc))
        self.assertEqual(restored, doc)


class DefensiveLoadTest(unittest.TestCase):
    """The canonical file may be hand-edited: every malformed entry must be
    reported with an explicit locator, never guessed."""

    def _write_and_load(self, payload) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.json"
            if isinstance(payload, str):
                path.write_text(payload, encoding="utf-8")
            else:
                path.write_text(json.dumps(payload), encoding="utf-8")
            load_extract(path)

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_extract(Path("nowhere/doc.json"))

    def test_broken_json_reports_line_and_column(self):
        with self.assertRaises(ExtractJsonError) as ctx:
            self._write_and_load("{broken")
        self.assertIn("invalid JSON", str(ctx.exception))
        self.assertIn("line 1", str(ctx.exception))

    def test_missing_identity_fields_are_required(self):
        with self.assertRaises(ExtractJsonError) as ctx:
            self._write_and_load({"chapters": []})
        self.assertIn("missing required", str(ctx.exception))

    def test_non_dict_root(self):
        with self.assertRaises(ExtractJsonError) as ctx:
            self._write_and_load([1, 2, 3])
        self.assertIn("root", str(ctx.exception))

    def test_malformed_chapter_carries_locator(self):
        payload = document_extract_to_json_dict(make_document())
        payload["chapters"] = [None]
        with self.assertRaises(ExtractJsonError) as ctx:
            self._write_and_load(payload)
        self.assertIn("chapters[0]", str(ctx.exception))

    def test_malformed_block_type_carries_locator(self):
        payload = document_extract_to_json_dict(make_document())
        payload["chapters"][0]["pages"][0]["sections"][0]["blocks"][0]["block_type"] = "hieroglyph"
        with self.assertRaises(ExtractJsonError) as ctx:
            self._write_and_load(payload)
        self.assertIn("block_type", str(ctx.exception))
        self.assertIn("hieroglyph", str(ctx.exception))

    def test_wrong_bbox_arity(self):
        payload = document_extract_to_json_dict(make_document())
        payload["chapters"][0]["pages"][0]["sections"][0]["bbox"] = [0, 0, 1]
        with self.assertRaises(ExtractJsonError) as ctx:
            self._write_and_load(payload)
        self.assertIn("bbox", str(ctx.exception))

    def test_wrong_scalar_type_named(self):
        payload = document_extract_to_json_dict(make_document())
        payload["total_pages"] = "two"
        with self.assertRaises(ExtractJsonError) as ctx:
            self._write_and_load(payload)
        self.assertIn("total_pages", str(ctx.exception))

    def test_missing_toc_entry_in_chapter(self):
        payload = document_extract_to_json_dict(make_document())
        del payload["chapters"][0]["toc_entry"]
        with self.assertRaises(ExtractJsonError) as ctx:
            self._write_and_load(payload)
        self.assertIn("toc_entry", str(ctx.exception))

    def test_wrong_list_type_carries_locator(self):
        payload = document_extract_to_json_dict(make_document())
        payload["toc"] = "not-a-list"
        with self.assertRaises(ExtractJsonError) as ctx:
            self._write_and_load(payload)
        self.assertIn("toc", str(ctx.exception))


class PathResolutionTest(unittest.TestCase):
    def _with_config(self, config):
        return mock.patch(
            "src.tools.config_loader.load_ingestion_config", return_value=config
        )

    def test_relative_dir_resolved_against_project_root(self):
        with self._with_config(
            {"documents_root": "d", "extensions": {},
             "extraction_output_dir": "custom/extracted"}
        ):
            self.assertEqual(extraction_output_dir(), PROJECT_ROOT / "custom" / "extracted")

    def test_absolute_dir_used_as_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._with_config(
                {"documents_root": "d", "extensions": {}, "extraction_output_dir": tmp}
            ):
                self.assertEqual(extraction_output_dir(), Path(tmp))

    def test_fail_open_to_default_when_config_unreadable(self):
        with mock.patch(
            "src.tools.config_loader.load_ingestion_config",
            side_effect=RuntimeError("boom"),
        ):
            self.assertEqual(extraction_output_dir(), PROJECT_ROOT / "data" / "extracted")

    def test_canonical_path_uses_document_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._with_config(
                {"documents_root": "d", "extensions": {}, "extraction_output_dir": tmp}
            ):
                self.assertEqual(
                    canonical_path_for(Path("data/sources/pdf/Dumas.pdf")),
                    Path(tmp) / "Dumas.json",
                )


if __name__ == "__main__":
    unittest.main()
