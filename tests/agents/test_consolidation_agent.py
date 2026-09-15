"""Tests for the ConsolidationAgent: context swap, cache JSON, config errors.

Hermetic: temp output dir injected (no real data/cache writes), documents
built from the models. Config errors are exercised through a patched
``load_ingestion_config`` — no real yaml is read.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.agents.agents.consolidation_agent import (
    ConsolidationAgent,
    consolidation_path_for,
    load_consolidated_extract,
    save_consolidated_extract,
)
from src.agents.contexts import IngestionContext
from src.agents.protocols import AgentStatus
from src.extraction.models import (
    BlockType,
    Chapter,
    DocumentExtract,
    PageContent,
    Section,
    TextBlock,
    TocEntry,
)


def _document() -> DocumentExtract:
    blocks = [
        TextBlock(block_id=0, page_number=1, bbox=(0.0, 0.0, 100.0, 20.0),
                  raw_text="miaou", block_type=BlockType.TEXT, id="txt:1"),
        TextBlock(block_id=1, page_number=1, bbox=(0.0, 0.0, 100.0, 20.0),
                  raw_text="meow", block_type=BlockType.TEXT, id="txt:2"),
        TextBlock(block_id=2, page_number=1, bbox=(0.0, 0.0, 100.0, 20.0),
                  raw_text="x" * 500, block_type=BlockType.TEXT, id="txt:3"),
    ]
    page = PageContent(page_number=1, width=595.0, height=842.0,
                       raw_text="page", sections=[
                           Section(section_id=1, blocks=blocks, page_number=1,
                                   bbox=(0.0, 0.0, 595.0, 842.0),
                                   raw_text="section", id="sec:1")],
                       id="pg:1")
    chapter = Chapter(
        toc_entry=TocEntry(level=1, title="Ch1", page_number=1, page_index=0),
        pages=[page], id="chp:1",
    )
    return DocumentExtract(id="doc:00000001", title="Test Doc",
                           total_pages=1, chapters=[chapter])


class ConsolidationAgentTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self._tmp.name) / "consolidation"
        self.source = Path(self._tmp.name) / "doc.pdf"
        self.agent = ConsolidationAgent(output_dir=self.out_dir)

    def tearDown(self):
        self._tmp.cleanup()

    def _context(self, document=None):
        context = IngestionContext(document_path=self.source)
        if document is not None:
            context.outputs["content_extraction"] = document
        return context

    def _patched_limits(self, min_length=100, max_length=1000):
        return mock.patch(
            "src.agents.agents.consolidation_agent.load_ingestion_config",
            return_value={
                "paragraph_min_length": min_length,
                "paragraph_max_length": max_length,
            },
        )

    def test_swaps_consolidated_view_into_content_extraction_key(self):
        context = self._context(_document())
        with self._patched_limits():
            result = self.agent.run(context)

        self.assertEqual(result.status, AgentStatus.OK)
        blocks = context.outputs["content_extraction"] \
            .chapters[0].pages[0].sections[0].blocks
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].id, "txt:1")
        self.assertIn("consolidator", context.outputs)
        self.assertEqual(context.outputs["consolidator"]["blocks_after"], 1)

    def test_cache_json_written_to_injected_dir(self):
        context = self._context(_document())
        with self._patched_limits():
            self.agent.run(context)

        cache_path = self.out_dir / "doc.json"
        self.assertTrue(cache_path.exists())
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], "wintermute-extract/1")
        self.assertEqual(payload["id"], "doc:00000001")

    def test_cache_failure_degrades_to_warning_not_fatal(self):
        from src.extraction.json_store import ExtractJsonError

        context = self._context(_document())
        with self._patched_limits(), \
                mock.patch(
                    "src.agents.agents.consolidation_agent"
                    ".save_consolidated_extract",
                    side_effect=ExtractJsonError("disk full"),
                ):
            result = self.agent.run(context)

        self.assertEqual(result.status, AgentStatus.OK)
        self.assertIn("consolidator", context.errors)

    def test_missing_input_is_input_data_failure(self):
        context = self._context()  # no content_extraction key
        result = self.agent.run(context)

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("content_extraction", result.detail)

    def test_missing_min_length_is_config_failure(self):
        context = self._context(_document())
        with mock.patch(
            "src.agents.agents.consolidation_agent.load_ingestion_config",
            return_value={"paragraph_max_length": 1000},
        ):
            result = self.agent.run(context)

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("paragraph_min_length", result.detail)

    def test_max_below_min_is_config_failure(self):
        context = self._context(_document())
        with self._patched_limits(min_length=500, max_length=100):
            result = self.agent.run(context)

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("paragraph_max_length", result.detail)

    def test_traces_are_emitted(self):
        context = self._context(_document())
        events = []
        context.emit = lambda kind, event, detail=None, data=None, **kw: \
            events.append(event)
        with self._patched_limits():
            self.agent.run(context)

        self.assertIn("consolidation_done", events)


class CacheStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.document = _document()

    def tearDown(self):
        self._tmp.cleanup()

    def test_roundtrip_preserves_the_document(self):
        path = self.dir / "doc.json"
        save_consolidated_extract(self.document, path)
        loaded = load_consolidated_extract(path)

        self.assertEqual(loaded.id, "doc:00000001")
        self.assertEqual(loaded.title, "Test Doc")
        blocks = loaded.chapters[0].pages[0].sections[0].blocks
        self.assertEqual([b.id for b in blocks], ["txt:1", "txt:2", "txt:3"])

    def test_load_missing_file_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            load_consolidated_extract(self.dir / "nope.json")

    def test_load_invalid_json_raises_extract_json_error(self):
        path = self.dir / "doc.json"
        path.write_text("{not json", encoding="utf-8")
        from src.extraction.json_store import ExtractJsonError
        with self.assertRaises(ExtractJsonError):
            load_consolidated_extract(path)

    def test_consolidation_path_for_uses_stem(self):
        path = consolidation_path_for(Path("data/sources/pdf/Gazette.pdf"))
        self.assertEqual(path.name, "Gazette.json")
        self.assertIn("consolidation", str(path))


if __name__ == "__main__":
    unittest.main()
