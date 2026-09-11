"""Tests for the extraction agent's canonical JSON save/resume workflow.

The two-store split: canonical extracted content (fixable, user-editable)
vs MinerU's working artifacts. Covers: fresh-extraction persistence,
canonical-first resume (the user-fix path), malformed-JSON self-healing,
force bypass, and the checkpoint ordering invariant.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.agents.agents.pdf_extraction_agent import (
    FORCE_EXTRACTION_KEY,
    PDFExtractionAgent,
)
from src.agents.contexts import IngestionContext
from src.extraction.document_extractor import DocumentExtractor
from src.helpers.document_extract_json_store import load_extract
from src.extraction.mineru_pdf_extractor import MineruPDFExtractor
from src.extraction.models import DocumentExtract
from src.tools.extraction_job_file import ExtractionJobFile


def make_document(path: Path, title: str = "Test Doc") -> DocumentExtract:
    from src.extraction.models import Chapter, PageContent, Section, TextBlock, TocEntry

    return DocumentExtract(
        source_path=str(path), title=title, author="", subject="", total_pages=2,
        chapters=[
            Chapter(
                toc_entry=TocEntry(1, "Ch1", 1, 0),
                pages=[
                    PageContent(
                        page_number=1, width=612, height=792, raw_text="page text",
                        sections=[
                            Section(
                                section_id=0, page_number=1, bbox=(0, 0, 100, 20),
                                raw_text="page text",
                                blocks=[TextBlock(
                                    block_id=0, page_number=1, bbox=(0, 0, 50, 10),
                                    raw_text="page text",
                                )],
                            )
                        ],
                    )
                ],
                full_text="page text",
            )
        ],
    )


class StubExtractor(DocumentExtractor):
    def __init__(self, doc=None):
        self.doc = doc

    def _run_backend(self, path):
        pass

    def _build_document(self, path):
        if self.doc is None:
            raise FileNotFoundError("nope")
        return self.doc


class ResumableExtractor(MineruPDFExtractor):
    """Mineru-derived stub: enables the artifact-rebuild (self-heal) branch."""

    def __init__(self, doc=None):
        self.bypass_ocr = False
        self.mineru_folder = None
        self.json_extension = "_content_list.json"
        self.doc = doc

    def _run_backend(self, path):
        pass

    def _build_document(self, path):
        if self.doc is None:
            raise FileNotFoundError("artifacts gone")
        return self.doc


class CanonicalSaveResumeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.pdf = self.tmp / "doc.pdf"
        self.pdf.write_bytes(b"%PDF-1.4 fake")
        self.job_file = ExtractionJobFile(self.tmp / "jobs.json")
        self.canonical_dir = self.tmp / "extracted"
        self.canonical = self.canonical_dir / "doc.json"

    def tearDown(self):
        self._tmp.cleanup()

    def _agent(self, extractor, job_file=None):
        return PDFExtractionAgent(
            extractor=extractor,
            job_file=job_file or self.job_file,
            canonical_dir=self.canonical_dir,
        )

    def _context(self):
        return IngestionContext(document_path=self.pdf, request="[test]")

    def test_fresh_extraction_saves_canonical_json_before_checkpoint(self):
        context = self._context()
        result = self._agent(StubExtractor(make_document(self.pdf))).run(context)
        self.assertEqual(result.status.value, "ok")
        self.assertTrue(self.canonical.exists())
        self.assertEqual(load_extract(self.canonical).title, "Test Doc")
        # Ordering invariant: job file entry only once the JSON is on disk.
        self.assertEqual(self.job_file.status_of("doc.pdf"), "already_done")
        self.assertEqual(result.payload["checkpoint"], "recorded")
        self.assertEqual(
            [t["kind"] for t in context.events],
            ["extracting", "canonical_saved", "extracted"],
        )

    def test_resume_from_canonical_json_skips_extractor(self):
        self._agent(StubExtractor(make_document(self.pdf))).run(self._context())

        class ExplodingExtractor(StubExtractor):
            def _run_backend(self, path):
                raise AssertionError("extractor must not run on canonical resume")

            def _build_document(self, path):
                raise AssertionError("extractor must not run on canonical resume")

        context = self._context()
        result = self._agent(ExplodingExtractor()).run(context)
        self.assertEqual(result.status.value, "ok")
        self.assertEqual(result.payload["resume"], "canonical_json")
        self.assertEqual(context.metadata["resume_source"], "canonical_json")
        self.assertEqual(
            [t["kind"] for t in context.events],
            ["canonical_found", "resumed"],
        )

    def test_user_fixed_content_is_honored(self):
        """The whole point of the canonical store: edit the JSON, re-ingest,
        get the corrected content through the pipeline — no re-extraction."""
        import json

        self._agent(StubExtractor(make_document(self.pdf))).run(self._context())
        data = json.loads(self.canonical.read_text(encoding="utf-8"))
        data["title"] = "Fixed Title"
        data["chapters"][0]["pages"][0]["raw_text"] = "corrected text"
        self.canonical.write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )

        class ExplodingExtractor(StubExtractor):
            def _run_backend(self, path):
                raise AssertionError("extractor must not run")

            def _build_document(self, path):
                raise AssertionError("extractor must not run")

        context = self._context()
        self._agent(ExplodingExtractor()).run(context)
        document = context.outputs["content_extraction"]
        self.assertEqual(document.title, "Fixed Title")
        self.assertEqual(
            document.chapters[0].pages[0].raw_text, "corrected text"
        )

    def test_malformed_canonical_falls_back_to_job_file_checkpoint(self):
        """With a stub (no artifacts), a broken canonical JSON is reported
        and the run falls through to the job-file checkpoint — honestly."""
        import json

        self._agent(StubExtractor(make_document(self.pdf))).run(self._context())
        self.canonical.write_text("{broken", encoding="utf-8")

        context = self._context()
        result = self._agent(StubExtractor()).run(context)
        kinds = [t["kind"] for t in context.events]
        self.assertEqual(
            kinds,
            ["canonical_found", "resume_failed", "checkpoint_hit", "already_done"],
        )
        self.assertEqual(result.status.value, "ok")  # honest already_done
        self.assertIn("use force", result.payload["message"])

    def test_malformed_canonical_self_heals_from_artifacts(self):
        """With a Mineru-derived extractor, the artifact-resume path rebuilds
        the canonical JSON — the store self-heals, no OCR re-run."""
        import json

        self._agent(StubExtractor(make_document(self.pdf))).run(self._context())
        self.canonical.write_text("{broken", encoding="utf-8")

        context = self._context()
        result = self._agent(
            ResumableExtractor(make_document(self.pdf, "Rebuilt"))
        ).run(context)
        kinds = [t["kind"] for t in context.events]
        self.assertEqual(
            kinds,
            ["canonical_found", "resume_failed", "checkpoint_hit",
             "resumed", "canonical_saved"],
        )
        self.assertEqual(result.payload["resume"], "mineru_artifacts")
        self.assertEqual(context.metadata["resume_source"], "mineru_artifacts")
        self.assertEqual(load_extract(self.canonical).title, "Rebuilt")

    def test_force_bypasses_canonical_json_and_reextracts(self):
        self._agent(StubExtractor(make_document(self.pdf, "Old"))).run(self._context())

        context = self._context()
        context.metadata[FORCE_EXTRACTION_KEY] = True
        result = self._agent(
            StubExtractor(make_document(self.pdf, "Fresh"))
        ).run(context)
        kinds = [t["kind"] for t in context.events]
        self.assertEqual(kinds[0], "extraction_forced")
        self.assertNotIn("canonical_found", kinds)
        self.assertEqual(load_extract(self.canonical).title, "Fresh")

    def test_canonical_save_failure_prevents_checkpoint(self):
        """A job file entry without its fixable JSON would resume into
        nothing — the checkpoint must only be recorded once the canonical
        JSON is on disk."""
        context = self._context()
        with mock.patch(
            "src.agents.agents.pdf_extraction_agent.save_extract",
            side_effect=OSError("disk full"),
        ):
            result = self._agent(StubExtractor(make_document(self.pdf))).run(context)
        self.assertEqual(result.status.value, "ok")  # work is kept
        self.assertIn(
            "canonical extraction not saved",
            context.errors["content_extraction"],
        )
        self.assertEqual(self.job_file.status_of("doc.pdf"), "new")
        self.assertIn(
            "canonical_save_failed", [t["kind"] for t in context.events]
        )


if __name__ == "__main__":
    unittest.main()
