"""Tests for the extraction agent's checkpoint/resume/force behavior."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.agents.contexts import IngestionContext
from src.agents.agents.pdf_extraction_agent import (
    FORCE_EXTRACTION_KEY,
    OUTPUT_KEY,
    PDFExtractionAgent,
)
from src.tools.extraction_job_file import ExtractionJobFile
from src.agents.protocols import AgentStatus
from src.extraction.document_extractor import DocumentExtractor
from src.extraction.models import DocumentExtract
from src.extraction.mineru_pdf_extractor import MineruPDFExtractor


def make_document(path: Path, total_pages: int = 2) -> DocumentExtract:
    return DocumentExtract(
        source_path=str(path), title="Test Doc", author="", subject="",
        total_pages=total_pages, toc=[], chapters=[], orphan_pages=[], metadata={},
    )


class StubExtractor(DocumentExtractor):
    def __init__(self, document=None, exc=None):
        self.document = document
        self.exc = exc
        self.calls = 0

    def _run_backend(self, path):
        pass

    def _build_document(self, path):
        self.calls += 1  # one extract() == one unit of work
        if self.exc is not None:
            raise self.exc
        return self.document


class FlakyMineruExtractor(MineruPDFExtractor):
    """MinerU-shaped extractor whose first call fails (missing artifacts)."""

    def __init__(self, document, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.document = document
        self.calls = 0

    def extract(self, path):
        self.calls += 1
        if self.calls == 1:
            raise FileNotFoundError("artifacts missing")
        return self.document


class CheckpointAgentTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.pdf = self.tmp / "doc.pdf"
        self.pdf.write_bytes(b"%PDF-1.4")
        self.job_file = ExtractionJobFile(self.tmp / "jobs.json")
        # Hermetic canonical store (never the real data/extracted).
        self.canonical_dir = self.tmp / "extracted"
        self.canonical = self.canonical_dir / "doc.json"

    def tearDown(self):
        self._tmp.cleanup()

    def _agent(self, extractor=None):
        return PDFExtractionAgent(
            extractor=extractor or StubExtractor(document=make_document(self.pdf)),
            job_file=self.job_file,
            canonical_dir=self.canonical_dir,
        )

    def _context(self, force=False):
        context = IngestionContext(document_path=self.pdf, request="[test]")
        if force:
            context.metadata[FORCE_EXTRACTION_KEY] = True
        return context

    # -- fresh extraction records the checkpoint ------------------------------

    def test_successful_extraction_records_job_file(self):
        agent = self._agent()
        result = agent.run(self._context())

        self.assertEqual(result.status, AgentStatus.OK)
        self.assertEqual(result.payload["checkpoint"], "recorded")
        self.assertEqual(self.job_file.entries(), ["doc.pdf"])
        # The fixable JSON exists before the entry was recorded.
        self.assertTrue(self.canonical.exists())

    def test_failed_extraction_does_not_record(self):
        agent = self._agent(StubExtractor(exc=ValueError("bad pdf")))
        result = agent.run(self._context())

        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(self.job_file.entries(), [])

    # -- checkpoint hit --------------------------------------------------------

    def test_second_run_short_circuits_for_stub_extractor(self):
        """The second run resumes from the canonical JSON (no extractor run,
        no checkpoint machinery needed)."""
        stub = StubExtractor(document=make_document(self.pdf))
        agent = self._agent(stub)
        agent.run(self._context())

        second_context = self._context()
        result = agent.run(second_context)
        self.assertEqual(result.status, AgentStatus.OK)
        self.assertEqual(result.payload["resume"], "canonical_json")
        self.assertEqual(stub.calls, 1, "extractor must not run again")
        self.assertEqual(second_context.metadata["resume_source"], "canonical_json")

    def test_checkpoint_hit_leaves_metadata_marker(self):
        """Legacy scenario (no canonical JSON): the job-file checkpoint is
        consulted and marks the context metadata."""
        agent = self._agent()
        agent.run(self._context())
        self.canonical.unlink()  # pre-two-store-split state

        context = self._context()
        agent.run(context)
        self.assertEqual(context.metadata.get("checkpoint_status"), "already_done")

    def test_stale_source_reported_in_payload(self):
        """Legacy scenario (no canonical JSON): a changed source is reported
        as stale by the job-file checkpoint."""
        agent = self._agent()
        agent.run(self._context())
        self.canonical.unlink()  # pre-two-store-split state
        # Simulate a source change after extraction.
        import os
        os.utime(self.pdf, (0, 0))

        context = self._context()
        result = agent.run(context)
        self.assertEqual(result.payload["checkpoint"], "stale")
        self.assertEqual(context.metadata["checkpoint_status"], "stale")

    # -- force bypass ----------------------------------------------------------

    def test_force_bypasses_checkpoint_and_reruns_extractor(self):
        stub = StubExtractor(document=make_document(self.pdf))
        agent = self._agent(stub)
        agent.run(self._context())

        result = agent.run(self._context(force=True))
        self.assertEqual(result.status, AgentStatus.OK)
        self.assertEqual(result.payload["checkpoint"], "recorded")
        self.assertEqual(stub.calls, 2, "force must re-run the extractor")
        # The entry is refreshed (mtime/size snapshot updated).

    def test_force_still_records_on_success(self):
        agent = self._agent()
        agent.run(self._context())
        agent.run(self._context(force=True))
        self.assertEqual(self.job_file.entries(), ["doc.pdf"])

    # -- resume (MinerU artifacts) ----------------------------------------------

    def test_checkpoint_hit_resumes_from_mineru_artifacts(self):
        """Legacy scenario (checkpoint recorded, canonical JSON missing):
        a checkpoint hit rebuilds the DocumentExtract from the artifacts via
        bypass_ocr (no MinerU run) and self-heals the canonical store."""
        # A REAL pdf is required: fitz opens it for metadata/TOC.
        import fitz

        real_pdf = self.tmp / "doc.pdf"
        with fitz.open() as doc:
            page = doc.new_page()
            page.insert_text((72, 72), "Hello World")
            doc.save(real_pdf)
        self.pdf = real_pdf

        mineru_json = [
            {"type": "text", "text": "Hello", "text_level": 1, "bbox": [0, 0, 10, 10], "page_idx": 0},
            {"type": "text", "text": "World", "bbox": [0, 5, 10, 15], "page_idx": 0},
        ]
        out_dir = self.tmp / "out" / "doc" / "auto"
        out_dir.mkdir(parents=True)
        (out_dir / "doc_content_list.json").write_text(json.dumps(mineru_json), encoding="utf-8")

        extractor = MineruPDFExtractor(
            mineru_folder=self.tmp / "out", bypass_ocr=True,
        )
        agent = PDFExtractionAgent(extractor=extractor, job_file=self.job_file,
                                   canonical_dir=self.canonical_dir)

        # First run: real extraction (bypassed OCR), records the checkpoint.
        first = agent.run(self._context())
        self.assertEqual(first.status, AgentStatus.OK)
        self.assertEqual(first.payload["checkpoint"], "recorded")

        # Second run with the canonical JSON wiped: legacy resume path.
        self.canonical.unlink()
        second_context = self._context()
        second = agent.run(second_context)
        self.assertEqual(second.status, AgentStatus.OK)
        self.assertEqual(second.payload["checkpoint"], "already_done")
        self.assertEqual(second.payload["resume"], "mineru_artifacts")
        self.assertIn(OUTPUT_KEY, second_context.outputs)
        self.assertEqual(second_context.outputs[OUTPUT_KEY].title, "doc")
        # The canonical store self-healed from the artifacts.
        self.assertTrue(self.canonical.exists())

    def test_checkpoint_hit_with_missing_artifacts_falls_back_to_extraction(self):
        """Job file says done but artifacts were wiped: re-extract."""
        # Record the checkpoint without ever running the extractor.
        self.job_file.record("doc.pdf", self.pdf)

        # Resume fails (no artifacts) -> the agent falls back to a fresh
        # extraction, which succeeds and refreshes the checkpoint.
        extractor = FlakyMineruExtractor(
            document=make_document(self.pdf), mineru_folder=self.tmp / "empty",
        )
        agent = PDFExtractionAgent(extractor=extractor, job_file=self.job_file,
                                   canonical_dir=self.canonical_dir)
        context = self._context()
        result = agent.run(context)

        self.assertEqual(result.status, AgentStatus.OK)
        self.assertEqual(result.payload["checkpoint"], "recorded")
        self.assertIn(OUTPUT_KEY, context.outputs)
        self.assertEqual(extractor.calls, 2, "resume attempt + fresh extraction")


if __name__ == "__main__":
    unittest.main()
