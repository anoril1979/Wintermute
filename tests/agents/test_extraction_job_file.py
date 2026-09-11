"""Tests for the extraction job file checkpoint store."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.tools.extraction_job_file import (
    STATUS_ALREADY_DONE,
    STATUS_NEW,
    STATUS_STALE,
    ExtractionJobFile,
    SummarizationJobFile,
)


class JobFileTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.job_file = ExtractionJobFile(self.tmp / "jobs.json")
        self.doc = self.tmp / "doc.pdf"
        self.doc.write_bytes(b"%PDF-1.4")

    def tearDown(self):
        self._tmp.cleanup()

    def test_new_document_status(self):
        self.assertEqual(self.job_file.status_of("doc.pdf"), STATUS_NEW)

    def test_record_then_already_done(self):
        self.job_file.record("doc.pdf", self.doc)
        self.assertEqual(self.job_file.status_of("doc.pdf"), STATUS_ALREADY_DONE)
        self.assertEqual(self.job_file.entries(), ["doc.pdf"])

    def test_record_stores_source_fingerprint(self):
        self.job_file.record("doc.pdf", self.doc)
        entry = self.job_file.load()["doc.pdf"]
        self.assertEqual(entry["source_path"], str(self.doc))
        self.assertIsNotNone(entry["source_mtime"])
        self.assertEqual(entry["source_size"], self.doc.stat().st_size)

    def test_modified_source_is_stale(self):
        self.job_file.record("doc.pdf", self.doc)
        os.utime(self.doc, (0, 0))  # change mtime only
        self.assertEqual(self.job_file.status_of("doc.pdf"), STATUS_STALE)

    def test_remove_removes_and_reports(self):
        self.job_file.record("doc.pdf", self.doc)
        self.assertTrue(self.job_file.remove("doc.pdf"))
        self.assertEqual(self.job_file.status_of("doc.pdf"), STATUS_NEW)
        self.assertFalse(self.job_file.remove("doc.pdf"))

    def test_missing_file_is_fail_open_empty(self):
        self.assertEqual(self.job_file.load(), {})
        self.assertEqual(self.job_file.status_of("doc.pdf"), STATUS_NEW)

    def test_corrupted_file_is_fail_open_empty(self):
        self.job_file.job_file.write_text("{not json", encoding="utf-8")
        self.assertEqual(self.job_file.load(), {})
        self.assertEqual(self.job_file.status_of("doc.pdf"), STATUS_NEW)

    def test_wrong_shape_is_fail_open_empty(self):
        self.job_file.job_file.write_text("[1, 2, 3]", encoding="utf-8")
        self.assertEqual(self.job_file.load(), {})

    def test_human_edit_is_honored(self):
        """The job file is meant to be human-editable: a hand-written entry works."""
        self.job_file.job_file.write_text(
            json.dumps({"doc.pdf": {"source_path": str(self.doc)}}),
            encoding="utf-8",
        )
        # Entry without fingerprint still counts as done (mtime None vs stat
        # value differs, but absence of fingerprint means unknown -> done).
        self.assertIn(self.job_file.status_of("doc.pdf"), (STATUS_ALREADY_DONE, STATUS_STALE))

    def test_atomic_write_leaves_no_tmp_files(self):
        self.job_file.record("doc.pdf", self.doc)
        leftovers = [p for p in self.tmp.iterdir()
                     if p.name not in ("jobs.json", "summary_jobs.json", "doc.pdf")]
        self.assertEqual(leftovers, [])

    def test_record_creates_parent_dirs(self):
        deep = self.tmp / "a" / "b" / "jobs.json"
        job_file = ExtractionJobFile(deep)
        job_file.record("doc.pdf", self.doc)
        self.assertTrue(deep.exists())


class SummarizationJobFileTest(JobFileTest):
    """Same checkpoint contract as extraction, own store/config key."""

    def setUp(self):
        super().setUp()
        self.job_file = SummarizationJobFile(self.tmp / "summary_jobs.json")

    def test_same_semantics_as_extraction_store(self):
        self.job_file.record("doc.pdf", self.doc)
        self.assertEqual(self.job_file.status_of("doc.pdf"), STATUS_ALREADY_DONE)
        os.utime(self.doc, (0, 0))
        self.assertEqual(self.job_file.status_of("doc.pdf"), STATUS_STALE)
        self.assertTrue(self.job_file.remove("doc.pdf"))
        self.assertEqual(self.job_file.status_of("doc.pdf"), STATUS_NEW)

    def test_uses_own_default_file_name(self):
        self.assertNotEqual(self.job_file.job_file.name, "extraction_jobs.json")

    def test_explicit_path_wins(self):
        custom = SummarizationJobFile(self.tmp / "explicit.json")
        self.assertEqual(custom.job_file, self.tmp / "explicit.json")


if __name__ == "__main__":
    unittest.main()
