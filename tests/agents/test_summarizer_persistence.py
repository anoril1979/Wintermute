"""Tests for the SummarizerAgent's summarized-store resume and persistence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.agents.agents.summarizer_agent import SummarizerAgent
from src.agents.contexts import IngestionContext
from src.extraction.models import (
    Chapter,
    DocumentExtract,
    PageContent,
    Section,
    TextBlock,
    TocEntry,
)
from src.summarization.summarized_store import load_summarized, save_summarized
from src.tools.extraction_job_file import SummarizationJobFile

LONG_TEXT = "Lorem ipsum dolor sit amet " * 60  # above summary_min_chars


def _document() -> DocumentExtract:
    block = TextBlock(block_id=0, page_number=1, bbox=(0, 0, 10, 10),
                      raw_text=LONG_TEXT)
    section = Section(section_id=0, blocks=[block], page_number=1,
                      bbox=(0, 0, 100, 20), raw_text=LONG_TEXT)
    page = PageContent(page_number=1, width=612, height=792, raw_text=LONG_TEXT,
                       sections=[section])
    chapter = Chapter(toc_entry=TocEntry(1, "Ch1", 1, 0), pages=[page],
                      full_text=LONG_TEXT)
    return DocumentExtract(
        source_path="doc.pdf", title="Doc", author="", subject="",
        total_pages=1, chapters=[chapter],
    )


class FakeLLM:
    """Counts calls, returns a short summary (below min_chars: containers copy)."""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt, max_tokens=None):
        self.calls += 1
        return "SUMMARY" + "x" * 20


class SummarizerPersistenceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.sum_dir = self.tmp / "sum"
        self.doc_path = self.tmp / "doc.pdf"
        self.doc_path.write_bytes(b"%PDF-1.4")  # real file: job-file mtimes

    def tearDown(self):
        self._tmp.cleanup()

    def _job_file(self) -> SummarizationJobFile:
        return SummarizationJobFile(self.tmp / "summary_jobs.json")

    def _run(self, document: DocumentExtract, metadata: dict | None = None):
        llm = FakeLLM()
        agent = SummarizerAgent(summarized_dir=self.sum_dir,
                                job_file=self._job_file())
        context = IngestionContext(request="test")
        context.document_path = self.doc_path
        context.outputs["content_extraction"] = document
        context.metadata.update(metadata or {})
        with mock.patch.object(SummarizerAgent, "llm_client", return_value=llm):
            result = agent.run(context)
        return agent, context, llm, result

    def test_fresh_run_saves_summarized_store(self):
        document = _document()
        agent, context, llm, result = self._run(document)
        self.assertEqual(result.status.value, "ok")
        self.assertEqual(llm.calls, 2)  # block + chapter (long); rest copied
        path = self.sum_dir / "doc.json"
        self.assertTrue(path.exists())
        stored, _ = load_summarized(path)
        self.assertEqual(stored, document)
        kinds = [t["kind"] for t in context.events]
        self.assertIn("summarized_saved", kinds)
        # The checkpoint is recorded too.
        self.assertEqual(self._job_file().status_of("doc.pdf"), "already_done")

    def test_resume_traces_the_checkpoint_hit(self):
        self._run(_document())
        agent, context, llm, result = self._run(_document())
        self.assertEqual(llm.calls, 0)
        kinds = [t["kind"] for t in context.events]
        self.assertIn("summarization_checkpoint_hit", kinds)
        self.assertEqual(
            context.metadata.get("summarization_checkpoint_status"),
            "already_done",
        )

    def test_force_flag_bypasses_resume_and_reruns(self):
        self._run(_document())
        document2 = _document()
        agent, context, llm, result = self._run(
            document2, {"force_summarization": True}
        )
        self.assertEqual(result.status.value, "ok")
        self.assertGreater(llm.calls, 0)  # re-summarized
        kinds = [t["kind"] for t in context.events]
        self.assertIn("summarization_forced", kinds)
        self.assertNotIn("summarized_resume", kinds)

    def test_resume_same_content_makes_zero_llm_calls(self):
        self._run(_document())

        document2 = _document()  # identical content, no summaries
        agent, context, llm, result = self._run(document2)
        self.assertEqual(result.status.value, "ok")
        self.assertEqual(llm.calls, 0)
        self.assertEqual(result.payload["resume"], "summarized_json")
        self.assertGreater(result.payload["applied_summaries"], 0)
        # Every level got its summary back, document included.
        self.assertIsNotNone(document2.summary)
        self.assertIsNotNone(document2.chapters[0].summary)
        self.assertIsNotNone(
            document2.chapters[0].pages[0].sections[0].blocks[0].summary
        )
        kinds = [t["kind"] for t in context.events]
        self.assertIn("summarized_found", kinds)
        self.assertIn("summarized_resume", kinds)

    def test_stale_content_is_resummarized(self):
        self._run(_document())

        document2 = _document()
        document2.chapters[0].pages[0].sections[0].blocks[0].raw_text += " FIXED"
        agent, context, llm, result = self._run(document2)
        self.assertEqual(result.status.value, "ok")
        self.assertGreater(llm.calls, 0)  # re-summarized
        kinds = [t["kind"] for t in context.events]
        self.assertIn("summarized_stale", kinds)
        self.assertNotIn("summarized_resume", kinds)
        # The store now matches the new content (overwritten at save time).
        stored, _ = load_summarized(self.sum_dir / "doc.json")
        self.assertEqual(stored, document2)

    def test_malformed_store_is_resummarized_and_overwritten(self):
        self._run(_document())
        path = self.sum_dir / "doc.json"
        path.write_text("{ broken", encoding="utf-8")

        document2 = _document()
        agent, context, llm, result = self._run(document2)
        self.assertEqual(result.status.value, "ok")
        self.assertGreater(llm.calls, 0)
        kinds = [t["kind"] for t in context.events]
        self.assertIn("summarized_unusable", kinds)
        self.assertTrue(path.exists())  # re-saved over the broken file

    def test_no_file_means_normal_summarization(self):
        document = _document()
        agent, context, llm, result = self._run(document)
        self.assertEqual(result.status.value, "ok")
        self.assertGreater(llm.calls, 0)
        kinds = [t["kind"] for t in context.events]
        self.assertNotIn("summarized_found", kinds)

    def test_carried_summaries_do_not_break_the_fingerprint(self):
        """A partially-summarized doc (graph retry) fingerprints like a clean one."""
        from src.agents.agents.summarizer_agent import (
            SummarizerAgent as _S,  # already imported; explicit for clarity
        )

        clean = _document()
        partial = _document()
        partial.chapters[0].pages[0].sections[0].blocks[0].summary = "partial"
        self.assertEqual(
            _S._pre_summarization_fingerprint(clean),
            _S._pre_summarization_fingerprint(partial),
        )
        # The carried summary is restored after the check.
        self.assertEqual(partial.chapters[0].pages[0].sections[0].blocks[0].summary,
                         "partial")

    def test_validate_passes_after_resume(self):
        self._run(_document())

        document2 = _document()
        agent, context, llm, result = self._run(document2)
        self.assertEqual(result.status.value, "ok")
        self.assertIsNone(agent.validate(context))

    def test_resume_does_not_re_record_checkpoint(self):
        """Same contract as the extraction agent's canonical resume."""
        self._run(_document())
        jobs = self._job_file()
        jobs.remove("doc.pdf")
        agent, context, llm, result = self._run(_document())
        self.assertEqual(result.status.value, "ok")
        self.assertEqual(llm.calls, 0)
        self.assertEqual(jobs.status_of("doc.pdf"), "new")


if __name__ == "__main__":
    unittest.main()
