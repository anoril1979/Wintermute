"""Tests for the SummarizerAgent (hierarchical summarization step).

The LLM is stubbed (the protocol is one ``complete`` call), so these tests
cover the agent's own logic: bottom-up order, copy-below-limit shortcut,
oversize warnings, resume behavior, failure domain mapping and validate().
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.agents.agents.summarizer_agent import (
    AGENT_NAME,
    DEFAULT_SUMMARY_MAX_CHARS,
    DEFAULT_SUMMARY_MIN_CHARS,
    UNREADABLE_MARKER,
    SummarizerAgent,
    combined_text,
    summary_limits,
)
from src.agents.contexts import IngestionContext
from src.extraction.models import (
    Chapter,
    DocumentExtract,
    PageContent,
    Section,
    TocEntry,
)


def make_block(block_id: int, text: str, page: int = 1):
    from src.extraction.models import TextBlock

    return TextBlock(block_id=block_id, page_number=page, bbox=(0, 0, 1, 1), raw_text=text)


def make_document(long_size: int = 1600) -> DocumentExtract:
    """One chapter / one page / one section / three blocks (short, long, long)."""
    short = make_block(0, "short text")
    long1 = make_block(1, "LONG" * (long_size // 4))
    long2 = make_block(2, "LONG2" * (long_size // 4))
    section = Section(
        section_id=0, blocks=[short, long1, long2],
        page_number=1, bbox=(0, 0, 1, 1), raw_text="section text",
    )
    page = PageContent(page_number=1, width=100, height=100, raw_text="page text",
                       sections=[section])
    chapter = Chapter(
        toc_entry=TocEntry(level=1, title="Chapitre I", page_number=1, page_index=0),
        pages=[page], full_text="",
    )
    return DocumentExtract(
        source_path="x.pdf", title="Doc", author="A", subject="S",
        total_pages=1, chapters=[chapter],
    )


class StubSummarizer(SummarizerAgent):
    """SummarizerAgent with a scripted LLM: counts calls, records prompts.

    ``reply`` is one string (every call returns it) or a list (calls consume
    the entries in order, cycling). Use distinctive LONG replies (> the copy
    limit) when a test must observe what one level fed to the next.

    The job file defaults to a throwaway path OUTSIDE the repo: without it,
    every green run recorded 'x.pdf' into the REAL data/cache/
    summarization_jobs.json (found while debugging a removal report).
    """

    # Lazily-created shared temp dir; cleaned up by the OS eventually.
    _hermetic_dir: tempfile.TemporaryDirectory | None = None

    @classmethod
    def _hermetic_job_file(cls):
        from src.tools.extraction_job_file import SummarizationJobFile

        if cls._hermetic_dir is None:
            cls._hermetic_dir = tempfile.TemporaryDirectory()
        return SummarizationJobFile(
            Path(cls._hermetic_dir.name) / "summarization_jobs.json"
        )

    def __init__(self, reply: str | list[str] = "SUMMARY.", **kwargs):
        kwargs.setdefault("job_file", self._hermetic_job_file())
        super().__init__(**kwargs)
        self.replies = list(reply) if isinstance(reply, list) else [reply]
        self.calls = 0
        self.prompts: list[str] = []

    def llm_client(self):  # noqa: D102 - protocol stand-in
        stub = self

        class _Client:
            def complete(self, prompt: str, max_tokens: int = 300) -> str:
                text = stub.replies[stub.calls % len(stub.replies)]
                stub.calls += 1
                stub.prompts.append(prompt)
                return text

        return _Client()


def run_agent(document, agent=None, min_max=None):
    context = IngestionContext(document_path=Path("x.pdf"))
    context.outputs["content_extraction"] = document
    if min_max is not None:
        context.metadata["summary_limits"] = min_max  # unused; limits come from yaml
    result = (agent or StubSummarizer()).run(context)
    return result, context


class SummarizerAgentTest(unittest.TestCase):
    """Core behaviors of the summarization step."""

    def setUp(self):
        # Hermetic summarized-store folder: tests must never touch (or
        # resume from) the real data/summarized store.
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.sum_dir = Path(self._tmp.name) / "sum"

    def make_agent(self, **kwargs) -> StubSummarizer:
        kwargs.setdefault("summarized_dir", self.sum_dir)
        return StubSummarizer(**kwargs)

    def test_bottom_up_summarizes_every_level(self):
        long_reply = "S" * 1200  # above the copy limit: every level calls the LLM
        agent = self.make_agent(reply=long_reply)
        doc = make_document()
        result, _ = run_agent(doc, agent)

        self.assertEqual(result.status.value, "ok")
        # 6 LLM units: 2 long blocks + section + page + chapter + document
        # (the short block is copied verbatim, no call).
        self.assertEqual(agent.calls, 6)
        section = doc.chapters[0].pages[0].sections[0]
        self.assertEqual(section.blocks[0].summary, "short text")  # copied
        self.assertEqual(section.blocks[1].summary, long_reply)
        self.assertEqual(section.blocks[2].summary, long_reply)
        self.assertEqual(section.summary, long_reply)
        self.assertEqual(doc.chapters[0].pages[0].summary, long_reply)
        self.assertEqual(doc.chapters[0].summary, long_reply)
        self.assertEqual(doc.summary, long_reply)

    def test_short_content_copied_without_llm_call(self):
        agent = self.make_agent()
        doc = make_document(long_size=0)  # every text is tiny
        doc.chapters[0].pages[0].sections[0].blocks[1].raw_text = "a" * 1200  # one long
        result, _ = run_agent(doc, agent)

        self.assertEqual(result.status.value, "ok")
        short_block = doc.chapters[0].pages[0].sections[0].blocks[0]
        self.assertEqual(short_block.summary, "short text")  # verbatim copy
        payload = result.payload
        self.assertGreaterEqual(payload["copied"], 1)
        # Only the one 1200-char block reaches the LLM; containers stay short.
        self.assertEqual(agent.calls, 1)

    def test_chapter_uses_pages_summaries_when_no_full_text(self):
        # Distinctive long replies: the page call must feed the chapter call.
        replies = [f"SUM-{i}".ljust(1100, "·") for i in range(1, 6)]
        agent = self.make_agent(reply=replies)
        doc = make_document(long_size=0)
        doc.chapters[0].pages[0].sections[0].blocks[1].raw_text = "y" * 1500
        result, _ = run_agent(doc, agent)

        self.assertEqual(result.status.value, "ok")
        # Calls: block(1) -> section(2) -> page(3) -> chapter(4) -> doc(5).
        chapter_prompt = agent.prompts[3]
        self.assertIn(replies[2], chapter_prompt)    # the page's summary
        self.assertNotIn("short text", chapter_prompt)  # not raw block text

    def test_document_input_is_chapter_summaries(self):
        # Distinctive long replies: the chapter call must feed the document call.
        replies = [f"SUM-{i}".ljust(1100, "·") for i in range(1, 6)]
        agent = self.make_agent(reply=replies)
        doc = make_document(long_size=0)
        doc.chapters[0].pages[0].sections[0].blocks[1].raw_text = "w" * 1500
        result, _ = run_agent(doc, agent)

        self.assertEqual(result.status.value, "ok")
        doc_prompt = agent.prompts[4]  # 5th call = the document
        self.assertIn(replies[3], doc_prompt)        # the chapter's summary
        self.assertNotIn("w" * 100, doc_prompt)      # not raw text

    def test_oversize_summary_kept_but_warned(self):
        oversize = "S" * (DEFAULT_SUMMARY_MAX_CHARS + 500)
        agent = self.make_agent(reply=oversize)
        doc = make_document()
        result, context = run_agent(doc, agent)

        self.assertEqual(result.status.value, "ok")
        self.assertGreaterEqual(result.payload["oversize"], 1)
        warning_kinds = [e["kind"] for e in context.events
                         if e["kind"] == "summarization_warning"]
        self.assertTrue(warning_kinds)
        # The oversize summary is still stored (warning, not failure).
        self.assertEqual(doc.chapters[0].pages[0].summary, oversize)

    def test_empty_llm_reply_fails_with_llm_response(self):
        doc = make_document()
        agent = self.make_agent(reply="   ")
        # A blank block is skipped (not failed); make block 2 blank on purpose:
        # with a blank-everything reply the first *content* unit fails the run.
        doc.chapters[0].pages[0].sections[0].blocks[2].raw_text = ""
        result, _ = run_agent(doc, agent)

        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.failure_domain.value, "llm_response")

    def test_unreadable_marker_fails(self):
        doc = make_document()
        agent = self.make_agent(reply=UNREADABLE_MARKER)
        doc.chapters[0].pages[0].sections[0].blocks[2].raw_text = ""
        result, _ = run_agent(doc, agent)

        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.failure_domain.value, "llm_response")

    def test_no_extraction_output_skips(self):
        agent = self.make_agent()
        context = IngestionContext(document_path=Path("x.pdf"))
        result = agent.run(context)

        self.assertEqual(result.status.value, "skipped")

    def test_resume_keeps_existing_summaries(self):
        doc = make_document()
        doc.chapters[0].pages[0].sections[0].blocks[0].summary = "KEPT"
        agent = self.make_agent()
        result, _ = run_agent(doc, agent)

        self.assertEqual(result.status.value, "ok")
        self.assertEqual(doc.chapters[0].pages[0].sections[0].blocks[0].summary, "KEPT")
        # Only the 2 long blocks hit the LLM: with a short stub reply, every
        # container text falls under the copy limit and is copied instead.
        self.assertEqual(agent.calls, 2)

    def test_validate_detects_missing_summaries(self):
        doc = make_document()
        agent = self.make_agent()
        context = IngestionContext(document_path=Path("x.pdf"))
        context.outputs["content_extraction"] = doc
        agent.run(context)
        self.assertIsNone(agent.validate(context))

        doc.chapters[0].pages[0].sections[0].blocks[1].summary = ""
        validation = agent.validate(context)
        self.assertIsNotNone(validation)
        self.assertEqual(validation.status.value, "failed")

    def test_prompt_embeds_size_target_and_delimiters(self):
        agent = self.make_agent()
        prompt = agent._build_prompt("le texte", 2500)
        self.assertIn("2500", prompt)
        self.assertIn("<<<<TEXT>>>>", prompt)
        self.assertIn("le texte", prompt)


class SummaryLimitsTest(unittest.TestCase):
    """Config resolution: values from yaml, fail-open to the defaults."""

    def test_limits_from_config(self):
        fake_config = {"summary_min_chars": 10, "summary_max_chars": 20}
        with mock.patch("src.agents.agents.summarizer_agent.load_ingestion_config",
                        return_value=fake_config):
            self.assertEqual(summary_limits(), (10, 20))

    def test_limits_fail_open_on_config_error(self):
        with mock.patch("src.agents.agents.summarizer_agent.load_ingestion_config",
                        side_effect=RuntimeError("broken yaml")):
            self.assertEqual(summary_limits(),
                             (DEFAULT_SUMMARY_MIN_CHARS, DEFAULT_SUMMARY_MAX_CHARS))

    def test_limits_fail_open_on_bad_values(self):
        with mock.patch("src.agents.agents.summarizer_agent.load_ingestion_config",
                        return_value={"summary_min_chars": -1, "summary_max_chars": 0}):
            self.assertEqual(summary_limits(),
                             (DEFAULT_SUMMARY_MIN_CHARS, DEFAULT_SUMMARY_MAX_CHARS))

    def test_combined_text_skips_blanks(self):
        self.assertEqual(combined_text(["a", None, "  ", "b"]), "a\n\nb")
        self.assertEqual(combined_text([None, ""]), "")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
