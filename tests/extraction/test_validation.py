"""Tests for the structural validator and the consistency pass."""

from __future__ import annotations

import unittest

from src.extraction.consistency import check_and_prune, has_content
from src.extraction.models import (
    Chapter,
    DocumentExtract,
    PageContent,
    Section,
    TextBlock,
    TocEntry,
)
from src.extraction.validation import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    structural_errors,
    structural_issues,
)


def _block(page: int = 1, text: str = "content", block_id: int = 0) -> TextBlock:
    return TextBlock(block_id=block_id, page_number=page, bbox=(0, 0, 10, 10), raw_text=text)


def _section(page: int = 1, text: str = "content", section_id: int = 0,
             blocks: list | None = None) -> Section:
    return Section(
        section_id=section_id, page_number=page, bbox=(0, 0, 100, 20),
        raw_text=text, blocks=blocks if blocks is not None else [_block(page, text, section_id)],
    )


def _page(number: int, *, text: str = "page text", sections: list | None = None) -> PageContent:
    return PageContent(
        page_number=number, width=612, height=792, raw_text=text,
        sections=sections if sections is not None else [_section(number)],
    )


def _document(pages: list, *, total_pages: int = 10, full_text: str = "aggregated") -> DocumentExtract:
    toc = TocEntry(1, "Ch1", 1, 0)
    return DocumentExtract(
        source_path="doc.pdf", title="Doc", author="", subject="",
        total_pages=total_pages,
        chapters=[Chapter(toc_entry=toc, pages=pages, full_text=full_text)],
    )


class StructuralIssuesTest(unittest.TestCase):
    def test_valid_document_has_no_issues(self):
        document = _document([_page(1), _page(2)])
        self.assertEqual(structural_issues(document), [])

    def test_negative_page_number_is_a_defect(self):
        document = _document([_page(0)])
        codes = {i.code for i in structural_issues(document)}
        self.assertIn("page_number_invalid", codes)

    def test_bad_title_is_a_defect(self):
        document = _document([_page(1)])
        document.title = "   "
        codes = {i.code for i in structural_issues(document)}
        self.assertIn("title_invalid", codes)

    def test_bad_total_pages_is_a_defect(self):
        document = _document([_page(1)], total_pages=0)
        codes = {i.code for i in structural_issues(document)}
        self.assertIn("total_pages_invalid", codes)

    def test_toc_level_zero_is_valid(self):
        """Level 0 is legitimate: 0-based levels and no-TOC fallback entries."""
        document = _document([_page(1)])
        document.chapters[0].toc_entry = TocEntry(level=0, title="No TOC", page_number=1, page_index=0)
        self.assertEqual(structural_errors(document), [])

    def test_bad_toc_entry_is_a_defect(self):
        document = _document([_page(1)])
        document.toc.append(TocEntry(level=-1, title="Bad", page_number=1, page_index=-1))
        codes = {i.code for i in structural_issues(document)}
        self.assertIn("toc_level_invalid", codes)
        self.assertIn("toc_page_index_invalid", codes)

    def test_wrong_child_type_is_a_defect(self):
        document = _document([_page(1)])
        document.chapters[0].pages.append("not a page")
        codes = {i.code for i in structural_issues(document)}
        self.assertIn("page_type_invalid", codes)

    def test_bad_section_shape_is_a_defect(self):
        document = _document([_page(1, sections=[
            _section(1, blocks=[_block(1, "ok")]),
            Section(section_id=1, page_number=1, bbox=(0, 0), raw_text="x", blocks=[]),
        ])])
        codes = {i.code for i in structural_issues(document)}
        self.assertIn("section_bbox_invalid", codes)

    def test_issue_paths_locate_the_object(self):
        document = _document([_page(1), _page(2)])
        document.chapters[0].pages[1].width = -3
        issues = structural_issues(document)
        self.assertEqual(issues[0].path, "chapters[0].pages[1]")
        self.assertEqual(issues[0].code, "page_width_invalid")

    def test_missing_geometry_is_a_warning_not_an_error(self):
        """MinerU leaves width/height unset: advisory only, never fatal."""
        document = _document([_page(1)])
        document.chapters[0].pages[0].width = None
        document.chapters[0].pages[0].height = None
        issues = structural_issues(document)
        self.assertEqual(
            {i.severity for i in issues}, {SEVERITY_WARNING},
        )
        self.assertEqual(structural_errors(document), [])

    def test_warnings_mixed_with_errors(self):
        """Errors and warnings coexist: structural_errors keeps only errors."""
        document = _document([_page(1)])
        document.chapters[0].pages[0].width = None
        document.chapters[0].pages[0].page_number = 0  # genuine defect
        issues = structural_issues(document)
        self.assertEqual(len(issues), 2)
        self.assertEqual(
            {i.code for i in structural_errors(document)}, {"page_number_invalid"},
        )
        self.assertEqual({i.severity for i in issues if i.code == "page_width_invalid"},
                         {SEVERITY_WARNING})

    def test_severity_defaults_to_error(self):
        document = _document([_page(0)])
        issues = structural_issues(document)
        self.assertEqual(issues[0].severity, SEVERITY_ERROR)


class ConsistencyPassTest(unittest.TestCase):
    def test_clean_document_is_untouched(self):
        document = _document([_page(1), _page(2)])
        report = check_and_prune(document)
        self.assertEqual(report.removals, 0)
        self.assertEqual(report.warnings, [])
        self.assertEqual([p.page_number for p in document.chapters[0].pages], [1, 2])

    def test_blank_text_objects_are_pruned(self):
        page = _page(1, sections=[
            _section(1, blocks=[
                _block(1, "real"),
                _block(1, "  \t\n ", block_id=1),
            ]),
            _section(1, text="   \n  ", section_id=1, blocks=[_block(1, " ", 9)]),
        ])
        document = _document([page])
        report = check_and_prune(document)
        self.assertEqual(report.removed_blocks, 1)
        self.assertEqual(report.removed_sections, 1)
        self.assertEqual(len(page.sections), 1)
        self.assertEqual(len(page.sections[0].blocks), 1)

    def test_out_of_bounds_page_removed(self):
        document = _document([_page(1), _page(15)], total_pages=10)
        report = check_and_prune(document)
        self.assertEqual(report.out_of_bounds_pages, 1)
        self.assertEqual(report.removed_pages, 1)
        self.assertEqual([p.page_number for p in document.chapters[0].pages], [1])

    def test_unsorted_pages_warned_but_kept(self):
        document = _document([_page(2), _page(1)])
        report = check_and_prune(document)
        self.assertEqual(report.unsorted_lists, 1)
        self.assertEqual([p.page_number for p in document.chapters[0].pages], [2, 1])

    def test_misaligned_section_realigned_to_parent(self):
        page = _page(3, sections=[_section(7, section_id=0)])
        document = _document([page])
        report = check_and_prune(document)
        self.assertEqual(report.realigned, 1)
        self.assertEqual(page.sections[0].page_number, 3)

    def test_empty_containers_pruned_cascading(self):
        # Section with only a blank block -> section empty -> page empty
        # -> chapter empty -> removed; document has no content left.
        page = _page(1, sections=[
            _section(1, text="visible", blocks=[_block(1, "\n \t", 1)]),
        ])
        document = _document([page])
        report = check_and_prune(document)
        self.assertEqual(report.removed_sections, 1)
        self.assertEqual(report.removed_pages, 1)
        self.assertEqual(report.removed_chapters, 1)
        self.assertFalse(has_content(document))

    def test_empty_chapter_removed(self):
        document = _document([])
        report = check_and_prune(document)
        self.assertEqual(report.removed_chapters, 1)
        self.assertFalse(has_content(document))

    def test_blank_full_text_with_content_warns_but_keeps(self):
        document = _document([_page(1)], full_text="   ")
        report = check_and_prune(document)
        self.assertTrue(has_content(document))
        self.assertEqual(report.removed_chapters, 0)
        self.assertTrue(any("full_text" in w for w in report.warnings))

    def test_toc_sorting_warned(self):
        document = _document([_page(1)])
        document.toc = [TocEntry(1, "B", 5, 4), TocEntry(1, "A", 2, 1)]
        report = check_and_prune(document)
        self.assertEqual(report.unsorted_lists, 1)

    def test_chapter_toc_entry_checked(self):
        document = _document([_page(1)])
        document.chapters[0].toc_entry.level = -1
        codes = {i.code for i in structural_issues(document)}
        self.assertIn("toc_level_invalid", codes)

    def test_orphan_pages_pruned_too(self):
        document = _document([_page(1)])
        document.orphan_pages = [_page(99)]
        report = check_and_prune(document)
        self.assertEqual(report.out_of_bounds_pages, 1)
        self.assertEqual(document.orphan_pages, [])

    def test_pass_is_idempotent(self):
        page = _page(1, sections=[
            _section(1, blocks=[_block(1, "keep"), _block(1, "  ", 1)]),
            _section(5, section_id=1),
        ])
        document = _document([page, _page(42)])
        first = check_and_prune(document)
        removals = first.removals
        second = check_and_prune(document)
        self.assertEqual(second.removals, 0)
        self.assertGreater(removals, 0)


class ExtractionValidationAgentTest(unittest.TestCase):
    def _run(self, document):
        import tempfile
        from pathlib import Path

        from src.agents.agents.extraction_validation_agent import ExtractionValidationAgent
        from src.agents.contexts import IngestionContext

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        context = IngestionContext(request="test")
        context.outputs["content_extraction"] = document
        return ExtractionValidationAgent().run(context)

    def test_ok_with_report_payload(self):
        result = self._run(_document([_page(1), _page(2)]))
        self.assertEqual(result.status.value, "ok")
        self.assertIn("report", result.payload)
        self.assertEqual(result.payload["report"]["warning_count"], 0)

    def test_missing_input_is_skipped(self):
        import tempfile

        from src.agents.agents.extraction_validation_agent import ExtractionValidationAgent
        from src.agents.contexts import IngestionContext

        context = IngestionContext(request="test")
        result = ExtractionValidationAgent().run(context)
        self.assertEqual(result.status.value, "skipped")

    def test_empty_after_pruning_fails(self):
        result = self._run(_document([]))
        self.assertEqual(result.status.value, "failed")
        self.assertIn("empty after consistency", result.detail)

    def test_warnings_emitted_as_traces(self):
        # One valid page keeps content alive; the out-of-bounds one warns.
        document = _document([_page(1), _page(15)], total_pages=10)
        import tempfile

        from src.agents.agents.extraction_validation_agent import ExtractionValidationAgent
        from src.agents.contexts import IngestionContext

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        context = IngestionContext(request="test")
        context.outputs["content_extraction"] = document
        result = ExtractionValidationAgent().run(context)
        kinds = [t["kind"] for t in context.events]
        self.assertIn("consistency_warning", kinds)
        self.assertIn("consistency_done", kinds)
        self.assertEqual(result.status.value, "ok")


class TwoStepGraphTest(unittest.TestCase):
    def test_default_registry_runs_both_steps(self):
        import tempfile
        from pathlib import Path

        from src.agents import build_default_agents
        from src.agents.agents.pdf_extraction_agent import PDFExtractionAgent
        from src.agents.contexts import IngestionContext
        from src.graphs import IngestionGraph
        from src.tools.extraction_job_file import ExtractionJobFile

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pdf = Path(tmp.name) / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        clean = DocumentExtract(
            source_path=str(pdf), title="Doc", author="", subject="", total_pages=2,
            chapters=[Chapter(toc_entry=TocEntry(1, "Ch1", 1, 0),
                              pages=[_page(1), _page(2)], full_text="text")],
        )

        class StubExtractor:
            def extract(self, path):
                return clean

        agents = build_default_agents()
        agents["content_extractor"] = PDFExtractionAgent(
            extractor=StubExtractor(),
            job_file=ExtractionJobFile(Path(tmp.name) / "jobs.json"),
            canonical_dir=Path(tmp.name) / "extracted",
        )
        context = IngestionContext(document_path=pdf, request="test")
        graph = IngestionGraph(
            agents=agents, steps=IngestionGraph.DEFAULT_STEPS[:2],
        )
        outcome = graph.run(context)
        self.assertTrue(outcome.accepted)
        self.assertEqual(
            outcome.completed_steps, ["content_extraction", "extraction_validation"]
        )
        pipeline_kinds = [t["kind"] for t in context.events if t["phase"] == "pipeline"]
        self.assertEqual(
            pipeline_kinds,
            ["step_started", "step_done", "step_started", "step_done"],
        )

    def test_structurally_broken_extraction_fails_first_step(self):
        import tempfile
        from pathlib import Path

        from src.agents.agents.pdf_extraction_agent import PDFExtractionAgent
        from src.agents.contexts import IngestionContext
        from src.graphs import IngestionGraph
        from src.tools.extraction_job_file import ExtractionJobFile

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pdf = Path(tmp.name) / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        broken = DocumentExtract(
            source_path=str(pdf), title="   ", author="", subject="", total_pages=2,
        )

        class StubExtractor:
            def extract(self, path):
                return broken

        context = IngestionContext(document_path=pdf, request="test")
        graph = IngestionGraph(
            agents={
                "content_extractor": PDFExtractionAgent(
                    extractor=StubExtractor(),
                    job_file=ExtractionJobFile(Path(tmp.name) / "jobs.json"),
                    canonical_dir=Path(tmp.name) / "extracted",
                ),
            },
            steps=IngestionGraph.DEFAULT_STEPS[:2],
        )
        outcome = graph.run(context)
        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.failed_step, "content_extraction")
        self.assertIn("structurally invalid", outcome.failure_detail)
        # the validation step never ran
        self.assertNotIn("extraction_validation", outcome.completed_steps)


if __name__ == "__main__":
    unittest.main()
