"""Tests for PDFExtractionAgent and its wiring into the orchestrator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.agents import PDFExtractionAgent, build_default_agents
from src.agents.contexts import IngestionContext
from src.agents.agents.pdf_extraction_agent import OUTPUT_KEY
from src.tools.extraction_job_file import ExtractionJobFile
from src.agents.protocols import AgentStatus, FailureDomain
from src.extraction.document_extractor import DocumentExtractor
from src.extraction.models import DocumentExtract


def make_document(path: Path, total_pages: int = 2, *, with_content: bool = True) -> DocumentExtract:
    """A canned valid extraction; ``with_content`` adds one chapter/page so
    the extraction_validation step finds content to keep."""
    from src.extraction.models import Chapter, PageContent, Section, TextBlock, TocEntry

    chapters = []
    if with_content:
        chapters = [
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
        ]
    return DocumentExtract(
        source_path=str(path), title="Test Doc", author="", subject="",
        total_pages=total_pages, toc=[], chapters=chapters, orphan_pages=[], metadata={},
    )


class StubExtractor(DocumentExtractor):
    """Extractor double: returns a canned document or raises the given error."""

    def __init__(self, document=None, exc: Exception | None = None) -> None:
        self.document = document
        self.exc = exc

    def _run_backend(self, path: Path) -> None:
        pass

    def _build_document(self, path: Path):
        if self.exc is not None:
            raise self.exc
        return self.document


class _StubSourceIndexer:
    """Indexing double: OK without touching Ollama or ChromaDB."""

    name = "source_indexer"

    def run(self, context):
        from src.agents.protocols import AgentResult

        return AgentResult(agent_name=self.name, status=AgentStatus.OK)

    def validate(self, context):
        return None


class RegistryTest(unittest.TestCase):
    def test_default_registry_has_content_extractor(self):
        registry = build_default_agents()
        self.assertIn("content_extractor", registry)
        self.assertIsInstance(registry["content_extractor"], PDFExtractionAgent)

    def test_default_registry_has_extraction_validator(self):
        from src.agents.agents.extraction_validation_agent import ExtractionValidationAgent

        registry = build_default_agents()
        self.assertIn("extraction_validator", registry)
        self.assertIsInstance(registry["extraction_validator"], ExtractionValidationAgent)

    def test_agent_key_matches_graph_step(self):
        # The graph looks agents up by key 'content_extractor' (step name
        # 'content_extraction') — the agent's declared name must match.
        self.assertEqual(build_default_agents()["content_extractor"].name, "content_extractor")


class AgentRunTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.pdf_path = Path(self._tmp.name) / "doc.pdf"
        self.pdf_path.write_bytes(b"%PDF-1.4 fake")  # never parsed by stubs
        # Hermetic job file + canonical store: never touch the real ones.
        self.job_file = ExtractionJobFile(Path(self._tmp.name) / "jobs.json")
        self.canonical_dir = Path(self._tmp.name) / "extracted"

    def tearDown(self):
        self._tmp.cleanup()

    def _context(self, path=None):
        return IngestionContext(
            document_path=path if path is not None else self.pdf_path,
            request="[test]",
        )

    def _agent(self, extractor):
        return PDFExtractionAgent(
            extractor=extractor, job_file=self.job_file,
            canonical_dir=self.canonical_dir,
        )

    def test_run_success_feeds_context(self):
        stub = StubExtractor(document=make_document(self.pdf_path))
        agent = self._agent(stub)
        context = self._context()

        result = agent.run(context)

        self.assertEqual(result.status, AgentStatus.OK)
        self.assertIn(OUTPUT_KEY, context.outputs)
        document = context.outputs[OUTPUT_KEY]
        self.assertEqual(document.title, "Test Doc")
        self.assertEqual(context.metadata["document_title"], "Test Doc")
        self.assertEqual(context.metadata["total_pages"], 2)
        self.assertEqual(result.payload["total_pages"], 2)

    def test_validate_ok_returns_none(self):
        stub = StubExtractor(document=make_document(self.pdf_path))
        agent = self._agent(stub)
        context = self._context()
        agent.run(context)
        self.assertIsNone(agent.validate(context))

    def test_validate_rejects_empty_document(self):
        stub = StubExtractor(document=make_document(self.pdf_path, total_pages=0))
        agent = self._agent(stub)
        context = self._context()
        agent.run(context)  # OK at run level
        validation = agent.validate(context)
        self.assertEqual(validation.status, AgentStatus.FAILED)
        self.assertEqual(validation.failure_domain, FailureDomain.INPUT_DATA)

    def test_missing_document_path(self):
        agent = self._agent(StubExtractor())
        context = IngestionContext(document_path=None, request="[test]")
        result = agent.run(context)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.failure_domain, FailureDomain.INPUT_DATA)

    def test_extract_returning_none_maps_to_input_data(self):
        # Out-of-contract extractor (returns None instead of DocumentExtract):
        # the agent must report a failure, not crash.
        agent = self._agent(StubExtractor(document=None))
        result = agent.run(self._context())
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.failure_domain, FailureDomain.INPUT_DATA)

    def test_file_not_found_maps_to_input_data(self):
        stub = StubExtractor(exc=FileNotFoundError("missing"))
        result = self._agent(stub).run(self._context())
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.failure_domain, FailureDomain.INPUT_DATA)

    def test_invalid_pdf_maps_to_input_data(self):
        stub = StubExtractor(exc=ValueError("not a pdf"))
        result = self._agent(stub).run(self._context())
        self.assertEqual(result.failure_domain, FailureDomain.INPUT_DATA)

    def test_mineru_failure_maps_to_external(self):
        stub = StubExtractor(exc=OSError("MinerU stopped with 1 error code."))
        result = self._agent(stub).run(self._context())
        self.assertEqual(result.failure_domain, FailureDomain.EXTERNAL)

    def test_unexpected_error_maps_to_unknown(self):
        stub = StubExtractor(exc=RuntimeError("boom"))
        result = self._agent(stub).run(self._context())
        self.assertEqual(result.failure_domain, FailureDomain.UNKNOWN)

    def test_default_extractor_is_mineru(self):
        from src.extraction.mineru_pdf_extractor import MineruPDFExtractor
        agent = PDFExtractionAgent()
        self.assertIsInstance(agent.extractor, MineruPDFExtractor)

    def test_default_job_file_is_project_checkpoint(self):
        from src.tools.extraction_job_file import DEFAULT_JOB_FILE
        self.assertEqual(PDFExtractionAgent().job_file.job_file, DEFAULT_JOB_FILE)


class OrchestratorPlugTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.pdf_path = Path(self._tmp.name) / "doc.pdf"
        self.pdf_path.write_bytes(b"%PDF-1.4 fake")
        # Hermetic job files for EVERY agent that records checkpoints: the
        # graph's default summarizer would otherwise record 'doc.pdf' into
        # the real data/cache/summarization_jobs.json.
        self._summarizer_jobs = ExtractionJobFile(
            self.pdf_path.parent / "summarization_jobs.json"
        )

    def _registry(self, **overrides):
        """The default registry with hermetic extraction + summarization
        stores and the real extractor/indexer swapped for stubs."""
        from src.agents.agents.summarizer_agent import SummarizerAgent

        registry = build_default_agents()
        registry["summarizer"] = SummarizerAgent(job_file=self._summarizer_jobs)
        registry.update(overrides)
        return registry

    def tearDown(self):
        self._tmp.cleanup()

    def test_run_ingestion_file_completes_extraction_step(self):
        """With the real registry shape (stubbed extractor), the graph runs
        content_extraction, extraction_validation, hierarchical_summarization
        AND source_indexing (all implemented), then stops at the
        not-yet-implemented knowledge_extraction step."""
        from src.ingestion.ingestion_orchestrator import run_ingestion_file

        stub = StubExtractor(document=make_document(self.pdf_path))
        registry = self._registry()
        registry["content_extractor"] = PDFExtractionAgent(
            extractor=stub,
            job_file=ExtractionJobFile(self.pdf_path.parent / "jobs.json"),
            canonical_dir=self.pdf_path.parent / "extracted",
        )
        # Indexing would reach the real Ollama/ChromaDB: stub it here — this
        # test pins the GRAPH wiring, not the agent internals.
        registry["source_indexer"] = _StubSourceIndexer()

        result = run_ingestion_file(self.pdf_path, agents=registry)

        self.assertEqual(result["status"], "not_implemented")
        self.assertEqual(
            result["completed_steps"],
            ["content_extraction", "extraction_validation",
             "hierarchical_summarization", "source_indexing"],
        )
        self.assertEqual(result["not_implemented_steps"], ["knowledge_extraction"])
        self.assertEqual(result["failed_step"], "knowledge_extraction")

    def test_content_free_extraction_is_rejected_by_validation(self):
        """An extraction with no content at all passes the structural gate
        but is rejected by the extraction_validation step."""
        from src.ingestion.ingestion_orchestrator import run_ingestion_file

        stub = StubExtractor(
            document=make_document(self.pdf_path, with_content=False)
        )
        registry = self._registry()
        registry["content_extractor"] = PDFExtractionAgent(
            extractor=stub,
            job_file=ExtractionJobFile(self.pdf_path.parent / "jobs.json"),
            canonical_dir=self.pdf_path.parent / "extracted",
        )

        result = run_ingestion_file(self.pdf_path, agents=registry)

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["failed_step"], "extraction_validation")
        self.assertIn("empty after consistency", result["failure_detail"])

    def test_extraction_failure_rejects_request(self):
        """A failing extraction (INPUT_DATA domain) stops the graph with a
        rejection, not a not-implemented status."""
        from src.ingestion.ingestion_orchestrator import run_ingestion_file

        stub = StubExtractor(exc=FileNotFoundError("corrupted"))
        registry = self._registry()
        registry["content_extractor"] = PDFExtractionAgent(
            extractor=stub,
            job_file=ExtractionJobFile(self.pdf_path.parent / "jobs.json"),
            canonical_dir=self.pdf_path.parent / "extracted",
        )

        result = run_ingestion_file(self.pdf_path, agents=registry)

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["failed_step"], "content_extraction")
        self.assertIn("corrupted", result["failure_detail"])

    def test_agents_none_resolves_to_default_registry(self):
        """``agents=None`` builds the default registry (post-paradigm
        wiring helper: :func:`_build_registry_gracefully`); an explicit
        (even empty) dict always wins."""
        from src.ingestion.ingestion_orchestrator import _build_registry_gracefully

        registry = _build_registry_gracefully(None)
        self.assertIn("content_extractor", registry)
        self.assertEqual(_build_registry_gracefully({}), {})


if __name__ == "__main__":
    unittest.main()
