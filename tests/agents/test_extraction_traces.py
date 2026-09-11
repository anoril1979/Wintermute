"""Tests for the extraction agent's and ingestion graph's trace events."""

from __future__ import annotations

import unittest
import unittest.mock
from pathlib import Path

from src.agents.agents.ingestion_task_agent import IngestionTaskAgent
from src.agents.agents.pdf_extraction_agent import (
    FORCE_EXTRACTION_KEY,
    PDFExtractionAgent,
)
from src.agents.contexts import IngestionContext, RoutingContext
from src.extraction.document_extractor import DocumentExtractor
from src.extraction.mineru_pdf_extractor import MineruPDFExtractor
from src.extraction.models import DocumentExtract
from src.graphs import IngestionGraph
from src.routing.models import AnalysisResult, RequestKind, UserRequest
from src.routing.routing_orchestrator import run_routing
from src.tools.extraction_job_file import ExtractionJobFile


class StubExtractor(DocumentExtractor):
    supported = (".pdf",)

    def _run_backend(self, path):
        return None

    def _build_document(self, path):
        return DocumentExtract(
            source_path=str(path), title="Doc", author="", subject="",
            total_pages=2,
        )


class ResumableExtractor(MineruPDFExtractor):
    """A Mineru-derived stub: enables the artifact-resume branch."""

    def __init__(self):  # skip MinerU's config/folder setup
        self.bypass_ocr = False

    def _run_backend(self, path):
        return None

    def _build_document(self, path):
        return DocumentExtract(
            source_path=str(path), title="Doc", author="", subject="",
            total_pages=2,
        )


class ExtractionAgentTraceTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.job_file = ExtractionJobFile(self._tmp.name + "/jobs.json")
        # Hermetic canonical store (never the real data/extracted).
        self.canonical_dir = Path(self._tmp.name) / "extracted"

    def tearDown(self):
        self._tmp.cleanup()

    def _agent(self, extractor):
        return PDFExtractionAgent(
            extractor=extractor, job_file=self.job_file,
            canonical_dir=self.canonical_dir,
        )

    def test_fresh_extraction_traces(self):
        import tempfile
        from pathlib import Path

        pdf = Path(self._tmp.name) / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        context = IngestionContext(document_path=pdf, request="test")
        result = self._agent(StubExtractor()).run(context)

        self.assertEqual(result.status.value, "ok")
        kinds = [t["kind"] for t in context.events]
        self.assertEqual(
            kinds, ["extracting", "canonical_saved", "extracted"],
            f"unexpected trace sequence: {kinds}",
        )
        extracted = context.events[-1]
        self.assertEqual(extracted["phase"], "task")
        self.assertEqual(extracted["data"]["pages"], 2)

    def test_checkpoint_hit_and_resume_traces(self):
        import tempfile
        from pathlib import Path

        pdf = Path(self._tmp.name) / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        self.job_file.record("doc.pdf", pdf)  # mark as extracted

        context = IngestionContext(document_path=pdf, request="test")
        agent = self._agent(ResumableExtractor())
        result = agent.run(context)

        self.assertEqual(result.status.value, "ok")
        kinds = [t["kind"] for t in context.events]
        self.assertEqual(kinds, ["checkpoint_hit", "resumed", "canonical_saved"])
        self.assertEqual(context.events[0]["data"]["checkpoint"], "already_done")

    def test_forced_extraction_traces(self):
        import tempfile
        from pathlib import Path

        pdf = Path(self._tmp.name) / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        self.job_file.record("doc.pdf", pdf)

        context = IngestionContext(document_path=pdf, request="test")
        context.metadata[FORCE_EXTRACTION_KEY] = True
        agent = self._agent(StubExtractor())
        result = agent.run(context)

        self.assertEqual(result.status.value, "ok")
        kinds = [t["kind"] for t in context.events]
        self.assertEqual(
            kinds, ["extraction_forced", "extracting", "canonical_saved", "extracted"]
        )

    def test_failed_extraction_traces(self):
        import tempfile
        from pathlib import Path

        class BoomExtractor(StubExtractor):
            def _run_backend(self, path):
                raise OSError("MinerU exploded")

        pdf = Path(self._tmp.name) / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        context = IngestionContext(document_path=pdf, request="test")
        agent = self._agent(BoomExtractor())
        result = agent.run(context)

        self.assertEqual(result.status.value, "failed")
        kinds = [t["kind"] for t in context.events]
        self.assertEqual(kinds, ["extracting", "extraction_failed"])

    def test_no_document_path_traces(self):
        context = IngestionContext(document_path=None, request="test")
        agent = self._agent(StubExtractor())
        agent.run(context)
        self.assertEqual(
            [t["kind"] for t in context.events], ["extraction_failed"]
        )


class IngestionGraphTraceTest(unittest.TestCase):
    def test_step_lifecycle_traces(self):
        from src.agents.protocols import AgentResult, AgentStatus

        class OkAgent:
            name = "ok"

            def run(self, context):
                return AgentResult(agent_name=self.name, status=AgentStatus.OK)

            def validate(self, context):
                return None

        context = IngestionContext(request="test")
        graph = IngestionGraph(
            agents={"content_extractor": OkAgent()},
            steps=IngestionGraph.DEFAULT_STEPS[:1],
        )
        outcome = graph.run(context)
        self.assertTrue(outcome.accepted)
        kinds = [t["kind"] for t in context.events if t["phase"] == "pipeline"]
        self.assertEqual(kinds, ["step_started", "step_done"])
        self.assertEqual(context.events[0]["data"]["step"], "content_extraction")

    def test_retry_and_not_implemented_traces(self):
        from src.agents.protocols import AgentResult, AgentStatus, FailureDomain

        class FlakyAgent:
            name = "flaky"
            calls = 0

            def run(self, context):
                FlakyAgent.calls += 1
                if FlakyAgent.calls == 1:
                    return AgentResult(
                        agent_name=self.name, status=AgentStatus.FAILED,
                        failure_domain=FailureDomain.LLM_RESPONSE, detail="bad json",
                    )
                return AgentResult(agent_name=self.name, status=AgentStatus.OK)

            def validate(self, context):
                return None

        context = IngestionContext(request="test")
        graph = IngestionGraph(
            agents={"content_extractor": FlakyAgent()},
            steps=IngestionGraph.DEFAULT_STEPS[:1],
        )
        outcome = graph.run(context)
        self.assertTrue(outcome.accepted)
        kinds = [t["kind"] for t in context.events if t["phase"] == "pipeline"]
        self.assertEqual(kinds, ["step_started", "step_retry", "step_done"])

        context2 = IngestionContext(request="test")
        graph2 = IngestionGraph(
            agents={}, steps=IngestionGraph.DEFAULT_STEPS[:1],
        )
        graph2.run(context2)
        kinds2 = [t["kind"] for t in context2.events if t["phase"] == "pipeline"]
        self.assertEqual(kinds2, ["not_implemented"])


class OrchestratorPassThroughTest(unittest.TestCase):
    def test_pipeline_traces_reach_routing_result(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "doc.pdf"
            pdf.write_bytes(b"%PDF-1.4 fake")

            def runner(path, force=False, on_event=None, **kwargs):
                context = IngestionContext(
                    document_path=path, request="[test]", on_event=on_event
                )
                outcome = IngestionGraph(
                    agents={"content_extractor": PDFExtractionAgent(
                        extractor=StubExtractor(),
                        job_file=ExtractionJobFile(Path(tmp) / "jobs.json"),
                        canonical_dir=Path(tmp) / "extracted",
                    )},
                    steps=IngestionGraph.DEFAULT_STEPS[:1],
                ).run(context)
                return {
                    "status": "accepted",
                    "completed_steps": outcome.completed_steps,
                    "traces": list(context.events),
                }

            class FakeAnalyzer:
                def analyze(self, prompt):
                    return AnalysisResult(requests=[
                        UserRequest(kind=RequestKind.INGESTION,
                                    utterance="ingest", document="doc.pdf")
                    ])

            with unittest.mock.patch(
                "src.tools.ingest_tool.ingest_document",
                return_value={"status": "ready", "path": str(pdf)},
            ):
                result = run_routing(
                    "ingest doc.pdf", analyzer=FakeAnalyzer(),
                    agents={"ingestion_task": IngestionTaskAgent(runner=runner)},
                )

            phases = {t["phase"] for t in result["traces"]}
            self.assertEqual(phases, {"analysis", "dispatch", "task", "pipeline"})
            kinds = [t["kind"] for t in result["traces"]]
            self.assertIn("extracting", kinds)
            self.assertIn("extracted", kinds)
            self.assertIn("step_started", kinds)
            # ingestion_done comes after the merged pipeline traces
            self.assertLess(kinds.index("step_done"), kinds.index("ingestion_done"))


if __name__ == "__main__":
    unittest.main()
