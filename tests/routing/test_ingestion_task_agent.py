"""Tests for the IngestionTaskAgent (stubbed ingestion runner)."""

from __future__ import annotations

import unittest
import unittest.mock

from src.agents.agents.ingestion_task_agent import IngestionTaskAgent
from src.agents.contexts import RoutingContext
from src.agents.protocols import AgentStatus, FailureDomain
from src.routing.models import IngestionRequest, RetrievalRequest


def _request(document="a.pdf", force=False, origin=None):
    return IngestionRequest(
        utterance=f"ingest {document}", document=document,
        force=force, origin=origin,
    )


class IngestionTaskAgentTest(unittest.TestCase):
    def setUp(self):
        self.context = RoutingContext(request="test")

    def test_accepted_ingestion(self):
        seen = {}

        def runner(path, force=False, **kwargs):
            seen["path"], seen["force"] = path, force
            return {"status": "accepted", "completed_steps": ["content_extraction"]}

        with unittest.mock.patch("src.tools.ingest_tool.ingest_document",
                                 return_value={"status": "ready", "path": "x/a.pdf"}):
            result = IngestionTaskAgent(runner=runner).run(self.context, _request(force=True))
        self.assertEqual(result.status, AgentStatus.OK)
        self.assertEqual(result.payload["ingestion"]["status"], "accepted")
        self.assertTrue(seen["force"])  # force option is forwarded

    def test_unknown_document_is_input_data_failure(self):
        def runner(path, force=False, **kwargs):  # pragma: no cover — must not be called
            raise AssertionError("runner must not be called for an unknown file")

        agent = IngestionTaskAgent(runner=runner)
        with unittest.mock.patch("src.tools.ingest_tool.ingest_document",
                                 return_value={"status": "no_file", "message": "nope"}):
            result = agent.run(self.context, _request())
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.failure_domain, FailureDomain.INPUT_DATA)
        self.assertEqual(result.detail, "nope")

    def test_rejected_ingestion_carries_reason(self):
        with unittest.mock.patch("src.tools.ingest_tool.ingest_document",
                                 return_value={"status": "ready", "path": "x/a.pdf"}):
            agent = IngestionTaskAgent(runner=lambda path, force=False, **kw:
                                       {"status": "rejected", "reason": "bad yaml"})
            result = agent.run(self.context, _request())
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.detail, "bad yaml")

    def test_config_error_maps_to_config_domain(self):
        with unittest.mock.patch("src.tools.ingest_tool.ingest_document",
                                 return_value={"status": "ready", "path": "x/a.pdf"}):
            agent = IngestionTaskAgent(runner=lambda path, force=False, **kw:
                                       {"status": "config_error", "message": "ingestion.yaml invalide"})
            result = agent.run(self.context, _request())
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.failure_domain, FailureDomain.CONFIG)

    def test_not_implemented_pipeline_reports_step(self):
        with unittest.mock.patch("src.tools.ingest_tool.ingest_document",
                                 return_value={"status": "ready", "path": "x/a.pdf"}):
            agent = IngestionTaskAgent(runner=lambda path, force=False, **kw:
                                       {"status": "not_implemented", "failed_step": "extraction_validation"})
            result = agent.run(self.context, _request())
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertIn("extraction_validation", result.detail)

    def test_origin_is_forwarded_to_the_pipeline(self):
        seen = {}

        def runner(path, force=False, **kwargs):
            seen.update(kwargs)
            return {"status": "accepted"}

        with unittest.mock.patch("src.tools.ingest_tool.ingest_document",
                                 return_value={"status": "ready", "path": "x/a.pdf"}):
            result = IngestionTaskAgent(runner=runner).run(
                self.context, _request(origin="canon"))
        self.assertEqual(result.status, AgentStatus.OK)
        self.assertEqual(seen.get("origin"), "canon")

    def test_out_of_contract_request_is_refused(self):
        retrieval = RetrievalRequest(utterance="u", question="q?")
        result = IngestionTaskAgent(runner=lambda p, force=False, **kw: {}).run(
            self.context, retrieval)
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.failure_domain, FailureDomain.INPUT_DATA)

    def test_real_orchestrator_import_path(self):
        # The lazy import must resolve to the real orchestrator function.
        agent = IngestionTaskAgent()
        self.assertIsNone(agent._runner)
        # call through the lazy path with a stub-free monkeypatch of the module attr
        import src.ingestion.ingestion_orchestrator as orch
        with unittest.mock.patch.object(orch, "run_ingestion_file",
                                        return_value={"status": "accepted"}) as fake:
            out = agent._run_ingestion("somewhere/a.pdf", force=True)
        fake.assert_called_once_with(
            "somewhere/a.pdf", force=True,
            force_summarization=False,
        )
        self.assertEqual(out, {"status": "accepted"})


if __name__ == "__main__":
    unittest.main()
