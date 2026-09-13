"""Tests for the ingestion orchestrator (deterministic flow).

Post-paradigm change: the orchestrator has ONE entry point
(:func:`run_ingestion_file`, called by scripts/ingest.py). No free-text
request path, no LLM router — a config gate, a file check, and the graph.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.agents import MissingLLMRoleError
from src.tools.config_loader import ConfigError, IngestionConfigError
from src.ingestion import ingestion_orchestrator as orch


class ConfigGateTest(unittest.TestCase):
    """A malformed ingestion.yaml ends the run gracefully with config_error."""

    def test_run_ingestion_file_returns_config_error_status(self):
        with mock.patch.object(
            orch,
            "load_ingestion_config",
            side_effect=IngestionConfigError("broken"),
        ):
            result = orch.run_ingestion_file("whatever.pdf")
        self.assertEqual(result["status"], orch.STATUS_CONFIG_ERROR)
        self.assertEqual(result["document"], "whatever.pdf")

    def test_yaml_syntax_error_is_graceful(self):
        """A ConfigError (not only IngestionConfigError) is caught too."""
        with mock.patch.object(
            orch,
            "load_ingestion_config",
            side_effect=ConfigError("syntax error"),
        ):
            result = orch.run_ingestion_file("whatever.pdf")
        self.assertEqual(result["status"], orch.STATUS_CONFIG_ERROR)
        self.assertIn("fix_hint", result)

    def test_agent_wiring_failure_reports_config_error(self):
        """A missing LLM role at agent construction is a graceful config_error."""
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "doc.pdf"
            existing.write_bytes(b"%PDF-1.4")
            with mock.patch.object(
                orch,
                "build_default_agents",
                side_effect=MissingLLMRoleError(
                    "LLM role 'summarizer' is required by an agent but unusable: ..."
                ),
            ):
                result = orch.run_ingestion_file(existing, agents=None)

        self.assertEqual(result["status"], orch.STATUS_CONFIG_ERROR)
        self.assertIn("summarizer", result["message"])
        self.assertIn("llm.yaml", result["fix_hint"])
        self.assertEqual(result["document"], str(existing))

    def test_agent_wiring_failure_only_with_default_registry(self):
        """An explicit agent dict never triggers the wiring gate."""
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "doc.pdf"
            existing.write_bytes(b"%PDF-1.4")
            with mock.patch.object(
                orch,
                "build_default_agents",
                side_effect=MissingLLMRoleError("boom"),
            ):
                result = orch.run_ingestion_file(existing, agents={})
        # Empty explicit registry: graph runs and reports not-implemented.
        self.assertEqual(result["status"], orch.STATUS_NOT_IMPLEMENTED)


class FileChecksTest(unittest.TestCase):
    def test_missing_file_is_rejected(self):
        result = orch.run_ingestion_file("no/such/file.pdf")
        self.assertEqual(result["status"], orch.STATUS_REJECTED)
        self.assertIn("not found", result["reason"])


class FlagForwardingTest(unittest.TestCase):
    """CLI flags land in the graph context metadata verbatim."""

    @staticmethod
    def _fake_outcome():
        from src.graphs import GraphOutcome
        return GraphOutcome(accepted=True, completed_steps=["content_extraction"])

    def test_run_ingestion_file_forwards_force_flags(self):
        captured = {}
        graph_mock = mock.MagicMock()
        graph_mock.run.side_effect = lambda context: (
            captured.update(
                force_summary=context.metadata.get("force_summarization"),
                force_extraction=context.metadata.get("force_extraction"),
            )
        ) or self._fake_outcome()

        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "doc.pdf"
            pdf.write_bytes(b"%PDF-1.4")
            with mock.patch.object(orch, "IngestionGraph", return_value=graph_mock):
                result = orch.run_ingestion_file(
                    pdf, agents={}, force=True, force_summarization=True,
                )

        self.assertEqual(result["status"], orch.STATUS_ACCEPTED)
        self.assertTrue(captured["force_summary"])
        self.assertTrue(captured["force_extraction"])

    def test_origin_is_normalized_and_forwarded(self):
        captured = {}
        graph_mock = mock.MagicMock()
        graph_mock.run.side_effect = lambda context: (
            captured.update(origin=context.metadata.get("document_origin"))
        ) or self._fake_outcome()

        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "doc.pdf"
            pdf.write_bytes(b"%PDF-1.4")
            with mock.patch.object(orch, "IngestionGraph", return_value=graph_mock):
                orch.run_ingestion_file(pdf, agents={}, origin="RPG")

        self.assertEqual(captured["origin"], "rpg")

    def test_no_origin_means_no_metadata_entry(self):
        """The CLI makes -o mandatory: without it, nothing is set — the
        extraction validation reports the unverified default visibly."""
        captured = {}
        graph_mock = mock.MagicMock()
        graph_mock.run.side_effect = lambda context: (
            captured.update(metadata=dict(context.metadata))
        ) or self._fake_outcome()

        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "doc.pdf"
            pdf.write_bytes(b"%PDF-1.4")
            with mock.patch.object(orch, "IngestionGraph", return_value=graph_mock):
                orch.run_ingestion_file(pdf, agents={})

        self.assertNotIn("document_origin", captured["metadata"])

    def test_unknown_origin_is_rejected_with_fix_hint(self):
        """A value outside the user-defined vocabulary is a CONFIG error
        naming the configured origins — never silently rewritten."""
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "doc.pdf"
            pdf.write_bytes(b"%PDF-1.4")
            result = orch.run_ingestion_file(
                pdf, agents={}, origin="galactic-empire"
            )

        self.assertEqual(result["status"], orch.STATUS_CONFIG_ERROR)
        self.assertIn("galactic-empire", result["message"])
        self.assertIn("canon", result["message"])  # configured vocabulary named
        self.assertIn("setup.yaml", result["fix_hint"])


class LegacyApiRemovedTest(unittest.TestCase):
    """The chat-facing ingestion API is gone — structurally."""

    def test_no_free_text_entry_point(self):
        self.assertFalse(hasattr(orch, "run_ingestion"))
        self.assertFalse(hasattr(orch, "STATUS_NEEDS_CLARIFICATION"))

    def test_no_keyword_validation_api(self):
        for legacy in ("validate_request", "force_requested",
                       "force_summarization_requested"):
            with self.subTest(legacy=legacy):
                self.assertFalse(hasattr(orch, legacy), legacy)


if __name__ == "__main__":
    unittest.main()
