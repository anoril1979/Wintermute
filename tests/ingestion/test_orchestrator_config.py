"""Tests for the ingestion orchestrator's configuration gate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.agents import MissingLLMRoleError
from src.tools.config_loader import IngestionConfigError
from src.ingestion import ingestion_orchestrator as orch


def _clarify_outcome():
    """A routing outcome that needs clarification (for gate-only tests)."""
    from src.graphs.ingestion_routing_graph import IngestionRoutingOutcome
    from src.ingestion.ingestion_router import (
        ROUTER_NEEDS_CLARIFICATION,
        RoutingDecision,
    )

    return IngestionRoutingOutcome(
        status=ROUTER_NEEDS_CLARIFICATION,
        decision=RoutingDecision(
            status=ROUTER_NEEDS_CLARIFICATION,
            explanation="gate test",
            question="Which document?",
        ),
    )


class ConfigGateTest(unittest.TestCase):
    """A malformed ingestion.yaml ends the run gracefully with config_error."""

    def test_run_ingestion_returns_config_error_status(self):
        with mock.patch.object(
            orch,
            "load_ingestion_config",
            side_effect=IngestionConfigError(
                "ingestion.yaml invalide : 'documents_root' doit être une chaîne."
            ),
        ):
            result = orch.run_ingestion("please ingest meow.pdf")

        self.assertEqual(result["status"], orch.STATUS_CONFIG_ERROR)
        self.assertIn("'documents_root'", result["message"])
        self.assertIn("fix", result["fix_hint"].lower())
        self.assertEqual(result["request"], "please ingest meow.pdf")

    def test_config_error_takes_precedence_over_request_validation(self):
        """Step 0 runs before request validation: a broken yaml wins."""
        with mock.patch.object(
            orch,
            "load_ingestion_config",
            side_effect=IngestionConfigError("broken"),
        ):
            result = orch.run_ingestion("this is not an ingestion request")
        self.assertEqual(result["status"], orch.STATUS_CONFIG_ERROR)

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
            side_effect=orch.ConfigError("syntax error"),
        ):
            result = orch.run_ingestion("ingest meow.pdf")
        self.assertEqual(result["status"], orch.STATUS_CONFIG_ERROR)
        self.assertIn("fix_hint", result)

    def test_valid_config_does_not_trigger_the_gate(self):
        """With a valid config the flow proceeds to normal rejection paths."""
        # A stub routing graph: the test is about the config gate, not the
        # router (a real one would call the live Ollama endpoint).
        routing = mock.MagicMock()
        routing.run.return_value = _clarify_outcome()
        with mock.patch.object(
            orch,
            "load_ingestion_config",
            return_value={
                "documents_root": "data/sources",
                "extensions": {".pdf": "pdf"},
            },
        ):
            result = orch.run_ingestion("please ingest meow.pdf",
                                        agents={}, routing_graph=routing)
        self.assertEqual(result["status"], orch.STATUS_NEEDS_CLARIFICATION)

    def test_real_routing_graph_construction_with_valid_config(self):
        """The default routing graph wires (LLM role resolved) without error."""
        with mock.patch.object(
            orch,
            "load_ingestion_config",
            return_value={
                "documents_root": "data/sources",
                "extensions": {".pdf": "pdf"},
            },
        ):
            graph = orch.IngestionRoutingGraph()
        # Strict role resolution happened at construction: the router holds
        # the llm.yaml config for the ingestion_router role.
        self.assertTrue(graph._router.llm_config)

    def test_cli_reports_config_error_human_readably(self):
        """The CLI prints message + hint and exits 2 on a config error."""
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.pdf"
            with mock.patch.object(
                orch,
                "load_ingestion_config",
                side_effect=IngestionConfigError("ingestion.yaml invalide : x"),
            ):
                with mock.patch("sys.stdout"):
                    code = orch.main(["-i", str(missing)])
            self.assertEqual(code, 2)

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


class ForceRoutingTest(unittest.TestCase):
    """Force intents are classified by the LLM router (keyword fallback).

    The orchestrator no longer sniffs keywords itself: the intent lives in
    src/ingestion/ingestion_router.py (tested in test_ingestion_router.py).
    Here we only lock the contract that the orchestrator module exposes no
    keyword API anymore — flags enter the flow through the routing graph.
    """

    def test_orchestrator_has_no_keyword_validation_api(self):
        """The legacy keyword helpers are gone (LLM router replaced them)."""
        for legacy in ("validate_request", "force_requested",
                       "force_summarization_requested"):
            with self.subTest(legacy=legacy):
                self.assertFalse(hasattr(orch, legacy), legacy)

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _fake_outcome():
        from src.graphs import GraphOutcome
        return GraphOutcome(accepted=True, completed_steps=["content_extraction"])

    @staticmethod
    def _proceed_routing_graph(flags):
        """A routing graph stub that proceeds with the given flags."""
        from src.graphs.ingestion_routing_graph import IngestionRoutingOutcome
        from src.ingestion.ingestion_router import (
            ROUTER_PROCEED,
            IngestionFacts,
            RoutingDecision,
        )

        outcome = IngestionRoutingOutcome(
            status=ROUTER_PROCEED,
            flags=dict(flags),
            decision=RoutingDecision(
                status=ROUTER_PROCEED, flags=dict(flags), explanation="test"
            ),
            facts=IngestionFacts(
                file_name="meow.pdf", found=True, source_path=Path("x/meow.pdf")
            ),
        )
        routing = mock.MagicMock()
        routing.run.return_value = outcome
        return routing

    def _run_with_routing(self, request, flags, captured):
        graph_mock = mock.MagicMock()
        graph_mock.run.side_effect = lambda context: (
            captured.update(metadata=dict(context.metadata))
        ) or self._fake_outcome()
        result = orch.run_ingestion(
            request, agents={},
            routing_graph=self._proceed_routing_graph(flags),
            graph=graph_mock,
        )
        return result

    def test_run_ingestion_sets_force_flag_in_context(self):
        captured = {}
        result = self._run_with_routing(
            "force extraction of meow.pdf",
            {"force_extraction": True, "force_summarization": True},
            captured,
        )
        self.assertEqual(result["status"], orch.STATUS_ACCEPTED)
        self.assertTrue(captured["metadata"]["force_extraction"])
        self.assertTrue(captured["metadata"]["force_summarization"])

    def test_run_ingestion_sets_summarization_flags_in_context(self):
        captured = {}
        result = self._run_with_routing(
            "re-ingest meow.pdf and redo the summaries",
            {"force_extraction": False, "force_summarization": True},
            captured,
        )
        self.assertEqual(result["status"], orch.STATUS_ACCEPTED)
        self.assertFalse(captured["metadata"]["force_extraction"])
        self.assertTrue(captured["metadata"]["force_summarization"])

    def test_run_ingestion_does_not_set_skip_summarization(self):
        """No skip mode by design: summaries follow the content."""
        captured = {}
        result = self._run_with_routing(
            "please ingest meow.pdf",
            {"force_extraction": False, "force_summarization": False},
            captured,
        )
        self.assertEqual(result["status"], orch.STATUS_ACCEPTED)
        self.assertFalse(captured["metadata"]["force_summarization"])
        self.assertNotIn("skip_summarization", captured["metadata"])

    def test_run_ingestion_file_forwards_summarization_flags(self):
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

    def test_needs_clarification_status_carries_the_question(self):
        """A non-proceed routing ends the run with the question for the user."""
        from src.graphs.ingestion_routing_graph import IngestionRoutingOutcome
        from src.ingestion.ingestion_router import (
            ROUTER_NEEDS_CLARIFICATION,
            IngestionFacts,
            RoutingDecision,
        )

        outcome = IngestionRoutingOutcome(
            status=ROUTER_NEEDS_CLARIFICATION,
            decision=RoutingDecision(
                status=ROUTER_NEEDS_CLARIFICATION,
                explanation="no such document",
                question="Which file?",
                suggestions=["list the available documents"],
            ),
            facts=IngestionFacts(file_name="meow.pdf", found=False),
        )
        routing = mock.MagicMock()
        routing.run.return_value = outcome

        result = orch.run_ingestion("ingest meow.pdf", agents={}, routing_graph=routing)

        self.assertEqual(result["status"], orch.STATUS_NEEDS_CLARIFICATION)
        self.assertEqual(result["question"], "Which file?")
        self.assertIn("suggestions", result)


if __name__ == "__main__":
    unittest.main()