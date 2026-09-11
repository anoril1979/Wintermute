"""Tests for the strict LLM role resolution used by agents."""

from __future__ import annotations

import unittest
from unittest import mock

from src.agents.contexts import IngestionContext
from src.agents.llm_roles import LLMRoleAgent, MissingLLMRoleError, require_llm_role
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
from src.graphs import GraphStep, IngestionGraph
from src.tools.config_loader import ConfigError

ROLE_CONFIG = {"model_name": "llama3:8b", "temperature": 0.0}


class RequireLLMRoleTest(unittest.TestCase):
    def test_returns_role_config_when_role_exists(self):
        with mock.patch(
            "src.agents.llm_roles.config_loader.get_model_config",
            return_value=ROLE_CONFIG,
        ):
            self.assertEqual(require_llm_role("router"), ROLE_CONFIG)

    def test_raises_when_role_missing(self):
        with mock.patch(
            "src.agents.llm_roles.config_loader.get_model_config",
            side_effect=ConfigError("Rôle de modèle 'router' introuvable dans llm.yaml."),
        ):
            with self.assertRaises(MissingLLMRoleError) as ctx:
                require_llm_role("router")
        self.assertIn("'router'", str(ctx.exception))

    def test_never_falls_back_to_default_role(self):
        """Even with a 'default' role configured, a missing role must raise."""
        with mock.patch(
            "src.agents.llm_roles.config_loader.get_model_config",
            side_effect=ConfigError("not found"),
        ):
            with self.assertRaises(MissingLLMRoleError):
                require_llm_role("summarizer")


class _RoleAgent(LLMRoleAgent):
    llm_role = "router"

    def run(self, context):
        return AgentResult(agent_name=self.__class__.__name__, status=AgentStatus.OK)


class _NoRoleAgent(LLMRoleAgent):
    def run(self, context):
        return AgentResult(agent_name="no_role", status=AgentStatus.OK)


class LLMRoleAgentMixinTest(unittest.TestCase):
    def test_resolves_role_eagerly_at_construction(self):
        with mock.patch(
            "src.agents.llm_roles.config_loader.get_model_config",
            return_value=ROLE_CONFIG,
        ):
            agent = _RoleAgent()
        self.assertEqual(agent.llm_config, ROLE_CONFIG)
        self.assertEqual(agent._llm_role, "router")

    def test_missing_role_fails_at_wiring_time(self):
        with mock.patch(
            "src.agents.llm_roles.config_loader.get_model_config",
            side_effect=ConfigError("missing"),
        ):
            with self.assertRaises(MissingLLMRoleError):
                _RoleAgent()

    def test_agent_without_role_declaration_is_rejected(self):
        with self.assertRaises(MissingLLMRoleError):
            _NoRoleAgent()

    def test_constructor_override_wins_over_class_attribute(self):
        with mock.patch(
            "src.agents.llm_roles.config_loader.get_model_config",
            return_value=ROLE_CONFIG,
        ) as lookup:
            _RoleAgent(llm_role="summarizer")
        lookup.assert_called_once_with("summarizer")


class ConfigFailureDomainTest(unittest.TestCase):
    """A lazy agent reporting a CONFIG failure is rejected, not retried."""

    def test_config_failure_is_not_retryable(self):
        calls = []

        class LazyAgent:
            name = "lazy"

            def run(self, context):
                calls.append(1)
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.FAILED,
                    failure_domain=FailureDomain.CONFIG,
                    detail="LLM role 'summarizer' is not configured",
                )

        graph = IngestionGraph(
            agents={"summarizer": LazyAgent()},
            steps=[GraphStep(name="summarize", agent_key="summarizer", max_retries=3)],
        )
        outcome = graph.run(IngestionContext(document_path=None, request="test"))

        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.failed_step, "summarize")
        self.assertEqual(outcome.failure_detail, "LLM role 'summarizer' is not configured")
        self.assertEqual(len(calls), 1, "CONFIG is not retryable: exactly one attempt")


if __name__ == "__main__":
    unittest.main()