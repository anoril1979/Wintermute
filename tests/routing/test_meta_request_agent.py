"""Tests for the MetaRequestAgent and the reply_to_meta_request switch.

The meta path: front-end auxiliary prompts (title/tags/follow-ups) are
intercepted by the guard, then either answered by the MetaRequestAgent
(switch on — fancy, one cheap LLM call) or by the fixed zero-cost text
(switch off / sentinel probe / any failure). A background task must
NEVER surface an error to the UI and NEVER reach the routing graph.
"""

from __future__ import annotations

import unittest
import unittest.mock

import app.api as api
from src.agents.agents import meta_request_agent as meta_module
from src.agents.agents.meta_request_agent import MetaRequestAgent, reset_meta_request_agent
from src.llm import guard
from src.tools.config_loader import ConfigError


TITLE_TASK = (
    "### Task: Generate a concise, 3-5 word title with an emoji "
    "summarizing the chat history."
)
FOLLOWUP_TASK = (
    "### Task: Suggest 3-5 relevant follow-up questions or prompts that "
    "the user might naturally ask next in this conversation."
)
TAGS_TASK = (
    "### Task: Generate 1-3 broad tags categorizing the main themes of "
    "the chat history."
)


def _config(value: object):
    return unittest.mock.patch(
        "src.tools.config_loader.load_setup_config",
        return_value={"reply_to_meta_request": value},
    )


class SwitchReaderTest(unittest.TestCase):
    """reply_to_meta_requests() — the setup.yaml switch."""

    def test_true_and_false_are_read(self):
        with _config(True):
            self.assertIs(guard.reply_to_meta_requests(), True)
        with _config(False):
            self.assertIs(guard.reply_to_meta_requests(), False)

    def test_absent_key_defaults_to_true(self):
        with _config(None):
            self.config_patch.return_value = {}
            self.assertIs(guard.reply_to_meta_requests(), True)

    def setUp(self):
        self.config_patch = unittest.mock.patch(
            "src.tools.config_loader.load_setup_config", return_value={}
        )
        self.config_mock = self.config_patch.start()
        self.addCleanup(self.config_patch.stop)

    def test_non_boolean_value_defaults_to_true(self):
        self.config_mock.return_value = {"reply_to_meta_request": "yes"}
        with self.assertLogs("src.llm.guard", level="WARNING"):
            self.assertIs(guard.reply_to_meta_requests(), True)

    def test_config_failure_defaults_to_true(self):
        self.config_mock.side_effect = RuntimeError("broken yaml")
        with self.assertLogs("src.llm.guard", level="WARNING"):
            self.assertIs(guard.reply_to_meta_requests(), True)


class MetaKindTest(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(guard.meta_kind(TITLE_TASK), "title")
        self.assertEqual(guard.meta_kind(FOLLOWUP_TASK), "followup")
        self.assertEqual(guard.meta_kind(TAGS_TASK), "tags")

    def test_unknown_kind(self):
        self.assertEqual(guard.meta_kind(guard.META_SENTINEL), "unknown")


class _FakeLLM:
    def __init__(self, answer="A fancy title"):
        self.answer = answer
        self.calls: list[str] = []
        self.error: Exception | None = None

    def complete(self, prompt, max_tokens=None):
        self.calls.append(prompt)
        if self.error is not None:
            raise self.error
        return self.answer


def _role_config_patch():
    """A resolvable meta_request role (skips the real llm.yaml)."""
    return unittest.mock.patch(
        "src.tools.config_loader.get_model_config",
        return_value={"model_name": "test-model"},
    )


class MetaRequestAgentTest(unittest.TestCase):
    def setUp(self):
        reset_meta_request_agent()
        self.addCleanup(reset_meta_request_agent)

    def _agent(self, llm):
        with _role_config_patch():
            return MetaRequestAgent(llm=llm)

    def test_non_meta_prompt_is_refused_without_llm_call(self):
        llm = _FakeLLM()
        agent = self._agent(llm)
        self.assertEqual(agent.run("Ingest meow.pdf, please."), guard.meta_answer())
        self.assertEqual(llm.calls, [])

    def test_success_returns_the_llm_answer(self):
        llm = _FakeLLM(answer="Ingesting the gazettes")
        agent = self._agent(llm)
        self.assertEqual(agent.run(TITLE_TASK), "Ingesting the gazettes")
        self.assertEqual(len(llm.calls), 1)
        # The prompt embeds the persona file and the kind hint.
        prompt = llm.calls[0]
        self.assertIn("Meta-request answers", prompt)
        self.assertIn("FRONT-END TASK (title)", prompt)
        self.assertIn(TITLE_TASK, prompt)

    def test_llm_failure_degrades_to_the_fixed_answer(self):
        llm = _FakeLLM()
        llm.error = RuntimeError("ollama down")
        agent = self._agent(llm)
        with self.assertLogs("src.agents.agents.meta_request_agent", level="WARNING"):
            self.assertEqual(agent.run(TITLE_TASK), guard.meta_answer())

    def test_empty_answer_degrades_to_the_fixed_answer(self):
        llm = _FakeLLM(answer="   ")
        agent = self._agent(llm)
        self.assertEqual(agent.run(TITLE_TASK), guard.meta_answer())

    def test_missing_role_degrades_to_the_fixed_answer(self):
        # The patch covers CONSTRUCTION (the role is resolved eagerly):
        # allow_missing_role swallows it, and run() then falls back.
        with unittest.mock.patch(
            "src.tools.config_loader.get_model_config",
            side_effect=ConfigError("no 'meta_request' role"),
        ):
            agent = MetaRequestAgent(allow_missing_role=True, llm=_FakeLLM())
            self.assertEqual(agent.run(TITLE_TASK), guard.meta_answer())

    def test_missing_role_fails_fast_by_default(self):
        with unittest.mock.patch(
            "src.tools.config_loader.get_model_config",
            side_effect=ConfigError("no 'meta_request' role"),
        ):
            with self.assertRaises(ConfigError):
                MetaRequestAgent()

    def test_process_wide_accessor_caches_one_instance(self):
        with _role_config_patch():
            first = meta_module.get_meta_request_agent()
            second = meta_module.get_meta_request_agent()
        self.assertIs(first, second)


class MetaAnswerDispatchTest(unittest.TestCase):
    """_meta_answer_for_prompt — the single switch decision point."""

    def setUp(self):
        self.agent = unittest.mock.Mock()
        self.agent.run.return_value = "A fancy title"
        self.agent_patch = unittest.mock.patch(
            "src.agents.agents.meta_request_agent.get_meta_request_agent",
            return_value=self.agent,
        )
        self.agent_patch.start()
        self.addCleanup(self.agent_patch.stop)

    def test_sentinel_stays_fixed_even_with_the_switch_on(self):
        with _config(True):
            self.assertEqual(
                api._meta_answer_for_prompt(guard.META_SENTINEL),
                guard.meta_answer(),
            )
        self.agent.run.assert_not_called()  # zero-cost probe preserved

    def test_switch_off_uses_the_fixed_answer_without_the_agent(self):
        with _config(False):
            self.assertEqual(
                api._meta_answer_for_prompt(TITLE_TASK), guard.meta_answer()
            )
        self.agent.run.assert_not_called()

    def test_switch_on_forwards_to_the_agent(self):
        with _config(True):
            self.assertEqual(
                api._meta_answer_for_prompt(TITLE_TASK), "A fancy title"
            )
        self.agent.run.assert_called_once_with(TITLE_TASK)

    def test_agent_failure_degrades_to_the_fixed_answer(self):
        self.agent.run.side_effect = RuntimeError("boom")
        with _config(True):
            self.assertEqual(
                api._meta_answer_for_prompt(TITLE_TASK), guard.meta_answer()
            )


class MetaEndpointTest(unittest.TestCase):
    """End to end through /v1/chat/completions (hermetic)."""

    def setUp(self):
        from fastapi.testclient import TestClient

        self.client = TestClient(api.app)
        self.agent = unittest.mock.Mock()
        self.agent.run.return_value = "A fancy title"
        self.agent_patch = unittest.mock.patch(
            "src.agents.agents.meta_request_agent.get_meta_request_agent",
            return_value=self.agent,
        )
        self.agent_patch.start()
        self.addCleanup(self.agent_patch.stop)

    def test_meta_prompt_gets_the_agent_answer(self):
        with _config(True):
            response = self.client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": TITLE_TASK}]},
            )
        self.assertEqual(response.status_code, 200)
        content = response.json()["choices"][0]["message"]["content"]
        self.assertEqual(content, "A fancy title")
        self.agent.run.assert_called_once_with(TITLE_TASK)

    def test_routing_is_never_called_for_a_meta_prompt(self):
        with _config(True), unittest.mock.patch(
            "src.routing.routing_orchestrator.run_routing"
        ) as run_routing:
            response = self.client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": TAGS_TASK}]},
            )
        self.assertEqual(response.status_code, 200)
        run_routing.assert_not_called()

    def test_sentinel_through_the_endpoint_is_zero_cost(self):
        with _config(True):
            response = self.client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": guard.META_SENTINEL}]},
            )
        content = response.json()["choices"][0]["message"]["content"]
        self.assertEqual(content, guard.meta_answer())
        self.agent.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
