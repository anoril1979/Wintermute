"""Tests for the GeneralTaskAgent (stubbed LLM) and the routing registry.

The agent answers ``general`` requests with the LLM's own knowledge, in
the Wintermute persona (the prompt's job). These tests stub the LLM
client — no live Ollama — and check the outcome mapping, the traces and
the registry wiring.
"""

from __future__ import annotations

import unittest
import unittest.mock

from src.agents.agents.general_task_agent import GeneralTaskAgent
from src.agents.contexts import RoutingContext
from src.agents.protocols import AgentStatus, FailureDomain
from src.routing.models import (
    RequestContextEntry,
    RequestKind,
    UserRequest,
)


def _general(utterance="What color is the sky over the Sprawl?"):
    return UserRequest(kind=RequestKind.GENERAL, utterance=utterance)


def _ingestion():
    return UserRequest(kind=RequestKind.INGESTION, utterance="u", document="a.pdf")


class _FakeLLM:
    """Stands in for the role's Ollama client."""

    def __init__(self, answer="The sky was the color of television, tuned to a dead channel."):
        self.answer = answer
        self.prompts: list[str] = []

    def complete(self, prompt, max_tokens=None):
        self.prompts.append(prompt)
        return self.answer


def _agent(llm, **kwargs) -> GeneralTaskAgent:
    return GeneralTaskAgent(allow_missing_role=True, llm=llm, **kwargs)


class GeneralTaskAgentTest(unittest.TestCase):
    def setUp(self):
        self.context = RoutingContext(request="test")

    def test_ok_answer_carries_payload_and_traces(self):
        llm = _FakeLLM("Ice-cold answer.")
        result = _agent(llm).run(self.context, _general())
        self.assertEqual(result.status, AgentStatus.OK)
        self.assertEqual(result.payload["answer"], "Ice-cold answer.")
        kinds = [e["kind"] for e in self.context.events if e["phase"] == "task"]
        self.assertEqual(kinds, ["general_start", "general_done"])

    def test_prompt_carries_persona_and_utterance(self):
        llm = _FakeLLM()
        _agent(llm).run(self.context, _general("hello there"))
        self.assertEqual(len(llm.prompts), 1)
        prompt = llm.prompts[0]
        self.assertIn("Wintermute", prompt)          # persona instructions
        self.assertIn("<<<<PROMPT>>>>", prompt)      # delimiter convention
        self.assertIn("hello there", prompt)         # the utterance, verbatim

    def test_llm_failure_maps_to_llm_response_domain(self):
        class DownLLM:
            def complete(self, prompt, max_tokens=None):
                raise RuntimeError("ollama down")

        result = _agent(DownLLM()).run(self.context, _general())
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.failure_domain, FailureDomain.LLM_RESPONSE)
        self.assertIn("ollama down", result.detail)

    def test_empty_answer_is_a_failure(self):
        result = _agent(_FakeLLM(answer="")).run(self.context, _general())
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.failure_domain, FailureDomain.LLM_RESPONSE)

    def test_wrong_kind_is_input_data_failure(self):
        llm = _FakeLLM()
        result = _agent(llm).run(self.context, _ingestion())
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.failure_domain, FailureDomain.INPUT_DATA)
        self.assertEqual(llm.prompts, [])  # no LLM call made

    def test_out_of_contract_request_is_refused(self):
        result = _agent(_FakeLLM()).run(self.context, object())
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.failure_domain, FailureDomain.INPUT_DATA)

    def test_missing_role_fails_cleanly_with_config_domain(self):
        # allow_missing_role=True: construction survives, run() reports.
        import src.agents.llm_roles as llm_roles

        with unittest.mock.patch(
            "src.agents.llm_roles.require_llm_role",
            side_effect=llm_roles.MissingLLMRoleError("nope"),
        ):
            agent = GeneralTaskAgent(allow_missing_role=True)
        result = agent.run(self.context, _general())
        self.assertEqual(result.status, AgentStatus.FAILED)
        self.assertEqual(result.failure_domain, FailureDomain.CONFIG)
        self.assertIn("general_task", result.detail)

    def test_strict_construction_raises_when_role_missing(self):
        # Production wiring stays fail-fast: no allow_missing_role, a
        # missing role raises at construction (wiring time).
        import src.agents.llm_roles as llm_roles

        with unittest.mock.patch(
            "src.agents.llm_roles.require_llm_role",
            side_effect=llm_roles.MissingLLMRoleError("nope"),
        ):
            with self.assertRaises(llm_roles.MissingLLMRoleError):
                GeneralTaskAgent()

    def test_validate_is_a_noop(self):
        self.assertIsNone(_agent(_FakeLLM()).validate(self.context, _general()))


class PromptLocalMemoryTest(unittest.TestCase):
    """The general agent renders same-prompt predecessors into its prompt."""

    def setUp(self):
        self.context = RoutingContext(request="test")

    def test_no_preceding_keeps_prompt_unchanged(self):
        llm = _FakeLLM()
        _agent(llm).run(self.context, _general())
        prompt = llm.prompts[0]
        # the *dynamic* block marker (the persona only *describes* the
        # block, without the ', in order' phrasing used at build time)
        self.assertNotIn("Earlier requests of this same user prompt, in order", prompt)

    def test_preceding_rendered_before_utterance(self):
        llm = _FakeLLM()
        request = UserRequest(
            kind=RequestKind.GENERAL,
            utterance="so, is it safe?",
            preceding=[
                RequestContextEntry(
                    kind="ingestion", utterance="ingest meow.pdf",
                    document="meow.pdf", status="done",
                ),
                RequestContextEntry(
                    kind="general", utterance="thanks!", status="rejected",
                    detail="boom",
                ),
            ],
        )
        _agent(llm).run(self.context, request)
        prompt = llm.prompts[0]
        self.assertIn("Earlier requests of this same user prompt, in order", prompt)
        self.assertIn('[ingestion] "ingest meow.pdf" — done', prompt)
        self.assertIn('[general] "thanks!" — rejected: boom', prompt)
        # context block injected after the persona rules, before the utterance
        self.assertGreater(
            prompt.index("Earlier requests of this same user prompt, in order"),
            prompt.index("## Output format"),  # the persona's last section
        )
        self.assertLess(
            prompt.index("Earlier requests of this same user prompt, in order"),
            prompt.index("User request:\n"),
        )

    def test_preceding_document_available_for_pronoun_resolution(self):
        llm = _FakeLLM()
        request = UserRequest(
            kind=RequestKind.GENERAL, utterance="is it indexed?",
            preceding=[RequestContextEntry(
                kind="ingestion", utterance="ingest meow.pdf",
                document="meow.pdf", status="done",
            )],
        )
        _agent(llm).run(self.context, request)
        self.assertIn("meow.pdf", llm.prompts[0])


class RegistryWiringTest(unittest.TestCase):
    def test_default_registry_contains_general_task(self):
        from src.agents.routing_registry import build_default_task_agents
        from src.agents.task_protocols import TASK_AGENT_KEYS

        agents = build_default_task_agents()
        self.assertIn(TASK_AGENT_KEYS["general"], agents)
        self.assertIsInstance(agents[TASK_AGENT_KEYS["general"]], GeneralTaskAgent)


if __name__ == "__main__":
    unittest.main()
