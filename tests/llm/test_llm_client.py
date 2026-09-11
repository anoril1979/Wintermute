"""Tests for LLMClientOllama role wiring and generation-budget delegation.

No live Ollama: an httpx mock transport captures the outgoing payloads, and
the llm.yaml role config is mocked (the real yaml is validated elsewhere).
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

import httpx

from src.llm.llm_client_ollama import LLMClientOllama


ROLE_CONFIG = {
    "model_name": "llama3:8b",
    "temperature": 0.0,
    "max_response_tokens": 256,
    "timeout_seconds": 15,
    "top_p": 0.95,
    "context_window": 8192,
    "keep_alive": "5m",
    "thinking": False,
    "max_retries": 2,
}


def _ollama_answer(request: httpx.Request) -> httpx.Response:
    """Fake /api/generate: echoes the request payload for inspection."""
    payload = json.loads(request.content.decode("utf-8"))
    return httpx.Response(
        200,
        json={"response": f"ok:{payload['options']['num_predict']}", "done": True},
        request=request,
    )


class _RecordingTransport(httpx.BaseTransport):
    """Mock transport remembering every request body."""

    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        payload = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"response": f"ok:{payload['options']['num_predict']}", "done": True},
            request=request,
        )


class RoleWiringTest(unittest.TestCase):
    """for_role() maps the renamed llm.yaml keys onto the client."""

    def setUp(self):
        self.transport = _RecordingTransport()

    def _for_role(self):
        with mock.patch("src.tools.config_loader.get_model_config",
                        return_value=dict(ROLE_CONFIG)), \
             mock.patch("src.tools.config_loader.load_setup_config",
                        return_value={"ollama": {"base_url": "http://localhost:11434"}}):
            client = LLMClientOllama.for_role("router")
        # Rebuild with the recording transport (for_role builds a real one).
        client._client = httpx.Client(base_url=client.base_url,
                                      timeout=client.timeout,
                                      transport=self.transport)
        return client

    def test_for_role_reads_context_window(self):
        client = self._for_role()
        self.assertEqual(client.context_window, 8192)

    def test_for_role_reads_max_response_tokens(self):
        client = self._for_role()
        self.assertEqual(client.max_response_tokens, 256)

    def test_payload_maps_num_ctx_and_num_predict(self):
        client = self._for_role()
        client.complete(prompt="hello")
        payload = json.loads(self.transport.calls[-1].content.decode("utf-8"))
        self.assertEqual(payload["options"]["num_ctx"], 8192)
        # No explicit budget -> the role's max_response_tokens is used.
        self.assertEqual(payload["options"]["num_predict"], 256)


class CompleteBudgetTest(unittest.TestCase):
    """complete(): explicit budget wins, None delegates to the default."""

    def setUp(self):
        self.transport = _RecordingTransport()
        self.client = LLMClientOllama(
            max_response_tokens=256, transport=self.transport
        )

    def _last_payload(self) -> dict:
        return json.loads(self.transport.calls[-1].content.decode("utf-8"))

    def test_default_budget_is_the_role_default(self):
        self.client.complete(prompt="hello")
        self.assertEqual(self._last_payload()["options"]["num_predict"], 256)

    def test_explicit_budget_overrides(self):
        self.client.complete(prompt="hello", max_tokens=42)
        self.assertEqual(self._last_payload()["options"]["num_predict"], 42)

    def test_none_means_default(self):
        self.client.complete(prompt="hello", max_tokens=None)
        self.assertEqual(self._last_payload()["options"]["num_predict"], 256)

    def test_non_positive_budget_rejected(self):
        with self.assertRaises(ValueError):
            self.client.complete(prompt="hello", max_tokens=0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
