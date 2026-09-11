"""Protocols (structural contracts) for LLM clients.

The protocol is the single contract the rest of the system codes against:
agents, summarizers and orchestrators depend on ``LLMClientProtocol``, never
on a concrete client. Concrete implementations live beside it
(``llm_client_ollama.py`` today; a future OpenAI or internal-endpoint client
would join them) and must satisfy this interface.

The contract is deliberately minimal: one method. Implementations MAY offer
more (connection lifecycle, streaming, ...); consumers that need extras
should feature-test for them rather than require them here.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Minimal interface expected from any LLM client.

    Examples of implementations: a wrapper around the Ollama API
    (``llm_client_ollama.LLMClientOllama``), a remote OpenAI-compatible
    endpoint, an internal model server, a test double returning canned
    text, ...

    The ``runtime_checkable`` decorator allows ``isinstance(client,
    LLMClientProtocol)`` sanity checks (method presence only — it does not
    verify signatures).
    """

    def complete(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Send a prompt to the model and return the generated text.

        Args:
            prompt: full prompt text (system instructions included by caller).
            max_tokens: generation budget for THIS call. ``None`` (default)
                delegates to the client's configured default — the role's
                ``max_response_tokens`` in llm.yaml for config-driven
                clients. Callers only pass a value to tighten the role's
                budget for one specific call.

        Returns:
            The generated text, already cleaned of model-internal
            artifacts (reasoning blocks, stray whitespace). No metadata
            is exposed through this contract.

        Raises:
            ValueError: on an unusable prompt or non-positive ``max_tokens``.
            LLMClientError: when the backend cannot produce an answer.
        """
        ...
