"""LLM layer: protocol contract + concrete clients.

Public API:
    LLMClientProtocol   — the contract every LLM client satisfies (src/llm/protocols.py)
    LLMClientOllama     — Ollama-backed implementation (src/llm/llm_client_ollama.py)
    get_llm_client      — config-driven, cached client factory per llm.yaml role
    LLMClientError,
    LLMRequestError     — exception hierarchy for client failures

Legacy: ``rag_llm_ollama.py`` (prototype client) is kept only for the
modules in ``src/ingestion/old/``; new code must import from here.
"""

from src.llm.protocols import LLMClientProtocol
from src.llm.llm_client_ollama import (
    LLMClientError,
    LLMClientOllama,
    LLMRequestError,
    get_llm_client,
)

__all__ = [
    "LLMClientError",
    "LLMClientOllama",
    "LLMClientProtocol",
    "LLMRequestError",
    "get_llm_client",
]
