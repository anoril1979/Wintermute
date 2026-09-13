"""Ollama-backed implementation of the LLM client protocol.

``LLMClientOllama`` merges the best of the two prototype versions:

* from ``rag_llm_ollama.py`` (raw httpx client): direct control of the
  ``/api/generate`` endpoint, correct option mapping (``num_predict``,
  ``num_ctx``, ``think``, ``keep_alive``), Qwen3 think-block cleanup,
  retries with a reusable keep-alive HTTP client, context-manager
  lifecycle;
* from ``llm_factory.py`` (LangChain factory): role-based instantiation
  driven by config/llm.yaml, one cached instance per role, dedicated
  exception types with actionable messages.

and drops their weaknesses: no more import-time config reads, no
module-level ``logging.basicConfig``, no dead code, a clear exception
hierarchy, and an injectable HTTP transport so the client is unit-testable
without a live Ollama.

Two ways to obtain a client:

* ``LLMClientOllama(...)`` — explicit construction with plain defaults
  (useful in tests and when tuning a specific call);
* ``get_llm_client(role)`` / ``LLMClientOllama.for_role(role)`` — the
  config-driven single entry point used everywhere else. One instance per
  role per run (cached): treat it as a shared connection, close it only
  at shutdown, or simply use it as a context manager.

Example:
    with get_llm_client("summarizer") as client:
        summary = client.complete(prompt=text, max_tokens=1024)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

import httpx

from src.llm.protocols import LLMClientProtocol
from src.tools import config_loader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions — dedicated types instead of bare RuntimeError, so callers can
# catch precisely and the failure domain (see src/agents.protocols) is obvious.
# ---------------------------------------------------------------------------

class LLMClientError(Exception):
    """Base class for LLM client failures."""


class LLMRequestError(LLMClientError):
    """The backend could not be reached or returned an unusable response."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@dataclass
class LLMClientOllama:
    """LLM client served by a local Ollama instance.

    Implements :class:`src.llm.protocols.LLMClientProtocol` (one method:
    ``complete``), plus connection lifecycle extras beyond the contract
    (``close()``, context-manager protocol).

    Attributes are plain, side-effect-free defaults; the config-driven
    construction path is :meth:`for_role` / :func:`get_llm_client`.
    """

    # Connection
    model: str = "qwen3"
    base_url: str = "http://localhost:11434"
    timeout: float = 30.0

    # Inference defaults (llm.yaml keys in parentheses where they differ)
    temperature: float = 0.2
    top_p: float = 0.95
    context_window: int = 8192            # Ollama option num_ctx — what the model can READ
    max_response_tokens: int = 1024       # Ollama option num_predict — what it can WRITE
    keep_alive: str = "5m"
    enable_thinking: bool = False  # llm.yaml "thinking" — reasoning mode

    # Robustness
    max_retries: int = 2

    # Extra options passed as-is to Ollama (repeat_penalty, seed, stop, ...)
    extra_options: dict[str, Any] = field(default_factory=dict)

    # Injectable transport (tests); a real connection is built in __post_init__
    transport: Optional[httpx.BaseTransport] = field(default=None, repr=False)

    # Internal reusable HTTP client (keep-alive)
    _client: Optional[httpx.Client] = field(default=None, init=False, repr=False)

    # Compiled regex stripping Qwen3 reasoning blocks. The tag strings are
    # assembled dynamically so the source never contains a literal
    # "<think>" that a careless parser (or copy-paste into a prompt) could
    # misinterpret.
    _THINK_TAG_OPEN = "<" + "think" + ">"
    _THINK_TAG_CLOSE = "</" + "think" + ">"
    _THINK_BLOCK_RE = re.compile(
        re.escape(_THINK_TAG_OPEN) + r".*?" + re.escape(_THINK_TAG_CLOSE),
        flags=re.DOTALL | re.IGNORECASE,
    )

    # -- Lifecycle ----------------------------------------------------------

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout, transport=self.transport)

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "LLMClientOllama":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- Public API (LLMClientProtocol) --------------------------------------

    def complete(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Send ``prompt`` to Ollama and return the cleaned generated text.

        Args:
            prompt: full prompt text.
            max_tokens: generation budget for THIS call (mapped to Ollama
                ``num_predict``). ``None`` (default) uses the role's
                ``max_response_tokens`` from llm.yaml.

        Returns:
            The generated text, stripped of reasoning blocks and whitespace.

        Raises:
            ValueError: empty prompt or non-positive ``max_tokens``.
            LLMRequestError: Ollama unreachable/unusable answer after retries.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty.")
        budget = self.max_response_tokens if max_tokens is None else max_tokens
        if budget <= 0:
            raise ValueError("max_tokens must be strictly positive.")

        payload = self._build_payload(prompt=prompt, max_tokens=budget)
        raw = self._post_with_retry("/api/generate", payload)

        text = raw.get("response", "")
        if not isinstance(text, str):
            raise LLMRequestError(
                f"Unexpected Ollama answer: 'response' field has type "
                f"{type(text).__name__}, expected str."
            )
        return self._postprocess(text)

    # -- Config-driven construction (the llm_factory heritage) ----------------

    @classmethod
    def for_role(cls, role: str) -> "LLMClientOllama":
        """Build a client from a functional role defined in config/llm.yaml.

        Args:
            role: a role under the ``models`` key of llm.yaml
                ("router", "summarizer", "default", "answerer", ...).

        Returns:
            A client whose defaults come from the role section, with the
            Ollama base URL from config/setup.yaml.

        Raises:
            LLMClientError: unknown role or invalid configuration.
        """
        try:
            model_cfg = config_loader.get_model_config(role)
            base_url = config_loader.load_setup_config()["ollama"]["base_url"]
        except Exception as exc:
            raise LLMClientError(
                f"Invalid configuration for role '{role}': {exc}"
            ) from exc

        try:
            return cls(
                model=model_cfg["model_name"],
                base_url=base_url,
                timeout=float(model_cfg.get("timeout_seconds", 30.0)),
                temperature=float(model_cfg.get("temperature", 0.2)),
                top_p=float(model_cfg.get("top_p", 0.95)),
                context_window=int(model_cfg.get("context_window", 8192)),
                max_response_tokens=int(model_cfg.get("max_response_tokens", 1024)),
                keep_alive=str(model_cfg.get("keep_alive", "5m")),
                enable_thinking=bool(model_cfg.get("thinking", False)),  # yaml key: thinking
                max_retries=int(model_cfg.get("max_retries", 2)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMClientError(
                f"Failed to build client for role '{role}' "
                f"(model_name='{model_cfg.get('model_name')}'): {exc}. "
                f"Check llm.yaml and that the model is installed (`ollama list`)."
            ) from exc

    # -- Request building -----------------------------------------------------

    def _build_payload(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_ctx": self.context_window,      # what the model can read
            "num_predict": max_tokens,          # what the model can write
            "think": self.enable_thinking,      # Qwen3 reasoning mode
        }
        options.update(self.extra_options)

        return {
            "model": self.model,
            "prompt": prompt.strip(),
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": options,
        }

    # -- HTTP with retries ------------------------------------------------------

    def _post_with_retry(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            raise LLMRequestError("HTTP client is not initialized or already closed.")

        last_exc: Optional[Exception] = None
        attempts = self.max_retries + 1

        for attempt in range(1, attempts + 1):
            try:
                logger.debug(
                    "Calling Ollama %s (attempt %d/%d) model=%s num_predict=%s",
                    path, attempt, attempts,
                    payload.get("model"),
                    payload.get("options", {}).get("num_predict"),
                )
                logger.debug(">>> prompt: %s", payload.get("prompt"))
                response = self._client.post(path, json=payload)
                response.raise_for_status()
                return response.json()

            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                last_exc = exc
                logger.warning("Ollama call failed (attempt %d/%d): %s", attempt, attempts, exc)

        raise LLMRequestError(
            f"No answer from Ollama after {attempts} attempts "
            f"({self.base_url}{path}, model={payload.get('model')}). "
            f"Check that Ollama is running (`ollama serve`) "
            f"and the model is installed (`ollama list`)."
        ) from last_exc

    # -- Post-processing ----------------------------------------------------------

    @classmethod
    def _postprocess(cls, text: str) -> str:
        """Strip Qwen3 internal reasoning blocks and stray whitespace."""
        if not text:
            return ""
        return cls._THINK_BLOCK_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Single entry point (the llm_factory heritage): one cached client per role.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def get_llm_client(role: str = "default") -> LLMClientOllama:
    """Return the shared LLM client for a functional role.

    Cached per role: the client (and its HTTP connection pool) is built once
    per run. Use it as a context manager if you need deterministic release.

    Raises:
        LLMClientError: unknown role or invalid configuration.
    """
    return LLMClientOllama.for_role(role)


# ---------------------------------------------------------------------------
# Manual check (direct module execution): needs a live Ollama.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = get_llm_client("default")
    assert isinstance(client, LLMClientProtocol)  # runtime_checkable protocol

    logger.info("Client ready: model=%s base_url=%s", client.model, client.base_url)

    with client:
        answer = client.complete(
            prompt=(
                "Summarize in exactly 5 words: 'Il était un petit chat qui "
                "buvait son lait dans la ferme. Et Paf le chat.'"
            ),
            max_tokens=2048,
        )
        print(answer)
