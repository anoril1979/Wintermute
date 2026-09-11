"""Ollama-backed implementation of the embedding client protocol.

Built like ``src/llm/llm_client_ollama.py`` (same anatomy, same
conventions):

* a plain dataclass with injectable httpx transport — hermetic tests, no
  live Ollama needed;
* retries on transport failures with a reusable keep-alive HTTP client and
  context-manager lifecycle;
* config-driven construction: the ``embedding`` role of config/llm.yaml
  (model, timeout, retries, keep_alive) + setup.yaml's Ollama base URL;
* a single cached entry point, :func:`get_embedding_client`, mirroring
  ``get_llm_client``.

An embedding model generates no text: no temperature, no generation budget.
The only things that matter are reliability (timeout, retries) and one
property the whole pipeline depends on — **dimensional stability**: a
collection's vector dimension is frozen by its first write, so the role's
model must not change casually (see config/llm.yaml).

Calls Ollama's ``/api/embed`` endpoint, batch-oriented: one request per
batch of texts, order-preserving.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional, Sequence

import httpx

from src.indexing.protocols import EmbeddingClientProtocol
from src.tools import config_loader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions — same philosophy as the LLM client: precise types, actionable
# messages, no bare RuntimeError.
# ---------------------------------------------------------------------------

class EmbeddingClientError(Exception):
    """Base class for embedding client failures."""


class EmbeddingRequestError(EmbeddingClientError):
    """The backend could not be reached or returned an unusable response."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@dataclass
class OllamaEmbeddingClient:
    """Embedding client served by a local Ollama instance.

    Implements :class:`src.indexing.protocols.EmbeddingClientProtocol`
    (one method: ``embed``), plus connection lifecycle extras beyond the
    contract (``close()``, context-manager protocol).

    Attributes are plain, side-effect-free defaults; the config-driven
    construction path is :meth:`for_role` / :func:`get_embedding_client`.
    """

    # Connection
    model: str = "qwen3-embedding"
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0

    # Robustness
    keep_alive: str = "10m"
    max_retries: int = 3

    # Injectable transport (tests); a real connection is built in __post_init__
    transport: Optional[httpx.BaseTransport] = field(default=None, repr=False)

    # Internal reusable HTTP client (keep-alive)
    _client: Optional[httpx.Client] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url, timeout=self.timeout, transport=self.transport
        )

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "OllamaEmbeddingClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- Public API (EmbeddingClientProtocol) --------------------------------

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, order-preserving.

        Args:
            texts: non-empty texts. A batch is rejected if empty or if any
                text is empty/blank — silently embedding garbage would
                poison the corpus with unfindable vectors.

        Returns:
            One vector per input text, in the same order. All vectors share
            the model's dimensionality.

        Raises:
            ValueError: empty batch, or an empty/blank text inside it.
            EmbeddingRequestError: Ollama unreachable/unusable answer after
                retries, or a malformed answer (count/dimension mismatch).
        """
        if not texts:
            raise ValueError("embed() requires a non-empty batch of texts.")
        cleaned = [t.strip() for t in texts]
        for index, text in enumerate(cleaned):
            if not text:
                raise ValueError(
                    f"embed() received an empty/blank text at index {index}; "
                    "blank chunks must be pruned before embedding."
                )

        payload: dict[str, Any] = {
            "model": self.model,
            "input": cleaned,
            "keep_alive": self.keep_alive,
        }
        raw = self._post_with_retry("/api/embed", payload)
        return self._parse_answer(raw, expected=len(cleaned))

    # -- Config-driven construction ------------------------------------------

    @classmethod
    def for_role(cls, role: str = "embedding") -> "OllamaEmbeddingClient":
        """Build a client from the ``embedding`` role of config/llm.yaml.

        Args:
            role: the llm.yaml role providing the embedding model
                (default: ``embedding``).

        Returns:
            A client whose connection settings come from the role section,
            with the Ollama base URL from config/setup.yaml.

        Raises:
            EmbeddingClientError: unknown role or invalid configuration.
        """
        try:
            model_cfg = config_loader.get_model_config(role)
            base_url = config_loader.load_setup_config()["ollama"]["base_url"]
        except Exception as exc:
            raise EmbeddingClientError(
                f"Invalid configuration for embedding role '{role}': {exc}"
            ) from exc

        try:
            return cls(
                model=model_cfg["model_name"],
                base_url=base_url,
                timeout=float(model_cfg.get("timeout_seconds", 120.0)),
                keep_alive=str(model_cfg.get("keep_alive", "10m")),
                max_retries=int(model_cfg.get("max_retries", 3)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingClientError(
                f"Failed to build embedding client for role '{role}' "
                f"(model_name='{model_cfg.get('model_name')}'): {exc}. "
                f"Check llm.yaml and that the model is installed (`ollama list`)."
            ) from exc

    # -- HTTP with retries -----------------------------------------------------

    def _post_with_retry(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            raise EmbeddingRequestError(
                "HTTP client is not initialized or already closed."
            )

        last_exc: Optional[Exception] = None
        attempts = self.max_retries + 1

        for attempt in range(1, attempts + 1):
            try:
                logger.debug(
                    "Calling Ollama %s (attempt %d/%d) model=%s texts=%d",
                    path, attempt, attempts, payload.get("model"),
                    len(payload.get("input", [])),
                )
                response = self._client.post(path, json=payload)
                response.raise_for_status()
                return response.json()

            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                last_exc = exc
                logger.warning(
                    "Embedding call failed (attempt %d/%d): %s", attempt, attempts, exc
                )

        raise EmbeddingRequestError(
            f"No answer from Ollama after {attempts} attempts "
            f"({self.base_url}{path}, model={payload.get('model')}). "
            f"Check that Ollama is running (`ollama serve`) "
            f"and the embedding model is installed (`ollama list`)."
        ) from last_exc

    # -- Answer validation -------------------------------------------------------

    def _parse_answer(
        self, raw: dict[str, Any], expected: int
    ) -> list[list[float]]:
        """Validate Ollama's /api/embed answer shape and return the vectors.

        A malformed answer is a hard error, not a partial result: storing a
        wrong-sized or missing vector would corrupt the collection.
        """
        embeddings = raw.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingRequestError(
                f"Unexpected Ollama answer: 'embeddings' field has type "
                f"{type(embeddings).__name__}, expected list."
            )
        if len(embeddings) != expected:
            raise EmbeddingRequestError(
                f"Ollama returned {len(embeddings)} embedding(s) for "
                f"{expected} input text(s) — count mismatch."
            )
        vectors: list[list[float]] = []
        for index, vector in enumerate(embeddings):
            if not isinstance(vector, list) or not vector:
                raise EmbeddingRequestError(
                    f"Ollama embedding at index {index} is not a non-empty "
                    "vector."
                )
            if not all(isinstance(v, (int, float)) for v in vector):
                raise EmbeddingRequestError(
                    f"Ollama embedding at index {index} contains "
                    "non-numeric values."
                )
            vectors.append([float(v) for v in vector])
        return vectors


# ---------------------------------------------------------------------------
# Single entry point: one cached client per role (get_llm_client heritage).
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def get_embedding_client(role: str = "embedding") -> OllamaEmbeddingClient:
    """Return the shared embedding client for an llm.yaml role.

    Cached per role: the client (and its HTTP connection pool) is built
    once per run. Use it as a context manager if you need deterministic
    release.

    Raises:
        EmbeddingClientError: unknown role or invalid configuration.
    """
    return OllamaEmbeddingClient.for_role(role)


# ---------------------------------------------------------------------------
# Manual check (direct module execution): needs a live Ollama.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = get_embedding_client()
    assert isinstance(client, EmbeddingClientProtocol)  # runtime_checkable

    logger.info("Embedding client ready: model=%s base_url=%s", client.model, client.base_url)

    with client:
        vectors = client.embed(["Bonjour le monde", "Second texte de test"])
        logger.info("Embedded 2 texts, dimensions: %d", len(vectors[0]))
