"""Strict LLM role resolution for agents.

Agents that call an LLM declare which functional role they need
(config/llm.yaml, ``models`` section). Resolving that role is STRICT: a
missing or malformed role raises — there is NEVER a silent fallback to the
``default`` role. If an agent wants the default model, it asks for the
``default`` role explicitly.

This module provides:

* ``MissingLLMRoleError`` — raised when the role cannot be resolved;
* ``require_llm_role(role)`` — strict lookup returning the role config dict;
* ``LLMRoleAgent`` — mixin for agents that use an LLM role: resolves the
  role eagerly at construction (fail fast at wiring time, before the graph
  burns any work) and exposes ``llm_config`` plus a cached ``llm_client()``.

Why strict, and why here rather than in a central list: role *existence* is
semantic knowledge that only the agent using the role possesses. A hardcoded
list of expected roles in the orchestrator would couple every new agent to
an orchestrator change; instead, each agent checks its own role at the
moment it resolves it (construction / first use) and reports the failure.
The orchestrator catches ``MissingLLMRoleError`` during agent wiring and
returns a graceful ``config_error`` status (see
``src/ingestion/ingestion_orchestrator``). Agents that resolve lazily must
catch it in ``run()`` and return a FAILED result with
``FailureDomain.CONFIG`` — the graph then rejects (CONFIG is not a
retryable domain).
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

from src.tools import config_loader
from src.tools.config_loader import ConfigError


class MissingLLMRoleError(ConfigError):
    """The LLM role an agent needs is not configured in llm.yaml."""


def require_llm_role(role: str) -> dict:
    """Strictly resolve an LLM role; raise MissingLLMRoleError when absent.

    There is NO fallback to the ``default`` role: a misspelled role name is
    a configuration error, not a silent downgrade. Ask for ``default``
    explicitly if that is the intended model.
    """
    try:
        return config_loader.get_model_config(role)
    except ConfigError as exc:
        raise MissingLLMRoleError(
            f"LLM role '{role}' is required by an agent but unusable: {exc}"
        ) from exc


class LLMRoleAgent:
    """Mixin for agents that use an LLM role (strict, fail-fast).

    Subclasses declare the role name as a class attribute::

        class MySummarizer(LLMRoleAgent, SummarizerAgent):
            llm_role = "summarizer"

    Construction resolves the role eagerly: ``self.llm_config`` holds the
    validated role config, and a missing role raises
    :class:`MissingLLMRoleError` at wiring time (when the agent registry is
    built), before the graph runs any step. ``llm_client()`` returns the
    shared, per-role cached client.
    """

    #: Role name under ``models`` in config/llm.yaml. Subclasses MUST set it
    #: (or pass ``llm_role=...`` to the constructor).
    llm_role: ClassVar[str] = ""

    def __init__(self, *args: Any, llm_role: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._llm_role = (llm_role or self.llm_role).strip()
        if not self._llm_role:
            raise MissingLLMRoleError(
                f"{type(self).__name__} uses an LLM but declares no 'llm_role'."
            )
        self.llm_config: dict = require_llm_role(self._llm_role)

    def llm_client(self):
        """Shared LLM client for this agent's role (cached per role)."""
        from src.llm.llm_client_ollama import get_llm_client

        return get_llm_client(self._llm_role)