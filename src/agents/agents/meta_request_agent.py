"""MetaRequestAgent — fancy answers to the front-end's background chatter.

Fix C proved where Open WebUI's phantom POSTs came from: title generation,
follow-up suggestions and topic tagging are auxiliary LLM tasks the
front-end fires at the same ``/v1/chat/completions`` endpoint the user's
real messages use. The guard (src/llm/guard.py) recognizes them so they
never reach the analyzer or the routing graph; this agent is what answers
them when ``reply_to_meta_request: true`` in setup.yaml — the front-end
gets a real (in-character) title, tag list or follow-up suggestions
instead of the fixed refusal text, and the UI stays consistent.

Cost profile, by design:

* the ``__WINTERMUTE_PING__`` sentinel is ALWAYS answered by the fixed
  text — zero-cost probing is its purpose;
* with the switch off, everything falls back to the fixed text too (the
  low-powered-machine mode);
* with the switch on, one cheap LLM call per auxiliary prompt (the
  ``meta_request`` role is a small fast model; answers are bounded by the
  role's ``max_response_tokens``).

Everything is best-effort: any failure (missing role, LLM down, empty
answer) degrades to the fixed :func:`meta_answer` text — a background
task must never surface an error to the user's UI.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.agents.llm_roles import LLMRoleAgent, MissingLLMRoleError
from src.llm.guard import meta_answer, meta_kind, prompt_is_meta

logger = logging.getLogger(__name__)

AGENT_NAME = "meta_request"

#: Where the meta-answer prompt lives (loaded lazily, once).
META_PROMPT_PATH = "prompts/answering/meta_request.md"


class MetaRequestAgent(LLMRoleAgent):
    """Answers front-end auxiliary tasks (title / tags / follow-ups).

    Not a routing-graph task agent: it is called directly by the API layer
    (``_meta_answer_for``) on prompts the guard intercepted — auxiliary
    traffic must never pay for analysis or dispatch.
    """

    name = AGENT_NAME
    llm_role = "meta_request"

    def __init__(
        self,
        *,
        allow_missing_role: bool = False,
        llm=None,
        prompt_path: Optional[str] = None,
    ) -> None:
        """Args:
        llm: pre-built LLM client override (tests); the shared role client
            is used otherwise.
        allow_missing_role: build the agent even when the ``meta_request``
            role is absent from llm.yaml — ``run`` then returns the fixed
            fallback text instead of raising. Intended for tests and for
            registries that must survive a partial configuration;
            production wiring stays fail-fast.
        prompt_path: prompt file override (tests).
    """
        self._llm = llm
        self._prompt_path = prompt_path or META_PROMPT_PATH
        self._prompt_template: Optional[str] = None
        try:
            super().__init__()
        except MissingLLMRoleError:
            if not allow_missing_role:
                raise
            # Defer the failure to run(): a half-configured llm.yaml must
            # not break the API's meta handling.
            self.llm_config = {}
            self._role_missing = True
        else:
            self._role_missing = False

    def run(self, prompt: str) -> str:
        """Answer one auxiliary prompt; never raises, never returns ''.

        Returns the LLM's answer, or the fixed :func:`meta_answer` text on
        ANY failure (missing role, unreachable model, empty answer) — a
        background task must degrade silently, not surface an error.
        """
        if not prompt_is_meta(prompt):
            # Contract guard: this agent only ever sees guard-classified
            # prompts. The sentinel keeps its zero-cost answer.
            return meta_answer()

        if self._role_missing:
            logger.warning(
                "MetaRequestAgent has no '%s' role in llm.yaml; "
                "answering with the fixed fallback", type(self).llm_role,
            )
            return meta_answer()

        kind = meta_kind(prompt)
        try:
            answer = (self.llm_client().complete(
                prompt=self._build_prompt(prompt, kind)
            ) or "").strip()
        except Exception as exc:  # LLMClientError / transport / whatever
            logger.warning("MetaRequestAgent LLM call failed: %s", exc)
            return meta_answer()

        if not answer:
            logger.warning("MetaRequestAgent received an empty answer")
            return meta_answer()
        return answer

    # -- internals ------------------------------------------------------------

    def llm_client(self):
        """The injected test client, or the shared role client."""
        if self._llm is not None:
            return self._llm
        return super().llm_client()

    def _load_prompt(self) -> str:
        if self._prompt_template is None:
            from pathlib import Path

            self._prompt_template = Path(self._prompt_path).read_text(
                encoding="utf-8"
            )
        return self._prompt_template

    def _build_prompt(self, prompt: str, kind: str) -> str:
        """The meta prompt: instructions + the front-end's raw task."""
        kind_hint = {
            "title": "a short title (a few words, no quotes)",
            "tags": "a short list of tags",
            "followup": "a short list of follow-up suggestions",
            "unknown": "a short, direct answer",
        }.get(kind, "a short, direct answer")
        return (
            f"{self._load_prompt().strip()}\n\n"
            f"---\n"
            f"FRONT-END TASK ({kind}): answer with {kind_hint}.\n\n"
            f"{prompt.strip()}"
        )


def get_meta_request_agent() -> MetaRequestAgent:
    """The process-wide agent (built once, fail-fast on a missing role).

    Production wiring: llm.yaml MUST declare the ``meta_request`` role,
    else construction raises — same philosophy as the other agents. The
    cached module global keeps construction lazy (the API stays bootable
    on a partially configured machine) and single-instance.
    """
    global _META_AGENT
    if _META_AGENT is None:
        _META_AGENT = MetaRequestAgent()
    return _META_AGENT


def reset_meta_request_agent() -> None:
    """Drop the cached agent (tests, config reload)."""
    global _META_AGENT
    _META_AGENT = None


#: The cached process-wide agent (None until first use).
_META_AGENT: Optional[MetaRequestAgent] = None
