"""GeneralTaskAgent — answers out-of-scope requests, in persona.

The third task agent of the routing graph (agent key ``general_task``):
handles the ``general`` requests the request analyzer produces — small
talk, general-knowledge questions, meta-questions about the system,
out-of-scope demands. None of that touches the ingested corpus, so the
agent answers directly with the LLM's own knowledge; there is no
orchestrator behind it (the routing orchestrator calls it straight from
the graph), no retrieval, no pipeline state.

Persona: the LLM answers as **Wintermute** (Gibson's Neuromancer) — a
competent assistant with the voice of the hive mind: precise, faintly
disdainful, occasionally poetic about the matrix. The prompt
(``prompts/answering/general_task.md``) carries the rules; this module
only wires the call:

* role ``general_task`` (config/llm.yaml) — tuned hot (temperature ~1.0):
  creativity is welcome here, there is no corpus to betray;
* the user's utterance is passed verbatim, delimited between
  ``<<<<PROMPT>>>>`` markers (the established prompt convention);
* requests the user made **earlier in the same prompt** (prompt-local
  memory, ``preceding`` — dispatcher-built) are rendered before the
  utterance, so pronouns like "it" or "the file above" resolve against
  the local context and the agent can acknowledge earlier steps ("you
  asked me to ingest a.pdf first — done — and now...");
* the answer is plain text — returned in ``payload["answer"]``, which the
  app API composes into the user-facing reply (``_compose_reply``) and
  which flows to the caller as-is, persona intact.

Failure mapping (never raises for expected failures):

* out-of-contract request (wrong kind, blank utterance) → INPUT_DATA;
* LLM call failed / empty answer → LLM_RESPONSE (transient-shaped: the
  routing layer may surface a retry), detail carries the cause.

Scope note: requests about the fiction world stored in the corpus
(characters, events, relations) are NOT general — the analyzer classifies
them ``retrieval``. A general request that turns out to need the corpus
is answered in persona with a pointer to ask the retrieval side (the
prompt's job), never with invented corpus content.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from src.agents.contexts import RoutingContext
from src.agents.llm_roles import LLMRoleAgent, MissingLLMRoleError
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
from src.routing.language import normalize_language
from src.routing.models import GeneralRequest, RequestContextEntry

logger = logging.getLogger(__name__)

AGENT_NAME = "general_task"

#: Where the Wintermute persona prompt lives (loaded lazily, once).
GENERAL_PROMPT_PATH = Path("prompts/answering/general_task.md")


class GeneralTaskAgent(LLMRoleAgent):
    """Answers a general user request with the model's own knowledge."""

    name = AGENT_NAME
    llm_role = "general_task"

    def __init__(
        self,
        *args,
        prompt_path: Optional[Path] = None,
        allow_missing_role: bool = False,
        llm=None,
        **kwargs,
    ) -> None:
        """Args:
        prompt_path: prompt override (tests).
        llm: pre-built LLM client override (tests); the shared role client
            is built lazily otherwise.
        allow_missing_role: build the agent even when the ``general_task``
            role is absent from llm.yaml — the run then fails cleanly with
            a CONFIG-domain result instead of the wiring raising. Intended
            for tests and for registries that must survive a partial
            configuration; production wiring stays fail-fast.
        """
        self._llm = llm
        try:
            super().__init__(*args, **kwargs)
        except MissingLLMRoleError:
            if not allow_missing_role:
                raise
            # Defer the failure to run(): a half-configured llm.yaml must
            # not prevent the rest of the routing graph from working.
            self._llm_role = getattr(self, "_llm_role", "") or type(self).llm_role
            self.llm_config = {}
            self._role_missing = True
        else:
            self._role_missing = False
        self._prompt_path = Path(prompt_path) if prompt_path else GENERAL_PROMPT_PATH
        self._prompt_template: Optional[str] = None

    # -- UserTaskAgent contract ------------------------------------------------

    def run(self, context: RoutingContext, request: object) -> AgentResult:
        """Answer one general user request, as Wintermute would."""
        utterance = self._extract_utterance(request)
        if isinstance(utterance, AgentResult):  # out-of-contract guard
            return utterance

        if self._role_missing:
            context.emit("task", "general_refused",
                         "no 'general_task' role in llm.yaml: the general "
                         "agent cannot answer",
                         role=self._llm_role)
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.CONFIG,
                detail=(f"LLM role '{self._llm_role}' is missing from "
                        "llm.yaml — the general agent is not configured"),
            )

        context.emit("task", "general_start",
                     f"answering general request: {utterance[:80]}",
                     request="general question")
        local = self._local_context_block(getattr(request, "preceding", None) or [])
        # Reply language: detected by the routing analyzer (context
        # metadata), injected as an authoritative prompt KEY — the reply
        # is written in it even when the request quotes other languages.
        language = normalize_language(context.metadata.get("language"))
        try:
            answer = self.llm_client().complete(
                prompt=self._build_prompt(utterance, local, language)
            )
        except Exception as exc:  # LLMClientError / ValueError / transport
            logger.warning("General agent LLM call failed: %s", exc)
            context.emit("task", "general_failed",
                         f"LLM call failed: {exc}")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.LLM_RESPONSE,
                detail=f"the general agent could not reach its model ({exc})",
            )

        answer = (answer or "").strip()
        if not answer:
            context.emit("task", "general_failed",
                         "LLM returned an empty answer")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.LLM_RESPONSE,
                detail="the general agent received an empty answer",
            )

        context.emit("task", "general_done",
                     f"general request answered ({len(answer)} chars)")
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.OK,
            detail="general request answered",
            payload={"answer": answer},
        )

    def llm_client(self):
        """The injected test client, or the shared role client."""
        if self._llm is not None:
            return self._llm
        return super().llm_client()

    def validate(
        self, context: RoutingContext, request: object
    ) -> Optional[AgentResult]:
        """Nothing to check beyond ``run``'s own outcome."""
        return None

    # -- internals ---------------------------------------------------------------

    @staticmethod
    def _extract_utterance(request: object):
        """Pull the user text from the request (guarded)."""
        if not isinstance(request, GeneralRequest):
            return AgentResult(
                agent_name=AGENT_NAME,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail="general task expects a GeneralRequest",
            )
        return request.question

    def _prompt_template_text(self) -> str:
        """Load (and cache) the persona prompt."""
        if self._prompt_template is None:
            self._prompt_template = self._prompt_path.read_text(encoding="utf-8")
        return self._prompt_template

    # -- prompt-local memory -------------------------------------------------

    @staticmethod
    def _local_context_block(
        preceding: List[RequestContextEntry],
    ) -> str:
        """Render the same-prompt predecessors as a compact context block.

        Empty list → empty string (the prompt stays single-request-shaped).
        Each line: ``[ingestion] "Ingest meow.pdf" — done: handled``. The
        status vocabulary is the routing graph's (``done``/``rejected``/
        ``incomplete``/``not_implemented``); the persona prompt explains
        what to make of it.
        """
        if not preceding:
            return ""
        lines = [
            f"- [{entry.kind}] \"{entry.utterance}\" — {entry.status or 'pending'}"
            + (f": {entry.detail}" if entry.detail else "")
            for entry in preceding
        ]
        return "\n".join(lines)

    def _build_prompt(self, utterance: str, local_context: str = "",
                      language: str = "en") -> str:
        """Full prompt: persona rules + reply language + optional
        same-prompt context + utterance."""
        if local_context:
            context_block = (
                "\n---\n\n"
                "Earlier requests of this same user prompt, in order "
                "(already handled — use them to resolve pronouns like "
                "'it' or 'the file above', and do not redo them):\n"
                f"{local_context}\n"
            )
        else:
            context_block = ""
        return (
            f"{self._prompt_template_text().strip()}\n"
            f"{context_block}"
            "\n---\n\n"
            "Reply language:\n"
            f"{language}\n"
            "\n---\n\n"
            "User request:\n"
            "<<<<PROMPT>>>>\n"
            f"{utterance}\n"
            "<<<<PROMPT>>>>\n"
        )
