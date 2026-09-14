"""AnswerAgent — phrases the final user reply from the retrieved chunks.

The last step of the retrieval graph (step ``answer``, agent key
``answerer``): runs after ``semantic_search`` and turns the scored hits
into the user-facing reply. The retrieval pipeline stays deterministic —
this agent only FORMULATES what was already fetched; it never searches,
never re-ranks, never adds knowledge:

* the hits (``context.outputs["hits"]``) are rendered as numbered
  excerpts with their citation metadata (document, page, level, origin);
* the prompt (``prompts/answering/retrieval_answer.md``) enforces the
  one rule that matters: every factual statement must come from those
  excerpts, cited ``[n]``; no corpus, no answer — a deterministic
  "nothing found" is returned before any LLM call when there are no
  hits to ground on;
* the answer goes back as ``context.outputs["answer"]`` and in the
  result payload — the orchestrator carries it, the task agent and the
  app API surface it verbatim.

Failure mapping (agents never raise for expected failures):

* no hits to phrase from          → OK/SKIPPED-shaped ``no_answer``
  payload (a legitimate outcome, not a failure);
* LLM call failed / empty answer  → ``LLM_RESPONSE`` (transient-shaped:
  the graph may retry; the search results are not lost — the caller can
  degrade to the raw-hit status line);
* out-of-contract context         → ``INPUT_DATA``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from src.agents.contexts import RetrievalContext
from src.agents.llm_roles import LLMRoleAgent, MissingLLMRoleError
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
from src.routing.language import nothing_found_reply, normalize_language

logger = logging.getLogger(__name__)

AGENT_NAME = "answerer"

#: Where the grounded-answering prompt lives (loaded lazily, once).
ANSWER_PROMPT_PATH = Path("prompts/answering/retrieval_answer.md")

#: Cap on excerpt characters fed to the model: the answerer must read
#: the sources, not drown in them (top_k is already clamped upstream).
MAX_EXCERPT_CHARS = 1500


class AnswerAgent(LLMRoleAgent):
    """Phrases the retrieval hits into the final user-facing answer."""

    name = AGENT_NAME
    llm_role = "answerer"

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
        llm: pre-built LLM client override (tests); the shared role
            client is built lazily otherwise.
        allow_missing_role: build the agent even when the ``answerer``
            role is absent from llm.yaml — the run then fails cleanly
            with a CONFIG-domain result instead of the wiring raising.
            For tests and registries that must survive a partial
            configuration; production wiring stays fail-fast.
        """
        self._llm = llm
        try:
            super().__init__(*args, **kwargs)
        except MissingLLMRoleError:
            if not allow_missing_role:
                raise
            # Defer the failure to run(): a half-configured llm.yaml must
            # not prevent the retrieval graph's search step from working.
            self._llm_role = getattr(self, "_llm_role", "") or type(self).llm_role
            self.llm_config = {}
            self._role_missing = True
        else:
            self._role_missing = False
        self._prompt_path = Path(prompt_path) if prompt_path else ANSWER_PROMPT_PATH
        self._prompt_template: Optional[str] = None

    # -- RetrievalGraph step contract -----------------------------------------

    def run(self, context: RetrievalContext) -> AgentResult:
        """Phrase ``context.outputs["hits"]`` into ``context.outputs["answer"]``."""
        hits = context.outputs.get("hits") or []

        if not hits:
            # Deterministic path: nothing to ground on, no LLM call — the
            # prompt's rule "no corpus, no answer" enforced in Python.
            # The fallback reply honors the detected reply language like
            # every LLM-phrased answer would.
            reply = nothing_found_reply(context.metadata.get("language"))
            context.emit(
                "task", "answer_no_source",
                "no hit to phrase an answer from — the corpus holds nothing relevant",
            )
            context.outputs["answer"] = reply
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.OK,
                detail="no relevant chunk found in the corpus",
                payload={"answer": reply, "no_answer": True},
            )

        if self._role_missing:
            context.emit(
                "task", "answer_refused",
                "no 'answerer' role in llm.yaml: the answer agent cannot phrase",
                role=self._llm_role,
            )
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.CONFIG,
                detail=(f"LLM role '{self._llm_role}' is missing from "
                        "llm.yaml — the answer agent is not configured"),
            )

        question = (context.question or "").strip()
        # Reply language: detected by the routing analyzer, forwarded by
        # the retrieval orchestrator. An authoritative prompt KEY — not a
        # hint buried in prose: the reply agent must answer in it even
        # when the sources read in another language.
        language = normalize_language(context.metadata.get("language"))
        sources_block = self._render_sources(hits)

        # Debug trail (log file): what the agent RECEIVED, chunk by chunk,
        # full metadata and full text — the user audits the grounding by
        # comparing this block with the LLM answer logged after the call.
        logger.info(
            "[answer] question: %s — %d chunk(s) received as sources",
            question, len(hits),
        )
        for index, hit in enumerate(hits, start=1):
            logger.info(
                "[answer] source [%d] score=%s id=%s metadata=%s\n[answer]   text: %s",
                index,
                getattr(hit, "score", None),
                getattr(hit, "id", "?"),
                dict(getattr(hit, "metadata", None) or {}),
                str(getattr(hit, "text", "") or ""),
            )
        # Thinking panel: one compact line per source (full detail lives
        # in the log file, not in the panel).
        context.emit("task", "answer_sources", self._sources_summary(hits),
                     sources=len(hits))

        prompt = (
            f"{self._prompt_template_text().strip()}\n"
            "\n---\n\n"
            "Reply language:\n"
            f"{language}\n"
            "\n---\n\n"
            "Sources retrieved from the ingested corpus:\n\n"
            f"{sources_block}\n"
            "\n---\n\n"
            "User question:\n"
            "<<<<PROMPT>>>>\n"
            f"{question}\n"
            "<<<<PROMPT>>>>\n"
        )

        try:
            logger.info(f"[llm] {prompt}")
            answer = self.llm_client().complete(prompt=prompt)
        except Exception as exc:  # LLMClientError / transport
            logger.warning("Answer agent LLM call failed: %s", exc)
            context.emit("task", "answer_failed", f"LLM call failed: {exc}")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.LLM_RESPONSE,
                detail=f"the answer agent could not reach its model ({exc})",
            )

        answer = (answer or "").strip()
        if not answer:
            context.emit("task", "answer_failed", "LLM returned an empty answer")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.LLM_RESPONSE,
                detail="the answer agent received an empty answer",
            )

        # Debug trail (log file): what the model RETURNED, verbatim — the
        # other half of the grounding audit (sources above, translation here).
        logger.info("[answer] LLM answer (%d chars):\n%s", len(answer), answer)

        context.emit(
            "task", "answer_done",
            f"answer phrased from {len(hits)} chunk(s) ({len(answer)} chars)",
            sources=len(hits),
        )
        context.outputs["answer"] = answer
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.OK,
            detail="answer phrased",
            payload={"answer": answer},
        )

    def validate(self, context: RetrievalContext) -> Optional[AgentResult]:
        """Post-run check: an answer exists when hits existed."""
        hits = context.outputs.get("hits") or []
        if hits and not context.outputs.get("answer"):
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail="answer step produced no answer despite retrieved hits",
            )
        return None

    # -- internals ------------------------------------------------------------

    def llm_client(self):
        """The injected test client, or the shared role client."""
        if self._llm is not None:
            return self._llm
        return super().llm_client()

    def _prompt_template_text(self) -> str:
        """Load (and cache) the answering prompt."""
        if self._prompt_template is None:
            self._prompt_template = self._prompt_path.read_text(encoding="utf-8")
        return self._prompt_template

    @staticmethod
    def _sources_summary(hits: list) -> str:
        """One compact line per source, for the thinking panel only."""
        parts: List[str] = []
        for index, hit in enumerate(hits, start=1):
            meta = dict(getattr(hit, "metadata", None) or {})
            title = str(meta.get("doc_title") or "?")
            page = meta.get("page_number")
            score = getattr(hit, "score", None)
            parts.append(
                f"[{index}] {title} p.{page} ({float(score):.2f})"
                if score is not None else f"[{index}] {title} p.{page}"
            )
        return "sources: " + ", ".join(parts)

    @staticmethod
    def _render_sources(hits: list) -> str:
        """Numbered excerpts with citation metadata, for the prompt block.

        Each entry: ``[n] (score 0.82) Document: "X", page 4 — excerpt``.
        Metadata is optional per chunk: missing facets are simply omitted
        (the prompt forbids citing what is not there).
        """
        lines: List[str] = []
        for index, hit in enumerate(hits, start=1):
            meta = dict(getattr(hit, "metadata", None) or {})
            score = getattr(hit, "score", None)
            parts = [f"[{index}]"]
            if score is not None:
                parts.append(f"(score {float(score):.2f})")
            citation = []
            title = str(meta.get("doc_title") or "").strip()
            if title:
                citation.append(f'Document: "{title}"')
            page = meta.get("page_number")
            if page not in (None, "", 0):
                citation.append(f"page {page}")
            origin = str(meta.get("origin") or "").strip()
            if origin:
                citation.append(f"origin: {origin}")
            if citation:
                parts.append("(" + ", ".join(citation) + ")")
            text = str(getattr(hit, "text", "") or "").strip()
            if len(text) > MAX_EXCERPT_CHARS:
                text = text[: MAX_EXCERPT_CHARS].rstrip() + "…"
            lines.append(" ".join(parts) + "\n" + text)
        return "\n\n".join(lines)
