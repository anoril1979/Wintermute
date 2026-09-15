"""KnowledgeLookupAgent — the entity-lookup worker of the retrieval graph.

Serves the ``lookup`` request kind: the analyzer identified that the user
asks about a *specific, identifiable entity* ("who is Marcus?", "tell me
about the sword Blackfang") — the answer lives in that entity's knowledge
file, not in a passage search. This agent resolves the entity against the
markdown knowledge base (src/knowledge/entity_lookup.py) and, when found,
emits its identity block as the grounded source the AnswerAgent phrases
from. Fully deterministic — no LLM anywhere in this step:

* resolution: direct slug → index scan (full names, then aliases,
  case/accent-insensitive) — the resolver's write-side rules mirrored;
* found      → ONE synthesized VectorChunk (the identity block: every
  known name with the content ids where it appears) placed in
  ``context.outputs["hits"]``, score 1.0 (an exact identity match —
  the grounding is total);
* not found  → a deterministic, localized "no entity named X" reply
  (with close candidates) placed directly in
  ``context.outputs["answer"]`` — the answer step then skips (the
  reply already exists) and the request is still ``served``: "this
  entity is unknown to me" is a legitimate answer, not a failure.

Failure mapping (agents never raise for expected failures):

* no entity to look for (analyzer bug: a lookup without a name — the
  question text is used as a last-resort fallback first) → ``INPUT_DATA``;
* a malformed knowledge file is logged and skipped by the resolver —
  the lookup degrades to "unknown entity + candidates", never crashes.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.agents.contexts import RetrievalContext
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
from src.indexing.chunks import VectorChunk
from src.knowledge.entity_lookup import close_candidates, resolve_entity
from src.routing.language import entity_unknown_reply, normalize_language

logger = logging.getLogger(__name__)

AGENT_NAME = "knowledge_lookup"

#: Metadata keys (what the orchestrator puts in the context).
ENTITY_KEY = "entity"

#: The synthesized chunk's score: an identity match is exact.
IDENTITY_SCORE = 1.0


def _identity_chunk_text(match) -> str:
    """The identity block the answer agent grounds on: every known name
    with the content ids where the knowledge base saw it."""
    lines = [f"{match.full_name} — known names and where they appear:"]
    for entry in match.names:
        alias = str(entry.get("alias", "")).strip()
        if not alias:
            continue
        ids = [str(i) for i in (entry.get("source_ids") or [])]
        if ids:
            lines.append(f"- {alias} [{'], ['.join(ids)}]")
        else:
            lines.append(f"- {alias} (no source location recorded)")
    return "\n".join(lines)


class KnowledgeLookupAgent:
    """Resolves the requested entity in the knowledge base, deterministically."""

    name = AGENT_NAME

    def __init__(
        self,
        *,
        base_dir: Optional[object] = None,
        entity_key: str = ENTITY_KEY,
    ) -> None:
        """Args:
        base_dir: knowledge-base override (tests); the config-driven
            folder is used otherwise.
        entity_key: metadata key holding the entity name (wiring seam).
        """
        self._base_dir = base_dir
        self._entity_key = entity_key

    # -- RetrievalGraph step contract -----------------------------------------

    def run(self, context: RetrievalContext) -> AgentResult:
        """Resolve the entity; emit the identity source or the unknown reply."""
        entity = str(context.metadata.get(self._entity_key) or "").strip()
        if not entity:
            # Degraded analyzer output (a lookup without a name): the
            # self-contained question is the best deterministic fallback.
            entity = (context.question or "").strip()
        if not entity:
            detail = "no entity to look up (metadata and question empty)"
            context.emit("task", "knowledge_lookup_failed", detail)
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail=detail,
            )

        context.metadata["entity"] = entity
        context.emit(
            "task", "knowledge_lookup_start",
            f"searching the knowledge base for {entity!r}",
            entity=entity,
        )

        match = resolve_entity(entity, base_dir=self._base_dir)

        if match is None:
            candidates = close_candidates(entity, base_dir=self._base_dir)
            language = normalize_language(context.metadata.get("language"))
            reply = entity_unknown_reply(entity, language, candidates)
            context.metadata["entity_candidates"] = candidates
            context.metadata["entity_unknown"] = True
            # Uniform output contract: every run defines "hits" (empty
            # here) and the reply is already written — the answer step
            # then keeps it verbatim and the request counts as served
            # (an honest "unknown" IS the answer).
            context.outputs["hits"] = []
            context.outputs["answer"] = reply
            context.emit(
                "task", "knowledge_lookup_miss",
                f"no entity {entity!r} in the knowledge base"
                + (f" — {len(candidates)} candidate(s) proposed" if candidates else ""),
                entity=entity,
                candidates=candidates,
            )
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.OK,
                detail=f"entity {entity!r} unknown to the knowledge base",
                payload={"entity": entity, "unknown": True,
                         "candidates": candidates},
            )

        context.metadata["resolved_entity"] = match.summary()
        chunk = VectorChunk(
            id=f"knowledge::{match.path.stem}",
            text=_identity_chunk_text(match),
            metadata={
                "doc_title": match.full_name,
                "kind": "entity",
                "level": "entity",
                "entity": match.full_name,
            },
            score=IDENTITY_SCORE,
        )
        context.outputs["hits"] = [chunk]
        context.emit(
            "task", "knowledge_lookup_hit",
            f"entity {match.full_name!r} found ({len(match.names)} known name(s), "
            f"{len(match.source_ids)} source location(s))",
            entity=match.full_name,
            names=[str(e.get("alias", "")) for e in match.names],
        )
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.OK,
            detail=f"entity {match.full_name!r} resolved",
            payload={"entity": match.full_name, "match": match.summary()},
        )

    def validate(self, context: RetrievalContext) -> Optional[AgentResult]:
        """Post-run check: a resolved entity must have produced its source."""
        if context.metadata.get("resolved_entity") and not context.outputs.get("hits"):
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail="entity resolved but no identity source was emitted",
            )
        return None
