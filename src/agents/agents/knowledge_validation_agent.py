"""KnowledgeValidationAgent — semantic gate on the discovered entities.

The ``knowledge_validation`` graph step (agent keys ``knowledge_validator``
/ ``place_validator``). Runs after knowledge extraction and BEFORE the
check-and-merge resolvers: the markdown knowledge base must only ever
receive well-formed entities, so this step validates every discovered
entry — deterministically, no LLM:

* **shape** — dict entry, non-empty string ``full_name``, optional string
  ``short_name``, list-of-strings ``aliases``, list-of-strings
  ``source_ids``;
* **provenance format** — every ``source_ids`` entry must be a full
  hierarchical unit id (``doc:<8hex>[::chp:1::pg:2::sec:1 ...]``,
  src/extraction/ids.py) — the resolvers write them into the knowledge
  base and the removal engine purges by the ``doc:<8hex>::`` prefix, so a
  malformed id here would poison both downstream behaviors;
* **model consistency** — the entry instantiates the type's knowledge
  model (``Character`` / ``Place``; id derivation from short/full name
  must succeed).

Architecture: the reusable check loop lives in
:class:`EntityValidatorAgent`; each entity type derives it with its
``INPUT_KEY`` / ``ENTRIES_KEY`` and model. Derivations:
:class:`CharacterValidatorAgent` (characters), :class:`PlaceValidatorAgent`
(places).

Failure mapping (the project's domain philosophy): a malformed entry is
an LLM misformed response that slipped past the extraction parse —
``FailureDomain.LLM_RESPONSE``, i.e. RETRYABLE: the graph restarts the
extraction step, which re-rolls the LLM passes. Nothing here is a data
failure: the input came from our own extraction, not from a human hand.

Warnings are non-fatal: duplicate full names in one payload (the
extraction dedup keys on full+short name, so same-full/different-short
variants can coexist — the resolver merges them anyway) and entries with
no source id at all (allowed: the resolver writes a ``(no source yet)``
bullet; flagged so the user knows the provenance is weak).

Traces (``task`` phase): ``knowledge_check`` start,
``knowledge_entry_warning`` per warning, ``knowledge_validated`` at the
end with the counts.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from src.agents.contexts import IngestionContext
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
from src.knowledge.models import Character, Place

logger = logging.getLogger(__name__)

#: Payload key this agent writes its report under.
OUTPUT_KEY = "knowledge_validation"

#: Full hierarchical unit id: ``doc:<8hex>`` followed by one or more
#: ``::<level>:<n>`` segments (chp/pg/sec/txt, 1-based counters). Every
#: extraction unit carries at least one level (chapter or page), so the
#: bare document id is NOT valid provenance here — the removal purge and
#: the vector-chunk join both rely on the unit-chain shape.
UNIT_ID_PATTERN = re.compile(
    r"^doc:[0-9a-fA-F]{8}(?:::(?:chp|pg|sec|txt):\d+)+$"
)


class EntityValidatorAgent:
    """Validates one entity type's discovered entries before check-and-merge.

    Subclasses declare the context payload key (``INPUT_KEY``), the entry
    list key (``ENTRIES_KEY``) and the pydantic model each entry must
    instantiate; the base class owns every check.
    """

    #: Context payload key this validator consumes (subclass MUST set it).
    INPUT_KEY = ""

    #: Key of the entry list inside the extraction payload ("characters").
    ENTRIES_KEY = "characters"

    #: Knowledge model every entry must satisfy (subclass MUST set it).
    MODEL: Type[BaseModel] = Character

    #: Human label in the traces ("character" / "place").
    ENTITY_LABEL = "character"

    def __init__(self) -> None:
        self.name = "knowledge_validator"

    # -- graph step -----------------------------------------------------------

    def run(self, context: IngestionContext) -> AgentResult:
        payload: Optional[Dict[str, Any]] = context.outputs.get(self.INPUT_KEY)

        if payload is None:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SKIPPED,
                detail="no knowledge extraction output to validate",
            )

        entries = payload.get(self.ENTRIES_KEY)
        if not isinstance(entries, list):
            return self._failed(
                context, f"extraction payload has no '{self.ENTRIES_KEY}' list"
            )

        context.emit(
            "task", "knowledge_check",
            f"validating {len(entries)} discovered {self.ENTITY_LABEL}(s)",
            entries=len(entries),
        )

        problems: List[str] = []
        warnings: List[str] = []
        seen_full_names: Dict[str, int] = {}

        for index, entry in enumerate(entries):
            where = f"{self.ENTRIES_KEY}[{index}]"
            if not isinstance(entry, dict):
                problems.append(f"{where}: entry must be an object, "
                                f"got {type(entry).__name__}")
                continue

            full_name = entry.get("full_name")
            if not isinstance(full_name, str) or not full_name.strip():
                problems.append(f"{where}: 'full_name' must be a non-empty string")
                continue

            short_name = entry.get("short_name")
            if short_name is not None and (
                not isinstance(short_name, str) or not short_name.strip()
            ):
                problems.append(f"{where} ({full_name!r}): 'short_name' must be "
                                "a non-empty string when present")

            aliases = entry.get("aliases", [])
            if aliases is None:
                aliases = []
            if not isinstance(aliases, list) or any(
                not isinstance(a, str) or not a.strip() for a in aliases
            ):
                problems.append(f"{where} ({full_name!r}): 'aliases' must be a "
                                "list of non-empty strings")

            source_ids = entry.get("source_ids", [])
            if source_ids is None:
                source_ids = []
            if not isinstance(source_ids, list) or any(
                not isinstance(s, str) or not UNIT_ID_PATTERN.match(s.strip())
                for s in source_ids
            ):
                problems.append(
                    f"{where} ({full_name!r}): 'source_ids' must be full unit ids "
                    "(doc:<8hex>[::chp:1::pg:2::sec:1 ...])"
                )
            elif not source_ids:
                warnings.append(
                    f"{where} ({full_name!r}): no source id — the knowledge base "
                    f"will record this {self.ENTITY_LABEL} without provenance"
                )

            # Model consistency: id derivation must succeed (short_name or
            # full_name slugified) and the prefix must agree with the type.
            try:
                self.MODEL(
                    full_name=full_name.strip(),
                    short_name=(short_name.strip() if isinstance(short_name, str)
                                and short_name.strip() else None),
                    aliases=[a.strip() for a in aliases if isinstance(a, str)],
                    source_ids=[s.strip() for s in source_ids if isinstance(s, str)],
                )
            except ValueError as exc:
                problems.append(f"{where} ({full_name!r}): invalid "
                                f"{self.ENTITY_LABEL} model: {exc}")
                continue

            key = full_name.strip().lower()
            seen_full_names[key] = seen_full_names.get(key, 0) + 1

        for name, count in seen_full_names.items():
            if count > 1:
                warnings.append(
                    f"{count} entries share the full name '{name}' — the resolver "
                    "will merge them (check the short names if they are distinct "
                    "entities)"
                )

        for warning in warnings:
            context.emit("task", "knowledge_entry_warning", warning)
            logger.info("Knowledge validation warning: %s", warning)

        if problems:
            # LLM misformed response that slipped past the extraction parse:
            # RETRYABLE — the graph restarts the extraction step.
            for problem in problems:
                logger.warning("Knowledge validation: %s", problem)
            detail = f"{len(problems)} invalid entr({'y' if len(problems) == 1 else 'ies'}): {problems[0]}"
            return self._failed(context, detail, problems)

        context.emit(
            "task", "knowledge_validated",
            f"all {len(entries)} discovered {self.ENTITY_LABEL}(s) valid "
            f"({len(warnings)} warning(s))",
            entries=len(entries), warnings=len(warnings),
        )
        report = {"entries": len(entries), "warnings": warnings,
                  "entity_type": self.ENTITY_LABEL}
        context.outputs[OUTPUT_KEY] = report
        return AgentResult(agent_name=self.name, status=AgentStatus.OK,
                           payload=report)

    def validate(self, context: IngestionContext) -> Optional[AgentResult]:
        """Nothing beyond the run's own checks (pure validation)."""
        return None

    # -- internals ------------------------------------------------------------

    def _failed(
        self, context: IngestionContext, detail: str,
        problems: Optional[List[str]] = None,
    ) -> AgentResult:
        """Step failure — LLM_RESPONSE (retryable) by design."""
        logger.warning("Knowledge validation failed: %s", detail)
        context.emit("task", "knowledge_validation_failed", detail)
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.FAILED,
            failure_domain=FailureDomain.LLM_RESPONSE,
            detail=f"knowledge validation failed: {detail}",
            payload={"problems": problems or [detail]},
        )


class CharacterValidatorAgent(EntityValidatorAgent):
    """Validates the discovered CHARACTERS (historical default)."""

    INPUT_KEY = "knowledge_characters"
    ENTRIES_KEY = "characters"
    MODEL = Character
    ENTITY_LABEL = "character"


class PlaceValidatorAgent(EntityValidatorAgent):
    """Validates the discovered PLACES."""

    INPUT_KEY = "knowledge_places"
    ENTRIES_KEY = "places"
    MODEL = Place
    ENTITY_LABEL = "place"

    def __init__(self) -> None:
        super().__init__()
        self.name = "place_validator"
