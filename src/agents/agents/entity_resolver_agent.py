"""EntityResolverAgent — the knowledge-base check-and-merge step.

The ``check_and_merge`` graph step (agent keys ``entity_resolver`` /
``place_resolver``). Runs after knowledge extraction: reads the discovered
entities (the extraction agent's stamped payload — ``full_name`` /
``short_name`` / ``aliases`` / ``source_ids``) from the context and
reconciles them into the markdown knowledge base
(``knowledge_base_dir``, config/ingestion.yaml, default
``data/knowledge``, one subfolder per entity type).

Matching — three steps, cheapest first:

1. **Direct file** — ``<slug-of-full-name>.md`` exists in the type's
   subfolder: match on the full name (the slug IS the identity convention).
2. **Index scan** — no direct file: the sidecar index (``characters.md`` /
   ``places.md``, one line per entity with all its aliases) is scanned for
   ANY name of the entry (full name included). A hit gives the owning
   file, which is read and merged. This is the "search by batch" of the
   design: the scan reads ONE small file instead of walking every entity
   file, so it stays O(1)-memory however big the corpus grows.
3. **Create** — no match anywhere: write the new entity file
   ``<slug>.md`` with the identity block — every known name (full name
   first) as a bullet, each followed by the content ids where that name
   was found, and register it in the index.

Merging appends to the matching name's bullet only the source ids not
already recorded (re-ingestion is idempotent) and adds brand-new aliases.

Identity convention (user-validated): the slugified full name IS the
identity — two discoveries with the same full name are the same entity
and merge into one file; finer identity arbitration (several holders of
a title, renamed entities...) belongs to a later pass, which will pin
the ambiguity explicitly.

Architecture: the reusable reconcile loop lives in
:class:`EntityResolverAgent`; each entity type derives it with its
``INPUT_KEY`` / ``ENTRIES_KEY`` / ``ENTITY_TYPE`` (prompt-free and
deterministic — no LLM here, the extraction was the only LLM step of the
knowledge layer). Derivations: :class:`CharacterResolver` (characters),
:class:`PlaceResolver` (places); organizations/objects/events follow the
same shape.

Failure handling: a malformed knowledge-base file (hand-edit gone wrong)
raises :class:`EntityMarkdownError` — surfaced as an INPUT_DATA step
failure, never silently skipped. There is no LLM-shaped failure here.

Traces (``task`` phase): ``resolver_start``, per entry
``resolver_created`` / ``resolver_merged`` / ``resolver_skipped``, and
``resolver_done`` with the counts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.agents.contexts import IngestionContext
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
from src.knowledge.character_markdown_store import (
    EntityMarkdownError,
    entities_dir,
    entity_path_for,
    index_path_for,
    load_index,
    read_entity,
    rebuild_index,
    write_entity,
)

logger = logging.getLogger(__name__)

#: Payload key this agent writes its report under.
OUTPUT_KEY = "knowledge_resolution"

AGENT_NAME = "entity_resolver"


class EntityResolverAgent:
    """Reconciles the discovered entities into the markdown knowledge base.

    Subclasses declare the payload/entry keys (``INPUT_KEY``,
    ``ENTRIES_KEY``), the knowledge-base subfolder (``ENTITY_TYPE``) and a
    human label for the traces; the base class owns the match (direct
    file -> index scan -> create), merge and index sync.
    """

    #: Context payload key this resolver consumes (subclass MUST set it).
    INPUT_KEY = ""

    #: Key of the entry list inside the extraction payload ("characters").
    ENTRIES_KEY = "characters"

    #: Knowledge-base subfolder of this type ("characters" / "places").
    ENTITY_TYPE = "characters"

    #: Human label in the traces ("character" / "place").
    ENTITY_LABEL = "entity"

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.name = AGENT_NAME
        self._base_dir_override = Path(base_dir) if base_dir else None

    # -- graph step -----------------------------------------------------------

    def run(self, context: IngestionContext) -> AgentResult:
        payload: Optional[Dict[str, Any]] = context.outputs.get(self.INPUT_KEY)

        if payload is None:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SKIPPED,
                detail="no knowledge extraction output to resolve",
            )

        entries = payload.get(self.ENTRIES_KEY)
        if not isinstance(entries, list):
            entries = []

        base_dir = self._base_dir(context)
        context.emit(
            "task", "resolver_start",
            f"resolving {self.ENTITY_LABEL} entries against the knowledge "
            f"base ({entities_dir(base_dir, self.ENTITY_TYPE)})",
            entries=len(entries),
        )

        try:
            index_snapshot = self._load_index_snapshot(base_dir)
        except EntityMarkdownError as exc:
            return self._failed(context, f"index unreadable: {exc}")

        created = 0
        merged = 0
        skipped = 0
        for entry in entries:
            try:
                outcome, detail = self._resolve_entry(
                    entry, base_dir, index_snapshot
                )
            except (EntityMarkdownError, OSError) as exc:
                return self._failed(context, str(exc))
            if outcome == "created":
                created += 1
                context.emit("task", "resolver_created", detail)
            elif outcome == "merged":
                merged += 1
                context.emit("task", "resolver_merged", detail)
            else:
                skipped += 1
                context.emit("task", "resolver_skipped", detail)

        try:
            index_path = self._sync_index(base_dir)
        except (OSError, EntityMarkdownError) as exc:
            return self._failed(context, f"index update failed: {exc}")

        summary = (
            f"{self.ENTITY_LABEL} resolution: {created} created, {merged} "
            f"merged, {skipped} skipped ({len(entries)} entr"
            f"{'y' if len(entries) == 1 else 'ies'})"
        )
        context.emit("task", "resolver_done", summary)
        result_payload = {
            "created": created,
            "merged": merged,
            "skipped": skipped,
            "entries": len(entries),
            "index": str(index_path),
            "entity_type": self.ENTITY_TYPE,
        }
        context.outputs[OUTPUT_KEY] = result_payload
        return AgentResult(agent_name=self.name, status=AgentStatus.OK,
                           payload=result_payload)

    def validate(self, context: IngestionContext) -> Optional[AgentResult]:
        """After a successful run the sidecar index must exist on disk."""
        if context.outputs.get(OUTPUT_KEY) is None:
            return None
        index_path = index_path_for(self._base_dir(context), self.ENTITY_TYPE)
        if not index_path.exists():
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail=f"knowledge-base index missing after the step: {index_path}",
            )
        return None

    # -- per-entity reconcile -------------------------------------------------

    def _resolve_entry(
        self,
        entry: Any,
        base_dir: Path,
        index_snapshot: List[Dict[str, object]],
    ) -> Tuple[str, str]:
        """Reconcile ONE entry against the base.

        Returns ``(outcome, trace detail)`` with outcome in
        ``'created'`` / ``'merged'`` / ``'skipped'``. Raises
        EntityMarkdownError / OSError for the step to map to a failure.
        """
        if not isinstance(entry, dict):
            return "skipped", f"non-object entry ignored: {entry!r}"

        full_name = str(entry.get("full_name") or "").strip()
        if not full_name:
            return "skipped", "entry without a full_name ignored"
        new_ids = [str(i).strip() for i in (entry.get("source_ids") or [])
                   if str(i).strip()]
        new_aliases = [
            str(a).strip() for a in (entry.get("aliases") or []) if str(a).strip()
        ]

        # 1. Direct file — full-name identity convention.
        entity_path = entity_path_for(full_name, base_dir, self.ENTITY_TYPE)
        if entity_path.exists():
            return self._merge_into(
                full_name, new_aliases, new_ids, entity_path
            )

        # 2. Index scan — any name of the entry may own a file.
        owner = self._find_owner(full_name, new_aliases, index_snapshot)
        if owner is not None:
            return self._merge_into(
                owner, new_aliases + [full_name], new_ids,
                entity_path_for(owner, base_dir, self.ENTITY_TYPE),
            )

        # 3. Create.
        names = [{"alias": full_name, "source_ids": list(new_ids)}]
        for alias in new_aliases:
            if alias != full_name:
                names.append({"alias": alias, "source_ids": list(new_ids)})
        write_entity(full_name, names, entity_path, self.ENTITY_TYPE)
        index_snapshot.append({"full_name": full_name, "aliases": list(new_aliases)})
        return "created", f"{self.ENTITY_LABEL} created: {entity_path.name}"

    def _find_owner(
        self,
        full_name: str,
        aliases: List[str],
        index_snapshot: List[Dict[str, object]],
    ) -> Optional[str]:
        """The full name of the entity owning ANY of the entry's names,
        via the index snapshot (None when nobody owns it)."""
        wanted = {full_name.strip().lower()} | {a.strip().lower() for a in aliases}
        for entity in index_snapshot:
            names = {str(entity["full_name"]).strip().lower()}
            names |= {str(a).strip().lower() for a in entity["aliases"]}  # type: ignore[union-attr]
            if wanted & names:
                return str(entity["full_name"])
        return None

    def _merge_into(
        self,
        owner_full_name: str,
        aliases: List[str],
        source_ids: List[str],
        entity_path: Path,
    ) -> Tuple[str, str]:
        """Merge the discovery into the owner's file (union of names and
        source ids; re-ingestion is idempotent)."""
        existing = read_entity(entity_path)
        names: List[Dict[str, object]] = existing["names"]  # type: ignore[assignment]
        by_alias = {str(n.get("alias", "")).strip().lower(): n for n in names}
        changed = False
        for alias in aliases + [owner_full_name]:
            key = alias.strip().lower()
            target = by_alias.get(key)
            if target is None:
                target = {"alias": alias, "source_ids": []}
                names.append(target)
                by_alias[key] = target
                changed = True
            for source_id in source_ids:
                if source_id not in target["source_ids"]:  # type: ignore[operator]
                    target["source_ids"].append(source_id)  # type: ignore[union-attr]
                    changed = True
        if changed:
            write_entity(owner_full_name, names, entity_path, self.ENTITY_TYPE)
        return "merged", f"{self.ENTITY_LABEL} merged: {entity_path.name}"

    # -- index ----------------------------------------------------------------

    def _load_index_snapshot(self, base_dir: Path) -> List[Dict[str, object]]:
        """The type's index at run start (empty when the base is new)."""
        return load_index(index_path_for(base_dir, self.ENTITY_TYPE))

    def _sync_index(self, base_dir: Path) -> Path:
        """End-of-pass index rebuild — delegated to the store's single-writer
        primitive: the base's entity FILES are the truth, the index their
        projection (stale lines for deleted files disappear; hand-created
        files are picked up; the in-run snapshot's create-merges are already
        on disk when this runs)."""
        return rebuild_index(base_dir, self.ENTITY_TYPE)

    def _base_dir(self, context: IngestionContext) -> Path:
        """Knowledge-base folder (constructor override > config)."""
        if self._base_dir_override is not None:
            return self._base_dir_override
        from src.knowledge.character_markdown_store import knowledge_base_dir

        return knowledge_base_dir()

    # -- internals ------------------------------------------------------------

    def _failed(self, context: IngestionContext, detail: str) -> AgentResult:
        """Step failure (INPUT_DATA — a base problem needs human eyes)."""
        logger.warning("Entity resolution failed: %s", detail)
        context.emit("task", "resolver_failed", detail)
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.FAILED,
            failure_domain=FailureDomain.INPUT_DATA,
            detail=f"entity resolution failed: {detail}",
        )


class CharacterResolver(EntityResolverAgent):
    """Resolves the discovered CHARACTERS into ``<base>/characters``."""

    INPUT_KEY = "knowledge_characters"
    ENTRIES_KEY = "characters"
    ENTITY_TYPE = "characters"
    ENTITY_LABEL = "character"


class PlaceResolver(EntityResolverAgent):
    """Resolves the discovered PLACES into ``<base>/places``."""

    INPUT_KEY = "knowledge_places"
    ENTRIES_KEY = "places"
    ENTITY_TYPE = "places"
    ENTITY_LABEL = "place"
