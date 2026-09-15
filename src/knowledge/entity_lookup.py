"""Entity lookup tools — search the markdown knowledge base for an entity.

The retrieval side of the knowledge base (the read counterpart of the
EntityResolver's writes): given the entity name the user asked about
(``lookup`` request kind, spelled as the user wrote it), find the
knowledge file holding its identity block.

Three matching strategies, cheapest-first:

1. **Direct slug** — ``<slug-of-name>.md`` exists in ``characters/``
   (one ``stat`` on the expected path);
2. **Index scan** — the ``characters.md`` sidecar (ONE small file, the
   resolver's batch surface): exact match on a full name or an alias,
   case- and accent-insensitive (the user rarely spells the corpus's
   names exactly);
3. **Unique containment** — the user wrote a fragment or a superset of
   exactly ONE known full name ("Joe" → "Joe le Clodo", "épée de
   vif-argent" → "Épée de vif-argent"): that unique entity is the
   answer. Ambiguous (0 or several matches) → the caller answers
   deterministically "no such entity", listing
   :func:`close_candidates` (containment on names and aliases).

No LLM anywhere: identity resolution is a file-system question. The
index is read, never written here — files are the truth, the index
their projection (the resolver and the removal purge own the writes).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.knowledge.character_markdown_store import (
    CharacterMarkdownError,
    character_path_for,
    characters_dir,
    index_path_for,
    load_index,
    read_character,
    rebuild_index,
)

logger = logging.getLogger(__name__)


@dataclass
class EntityMatch:
    """One resolved entity: identity file + parsed names with sources."""

    #: Canonical full name (the file's title line — the identity).
    full_name: str
    #: The character markdown file (``<base>/characters/<slug>.md``).
    path: Path
    #: Parsed ``Known names:`` entries — ``{"alias", "source_ids": [str]}``
    #: (first entry is the full name itself), straight from
    #: :func:`read_character`.
    names: List[Dict[str, object]] = field(default_factory=list)

    @property
    def source_ids(self) -> List[str]:
        """Every content id the entity's names were found at (deduped)."""
        seen: List[str] = []
        for entry in self.names:
            for source_id in entry.get("source_ids") or []:
                if source_id not in seen:
                    seen.append(source_id)
        return seen

    def summary(self) -> Dict[str, object]:
        """Compact dict for payloads and traces."""
        return {
            "full_name": self.full_name,
            "path": str(self.path),
            "names": [str(e.get("alias", "")) for e in self.names],
            "source_ids": self.source_ids,
        }


def fold_name(name: str) -> str:
    """Case-, accent- and punctuation-insensitive form of a name.

    'Épée de Vif-Argent' and 'epée de vif argent' fold to the same key —
    user input must match corpus names that are rarely typed identically.
    """
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def resolve_entity(
    entity: str, base_dir: Optional[Path] = None
) -> Optional[EntityMatch]:
    """Resolve a user-named entity to its knowledge file.

    Strategies (in order): direct slug file → index scan (full names,
    then aliases — case/accent-insensitive) → unique containment (the
    user's words span exactly one known full name). ``None`` when
    nothing matches unambiguously; the caller then renders the
    deterministic "unknown entity" reply (with :func:`close_candidates`).

    A stale index entry (its file was hand-deleted) triggers one
    ``rebuild_index()`` — files are the truth — and the resolution is
    retried once against the repaired base.
    """
    query = (entity or "").strip()
    if not query:
        return None

    match = _resolve_once(query, base_dir)
    if match is not None:
        return match

    # One repair pass: the index may be stale relative to the files
    # (hand-edited base). Files are the truth; rebuild and retry once.
    try:
        rebuild_index(base_dir)
    except CharacterMarkdownError as exc:
        logger.warning("Knowledge-base index rebuild failed: %s", exc)
        return None
    match = _resolve_once(query, base_dir)
    if match is not None:
        return match
    return _resolve_by_unique_containment(query, base_dir)


def _resolve_by_unique_containment(
    query: str, base_dir: Optional[Path]
) -> Optional[EntityMatch]:
    """Strategy 3: a UNIQUE full-name containment match resolves.

    ``full_name_hits`` from :func:`close_candidates` — containment in
    either direction, FULL NAMES only (aliases are too fuzzy to
    auto-resolve). Exactly one hit → that entity; otherwise None (the
    caller lists the candidates)."""
    folded = fold_name(query)
    if not folded:
        return None
    try:
        entries = load_index(index_path_for(base_dir))
    except CharacterMarkdownError as exc:
        logger.warning("Knowledge-base index unreadable: %s", exc)
        return None
    hits = []
    for entry in entries:
        full_name = str(entry["full_name"])
        target = fold_name(full_name)
        if folded in target or target in folded:
            hits.append(full_name)
    if len(hits) != 1:
        return None
    return _load_match(character_path_for(hits[0], base_dir))


def _resolve_once(query: str, base_dir: Optional[Path]) -> Optional[EntityMatch]:
    folded = fold_name(query)

    # 1. Direct slug: the user's spelling may BE the file name.
    direct = character_path_for(query, base_dir)
    match = _load_match(direct)
    if match is not None:
        return match

    # 2. Index scan (one small file): full names first, then aliases.
    try:
        entries = load_index(index_path_for(base_dir))
    except CharacterMarkdownError as exc:
        logger.warning("Knowledge-base index unreadable: %s", exc)
        return None
    for entry in entries:
        if fold_name(str(entry["full_name"])) == folded:
            found = _load_match(character_path_for(str(entry["full_name"]), base_dir))
            if found is not None:
                return found
    for entry in entries:
        for alias in entry.get("aliases") or []:
            if fold_name(str(alias)) == folded:
                found = _load_match(
                    character_path_for(str(entry["full_name"]), base_dir)
                )
                if found is not None:
                    return found
    return None


def _load_match(path: Path) -> Optional[EntityMatch]:
    """Parse a character file into a match; None when absent/broken.

    A malformed file (hand-edit gone wrong) is logged and skipped here:
    the lookup must degrade to "unknown entity + candidates", not crash
    the retrieval graph — the resolver/validator surface the damage.
    """
    if not path.is_file():
        return None
    try:
        data = read_character(path)
    except CharacterMarkdownError as exc:
        logger.warning("Knowledge file unreadable (%s): %s", path, exc)
        return None
    return EntityMatch(
        full_name=str(data["full_name"]),
        path=path,
        names=list(data["names"]),  # type: ignore[arg-type]
    )


def close_candidates(
    entity: str,
    base_dir: Optional[Path] = None,
    *,
    limit: int = 5,
) -> List[str]:
    """Names of the knowledge base close to ``entity``, best first.

    Containment matching on folded full names and aliases (either
    direction: the query may be a fragment of a name or contain it —
    "vif-argent" finds "Épée de vif-argent"). Full-name hits rank above
    alias hits; equal rank keeps the index order (alphabetical files).
    """
    query = (entity or "").strip()
    if not query:
        return []
    folded = fold_name(query)
    if not folded:
        return []

    try:
        entries = load_index(index_path_for(base_dir))
    except CharacterMarkdownError as exc:
        logger.warning("Knowledge-base index unreadable: %s", exc)
        return []

    full_name_hits: List[str] = []
    alias_hits: List[str] = []
    for entry in entries:
        full_name = str(entry["full_name"])
        if folded and (folded in fold_name(full_name) or fold_name(full_name) in folded):
            if full_name not in full_name_hits:
                full_name_hits.append(full_name)
            continue
        for alias in entry.get("aliases") or []:
            alias_folded = fold_name(str(alias))
            if folded in alias_folded or alias_folded in folded:
                if full_name not in alias_hits:
                    alias_hits.append(full_name)
                break
    return (full_name_hits + alias_hits)[: max(1, limit)]


def known_entity_count(base_dir: Optional[Path] = None) -> int:
    """How many entities the base lists (the sidecar index's length)."""
    try:
        return len(load_index(index_path_for(base_dir)))
    except CharacterMarkdownError:
        return 0


# ---------------------------------------------------------------------------
# The vector companion — opaque source ids -> actual corpus content
# ---------------------------------------------------------------------------

#: Default cap on content chunks fetched per unit, when no store override
#: carries one (must stay in sync with config/retrieval.yaml's
#: ``lookup_max_content_hits``).
DEFAULT_MAX_CONTENT_HITS = 8


def fetch_unit_content(
    source_ids: List[str],
    store=None,
    *,
    max_hits: Optional[int] = None,
) -> List[object]:
    """Expand knowledge source ids into their actual stored content.

    The deterministic vector companion of the markdown lookup: an alias's
    ``source_ids`` (``doc:<hex>::chp:1::pg:1::sec:2``) are OPAQUE chains
    — exact, stable, unreadable for an LLM phrasing an answer. Each chain
    is the id-prefix of that unit's stored chunks (its text blocks and its
    ``::sum`` summary — the unified id scheme restarts every counter at
    its parent), so the vector store fetches the unit's whole projection
    by identity, no similarity involved.

    Deterministic and quiet by design:

    * ids are deduped (the same unit may back several aliases) and the
      first-seen order of ``source_ids`` is kept — that order is the
      knowledge base's own reading order;
    * a chain that stores nothing (unknown unit, never-indexed document,
      pruned corpus) is logged and skipped — content the corpus no longer
      holds must not fail a lookup, the identity block already grounds
      the answer;
    * a vector-store failure is logged and swallowed the same way: a
      lookup answers even with the store down (it reads the markdown
      base, the store is an enrichment);
    * the global cap keeps one broad character (hundreds of aliases,
      many units) from flooding the answerer's context — the first units
      of the knowledge base's order win, the identity card is never
      truncated.

    Args:
        source_ids: the unified id chains recorded on the entity (any
            order, duplicates allowed).
        store: the vector store client; the retrieval.yaml collection is
            built lazily when omitted (tests inject a stub).
        max_hits: cap on the content chunks returned; ``None`` reads
            retrieval.yaml's ``lookup_max_content_hits``.

    Returns:
        The unit chunks (reading order within a unit, ``::sum`` last),
        scores unset. Empty list when nothing is stored for any of the
        ids.
    """
    seen: List[str] = []
    for source_id in source_ids:
        chain = str(source_id or "").strip()
        if chain and chain not in seen:
            seen.append(chain)

    if not seen:
        return []

    if max_hits is None:
        max_hits = _config_max_content_hits()

    if store is None:
        from src.indexing.chroma_client import ChromaVectorClient
        from src.tools.config_loader import load_retrieval_config

        key = load_retrieval_config().get(
            "source_collection_key", "source_chunks"
        )
        store = ChromaVectorClient(str(key))

    chunks: List[object] = []
    for chain in seen:
        try:
            found = store.get_unit_chunks(chain, limit=max(1, int(max_hits)))
        except Exception as exc:  # noqa: BLE001 — enrichment, never a failure
            logger.warning(
                "Unit content fetch failed for '%s' (skipped): %s", chain, exc
            )
            continue
        if not found:
            logger.info(
                "No stored content for knowledge unit '%s' "
                "(never-indexed or pruned corpus).",
                chain,
            )
            continue
        chunks.extend(found)
        if len(chunks) >= max_hits:
            break

    return chunks[:max_hits]


def _config_max_content_hits() -> int:
    """``lookup_max_content_hits`` from retrieval.yaml, fail-open."""
    try:
        from src.tools.config_loader import load_retrieval_config

        value = int(load_retrieval_config().get(
            "lookup_max_content_hits", DEFAULT_MAX_CONTENT_HITS
        ))
    except Exception as exc:  # noqa: BLE001 — fail-open to the default
        logger.warning(
            "Could not read retrieval.yaml for lookup_max_content_hits; "
            "using default %d: %s", DEFAULT_MAX_CONTENT_HITS, exc,
        )
        return DEFAULT_MAX_CONTENT_HITS
    return value if value > 0 else DEFAULT_MAX_CONTENT_HITS


__all__ = [
    "DEFAULT_MAX_CONTENT_HITS",
    "EntityMatch",
    "characters_dir",
    "close_candidates",
    "fetch_unit_content",
    "fold_name",
    "known_entity_count",
    "resolve_entity",
]
