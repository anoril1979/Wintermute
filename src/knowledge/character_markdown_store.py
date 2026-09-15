"""Markdown knowledge base — the resolvers' per-entity store.

Second persistence layer of the knowledge ingestion (after the per-document
extraction cache): the EntityResolvers project the discovered entities into
a human-readable, human-editable markdown base under ``knowledge_base_dir``
(config/ingestion.yaml, default ``data/knowledge``), ONE SUBFOLDER PER
ENTITY TYPE:

    data/knowledge/
      characters/          one <slug>.md per character + characters.md
      places/              one <slug>.md per place + places.md
      organizations/       (future)
      sources/             per-document registrations (own store module)

Every entity file holds the identity block: every known name (full name
first) as a bullet, each followed by the content ids where that name was
found::

    # Character : Joe le Clodo

    Known names:

    - Joe le Clodo
      - [doc:36a911e2::chp:1::pg:1::sec:2]
    - Bobby
      - [doc:36a911e2::chp:1::pg:1::sec:4], [doc:36a911e2::chp:1::pg:2::sec:1]

The ids are the extraction layer's full hierarchical unit ids
(src/extraction/ids.py) — they link back into the canonical JSON store
and, through it, into the vector chunks.

Each type's sidecar index (``characters.md``, ``places.md``) — one line
per entity with its aliases (first name = full name = file name)::

    - Joe le Clodo (aka: Bobby, le Clodo)

The index is the resolver's batch-scan surface: identity lookups read
ONE small file instead of walking every entity file, and it doubles
as the user-facing listing of the known entities.

Design mirrors the other knowledge stores: config-driven folder (relative
paths anchored to the project root), atomic writes (tempfile + os.replace),
and a defensive loader — a malformed entity file or index (hand-edit gone
wrong) raises :class:`EntityMarkdownError` with the path, reported instead
of silently ignored.

All primitives are TYPE-PARAMETERIZED (``entity_type`` first argument,
defaulting to ``"characters"``): adding an entity kind needs no new store
code — just a new subfolder convention and callers passing the type.
Backwards-compatibility aliases keep the historical character-only names
(``characters_dir``, ``character_path_for``, ``write_character``...)
working for existing callers.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from src.helpers.strings import slugify
from src.tools.config_loader import PROJECT_ROOT

logger = logging.getLogger(__name__)

#: Fallback when ingestion.yaml does not set ``knowledge_base_dir``.
DEFAULT_KNOWLEDGE_BASE_DIR = "data/knowledge"

#: Per-entity-type subfolders of the knowledge base. The entity kinds the
#: system extracts today and tomorrow: one subfolder + one sidecar index
#: each, identical structure.
ENTITY_TYPES = ("characters", "places", "organizations", "objects", "events")

#: Default entity type for every primitive (historical callers).
DEFAULT_ENTITY_TYPE = "characters"

#: Name of the sidecar index inside an entity-type folder.
INDEX_FILENAME = "{type}.md"

#: Fixed header of the sidecar index.
_INDEX_HEADER_TEMPLATE = [
    "# {Title}",
    "",
    "One entry per {singular}; the first name is",
    "the full name used as the file name.",
    "",
]

#: Markdown structure keys (fixed English — the files are consumed by the
#: system first, read by humans second; content stays in its own language).
TITLE_PREFIX_TEMPLATE = "# {Title} : "
NAMES_HEADER = "Known names:"


class EntityMarkdownError(ValueError):
    """A knowledge-base markdown file is malformed (hand-edit gone wrong)."""


#: Historical name of the error (character-only era).
CharacterMarkdownError = EntityMarkdownError


def _type_plural(entity_type: str) -> str:
    """Normalized plural folder name (``"place"`` -> ``"places"``)."""
    plural = (entity_type or "").strip().lower()
    if not plural:
        return DEFAULT_ENTITY_TYPE
    if not plural.endswith("s"):
        plural += "s"
    return plural


def _title_case(plural: str) -> str:
    """"characters" -> "Characters" (index header / file title)."""
    return plural[:-1].capitalize() if plural.endswith("s") else plural.capitalize()


# ---------------------------------------------------------------------------
# Path resolution (config-driven, fail-open like every other store)
# ---------------------------------------------------------------------------

def knowledge_base_dir() -> Path:
    """Knowledge BASE folder from ingestion.yaml (``knowledge_base_dir``;
    default ``data/knowledge``).

    The base hosts one subfolder per entity type (``characters/``,
    ``places/``, future siblings). Relative values are resolved against
    the project root; absolute values are used as-is. Fails open to the
    default on a broken config.
    """
    raw = DEFAULT_KNOWLEDGE_BASE_DIR
    try:
        from src.tools.config_loader import load_ingestion_config

        value = load_ingestion_config().get("knowledge_base_dir")
        if isinstance(value, str) and value.strip():
            raw = value
    except Exception as exc:  # noqa: BLE001 — fail-open to the default folder
        logger.warning(
            "Could not read ingestion.yaml for knowledge_base_dir; using "
            "default %s (%s)", DEFAULT_KNOWLEDGE_BASE_DIR, exc,
        )
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def entities_dir(base_dir: Optional[Path] = None,
                 entity_type: str = DEFAULT_ENTITY_TYPE) -> Path:
    """Per-type subfolder of the knowledge base: ``<base>/<type>``."""
    base = Path(base_dir) if base_dir else knowledge_base_dir()
    return base / _type_plural(entity_type)


def slugify_filename(name: str) -> str:
    """File-system slug of an entity full name, e.g.
    'Édmond Dantès' -> 'edmond-dantes' (ascii, lowercase, hyphenated).

    Thin wrapper over the canonical helper (src/helpers/strings.py);
    the '' fallback becomes 'unnamed' so a file name always exists.
    """
    return slugify(name, joiner="-") or "unnamed"


def entity_path_for(full_name: str, base_dir: Optional[Path] = None,
                    entity_type: str = DEFAULT_ENTITY_TYPE) -> Path:
    """Entity file path for a full name: ``<base>/<type>/<slug>.md``."""
    return entities_dir(base_dir, entity_type) / f"{slugify_filename(full_name)}.md"


def index_path_for(base_dir: Optional[Path] = None,
                   entity_type: str = DEFAULT_ENTITY_TYPE) -> Path:
    """Sidecar index path: ``<base>/<type>/<type>.md``."""
    folder = entities_dir(base_dir, entity_type)
    plural = folder.name
    return folder / INDEX_FILENAME.format(type=plural)


# ---------------------------------------------------------------------------
# Entity files
# ---------------------------------------------------------------------------

def write_entity(
    full_name: str,
    names: List[Dict[str, object]],
    path: Path,
    entity_type: str = DEFAULT_ENTITY_TYPE,
) -> Path:
    """(Re)write one entity file (pretty UTF-8, atomic).

    Args:
        full_name: the entity's full name (title of the file).
        names: ordered name entries — ``{"alias": str, "source_ids": [str]}``;
            the FIRST entry is conventionally the full name itself.
        path: target ``<slug>.md`` path.
        entity_type: title word of the file (``"# Character : ..."`` /
            ``"# Place : ..."``).

    Returns the written path.
    """
    title = _title_case(_type_plural(entity_type))
    lines = [f"{TITLE_PREFIX_TEMPLATE.format(Title=title)}{full_name.strip()}",
             "", NAMES_HEADER, ""]
    for entry in names:
        alias = str(entry.get("alias", "")).strip()
        ids = [str(i) for i in (entry.get("source_ids") or [])]
        lines.append(f"- {alias}")
        if ids:
            lines.append(f"  - [{'], ['.join(ids)}]")
        else:
            lines.append("  - (no source yet)")
        lines.append("")
    _atomic_write(Path(path), "\n".join(lines).rstrip() + "\n")
    return Path(path)


def read_entity(path: Path) -> Dict[str, object]:
    """Parse one entity file back into
    ``{"full_name": str, "names": [{"alias", "source_ids"}]}``.

    Raises:
        FileNotFoundError: the file does not exist.
        EntityMarkdownError: the structure is not an entity file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Entity file not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()

    full_name = ""
    for line in lines:
        if line.startswith("# ") and " : " in line:
            full_name = line.split(" : ", 1)[1].strip()
            break
    if not full_name:
        raise EntityMarkdownError(
            f"{path}: missing '# <Type> : <name>' title line"
        )

    names: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(NAMES_HEADER) or stripped.startswith("#"):
            continue
        if line.startswith("- ") and not stripped.startswith("- ["):
            current = {"alias": stripped[2:].strip(), "source_ids": []}
            names.append(current)
        elif stripped.startswith("-") and "[" in stripped:
            # Ids line:  "- [id], [id]"  or  "(no source yet)" placeholder
            # (indented sub-bullet of the name above it).
            if current is None:
                raise EntityMarkdownError(
                    f"{path}: source-id line without a name bullet above it: {line!r}"
                )
            for source_id in re.findall(r"\[([^\[\]]+)\]", stripped):
                if ":" in source_id:  # "(no source yet)" has no id colon
                    current["source_ids"].append(source_id)  # type: ignore[union-attr]
        # Any other line: prose/hand-notes — ignored.
    return {"full_name": full_name, "names": names}


# ---------------------------------------------------------------------------
# Sidecar index (one line per entity)
# ---------------------------------------------------------------------------

def format_index_line(full_name: str, aliases: List[str]) -> str:
    """``- Full Name (aka: a1, a2)`` — aliases beyond the full name."""
    extras = [a for a in aliases if a and a != full_name]
    if extras:
        return f"- {full_name} (aka: {', '.join(extras)})"
    return f"- {full_name}"


def load_index(path: Path) -> List[Dict[str, object]]:
    """Parse the sidecar index into
    ``[{"full_name": str, "aliases": [str]}]`` (reading order).

    A missing index is an empty knowledge base, not an error.
    """
    path = Path(path)
    if not path.exists():
        return []
    entities: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue  # headers, blanks, hand-notes
        body = stripped[2:].strip()
        match = re.match(r"^(.*?)\s*\(aka:\s*(.*)\)\s*$", body)
        if match:
            full_name = match.group(1).strip()
            aliases = [a.strip() for a in match.group(2).split(",") if a.strip()]
        else:
            full_name = body
            aliases = []
        if not full_name:
            raise EntityMarkdownError(f"{path}: empty entity line: {line!r}")
        entities.append({"full_name": full_name, "aliases": aliases})
    return entities


def write_index(path: Path, entities: List[Dict[str, object]],
                entity_type: str = DEFAULT_ENTITY_TYPE) -> Path:
    """Rewrite the WHOLE index from entity dicts
    (``[{'full_name': str, 'aliases': [str]}]``, in order) — the rebuild
    primitive used by the resolvers' end-of-pass sync (files are the
    truth, the index their projection: stale lines for deleted files
    disappear here)."""
    plural = _type_plural(entity_type)
    lines = [
        line.format(Title=_title_case(plural), singular=plural[:-1])
        for line in _INDEX_HEADER_TEMPLATE
    ]
    for entity in entities:
        lines.append(format_index_line(
            entity["full_name"], entity["aliases"]  # type: ignore[arg-type]
        ))
    lines.append("")
    _atomic_write(Path(path), "\n".join(lines))
    return Path(path)


def rebuild_index(base_dir: Optional[Path] = None,
                  entity_type: str = DEFAULT_ENTITY_TYPE) -> Path:
    """Rebuild the WHOLE sidecar index from the entity files of one type.

    Single writer of ``<type>.md``: every flow that touches the base
    (the resolver's end-of-pass sync, the removal engine's purge) ends by
    calling this — the files are the truth, the index their projection, so
    a flow cannot leave the listing inconsistent by forgetting a step.

    Skips the index file itself; an entity file that fails to parse is
    reported (EntityMarkdownError) instead of silently dropped from the
    listing. Entity files with empty ``Known names:`` (should not exist:
    a purged entity loses its file) are skipped defensively.
    """
    folder = entities_dir(base_dir, entity_type)
    index_path = index_path_for(base_dir, entity_type)
    entities: List[Dict[str, object]] = []
    if folder.is_dir():
        for path in sorted(folder.glob("*.md")):
            if path.name == index_path.name:
                continue
            data = read_entity(path)  # may raise EntityMarkdownError
            full_name = str(data["full_name"])
            names = data["names"]  # type: ignore[union-attr]
            if not names:
                continue
            entities.append({
                "full_name": full_name,
                "aliases": [
                    str(n.get("alias", "")).strip()
                    for n in names  # type: ignore[union-attr]
                    if str(n.get("alias", "")).strip() != full_name
                ],
            })
    return write_index(index_path, entities, entity_type)


def upsert_index(path: Path, full_name: str, aliases: List[str],
                 entity_type: str = DEFAULT_ENTITY_TYPE) -> Path:
    """Insert or replace the entity's index line (atomic rewrite).

    The line is matched on the exact full name (case-sensitive, the
    identity convention); aliases are stored verbatim minus the full name.
    A replacement keeps the line's original position — re-merging an old
    entity never reorders the listing.
    """
    path = Path(path)
    existing = load_index(path)
    position = next(
        (i for i, c in enumerate(existing) if c["full_name"] == full_name),  # type: ignore[union-attr]
        None,
    )
    kept = [c for c in existing if c["full_name"] != full_name]  # type: ignore[union-attr]
    known_aliases = [
        a for c in existing  # type: ignore[union-attr]
        if c["full_name"] == full_name  # type: ignore[union-attr]
        for a in (c["aliases"] + [full_name])  # type: ignore[union-attr]
    ]
    merged_aliases: List[str] = []
    for alias in list(aliases) + known_aliases:
        if alias and alias not in merged_aliases:
            merged_aliases.append(alias)
    new_entry = {"full_name": full_name, "aliases": merged_aliases}
    if position is None:
        kept.append(new_entry)
    else:
        kept.insert(min(position, len(kept)), new_entry)
    return write_index(path, kept, entity_type)


def remove_from_index(path: Path, full_name: str) -> bool:
    """Drop an entity's index line. True when something was removed."""
    path = Path(path)
    existing = load_index(path)
    kept = [c for c in existing if c["full_name"] != full_name]  # type: ignore[union-attr]
    if len(kept) == len(existing):
        return False
    write_index(path, kept, DEFAULT_ENTITY_TYPE)
    return True


# ---------------------------------------------------------------------------
# Backwards-compatibility aliases (character-only era)
# ---------------------------------------------------------------------------

def characters_dir(base_dir: Optional[Path] = None) -> Path:
    """Historical alias of :func:`entities_dir` for characters."""
    return entities_dir(base_dir, "characters")


def character_path_for(full_name: str, base_dir: Optional[Path] = None) -> Path:
    """Historical alias of :func:`entity_path_for` for characters."""
    return entity_path_for(full_name, base_dir, "characters")


def write_character(full_name: str, names: List[Dict[str, object]],
                    path: Path) -> Path:
    """Historical alias of :func:`write_entity` for characters."""
    return write_entity(full_name, names, path, "characters")


def read_character(path: Path) -> Dict[str, object]:
    """Historical alias of :func:`read_entity`."""
    return read_entity(path)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, content: str) -> None:
    """Atomic UTF-8 write (tempfile in the target folder + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str = ""
    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.stem}_", suffix=".tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_name, path)
    except OSError:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
        raise
