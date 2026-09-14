"""Markdown knowledge base — the resolver's per-character store.

Second persistence layer of the knowledge ingestion (after the per-document
extraction cache): the EntityResolver projects the discovered entities into
a human-readable, human-editable markdown base under ``knowledge_base_dir``
(config/ingestion.yaml, default ``data/knowledge``):

* one file per character — ``<slug-of-full-name>.md`` — holding the
  identity block: every known name (full name first) as a bullet, each
  followed by the content ids where that name was found::

      # Character : Joe le Clodo

      Known names:

      - Joe le Clodo
        - [doc:36a911e2::chp:1::pg:1::sec:2]
      - Bobby
        - [doc:36a911e2::chp:1::pg:1::sec:4], [doc:36a911e2::chp:1::pg:2::sec:1]

  The ids are the extraction layer's full hierarchical unit ids
  (src/extraction/ids.py) — they link back into the canonical JSON store
  and, through it, into the vector chunks.

* a sidecar index, ``characters.md`` — one line per character with its
  aliases (first name = full name = file name)::

      - Joe le Clodo (aka: Bobby, le Clodo)

  The index is the resolver's batch-scan surface: identity lookups read
  ONE small file instead of walking every character file, and it doubles
  as the user-facing listing of the known entities.

Design mirrors the other knowledge stores: config-driven folder (relative
paths anchored to the project root), atomic writes (tempfile + os.replace),
and a defensive loader — a malformed character file or index (hand-edit
gone wrong) raises :class:`CharacterMarkdownError` with the path, reported
instead of silently ignored.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

from src.tools.config_loader import PROJECT_ROOT

logger = logging.getLogger(__name__)

#: Fallback when ingestion.yaml does not set ``knowledge_base_dir``.
DEFAULT_KNOWLEDGE_BASE_DIR = "data/knowledge"

#: Per-entity-type subfolder of the knowledge base (characters today;
#: future siblings: places/, organizations/, ...).
CHARACTERS_SUBDIR = "characters"

#: Name of the sidecar index inside the knowledge base folder.
INDEX_FILENAME = "characters.md"

#: Fixed header of the sidecar index.
_INDEX_HEADER = [
    "# Characters",
    "",
    "One entry per character; the first name is",
    "the full name used as the file name.",
    "",
]

#: Markdown structure keys (fixed English — the files are consumed by the
#: system first, read by humans second; content stays in its own language).
TITLE_PREFIX = "# Character : "
NAMES_HEADER = "Known names:"


class CharacterMarkdownError(ValueError):
    """A knowledge-base markdown file is malformed (hand-edit gone wrong)."""


# ---------------------------------------------------------------------------
# Path resolution (config-driven, fail-open like every other store)
# ---------------------------------------------------------------------------

def knowledge_base_dir() -> Path:
    """Knowledge BASE folder from ingestion.yaml (``knowledge_base_dir``;
    default ``data/knowledge``).

    The base hosts one subfolder per entity type (``characters/`` today;
    future siblings: ``places/``, ``organizations/``, ...). Relative values
    are resolved against the project root; absolute values are used as-is.
    Fails open to the default on a broken config.
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


def characters_dir(base_dir: Optional[Path] = None) -> Path:
    """Characters subfolder of the knowledge base: ``<base>/characters``."""
    base = Path(base_dir) if base_dir else knowledge_base_dir()
    return base / CHARACTERS_SUBDIR


def slugify_filename(name: str) -> str:
    """File-system slug of a character full name, e.g.
    'Édmond Dantès' -> 'edmond-dantes' (ascii, lowercase, hyphenated)."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return text or "unnamed"


def character_path_for(full_name: str, base_dir: Optional[Path] = None) -> Path:
    """Character file path for a full name: ``<base>/characters/<slug>.md``."""
    return characters_dir(base_dir) / f"{slugify_filename(full_name)}.md"


def index_path_for(base_dir: Optional[Path] = None) -> Path:
    """Sidecar index path: ``<base>/characters/characters.md``."""
    return characters_dir(base_dir) / INDEX_FILENAME


# ---------------------------------------------------------------------------
# Character files
# ---------------------------------------------------------------------------

def write_character(
    full_name: str,
    names: List[Dict[str, object]],
    path: Path,
) -> Path:
    """(Re)write one character file (pretty UTF-8, atomic).

    Args:
        full_name: the character's full name (title of the file).
        names: ordered name entries — ``{"alias": str, "source_ids": [str]}``;
            the FIRST entry is conventionally the full name itself.
        path: target ``<slug>.md`` path.

    Returns the written path.
    """
    lines = [f"{TITLE_PREFIX}{full_name.strip()}", "", NAMES_HEADER, ""]
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


def read_character(path: Path) -> Dict[str, object]:
    """Parse one character file back into
    ``{"full_name": str, "names": [{"alias", "source_ids"}]}``.

    Raises:
        FileNotFoundError: the file does not exist.
        CharacterMarkdownError: the structure is not a character file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Character file not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()

    full_name = ""
    for line in lines:
        if line.startswith(TITLE_PREFIX):
            full_name = line[len(TITLE_PREFIX):].strip()
            break
    if not full_name:
        raise CharacterMarkdownError(
            f"{path}: missing '{TITLE_PREFIX}<name>' title line"
        )

    names: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(NAMES_HEADER) or stripped.startswith("#"):
            continue
        if line.startswith("- "):
            current = {"alias": stripped[2:].strip(), "source_ids": []}
            names.append(current)
        elif stripped.startswith("-") and "[" in stripped:
            # Ids line:  "- [id], [id]"  or  "(no source yet)" placeholder
            # (indented sub-bullet of the name above it).
            if current is None:
                raise CharacterMarkdownError(
                    f"{path}: source-id line without a name bullet above it: {line!r}"
                )
            for source_id in re.findall(r"\[([^\[\]]+)\]", stripped):
                if ":" in source_id:  # "(no source yet)" has no id colon
                    current["source_ids"].append(source_id)  # type: ignore[union-attr]
        # Any other line: prose/hand-notes — ignored.
    return {"full_name": full_name, "names": names}


# ---------------------------------------------------------------------------
# Sidecar index (one line per character)
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
    characters: List[Dict[str, object]] = []
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
            raise CharacterMarkdownError(f"{path}: empty character line: {line!r}")
        characters.append({"full_name": full_name, "aliases": aliases})
    return characters


def write_index(path: Path, characters: List[Dict[str, object]]) -> Path:
    """Rewrite the WHOLE index from character dicts
    (``[{'full_name': str, 'aliases': [str]}]``, in order) — the rebuild
    primitive used by the resolver's end-of-pass sync (files are the
    truth, the index their projection: stale lines for deleted files
    disappear here)."""
    lines = list(_INDEX_HEADER)
    for character in characters:
        lines.append(format_index_line(
            character["full_name"], character["aliases"]  # type: ignore[arg-type]
        ))
    lines.append("")
    _atomic_write(Path(path), "\n".join(lines))
    return Path(path)


def rebuild_index(base_dir: Optional[Path] = None) -> Path:
    """Rebuild the WHOLE sidecar index from the character files of the base.

    Single writer of ``characters.md``: every flow that touches the base
    (the resolver's end-of-pass sync, the removal engine's purge) ends by
    calling this — the files are the truth, the index their projection, so
    a flow cannot leave the listing inconsistent by forgetting a step.

    Skips the index file itself; a character file that fails to parse is
    reported (CharacterMarkdownError) instead of silently dropped from the
    listing. Character files with empty ``Known names:`` (should not exist:
    a purged character loses its file) are skipped defensively.
    """
    folder = characters_dir(base_dir)
    index_path = index_path_for(base_dir)
    characters: List[Dict[str, object]] = []
    if folder.is_dir():
        for path in sorted(folder.glob("*.md")):
            if path.name == INDEX_FILENAME:
                continue
            data = read_character(path)  # may raise CharacterMarkdownError
            full_name = str(data["full_name"])
            names = data["names"]  # type: ignore[union-attr]
            if not names:
                continue
            characters.append({
                "full_name": full_name,
                "aliases": [
                    str(n.get("alias", "")).strip()
                    for n in names  # type: ignore[union-attr]
                    if str(n.get("alias", "")).strip() != full_name
                ],
            })
    return write_index(index_path, characters)


def upsert_index(path: Path, full_name: str, aliases: List[str]) -> Path:
    """Insert or replace the character's index line (atomic rewrite).

    The line is matched on the exact full name (case-sensitive, the
    identity convention); aliases are stored verbatim minus the full name.
    A replacement keeps the line's original position — re-merging an old
    character never reorders the listing.
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
    return write_index(path, kept)


def remove_from_index(path: Path, full_name: str) -> bool:
    """Drop a character's index line. True when something was removed."""
    path = Path(path)
    existing = load_index(path)
    kept = [c for c in existing if c["full_name"] != full_name]  # type: ignore[union-attr]
    if len(kept) == len(existing):
        return False
    write_index(path, kept)
    return True


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
