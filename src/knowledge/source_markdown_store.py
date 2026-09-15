"""Markdown knowledge base — the per-SOURCE registration store.

Third projection of the knowledge ingestion, beside the per-character
files: every ingested document gets its own markdown registration — the
document as a knowledge entity of its own (its identity, its shape, where
it came from) — so the corpus's sources can be listed, inspected and
annotated like every other entity of the base.

Layout (under ``knowledge_base_dir``, config/ingestion.yaml, default
``data/knowledge``)::

    data/knowledge/
    ├── characters/          (per-character files + characters.md)
    ├── sources/             (per-document registrations)
    │   ├── doc_3fa2b81c.md  (one file per ingested document)
    │   └── sources.md       (the sidecar listing)
    └── sources.md           (legacy duplicate guard — see below)

The per-document file is named by the document's UNIFIED id (``doc:<8hex>``
→ ``doc_3fa2b81c.md`` — ``:`` is not file-system-friendly): the same id
the extraction stores, the vector chunks and the character provenance
use. Re-ingesting the same source file always lands on the same file.

File structure (fixed English structural keys — content stays in its own
language)::

    # Source : Dark Earth - Le marcheur (Gazette #1).pdf

    - Id: doc:3fa2b81c
    - Type: pdf
    - Title: Dark Earth - Le marcheur (Gazette #1)
    - File: Dark Earth - Le marcheur (Gazette #1).pdf
    - Path: data/sources/pdf/Dark Earth - Le marcheur (Gazette #1).pdf
    - Origin: community
    - Chapters: 5
    - Pages: 17

    Notes:
    (added by the user — preserved by the system)

The ``Notes:`` section is the USER's scope: hand-written annotations are
preserved verbatim across re-registrations (the system rewrites everything
above the section, never below). Everything else is system-owned and
regenerated at each ingestion.

The sidecar listing ``sources.md`` — one line per document,
``- doc:3fa2b81c — <stem>`` — follows the same single-writer rule as
``characters.md``: the FILES are the truth, the listing their projection,
rebuilt from the files by every flow that touches the base (registration,
removal purge), so a deleted document's line cannot survive.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.knowledge.character_markdown_store import (
    knowledge_base_dir,
    _atomic_write,
)

logger = logging.getLogger(__name__)

#: Subfolder of the knowledge base hosting the per-document registrations.
SOURCES_SUBDIR = "sources"

#: Name of the sidecar listing inside that subfolder.
LISTING_FILENAME = "sources.md"

#: Fixed header of the sidecar listing.
_LISTING_HEADER = [
    "# Sources",
    "",
    "One entry per ingested document: id and file stem.",
    "The file name is the document's unified id (``_`` for ``:``).",
    "",
]

#: Markdown structural keys (fixed English, like the character store).
SOURCE_TITLE_PREFIX = "# Source : "
NOTES_HEADER = "Notes:"

#: The user's section marker: everything below is preserved on rewrite.
_NOTES_SENTINEL = NOTES_HEADER

#: Regex for one listing line: ``- doc:<8hex> — <stem>``.
_LISTING_LINE = re.compile(r"^-\s+(doc:[0-9a-f]{8})\s+—\s+(.+)$")

#: Fallback stem when a document carries no usable title.
_UNTITLED = "untitled"


class SourceMarkdownError(ValueError):
    """A source registration file is malformed (hand-edit gone wrong)."""


# ---------------------------------------------------------------------------
# Document type (readable, extension-derived)
# ---------------------------------------------------------------------------

#: Extension -> document type (the knowledge layer's ``SourceType``
#: vocabulary, src/knowledge/models.py). Keys lowercase, with the dot;
#: unknown/absent extensions fall back to ``other``.
_EXTENSION_TYPES: Dict[str, str] = {
    ".pdf": "pdf",
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
    ".docx": "word",
    ".doc": "word",
    ".odt": "openoffice",
    ".url": "url",
}


def document_type_for(source_path: str) -> str:
    """Readable document type for a source path (``SourceType`` vocabulary).

    Derived from the file extension — ``Gazette.pdf`` → ``pdf`` — with
    ``other`` for unknown or extension-less names (the readable field must
    always exist, even when the derivation has nothing to bite on).
    """
    suffix = Path(str(source_path or "")).suffix.lower()
    return _EXTENSION_TYPES.get(suffix, "other")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def sources_dir(base_dir: Optional[Path] = None) -> Path:
    """Sources subfolder of the knowledge base: ``<base>/sources``."""
    base = Path(base_dir) if base_dir else knowledge_base_dir()
    return base / SOURCES_SUBDIR


def filename_for_doc_id(doc_id: str) -> str:
    """File name of a source registration: ``doc:<8hex>`` → ``doc_<8hex>.md``.

    The colon is swapped for an underscore (Windows-safe, shell-safe);
    the shape stays recognizable and reversible.
    """
    clean = (doc_id or "").strip()
    if not re.fullmatch(r"doc:[0-9a-f]{8}", clean):
        raise SourceMarkdownError(
            f"invalid document id {doc_id!r}: expected 'doc:<8 hex chars>'"
        )
    return f"{clean.replace(':', '_')}.md"


def source_path_for(doc_id: str, base_dir: Optional[Path] = None) -> Path:
    """Registration file for a document id: ``<base>/sources/<doc_id>.md``."""
    return sources_dir(base_dir) / filename_for_doc_id(doc_id)


def listing_path_for(base_dir: Optional[Path] = None) -> Path:
    """Sidecar listing path: ``<base>/sources/sources.md``."""
    return sources_dir(base_dir) / LISTING_FILENAME


# ---------------------------------------------------------------------------
# Per-document registration files
# ---------------------------------------------------------------------------

def write_source(
    doc_id: str,
    title: str,
    file_name: str,
    path: str,
    origin: str,
    chapters: int,
    pages: int,
    registration_path: Path,
    doc_type: str = "",
) -> Path:
    """(Re)write one source registration (pretty UTF-8, atomic).

    System-owned fields are regenerated; an existing file's ``Notes:``
    section (everything below the marker) is preserved verbatim — the
    user's annotations survive re-ingestion.

    Args:
        doc_id: the document's unified id (``doc:<8hex>``).
        doc_type: readable document type (``SourceType`` vocabulary, e.g.
            ``pdf``); empty writes ``other`` (the field always exists).
        title: extraction title (usually the file stem).
        file_name: source file name with extension.
        path: source path as resolved at ingestion (project-relative when
            possible).
        origin: governance label (user-defined vocabulary, setup.yaml).
        chapters: number of chapters in the extraction (orphan pages are
            NOT chapters).
        pages: total page count of the extraction (``total_pages``).
        registration_path: target ``<base>/sources/<doc_id>.md`` path.

    Returns the written path.
    """
    doc_id = doc_id.strip()
    doc_type = (doc_type or "").strip().lower() or "other"
    title = (title or "").strip() or _UNTITLED
    file_name = (file_name or "").strip()
    path = (path or "").strip()
    origin = (origin or "").strip().lower()
    chapters = max(0, int(chapters or 0))
    pages = max(0, int(pages or 0))

    notes = read_notes(registration_path)
    lines = [
        f"{SOURCE_TITLE_PREFIX}{title}",
        "",
        f"- Id: {doc_id}",
        f"- Type: {doc_type}",
        f"- Title: {title}",
        f"- File: {file_name}",
        f"- Path: {path}",
        f"- Origin: {origin}",
        f"- Chapters: {chapters}",
        f"- Pages: {pages}",
        "",
        _NOTES_SENTINEL,
    ]
    if notes:
        lines.append(notes)
    _atomic_write(Path(registration_path), "\n".join(lines).rstrip() + "\n")
    return Path(registration_path)


def read_notes(path: Path) -> str:
    """The user's ``Notes:`` section of an existing registration (or '').

    Everything strictly below the ``Notes:`` line is the user's scope and
    comes back verbatim (blank-line trimmed, content preserved).
    """
    path = Path(path)
    if not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("Could not read notes from %s: %s", path, exc)
        return ""
    for index, line in enumerate(lines):
        if line.strip() == _NOTES_SENTINEL:
            return "\n".join(lines[index + 1:]).strip("\n")
    return ""


def read_source(path: Path) -> Dict[str, object]:
    """Parse one registration file back into a flat dict.

    Returns the system fields (``doc_id``, ``title``, ``file``, ``path``,
    ``origin``, ``chapters``, ``pages``) plus the user's ``notes``.

    Raises:
        FileNotFoundError: the file does not exist.
        SourceMarkdownError: the structure is not a source registration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source registration not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()

    title = ""
    fields: Dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(SOURCE_TITLE_PREFIX):
            title = stripped[len(SOURCE_TITLE_PREFIX):].strip()
            continue
        if stripped.startswith("- ") and ":" in stripped:
            key, _, value = stripped[2:].partition(":")
            fields[key.strip().lower()] = value.strip()

    doc_id = fields.get("id", "")
    if not doc_id:
        raise SourceMarkdownError(f"{path}: missing '- Id: doc:<8hex>' line")
    return {
        "doc_id": doc_id,
        "type": fields.get("type", ""),
        "title": title or fields.get("title", ""),
        "file": fields.get("file", ""),
        "path": fields.get("path", ""),
        "origin": fields.get("origin", ""),
        "chapters": _int_field(fields.get("chapters")),
        "pages": _int_field(fields.get("pages")),
        "notes": read_notes(path),
    }


def _int_field(raw: Optional[str]) -> int:
    """A parsed int field; -1 when unreadable (hand-edit damage visible)."""
    try:
        return int((raw or "").strip())
    except (TypeError, ValueError):
        return -1


# ---------------------------------------------------------------------------
# Sidecar listing (one line per document)
# ---------------------------------------------------------------------------

def format_listing_line(doc_id: str, stem: str) -> str:
    """``- doc:<8hex> — <stem>`` — the id, then the human-readable stem."""
    return f"- {doc_id} — {stem or _UNTITLED}"


def load_listing(path: Path) -> List[Dict[str, str]]:
    """Parse the sidecar listing into ``[{"doc_id", "stem"}]`` (order kept).

    A missing listing is an empty base, not an error. A line that does not
    match the id-stem shape raises (hand-edit damage must be visible, the
    same defensive rule as the character index).
    """
    path = Path(path)
    if not path.exists():
        return []
    entries: List[Dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue  # headers, blanks, hand-notes
        match = _LISTING_LINE.match(stripped)
        if not match:
            raise SourceMarkdownError(
                f"{path}: malformed source line: {line!r} "
                "(expected '- doc:<8hex> — <stem>')"
            )
        entries.append({"doc_id": match.group(1), "stem": match.group(2).strip()})
    return entries


def rebuild_listing(base_dir: Optional[Path] = None) -> Path:
    """Rebuild the WHOLE sidecar listing from the registration files.

    Single writer of ``sources.md`` (the files are the truth, the listing
    their projection): every flow touching the base ends by calling this —
    the registration agent after an ingestion, the removal engine after a
    purge — so a deleted document's line cannot survive by construction.
    Files are read in id order (stable listing); the ``sources.md`` listing
    itself is skipped.
    """
    folder = sources_dir(base_dir)
    entries: List[Dict[str, str]] = []
    if folder.is_dir():
        for path in sorted(folder.glob("*.md")):
            if path.name == LISTING_FILENAME:
                continue
            data = read_source(path)  # may raise SourceMarkdownError
            entries.append({
                "doc_id": str(data["doc_id"]),
                "stem": str(data["title"]) or _UNTITLED,
            })
    lines = list(_LISTING_HEADER)
    for entry in entries:
        lines.append(format_listing_line(entry["doc_id"], entry["stem"]))
    lines.append("")
    _atomic_write(listing_path_for(base_dir), "\n".join(lines))
    return listing_path_for(base_dir)


def remove_registration(doc_id: str, base_dir: Optional[Path] = None) -> bool:
    """Delete one document's registration file and refresh the listing.

    Returns True when a file was deleted; the listing rebuild happens in
    every case (it may still sweep other stale lines). An unknown/invalid
    id is a no-op returning False.
    """
    try:
        path = source_path_for(doc_id, base_dir)
    except SourceMarkdownError:
        return False
    if path.is_file():
        path.unlink()
        logger.info("Source registration removed: %s", path.name)
        removed = True
    else:
        removed = False
    try:
        rebuild_listing(base_dir)
    except (SourceMarkdownError, OSError) as exc:
        logger.warning("Source listing rebuild failed: %s", exc)
    return removed


__all__ = [
    "LISTING_FILENAME",
    "SOURCES_SUBDIR",
    "SOURCE_TITLE_PREFIX",
    "SourceMarkdownError",
    "document_type_for",
    "filename_for_doc_id",
    "format_listing_line",
    "listing_path_for",
    "load_listing",
    "read_notes",
    "read_source",
    "rebuild_listing",
    "remove_registration",
    "source_path_for",
    "sources_dir",
    "write_source",
]
