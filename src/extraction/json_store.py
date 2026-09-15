"""Canonical extracted-content store — the fixable JSON persistence layer.

Home: ``src.extraction`` — the store is the extraction layer's persistence
face (models + config loader only, no extraction logic): the pipeline's
consumers import it from the extraction package like every other piece of
the layer.

When a document is extracted, its full ``DocumentExtract`` is serialized as
JSON under ``extraction_output_dir`` (config/ingestion.yaml, default
``data/extracted``) as ``<document stem>.json``. This is the **canonical**
extracted content, distinct from MinerU's raw working artifacts
(``extraction_mineru_output_dir`` — MinerU's own sandbox):

* the ingestion pipeline (summary, embedding, knowledge...) works from the
  canonical JSON's content;
* when a source document cannot be fixed (commercial PDF, remote file...),
  the user edits the canonical JSON instead and asks for a re-ingestion:
  the extraction agent then resumes from that JSON, bypassing MinerU
  entirely.

Design choices:

* **Full fidelity** — every field of every model is serialized, so a
  round trip ``extract -> save -> load`` yields an equal ``DocumentExtract``
  (unlike ``document_extract_to_dict``, which is a human-readable *summary*
  used for debug payloads) — including the stable ids
  (src/extraction/ids.py), preserved from extraction onward.
* **Id-tolerant loading** — files written before the id scheme existed
  (or stripped of ids by hand) load fine with empty ids; the extraction
  agent assigns them on first touch.
* **Human-editable** — pretty-printed UTF-8 JSON; block/section/page
  numbers stay explicit; ``block_type`` is stored as its string value.
* **Defensive loading** — the file may have been edited by hand: a
  malformed entry raises :class:`ExtractJsonError` (a ``ValueError``) with
  an explicit path locator, so the failure is reported, never guessed.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.extraction.models import (
    BlockType,
    Chapter,
    DocumentExtract,
    PageContent,
    Section,
    TextBlock,
    TocEntry,
)
from src.tools.config_loader import (
    PROJECT_ROOT,
    coerce_origin,
    get_default_origin,
    get_valid_origins as get_valid_origins_list,
)

logger = logging.getLogger(__name__)

#: Marker written into canonical files; tolerated absent on load (forward
#: compatibility with hand-written or older files).
SCHEMA_MARKER = "wintermute-extract/1"

#: Fallback when ingestion.yaml does not set ``extraction_output_dir``.
DEFAULT_EXTRACTION_OUTPUT_DIR = "data/extracted"


class ExtractJsonError(ValueError):
    """A canonical extracted-content JSON file is malformed."""


# ---------------------------------------------------------------------------
# Serialization (full fidelity — the mirror of document_extract_from_json_dict)
# ---------------------------------------------------------------------------

def _bbox_to_list(bbox: tuple) -> List[float]:
    return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]


def _block_to_dict(block: TextBlock) -> Dict[str, Any]:
    return {
        "id": block.id,
        "block_id": block.block_id,
        "page_number": block.page_number,
        "bbox": _bbox_to_list(block.bbox),
        "raw_text": block.raw_text,
        "summary": block.summary,
        "block_type": block.block_type.value,
        "text_level": block.text_level,
    }


def _section_to_dict(section: Section) -> Dict[str, Any]:
    return {
        "id": section.id,
        "section_id": section.section_id,
        "blocks": [_block_to_dict(b) for b in section.blocks],
        "page_number": section.page_number,
        "bbox": _bbox_to_list(section.bbox),
        "raw_text": section.raw_text,
        "summary": section.summary,
        "section_title": section.section_title,
        "section_level": section.section_level,
        "is_orphan": section.is_orphan,
    }


def _page_to_dict(page: PageContent) -> Dict[str, Any]:
    return {
        "id": page.id,
        "page_number": page.page_number,
        "width": page.width,
        "height": page.height,
        "raw_text": page.raw_text,
        "summary": page.summary,
        "sections": [_section_to_dict(s) for s in page.sections],
        "chapter_title": page.chapter_title,
    }


def _toc_entry_to_dict(entry: TocEntry) -> Dict[str, Any]:
    return {
        "level": entry.level,
        "title": entry.title,
        "page_number": entry.page_number,
        "page_index": entry.page_index,
    }


def _chapter_to_dict(chapter: Chapter) -> Dict[str, Any]:
    return {
        "id": chapter.id,
        "toc_entry": _toc_entry_to_dict(chapter.toc_entry),
        "pages": [_page_to_dict(p) for p in chapter.pages],
        "full_text": chapter.full_text,
        "summary": chapter.summary,
        "metadata": chapter.metadata,
    }


def document_extract_to_json_dict(doc: DocumentExtract) -> Dict[str, Any]:
    """Full-fidelity, JSON-serializable form of a ``DocumentExtract``."""
    return {
        "schema": SCHEMA_MARKER,
        "id": doc.id,
        "source_path": doc.source_path,
        "title": doc.title,
        "author": doc.author,
        "subject": doc.subject,
        "total_pages": doc.total_pages,
        "origin": doc.origin,
        "toc": [_toc_entry_to_dict(t) for t in doc.toc],
        "chapters": [_chapter_to_dict(c) for c in doc.chapters],
        "summary": doc.summary,
        "orphan_pages": [_page_to_dict(p) for p in doc.orphan_pages],
        "metadata": doc.metadata,
    }


# ---------------------------------------------------------------------------
# Deserialization (defensive: the file may be human-edited)
# ---------------------------------------------------------------------------

def _require_mapping(value: Any, path: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ExtractJsonError(f"{path}: expected an object, got {type(value).__name__}")
    return value


def _require_list(value: Any, path: str) -> List[Any]:
    if not isinstance(value, list):
        raise ExtractJsonError(f"{path}: expected a list, got {type(value).__name__}")
    return value


def _require_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExtractJsonError(f"{path}: expected an integer, got {value!r}")
    return value


def _require_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ExtractJsonError(f"{path}: expected a string, got {value!r}")
    return value


def _opt_str(value: Any, path: str) -> Optional[str]:
    if value is None:
        return None
    return _require_str(value, path)


def _bbox_from_list(value: Any, path: str) -> tuple:
    items = _require_list(value, path)
    if len(items) != 4:
        raise ExtractJsonError(f"{path}: bbox must hold exactly 4 numbers, got {len(items)}")
    numbers = []
    for i, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ExtractJsonError(f"{path}[{i}]: expected a number, got {item!r}")
        numbers.append(float(item))
    return tuple(numbers)


def _block_from_dict(data: Any, path: str) -> TextBlock:
    data = _require_mapping(data, path)
    block_type = data.get("block_type", BlockType.TEXT)
    try:
        block_type = BlockType(block_type)
    except ValueError as exc:
        raise ExtractJsonError(
            f"{path}.block_type: unknown block type {block_type!r}"
        ) from exc
    return TextBlock(
        id=_require_str(data.get("id", ""), f"{path}.id"),
        block_id=_require_int(data.get("block_id", 0), f"{path}.block_id"),
        page_number=_require_int(data.get("page_number", 0), f"{path}.page_number"),
        bbox=_bbox_from_list(data.get("bbox", [0, 0, 0, 0]), f"{path}.bbox"),
        raw_text=_require_str(data.get("raw_text", ""), f"{path}.raw_text"),
        summary=_opt_str(data.get("summary"), f"{path}.summary"),
        block_type=block_type,
        text_level=_require_int(data.get("text_level", 0), f"{path}.text_level"),
    )


def _section_from_dict(data: Any, path: str) -> Section:
    data = _require_mapping(data, path)
    blocks = [
        _block_from_dict(item, f"{path}.blocks[{i}]")
        for i, item in enumerate(_require_list(data.get("blocks", []), f"{path}.blocks"))
    ]
    return Section(
        id=_require_str(data.get("id", ""), f"{path}.id"),
        section_id=_require_int(data.get("section_id", 0), f"{path}.section_id"),
        blocks=blocks,
        page_number=_require_int(data.get("page_number", 0), f"{path}.page_number"),
        bbox=_bbox_from_list(data.get("bbox", [0, 0, 0, 0]), f"{path}.bbox"),
        raw_text=_require_str(data.get("raw_text", ""), f"{path}.raw_text"),
        summary=_opt_str(data.get("summary"), f"{path}.summary"),
        section_title=_opt_str(data.get("section_title"), f"{path}.section_title"),
        section_level=_require_int(data.get("section_level", 0), f"{path}.section_level"),
        is_orphan=bool(data.get("is_orphan", False)),
    )


def _page_from_dict(data: Any, path: str) -> PageContent:
    data = _require_mapping(data, path)
    width, height = data.get("width"), data.get("height")
    for name, value in (("width", width), ("height", height)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ExtractJsonError(f"{path}.{name}: expected a number or null, got {value!r}")
    sections = [
        _section_from_dict(item, f"{path}.sections[{i}]")
        for i, item in enumerate(_require_list(data.get("sections", []), f"{path}.sections"))
    ]
    return PageContent(
        id=_require_str(data.get("id", ""), f"{path}.id"),
        page_number=_require_int(data.get("page_number", 0), f"{path}.page_number"),
        width=float(width) if width is not None else None,
        height=float(height) if height is not None else None,
        raw_text=_require_str(data.get("raw_text", ""), f"{path}.raw_text"),
        summary=_opt_str(data.get("summary"), f"{path}.summary"),
        sections=sections,
        chapter_title=_opt_str(data.get("chapter_title"), f"{path}.chapter_title"),
    )


def _toc_entry_from_dict(data: Any, path: str) -> TocEntry:
    data = _require_mapping(data, path)
    return TocEntry(
        level=_require_int(data.get("level", 0), f"{path}.level"),
        title=_require_str(data.get("title", ""), f"{path}.title"),
        page_number=_require_int(data.get("page_number", 0), f"{path}.page_number"),
        page_index=_require_int(data.get("page_index", 0), f"{path}.page_index"),
    )


def _chapter_from_dict(data: Any, path: str) -> Chapter:
    data = _require_mapping(data, path)
    if "toc_entry" not in data:
        raise ExtractJsonError(f"{path}: missing required 'toc_entry'")
    pages = [
        _page_from_dict(item, f"{path}.pages[{i}]")
        for i, item in enumerate(_require_list(data.get("pages", []), f"{path}.pages"))
    ]
    return Chapter(
        id=_require_str(data.get("id", ""), f"{path}.id"),
        toc_entry=_toc_entry_from_dict(data["toc_entry"], f"{path}.toc_entry"),
        pages=pages,
        full_text=_require_str(data.get("full_text", ""), f"{path}.full_text"),
        summary=_opt_str(data.get("summary"), f"{path}.summary"),
        metadata=_require_mapping(data.get("metadata", {}), f"{path}.metadata"),
    )


def document_extract_from_json_dict(data: Any) -> DocumentExtract:
    """Rebuild a ``DocumentExtract`` from :func:`document_extract_to_json_dict` output.

    Raises:
        ExtractJsonError: with an explicit path locator for every malformed
            entry (the file may have been edited by hand).
    """
    data = _require_mapping(data, "root")
    for required in ("source_path", "title", "total_pages"):
        if required not in data:
            raise ExtractJsonError(
                f"root: missing required '{required}' — this does not look "
                "like a canonical extracted-content file"
            )
    chapters = [
        _chapter_from_dict(item, f"chapters[{i}]")
        for i, item in enumerate(_require_list(data.get("chapters", []), "chapters"))
    ]
    orphans = [
        _page_from_dict(item, f"orphan_pages[{i}]")
        for i, item in enumerate(_require_list(data.get("orphan_pages", []), "orphan_pages"))
    ]
    toc = [
        _toc_entry_from_dict(item, f"toc[{i}]")
        for i, item in enumerate(_require_list(data.get("toc", []), "toc"))
    ]
    metadata = _require_mapping(data.get("metadata", {}), "metadata")
    # Origin: tolerated absent (legacy files) — the configured default
    # (first entry of setup.yaml documents.origins) then applies; a value
    # OUTSIDE the user-defined vocabulary is rejected, not guessed.
    origin_raw = data.get("origin")
    if origin_raw is None:
        origin = get_default_origin()
    else:
        origin = coerce_origin(_require_str(origin_raw, "root.origin"))
        if origin is None:
            raise ExtractJsonError(
                f"root.origin: unknown document origin {origin_raw!r} "
                f"(configured origins: {', '.join(get_valid_origins_list())})"
            )
    return DocumentExtract(
        id=_require_str(data.get("id", ""), "root.id"),
        source_path=_require_str(data.get("source_path", ""), "source_path"),
        title=_require_str(data.get("title", ""), "title"),
        author=_require_str(data.get("author", ""), "author"),
        subject=_require_str(data.get("subject", ""), "subject"),
        total_pages=_require_int(data.get("total_pages", 0), "total_pages"),
        origin=origin,
        toc=toc,
        chapters=chapters,
        summary=_opt_str(data.get("summary"), "summary"),
        orphan_pages=orphans,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_extract(doc: DocumentExtract, path: Path) -> Path:
    """Serialize ``doc`` to a canonical JSON file (pretty UTF-8, atomic write).

    Returns the written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = document_extract_to_json_dict(doc)
    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.stem}_", suffix=".tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(tmp_name, path)
    except OSError:
        try:
            os.unlink(tmp_name)  # noqa: F821 — bound in the try block
        except OSError:
            pass
        raise
    logger.debug("Canonical extraction saved: %s", path)
    return path


def load_extract(path: Path) -> DocumentExtract:
    """Load a canonical extracted-content JSON file back into a ``DocumentExtract``.

    Raises:
        FileNotFoundError: the file does not exist.
        ExtractJsonError: the file is not valid JSON or violates the schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Canonical extraction not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ExtractJsonError(
            f"{path}: invalid JSON (line {exc.lineno}, column {exc.colno}) — "
            "the file may have been edited by hand; fix it and retry."
        ) from exc
    return document_extract_from_json_dict(data)


# ---------------------------------------------------------------------------
# Path resolution (config-driven)
# ---------------------------------------------------------------------------

def extraction_output_dir() -> Path:
    """Canonical extracted-content folder from ingestion.yaml
    (``extraction_output_dir``; default ``data/extracted``).

    Relative values are resolved against the project root; absolute values
    are used as-is.
    """
    from src.tools.config_loader import load_ingestion_config

    raw = DEFAULT_EXTRACTION_OUTPUT_DIR
    try:
        config = load_ingestion_config()
        value = config.get("extraction_output_dir")
        if isinstance(value, str) and value.strip():
            raw = value
    except Exception as exc:  # noqa: BLE001 — fail-open to the default folder
        logger.warning("Could not read ingestion.yaml for extraction_output_dir; "
                       "using default %s (%s)", DEFAULT_EXTRACTION_OUTPUT_DIR, exc)
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def canonical_path_for(source_path: Path) -> Path:
    """Canonical JSON path for a source document: ``<out>/<stem>.json``."""
    return extraction_output_dir() / f"{Path(source_path).stem}.json"
