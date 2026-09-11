"""Stable element identities for extracted documents.

The unified id scheme shared by the extraction stores (JSON), the vector
store and the knowledge layer (``SourceLocator.id``):

* the **document** id — ``doc:<8 hex chars>`` — derives from a SHA-256 of
  the lowercased source filename *with extension*
  (``Dark Earth - Gazette #1.pdf`` → ``doc:3fa2b81c``). Stateless and
  deterministic: no counter to corrupt, same id on any machine, stable
  across re-extraction; the extension in the digest keeps
  ``Gazette.pdf`` and ``Gazette.md`` distinct.
* every **element** id — chapters ``chp:x``, pages ``pg:x``, sections
  ``sec:x``, text blocks ``txt:x`` — is *flat per parent*: numbered within
  its immediate parent, 1-based, in reading order. Short and stable: a
  hand-edit deep in chapter 1 never renumbers chapter 2 (``data/extracted``
  files are meant to be human-edited — global numbering would shift
  hundreds of ids on a one-block insertion).
* the **full hierarchical id** is built *on the fly* when a unique,
  self-locating reference is needed (vector chunk ids, metadata,
  citations):

      ``doc:3fa2b81c::chp:1::pg:2::sec:1::txt:3``

  Each segment's counter restarts at its parent, so the chain alone tells
  where the element lives.

Ids are assigned once, by the extraction layer (:func:`assign_extract_ids`),
and persisted by the canonical JSON store — they then ride the whole
ingestion pipeline (summarization mutates in place and preserves them) and
become the knowledge layer's source ids.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from src.extraction.models import DocumentExtract

#: Flat, per-parent element prefixes (user-facing scheme — keep stable).
DOC_PREFIX = "doc"
CHP_PREFIX = "chp"
PG_PREFIX = "pg"
SEC_PREFIX = "sec"
TXT_PREFIX = "txt"


# ---------------------------------------------------------------------------
# Document id
# ---------------------------------------------------------------------------

def doc_id_from_filename(filename: str) -> str:
    """``doc:<8 hex>`` — SHA-256 of the lowercased filename with extension.

    Deterministic and collision-safe for a personal corpus; the extension
    participates so two formats of the same stem stay distinct documents.
    """
    digest = hashlib.sha256(filename.strip().lower().encode("utf-8")).hexdigest()
    return f"{DOC_PREFIX}:{digest[:8]}"


# ---------------------------------------------------------------------------
# Flat, per-parent element ids (stored on the models)
# ---------------------------------------------------------------------------

def flat_id(prefix: str, index: int) -> str:
    """``<prefix>:<1-based index>`` — the stored, flat per-parent id."""
    return f"{prefix}:{index + 1}"


# ---------------------------------------------------------------------------
# Full hierarchical id (built on the fly — never stored on the models)
# ---------------------------------------------------------------------------

def full_id(
    doc_id: str,
    *,
    chapter_index: Optional[int] = None,
    page_index: Optional[int] = None,
    section_index: Optional[int] = None,
    block_index: Optional[int] = None,
) -> str:
    """Build the full hierarchical id from 0-based list positions.

    Every present level is 1-based in the produced chain; counters restart
    at each parent, so ``doc:x::chp:1::pg:2::sec:1::txt:3`` reads
    "chapter 1, page 2, section 1, third text block".

    Orphan pages pass ``chapter_index=None``: ``doc:x::pg:2`` — a chain
    without a chapter segment, distinct from any in-chapter page.
    """
    parts = [doc_id]
    if chapter_index is not None:
        parts.append(flat_id(CHP_PREFIX, chapter_index))
    if page_index is not None:
        parts.append(flat_id(PG_PREFIX, page_index))
    if section_index is not None:
        parts.append(flat_id(SEC_PREFIX, section_index))
    if block_index is not None:
        parts.append(flat_id(TXT_PREFIX, block_index))
    return "::".join(parts)


# ---------------------------------------------------------------------------
# Assignment (extraction layer's job — once, before first persistence)
# ---------------------------------------------------------------------------

def assign_extract_ids(doc: DocumentExtract) -> DocumentExtract:
    """Assign every missing id on a ``DocumentExtract`` (in place).

    Called by the extraction agent on a fresh extraction AND on resume from
    an id-less store (pre-unification JSON), so a document always carries
    its ids from the moment it enters the pipeline. Idempotent: existing
    ids (including the document's) are never overwritten — a re-ingestion
    keeps the identity a user may already have referenced.

    The document id derives from the source filename (``source_path``);
    falls back to the title when the path is unknown. Raises ``ValueError``
    when neither can produce a name — no id, no indexing.

    Returns the same document, for call-site convenience.
    """
    if not doc.id:
        name = doc.source_path.strip() if doc.source_path else ""
        if not name:
            name = doc.title.strip() if doc.title else ""
        if not name:
            raise ValueError(
                "Cannot derive the document id: DocumentExtract has neither "
                "a source_path nor a title."
            )
        # Filename with extension, not the full path: the id must not move
        # when the file is reorganized within data/sources.
        doc.id = doc_id_from_filename(name.replace("\\", "/").rsplit("/", 1)[-1])

    for chapter_index, chapter in enumerate(doc.chapters):
        if not chapter.id:
            chapter.id = flat_id(CHP_PREFIX, chapter_index)
        for page_index, page in enumerate(chapter.pages):
            if not page.id:
                page.id = flat_id(PG_PREFIX, page_index)
            for section_index, section in enumerate(page.sections):
                if not section.id:
                    section.id = flat_id(SEC_PREFIX, section_index)
                for block_index, block in enumerate(section.blocks):
                    if not block.id:
                        block.id = flat_id(TXT_PREFIX, block_index)

    for orphan_index, orphan in enumerate(doc.orphan_pages):
        if not orphan.id:
            orphan.id = flat_id(PG_PREFIX, orphan_index)
        for section_index, section in enumerate(orphan.sections):
            if not section.id:
                section.id = flat_id(SEC_PREFIX, section_index)
            for block_index, block in enumerate(section.blocks):
                if not block.id:
                    block.id = flat_id(TXT_PREFIX, block_index)

    return doc
