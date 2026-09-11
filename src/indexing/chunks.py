"""Chunks: the unit of vector storage, built from the extracted documents.

A :class:`VectorChunk` is what actually goes into a vector store: a text,
its embedding (filled by the caller — the store never embeds), and the
scalar metadata that ChromaDB keeps alongside the vector for filtering and
citations later.

Two families of chunks, matching the two collections of setup.yaml:

* **source chunks** (collection ``source_chunks``) — built from a
  ``DocumentExtract`` by :func:`build_source_chunks`:

      - one chunk per **TextBlock** raw_text — the document's true content
        (blocks == paragraphs everywhere else in the system);
      - one chunk per **summarized level that is not a block**: section,
        page, chapter and whole-document summaries (a block's own summary
        is deliberately NOT indexed — the block's raw_text is the content;
        its summary is a pipeline-internal view).

* **knowledge chunks** (collection ``knowledge_chunks``) — chunks of the
  knowledge markdown files. Producer (knowledge extraction) not built yet:
  :func:`build_knowledge_chunks` is a stub raising ``NotImplementedError``.

Chunk ids are **deterministic** — derived from the document id and the
chunk's position in the structure, never from a counter or the wall clock.
Re-ingesting (or re-indexing) a document therefore *updates* its chunks
instead of duplicating them: the vector store stays consistent with the
same idempotent spirit as the extraction/summarization job files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.extraction.models import DocumentExtract

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The chunk
# ---------------------------------------------------------------------------

# ChromaDB metadata values must be scalars (str / int / float / bool) —
# anything else is rejected by the store, so the type is pinned here.
MetadataValue = str | int | float | bool


@dataclass
class VectorChunk:
    """One unit of vector storage: text + metadata + (optional) embedding.

    Attributes:
        id: deterministic, unique within a collection (see module docstring).
        text: the text the vector was computed from.
        metadata: scalar (Chroma-compatible) facets: document identity,
            hierarchical position, level/kind. Used later for filtered
            retrieval and human-readable citations.
        vector: the embedding, filled by the caller before storage.
            ``None`` until embedded.
    """

    id: str
    text: str
    metadata: dict[str, MetadataValue] = field(default_factory=dict)
    vector: Optional[list[float]] = None


# ---------------------------------------------------------------------------
# Deterministic ids
# ---------------------------------------------------------------------------

#: Level names used in ids and metadata — keep in sync with the builders.
LEVEL_BLOCK = "block"
LEVEL_SECTION = "section"
LEVEL_PAGE = "page"
LEVEL_CHAPTER = "chapter"
LEVEL_DOCUMENT = "document"

#: Kind of content a chunk carries (blocks are content; summaries are
#: LLM-produced digests of a level above the blocks).
KIND_CONTENT = "content"
KIND_SUMMARY = "summary"


def doc_id_from_source_path(source_path: str) -> str:
    """The document id used in chunk ids: the source file's stem.

    ``data/sources/pdf/Dark Earth - Gazette #1.pdf`` →
    ``Dark Earth - Gazette #1``. Readable, stable across re-ingestions of
    the same file, and unique enough within the documents root.
    """
    return Path(source_path).stem


def _chunk_id(doc_id: str, parts: list[str]) -> str:
    """Join the deterministic id: ``<doc_id>::<part>::<part>::...``."""
    return "::".join([doc_id, *parts])


# ---------------------------------------------------------------------------
# Source chunks (from a DocumentExtract)
# ---------------------------------------------------------------------------

def build_source_chunks(doc: DocumentExtract) -> list[VectorChunk]:
    """Build every vector chunk of one summarized ``DocumentExtract``.

    Bottom-up over the structure (blocks, then sections, pages, chapters,
    document), skipping anything whose text is empty/blank — the extraction
    validator already prunes empty content, but indexing must stay safe on
    any input. Expected volume per document:

        1 chunk per text block            (kind=content, level=block)
        + 1 per section summary           (kind=summary, level=section)
        + 1 per page summary              (kind=summary, level=page)
        + 1 per chapter summary           (kind=summary, level=chapter)
        + 1 for the document summary      (kind=summary, level=document)

    Returns:
        The chunks, vectors unset (embedding is the caller's job).

    Raises:
        ValueError: if the document has neither a usable source_path nor a
            title (no id can be derived).
    """
    doc_id = doc_id_from_source_path(doc.source_path or doc.title or "")
    if not doc_id.strip():
        raise ValueError(
            "Cannot derive a document id: DocumentExtract has neither a "
            "source_path nor a title."
        )

    base_meta: dict[str, MetadataValue] = {
        "doc_id": doc_id,
        "doc_title": doc.title,
        "source_path": doc.source_path,
        "total_pages": doc.total_pages,
    }

    chunks: list[VectorChunk] = []
    skipped = 0

    for page in doc.all_pages():
        for section in page.sections:
            for block in section.blocks:
                text = block.raw_text.strip()
                if not text:
                    skipped += 1
                    continue
                meta = dict(base_meta)
                meta.update(
                    level=LEVEL_BLOCK,
                    kind=KIND_CONTENT,
                    page_number=block.page_number,
                    block_id=block.block_id,
                    block_type=str(block.block_type.value),
                    section_id=section.section_id,
                    chapter_title=page.chapter_title or "",
                )
                chunks.append(
                    VectorChunk(
                        id=_chunk_id(
                            doc_id,
                            [LEVEL_BLOCK, str(block.page_number), str(block.block_id)],
                        ),
                        text=text,
                        metadata=meta,
                    )
                )

            section_text = (section.summary or "").strip()
            if not section_text:
                continue
            meta = dict(base_meta)
            meta.update(
                level=LEVEL_SECTION,
                kind=KIND_SUMMARY,
                page_number=section.page_number,
                section_id=section.section_id,
                section_title=section.section_title or "",
                chapter_title=page.chapter_title or "",
            )
            chunks.append(
                VectorChunk(
                    id=_chunk_id(
                        doc_id,
                        [KIND_SUMMARY, LEVEL_SECTION,
                         str(section.page_number), str(section.section_id)],
                    ),
                    text=section_text,
                    metadata=meta,
                )
            )

        page_text = (page.summary or "").strip()
        if not page_text:
            continue
        meta = dict(base_meta)
        meta.update(
            level=LEVEL_PAGE,
            kind=KIND_SUMMARY,
            page_number=page.page_number,
            chapter_title=page.chapter_title or "",
        )
        chunks.append(
            VectorChunk(
                id=_chunk_id(doc_id, [KIND_SUMMARY, LEVEL_PAGE, str(page.page_number)]),
                text=page_text,
                metadata=meta,
            )
        )

    for chapter_index, chapter in enumerate(doc.chapters):
        chapter_text = (chapter.summary or "").strip()
        if not chapter_text:
            continue
        meta = dict(base_meta)
        meta.update(
            level=LEVEL_CHAPTER,
            kind=KIND_SUMMARY,
            chapter_index=chapter_index,
            chapter_title=chapter.toc_entry.title,
            start_page=chapter.start_page,
            end_page=chapter.end_page,
        )
        chunks.append(
            VectorChunk(
                id=_chunk_id(
                    doc_id, [KIND_SUMMARY, LEVEL_CHAPTER, str(chapter_index)]
                ),
                text=chapter_text,
                metadata=meta,
            )
        )

    doc_text = (doc.summary or "").strip()
    if doc_text:
        meta = dict(base_meta)
        meta.update(level=LEVEL_DOCUMENT, kind=KIND_SUMMARY)
        chunks.append(
            VectorChunk(
                id=_chunk_id(doc_id, [KIND_SUMMARY, LEVEL_DOCUMENT]),
                text=doc_text,
                metadata=meta,
            )
        )

    if skipped:
        logger.debug(
            "build_source_chunks(%s): %d blank block(s) skipped", doc_id, skipped
        )
    return chunks


# ---------------------------------------------------------------------------
# Knowledge chunks (markdown files) — deferred
# ---------------------------------------------------------------------------

def build_knowledge_chunks(*args: object, **kwargs: object) -> list[VectorChunk]:
    """STUB — chunk the knowledge markdown files into knowledge chunks.

    The producer of these files (the knowledge-extraction and check-n-merge
    steps) is not built yet, so the chunking policy (heading-aware split?)
    is deliberately undecided. This stub documents the contract point and
    fails loudly if something reaches it before its time.
    """
    raise NotImplementedError(
        "Knowledge chunks have no producer yet: the markdown chunker lands "
        "with the knowledge-extraction step (see src/indexing/chunks.py)."
    )
