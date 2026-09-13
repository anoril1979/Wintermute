"""Chunks: the unit of vector storage, built from the extracted documents.

A :class:`VectorChunk` is what actually goes into a vector store: a text,
its embedding (filled by the caller — the store never embeds), and the
scalar metadata that ChromaDB keeps alongside the vector for filtering and
citations later.

**Ids ride the unified scheme** (src/extraction/ids.py): the document id
``doc:<8hex>`` is the ``DocumentExtract.id`` assigned by the extraction
layer — the same id the knowledge layer's ``SourceLocator`` uses — and a
chunk id is the element's full hierarchical chain:

    content, one chunk per text block:
        ``doc:3fa2b81c::chp:1::pg:2::sec:1::txt:3``
    summaries (every level above the blocks):
        ``doc:3fa2b81c::chp:1::pg:2::sec:1::sum``   (section summary)
        ``doc:3fa2b81c::chp:1::pg:2::sum``          (page summary)
        ``doc:3fa2b81c::chp:1::sum``                (chapter summary)
        ``doc:3fa2b81c::sum``                       (document summary)

Deterministic by construction → re-indexing a document *updates* its
chunks instead of duplicating them. A block's own ``summary`` field is
deliberately NOT indexed: the block's raw_text is the content, its summary
is a pipeline-internal view.

The builder requires ids to be assigned (``assign_extract_ids`` — the
extraction agent's job): identity decisions belong to the extraction
layer, indexing only consumes them.

Knowledge chunks (collection ``knowledge_chunks``) — chunks of the
knowledge markdown files — have no producer yet: :func:`build_knowledge_chunks`
is a stub raising ``NotImplementedError``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.extraction.ids import (
    CHP_PREFIX,
    DOC_PREFIX,
    PG_PREFIX,
    SEC_PREFIX,
    TXT_PREFIX,
    doc_id_from_filename,
    full_id,
)
from src.extraction.models import DocumentExtract

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The chunk
# ---------------------------------------------------------------------------

# ChromaDB metadata values must be scalars (str / int / float / bool) —
# anything else is rejected by the store, so the type is pinned here.
MetadataValue = str | int | float | bool

#: Suffix segment marking a summary chunk (appended to the element's chain).
SUM_SEGMENT = "sum"


@dataclass
class VectorChunk:
    """One unit of vector storage: text + metadata + (optional) embedding.

    Attributes:
        id: deterministic, unique within a collection — the element's full
            hierarchical id (``doc:x::chp:1::pg:2::sec:1::txt:3``), with a
            ``::sum`` suffix for summary chunks.
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
    #: Similarity of a retrieved chunk against its query (cosine, 0..1,
    #: higher is better). ``None`` until a query fills it — storage never
    #: sets it, it is a retrieval-time attribute only.
    score: Optional[float] = None


# ---------------------------------------------------------------------------
# Level / kind vocabulary (metadata facets)
# ---------------------------------------------------------------------------

LEVEL_BLOCK = "block"
LEVEL_SECTION = "section"
LEVEL_PAGE = "page"
LEVEL_CHAPTER = "chapter"
LEVEL_DOCUMENT = "document"

KIND_CONTENT = "content"
KIND_SUMMARY = "summary"


# ---------------------------------------------------------------------------
# Document id (consumed from the models — never invented here)
# ---------------------------------------------------------------------------

def doc_id_of(doc: DocumentExtract) -> str:
    """The document's unified id: ``DocumentExtract.id`` when assigned.

    Raises ``ValueError`` when it is not — indexing must never silently
    invent an identity that would diverge from the stores and the
    knowledge layer. The message points at the one-call fix.
    """
    if doc.id and doc.id.strip():
        return doc.id.strip()
    raise ValueError(
        "DocumentExtract has no id: run assign_extract_ids(doc) "
        "(src/extraction/ids.py) before building chunks — the vector ids "
        "must use the same unified identity as the stores and the "
        "knowledge layer."
    )


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

    Chunk ids and metadata use the unified id scheme (module docstring):
    the document's ``doc:<8hex>`` id and the flat per-parent element ids
    already stored on the models.

    Returns:
        The chunks, vectors unset (embedding is the caller's job).

    Raises:
        ValueError: if the document id is not assigned (see :func:`doc_id_of`).
    """
    doc_id = doc_id_of(doc)

    base_meta: dict[str, MetadataValue] = {
        "doc_id": doc_id,
        "doc_title": doc.title,
        "source_path": doc.source_path,
        "total_pages": doc.total_pages,
        # Governance metadata: retrieval uses this to weigh/trust results
        # (canon vs community vs user-made) and to arbitrate conflicts.
        "origin": doc.origin,
    }

    def element_meta(
        base: dict[str, MetadataValue],
        *,
        level: str,
        kind: str,
        elem_id: str,
        chain: str,
        **extra: MetadataValue,
    ) -> dict[str, MetadataValue]:
        meta = dict(base)
        meta.update(
            level=level,
            kind=kind,
            elem_id=elem_id,
            full_id=chain,
        )
        meta.update(extra)
        return meta

    chunks: list[VectorChunk] = []
    skipped = 0

    # -- in-chapter pages -----------------------------------------------------
    for chapter_index, chapter in enumerate(doc.chapters):
        chapter_id = chapter.id
        for page_index, page in enumerate(chapter.pages):
            page_chain = full_id(doc_id, chapter_index=chapter_index, page_index=page_index)

            for section_index, section in enumerate(page.sections):
                section_chain = f"{page_chain}::{section.id or full_id(doc_id, section_index=section_index).split('::')[-1]}"

                for block_index, block in enumerate(section.blocks):
                    text = block.raw_text.strip()
                    if not text:
                        skipped += 1
                        continue
                    block_elem = block.id or full_id(
                        doc_id, block_index=block_index
                    ).split("::")[-1]
                    chain = f"{section_chain}::{block_elem}"
                    chunks.append(
                        VectorChunk(
                            id=chain,
                            text=text,
                            metadata=element_meta(
                                base_meta,
                                level=LEVEL_BLOCK,
                                kind=KIND_CONTENT,
                                elem_id=block_elem,
                                chain=chain,
                                page_number=block.page_number,
                                block_type=str(block.block_type.value),
                                chapter_id=chapter_id or "",
                                chapter_title=page.chapter_title or "",
                                page_id=page.id or "",
                                section_id=section.id or "",
                            ),
                        )
                    )

                section_text = (section.summary or "").strip()
                if not section_text:
                    continue
                section_elem = section.id or full_id(
                    doc_id, section_index=section_index
                ).split("::")[-1]
                chain = f"{page_chain}::{section_elem}::{SUM_SEGMENT}"
                chunks.append(
                    VectorChunk(
                        id=chain,
                        text=section_text,
                        metadata=element_meta(
                            base_meta,
                            level=LEVEL_SECTION,
                            kind=KIND_SUMMARY,
                            elem_id=section_elem,
                            chain=chain,
                            page_number=section.page_number,
                            section_title=section.section_title or "",
                            chapter_id=chapter_id or "",
                            chapter_title=page.chapter_title or "",
                            page_id=page.id or "",
                        ),
                    )
                )

            page_text = (page.summary or "").strip()
            if not page_text:
                continue
            page_elem = page.id or full_id(
                doc_id, page_index=page_index
            ).split("::")[-1]
            chain = f"{page_chain}::{SUM_SEGMENT}"
            chunks.append(
                VectorChunk(
                    id=chain,
                    text=page_text,
                    metadata=element_meta(
                        base_meta,
                        level=LEVEL_PAGE,
                        kind=KIND_SUMMARY,
                        elem_id=page_elem,
                        chain=chain,
                        page_number=page.page_number,
                        chapter_id=chapter_id or "",
                        chapter_title=page.chapter_title or "",
                    ),
                )
            )

        chapter_text = (chapter.summary or "").strip()
        if not chapter_text:
            continue
        chapter_elem = chapter_id or full_id(
            doc_id, chapter_index=chapter_index
        ).split("::")[-1]
        chain = full_id(doc_id, chapter_index=chapter_index) + f"::{SUM_SEGMENT}"
        chunks.append(
            VectorChunk(
                id=chain,
                text=chapter_text,
                metadata=element_meta(
                    base_meta,
                    level=LEVEL_CHAPTER,
                    kind=KIND_SUMMARY,
                    elem_id=chapter_elem,
                    chain=chain,
                    chapter_title=chapter.toc_entry.title,
                    start_page=chapter.start_page,
                    end_page=chapter.end_page,
                ),
            )
        )

    # -- orphan pages -----------------------------------------------------------
    for orphan_index, orphan in enumerate(doc.orphan_pages):
        orphan_chain = full_id(doc_id, page_index=orphan_index)

        for section_index, section in enumerate(orphan.sections):
            section_elem = section.id or full_id(
                doc_id, section_index=section_index
            ).split("::")[-1]

            for block_index, block in enumerate(section.blocks):
                text = block.raw_text.strip()
                if not text:
                    skipped += 1
                    continue
                block_elem = block.id or full_id(
                    doc_id, block_index=block_index
                ).split("::")[-1]
                chain = f"{orphan_chain}::{section_elem}::{block_elem}"
                chunks.append(
                    VectorChunk(
                        id=chain,
                        text=text,
                        metadata=element_meta(
                            base_meta,
                            level=LEVEL_BLOCK,
                            kind=KIND_CONTENT,
                            elem_id=block_elem,
                            chain=chain,
                            page_number=block.page_number,
                            block_type=str(block.block_type.value),
                            chapter_id="",
                            chapter_title=orphan.chapter_title or "",
                            page_id=orphan.id or "",
                            section_id=section_elem,
                        ),
                    )
                )

        orphan_text = (orphan.summary or "").strip()
        if not orphan_text:
            continue
        orphan_elem = orphan.id or full_id(
            doc_id, page_index=orphan_index
        ).split("::")[-1]
        chain = f"{orphan_chain}::{SUM_SEGMENT}"
        chunks.append(
            VectorChunk(
                id=chain,
                text=orphan_text,
                metadata=element_meta(
                    base_meta,
                    level=LEVEL_PAGE,
                    kind=KIND_SUMMARY,
                    elem_id=orphan_elem,
                    chain=chain,
                    page_number=orphan.page_number,
                    chapter_id="",
                    chapter_title=orphan.chapter_title or "",
                ),
            )
        )

    # -- document summary ---------------------------------------------------------
    doc_text = (doc.summary or "").strip()
    if doc_text:
        chain = f"{doc_id}::{SUM_SEGMENT}"
        chunks.append(
            VectorChunk(
                id=chain,
                text=doc_text,
                metadata=element_meta(
                    base_meta,
                    level=LEVEL_DOCUMENT,
                    kind=KIND_SUMMARY,
                    elem_id="",
                    chain=chain,
                ),
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
