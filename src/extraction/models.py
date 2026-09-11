"""Document models produced by the extraction step.

These dataclasses describe a structured document extraction: atomic text
blocks, grouped into sections, assembled into pages, aligned on the
document's table of contents to form chapters. ``DocumentExtract`` is the
root object exchanged with the rest of the ingestion pipeline.

Stable identities (src/extraction/ids.py): every element carries an ``id``
— the document ``doc:<8hex>`` (hash of the source filename), and flat
per-parent element ids (``chp:x``, ``pg:x``, ``sec:x``, ``txt:x``, 1-based
within the parent). Assigned by the extraction layer (``assign_extract_ids``),
persisted by the canonical JSON store, and used downstream as vector chunk
ids/metadata and as the knowledge layer's source ids. ``""`` (empty) means
"not yet assigned" — legacy files without ids stay loadable; the agent
fills them in on first touch. The full hierarchical id
(``doc:x::chp:1::pg:2::sec:1::txt:3``) is built on the fly, never stored.

Moved from the prototype (``src/ingestion/old/rag_models.py``) — the legacy
module still exists for the old pipeline; new code imports from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BlockType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    HEADER = "header"
    FOOTER = "footer"


@dataclass
class TocEntry:
    """Entrée du sommaire (Table of Contents)."""
    level: int                  # Niveau hiérarchique (1 = chapitre, 2 = section…)
    title: str                  # Titre tel qu'il apparaît dans le TOC
    page_number: int            # Page de début (index 1)
    page_index: int             # Page de début (index 0, pour PyMuPDF)


@dataclass
class TextBlock:
    """Bloc de texte atomique extrait d'une page.

    ``id``: flat per-parent id ``txt:x`` (1-based within its section),
    assigned by the extraction layer — empty until then.
    """
    block_id: int
    page_number: int            # Index 1
    bbox: tuple[float, float, float, float]   # (x0, y0, x1, y1)
    raw_text: str
    summary: Optional[str] = None  # Résumé (produit par étape suivante du pipeline)
    block_type: BlockType = BlockType.TEXT
    text_level: int = 0
    id: str = ""


@dataclass
class Section:
    """Section/Partie d'une page.

    ``id``: flat per-parent id ``sec:x`` (1-based within its page).
    """
    section_id: int
    blocks: list[TextBlock]
    page_number: int            # Index 1
    bbox: tuple[float, float, float, float]   # (x0, y0, x1, y1)
    raw_text: str
    summary: Optional[str] = None  # Résumé (produit par étape suivante du pipeline)
    section_title: Optional[str] = None
    section_level: int = 0
    is_orphan: bool = False
    id: str = ""


@dataclass
class PageContent:
    """Contenu complet d'une page.

    ``id``: flat per-parent id ``pg:x`` (1-based within its chapter, or
    within the orphan list for orphan pages).
    """
    page_number: int            # Index 1
    width: float
    height: float
    raw_text: str               # Texte brut concaténé de la page
    summary: Optional[str] = None  # Résumé (produit par étape suivante du pipeline)
    sections: list[Section] = field(default_factory=list)
    chapter_title: Optional[str] = None
    id: str = ""


@dataclass
class Chapter:
    """Section/Chapitre logique du document, alignée sur une entrée TOC.

    ``id``: flat per-parent id ``chp:x`` (1-based within the document's
    chapter list).
    """
    toc_entry: TocEntry
    pages: list[PageContent] = field(default_factory=list)
    full_text: str = ""          # Texte brut agrégé du chapitre
    summary: Optional[str] = None  # Résumé (produit par étape suivante du pipeline)
    metadata: dict = field(default_factory=dict)
    id: str = ""

    @property
    def start_page(self) -> int:
        return self.toc_entry.page_number

    @property
    def end_page(self) -> int:
        if self.pages:
            return self.pages[-1].page_number
        return self.start_page


@dataclass
class DocumentExtract:
    """
    Objet racine produit par le PDFExtractor.
    C'est l'unité d'échange avec le reste du pipeline RAG.

    ``id``: the document's stable identity ``doc:<8hex>`` — SHA-256 of the
    lowercased source filename with extension (see src/extraction/ids.py).
    This is THE unified source id: the knowledge layer's SourceLocator.id,
    the vector chunk id prefix, and the stores' cross-reference. Empty until
    the extraction layer assigns it.
    """
    id: str = ""
    source_path: str = ""
    title: str = ""
    author: str = ""
    subject: str = ""
    total_pages: int = 0
    toc: list[TocEntry] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    summary: Optional[str] = None  # Résumé (produit par étape suivante du pipeline)
    orphan_pages: list[PageContent] = field(default_factory=list)  # Pages hors TOC
    metadata: dict = field(default_factory=dict)

    def all_pages(self) -> list[PageContent]:
        """Itère sur toutes les pages dans l'ordre, toutes sections confondues."""
        pages = []
        for chapter in self.chapters:
            pages.extend(chapter.pages)
        pages.extend(self.orphan_pages)
        return sorted(pages, key=lambda p: p.page_number)


# ---------------------------------------------------------------------------
# Sérialisation utilitaire (debug / persistance inter-étapes)
# ---------------------------------------------------------------------------

def document_extract_to_dict(doc: DocumentExtract) -> dict:
    """Convertit un DocumentExtract en dictionnaire JSON-sérialisable."""

    def toc_to_dict(t: TocEntry) -> dict:
        return {"level": t.level, "title": t.title, "page_number": t.page_number}

    def section_to_dict(s: Section) -> dict:
        return {
            "section_title": s.section_title,
            "block_count": len(s.blocks),
            "page_number": s.page_number,
            "bbox": [s.bbox[0], s.bbox[1], s.bbox[2], s.bbox[3]],
            "raw_text": s.raw_text,
        }

    def page_to_dict(p: PageContent) -> dict:
        return {
            "page_number": p.page_number,
            "chapter_title": p.chapter_title,
            "raw_text": p.raw_text,
            "section_count": len(p.sections),
            "sections": [section_to_dict(s) for s in p.sections],
        }

    def chapter_to_dict(c: Chapter) -> dict:
        return {
            "title": c.toc_entry.title,
            "level": c.toc_entry.level,
            "start_page": c.start_page,
            "end_page": c.end_page,
            "page_count": len(c.pages),
            "full_text_length": len(c.full_text),
            "summary": c.summary,
            "pages": [page_to_dict(p) for p in c.pages],
            "metadata": c.metadata,
        }

    return {
        "source_path": doc.source_path,
        "title": doc.title,
        "author": doc.author,
        "subject": doc.subject,
        "total_pages": doc.total_pages,
        "toc": [toc_to_dict(t) for t in doc.toc],
        "chapters": [chapter_to_dict(c) for c in doc.chapters],
        "orphan_page_count": len(doc.orphan_pages),
        "orphans": [page_to_dict(p) for p in doc.orphan_pages],
        "metadata": doc.metadata,
    }
