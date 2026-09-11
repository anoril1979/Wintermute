# models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


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
    """Bloc de texte atomique extrait d'une page."""
    block_id: int
    page_number: int            # Index 1
    bbox: tuple[float, float, float, float]   # (x0, y0, x1, y1)
    raw_text: str
    summary: Optional[str] = None  # Résumé (produit par étape suivante du pipeline)
    block_type: BlockType = BlockType.TEXT
    text_level: int = 0

@dataclass
class Section:
    """Section/Partie d'une page."""
    section_id: int
    blocks: list[TextBlock]
    page_number: int            # Index 1
    bbox: tuple[float, float, float, float]   # (x0, y0, x1, y1)
    raw_text: str
    summary: Optional[str] = None  # Résumé (produit par étape suivante du pipeline)
    section_title: Optional[str] = None
    section_level: int = 0
    is_orphan: bool = False

@dataclass
class PageContent:
    """Contenu complet d'une page."""
    page_number: int            # Index 1
    width: float
    height: float
    raw_text: str               # Texte brut concatené de la page
    summary: Optional[str] = None  # Résumé (produit par étape suivante du pipeline)
    sections: list[Section] = field(default_factory=list)
    chapter_title: Optional[str] = None

@dataclass
class Chapter:
    """Section/Chapitre logique du document, alignée sur une entrée TOC."""
    toc_entry: TocEntry
    pages: list[PageContent] = field(default_factory=list)
    full_text: str = ""          # Texte brut agrégé du chapitre
    summary: Optional[str] = None  # Résumé (produit par étape suivante du pipeline)
    metadata: dict = field(default_factory=dict)

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
    """
    source_path: str
    title: str
    author: str
    subject: str
    total_pages: int
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