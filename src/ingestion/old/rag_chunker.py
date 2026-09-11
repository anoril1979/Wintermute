"""
Chunker RAG.

Ce chunker va prendre le document et découper son contenu en blocs (chunks) pour pouvoir
les stocker par la suite.



"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from src.ingestion.old import rag_config

from src.ingestion.old.rag_models import DocumentExtract, Chapter, Section


# ---------------------------------------------------------------------------
# Protocoles des étapes aval (contrats d'interface)
# ---------------------------------------------------------------------------

@runtime_checkable
class ChunkerProtocol(Protocol):
    def chunk(self, document: DocumentExtract) -> list[dict[str, Any]]:
        """Découpe un DocumentExtract en chunks prêts à l'embedding."""
        ...


# ---------------------------------------------------------------------------
# Chunker par défaut — découpe par section + respect d'une taille max
# ---------------------------------------------------------------------------

@dataclass
class ChapterChunker:
    """
    Stratégie de chunking : une section = un chunk de base.
    Si le texte dépasse max_chars, il est subdivisé par paragraphes.
    """

    def chunk(self, document: DocumentExtract) -> list[dict[str, Any]]:
        chunks = []
        chunk_id = 0

        all_chapters = document.chapters

        # Les pages orphelines forment un chapitre virtuel
        if document.orphan_pages:
            orphan_text = "\n\n".join(
                p.raw_text for p in document.orphan_pages if p.raw_text
            )
            if orphan_text.strip():
                all_chapters_orphan_text = orphan_text
            else:
                all_chapters_orphan_text = None
        else:
            all_chapters_orphan_text = None

        for chapter in all_chapters:
            sub_chunks = self._split_text(chapter.full_text)
            for idx, text in enumerate(sub_chunks):
                chunks.append({
                    "chunk_id": f"{document.source_path}__s{chapter.start_page}_{idx}",
                    "source": document.source_path,
                    "doc_title": document.title,
                    "section_title": chapter.toc_entry.title,
                    "section_level": chapter.toc_entry.level,
                    "start_page": chapter.start_page,
                    "end_page": chapter.end_page,
                    "chunk_index": idx,
                    "text": text,
                    "summary": chapter.summary,
                    "metadata": chapter.metadata,
                })
                chunk_id += 1

        # Chunk des orphelins
        if all_chapters_orphan_text:
            for idx, text in enumerate(self._split_text(all_chapters_orphan_text)):
                chunks.append({
                    "chunk_id": f"{document.source_path}__orphan_{idx}",
                    "source": document.source_path,
                    "doc_title": document.title,
                    "section_title": "preamble",
                    "section_level": 0,
                    "start_page": 1,
                    "end_page": document.orphan_pages[-1].page_number,
                    "chunk_index": idx,
                    "text": text,
                    "summary": None,
                    "metadata": {},
                })

        return chunks

    def _split_text(self, text: str) -> list[str]:
        """Subdivise un texte long en segments avec overlap."""
        if len(text) <= rag_config.MAX_CHUNK_SIZE:
            return [text.strip()]

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = []
        current_len = 0

        for para in paragraphs:
            if current_len + len(para) > rag_config.MAX_CHUNK_SIZE and current:
                chunk_text = "\n\n".join(current)
                chunks.append(chunk_text)
                # Overlap : on garde le dernier paragraphe
                overlap_text = current[-1] if rag_config.OVERLAP_CHAR_SIZE > 0 else ""
                current = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)
            current.append(para)
            current_len += len(para)

        if current:
            chunks.append("\n\n".join(current))

        return chunks


if __name__ == "__main__":

    import json
    from argparse import ArgumentParser
    from pathlib import Path
    from src.extraction.rag_pdf_extractor import MineruPDFExtractor

    parser = ArgumentParser()
    parser.add_argument("-i", "--input", dest="input_file", required=True,
                        help="The file to load", metavar="FILE")
    parser.add_argument("-j", "--json", dest="output_file", default=None,
                        help="Output JSON file", metavar="FILE")

    args = parser.parse_args()

    extractor = MineruPDFExtractor(Path("mineru"), bypass_ocr=True)
    result = extractor.extract(Path(args.input_file))

    chunker = ChapterChunker()
    chunks = chunker.chunk(result)

    if args.output_file is not None:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=4)
            print(f"Résultat écrit dans : {args.output_file}")
    else:
        print(json.dumps(chunks, ensure_ascii=False, indent=4))
