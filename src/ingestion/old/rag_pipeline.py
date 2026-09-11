# rag_pipeline.py
"""
Pipeline RAG minimal.

Étapes :
    1. Extract   → PDFExtractor
    2. Chunk     → RAGChunker          (étape suivante)
    3. Summarize → SectionSummarizer   (étape suivante, ex: appel LLM)
    4. Embed     → VectorEmbedder      (ex: OpenAI, sentence-transformers)
    5. Store     → VectorStore         (ex: Qdrant, Chroma, Pinecone)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from src.ingestion.old import rag_config

from src.ingestion.old.rag_models import DocumentExtract, Section
from src.extraction.rag_pdf_extractor import MineruPDFExtractor
from src.ingestion.old.rag_chunker import ChapterChunker
# Legacy note: the old client (src/llm/rag_llm_ollama.py) was replaced by
# LLMClientOllama, which satisfies the same complete(prompt, max_tokens) contract.
from src.llm.llm_client_ollama import LLMClientOllama as OllamaLLMClient
from src.ingestion.old.rag_summerizer import LLMSectionSummarizer


# ---------------------------------------------------------------------------
# Protocoles des étapes aval (contrats d'interface)
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbedderProtocol(Protocol):
    def embed(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ajoute un vecteur d'embedding à chaque chunk."""
        ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    def upsert(self, chunks: list[dict[str, Any]]) -> None:
        """Persiste les chunks vectorisés dans le store."""
        ...



# ---------------------------------------------------------------------------
# Pipeline orchestrateur
# ---------------------------------------------------------------------------

@dataclass
class RAGIngestionPipeline:
    """
    Orchestre les étapes Extract → Chunk → Summarize → Embed → Store.

    Seuls l'extractor et le chunker sont obligatoires.
    Les étapes summarizer, embedder et store sont optionnelles
    et peuvent être branchées progressivement.
    """
    extractor: MineruPDFExtractor = MineruPDFExtractor(Path(rag_config.DEFAULT_MINERU_FOLDER), True)
    chunker: ChapterChunker = field(default_factory=ChapterChunker)
    summarizer = LLMSectionSummarizer(llm_client=OllamaLLMClient(model=rag_config.OLLAMA_MODEL_SUMMARY, enable_thinking=False))
    embedder: EmbedderProtocol | None = None
    store: VectorStoreProtocol | None = None

    def run(self, pdf_path: str | Path) -> list[dict[str, Any]]:
        """
        Exécute le pipeline complet sur un fichier PDF.

        Args:
            pdf_path : chemin vers le PDF source

        Returns:
            Liste de chunks enrichis (avec vecteurs si embedder branché)
        """
        pdf_path = Path(pdf_path)

        # ── Étape 1 : Extraction ──────────────────────────────────────
        print(f"[1/4] Extraction PDF : {pdf_path.name}")
        document = self.extractor.extract(pdf_path)
        print(
            f"      → {document.total_pages} pages, "
            f"{len(document.chapters)} chapters, "
            f"{len(document.toc)} entrées TOC"
        )

        # ── Étape 2 : Résumés par section (optionnel) ─────────────────
        if self.summarizer:
            print(f"[2/4] Résumé de {len(document.chapters)} chapitres…")
            for chapter in document.chapters:
                chapter.summary = self.summarizer.summarize(chapter)
        else:
            print("[2/4] Résumé ignoré (pas de summarizer branché)")

        # ── Étape 3 : Chunking ────────────────────────────────────────
        print("[3/4] Chunking…")
        chunks = self.chunker.chunk(document)
        print(f"      → {len(chunks)} chunks générés")

        # ── Étape 4 : Embedding + stockage (optionnel) ────────────────
        if self.embedder:
            print("[4/4] Embedding…")
            chunks = self.embedder.embed(chunks)
            if self.store:
                print("      → Upsert dans le vector store…")
                self.store.upsert(chunks)
        else:
            print("[4/4] Embedding ignoré (pas d'embedder branché)")

        return chunks


# ---------------------------------------------------------------------------
# Point d'entrée CLI du pipeline complet
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys
    import logging
    from argparse import ArgumentParser

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    parser = ArgumentParser()
    parser.add_argument("-i", "--input", dest="input_file", required=True,
                        help="The file to load", metavar="FILE")

    args = parser.parse_args()

    pipeline = RAGIngestionPipeline()
    output_chunks = pipeline.run(args.input_file)