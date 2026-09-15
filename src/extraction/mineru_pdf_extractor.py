"""MinerU-backed PDF extractor.

Elaborated from the prototype ``src/extraction/rag_pdf_extractor.py``,
keeping its members, methods and pipeline logic while cleaning it up:

* ``extract()`` follows the ``DocumentExtractor`` template (validate →
  run backend → build); no more dead "error code" booleans;
* ``construct_pdf_content`` raises ``FileNotFoundError`` when MinerU's
  output JSON is missing instead of returning ``None`` (the protocol
  promises a ``DocumentExtract``, never a silent empty result);
* the output folder and JSON suffix come from config/ingestion.yaml
  (``extraction_mineru_output_dir``, ``mineru_json_extension``) — no more
  hard-coded defaults in the code;
* ``source_path`` now carries the full path (the prototype stored only
  the file stem);
* chapter ranges are cleaned: a chapter owns pages
  ``[toc_entry.page_number, next_entry.page_number)`` — the prototype's
  inclusive upper bound double-assigned boundary pages to two chapters;
* MinerU output blocks are sorted by page index before grouping (the
  prototype relied on the JSON being already ordered);
* page width/height (in fitz points) are read from the PDF via PyMuPDF —
  MinerU's JSON carries no page geometry.

Pipeline logic (kept from the prototype):

    MinerU subprocess (OCR/layout) → content_list JSON + PyMuPDF (TOC,
    metadata) → pages of blocks → sections per page (a block carrying a
    ``text_level`` starts a new section) → chapters aligned on the TOC,
    orphan pages before the first TOC entry grouped in a fallback chapter.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import subprocess
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from src.extraction.document_extractor import PDFExtractor
from src.extraction.models import (
    BlockType,
    Chapter,
    DocumentExtract,
    PageContent,
    Section,
    TextBlock,
    TocEntry,
)
from src.tools import config_loader
from src.tools.config_loader import PROJECT_ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MinerU hyphenation cleanup
# ---------------------------------------------------------------------------

# MinerU's layout analysis emits an ASCII STX control character (\x02) where
# a word was hyphenated across a line break, followed by the line break and
# indentation whitespace: "cre\x02 vacier" is really "crevacier". The data
# lives in MinerU's own JSON, so the cleanup happens at the single entry
# point where MinerU blocks enter our models.
STX = "\x02"


def strip_mineru_hyphenation(text: str) -> str:
    """Rejoin words MinerU hyphenated across line breaks and drop STX markers.

    MinerU marks a line-break hyphenation with a STX control character
    (``\x02``) before the break, e.g. ``cre\x02\nvacier``. We rejoin the
    fragments — ``crevacier`` — and strip any leftover marker (a bare STX
    with no hyphenation context) to keep the stored text clean.

    The rejoining rule (\x02 + any whitespace incl. line breaks → '') is
    applied only between a lower-case letter and a letter: this preserves
    legitimate dashes (``financiers — un``) and avoids gluing the second
    half onto the previous word when the break is mid-word in the source.
    """
    if STX not in text:
        return text
    # Primary rule: STX between two letters (with optional whitespace) is a
    # line-break hyphenation → rejoin without hyphen or space.
    text = re.sub(
        r"(?<=[a-zà-öø-ÿ])\x02\s*(?=[a-zà-öø-ÿ])",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Fallback: any leftover STX (no letter context, no hyphenation) is
    # layout noise → remove it and the whitespace it introduced.
    return text.replace(STX, "")


# Fallbacks used when the keys are missing from config/ingestion.yaml.
# The configuration is the intended source of truth — set them there.
MINERU_JSON_FILE_EXTENSION = "_content_list.json"
DEFAULT_MINERU_OUTPUT_DIR = "data/extracted/mineru"


def _default_output_dir() -> Path:
    """MinerU working folder from ingestion.yaml.

    ``extraction_mineru_output_dir`` is the intended key (MinerU owns its
    raw sandbox there; the canonical JSON extracted from it lives in
    ``extraction_output_dir`` — see src/extraction/json_store.py). The
    legacy ``extraction_output_dir`` key is honored as a fallback for
    configurations written before the two-store split. Relative values are
    resolved against the project root; absolute values are used as-is.
    """
    config = config_loader.load_ingestion_config()
    raw = config.get("extraction_mineru_output_dir")
    if not isinstance(raw, str) or not raw.strip():
        raw = config.get("extraction_output_dir", DEFAULT_MINERU_OUTPUT_DIR)
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _json_extension() -> str:
    """MinerU JSON file suffix from ingestion.yaml (``mineru_json_extension``)."""
    config = config_loader.load_ingestion_config()
    return str(config.get("mineru_json_extension", MINERU_JSON_FILE_EXTENSION))


class MineruPDFExtractor(PDFExtractor):
    """Concrete ``PDFExtractor`` based on MinerU.

    Config:
        mineru_folder: working folder where MinerU writes its output;
            defaults to ingestion.yaml ``extraction_mineru_output_dir``
            (relative values resolved against the project root);
        bypass_ocr: when True, skip the MinerU call and reuse the
            artifacts already present in ``mineru_folder`` (test mode).

    The JSON output suffix also comes from ingestion.yaml
    (``mineru_json_extension``), stored as ``self.json_extension``.
    """

    def __init__(
        self,
        mineru_folder: Optional[Path] = None,
        bypass_ocr: bool = False,
    ) -> None:
        self.bypass_ocr = bypass_ocr
        # Config-driven defaults (config/ingestion.yaml); an explicit
        # mineru_folder argument always wins.
        self.mineru_folder = Path(mineru_folder) if mineru_folder else _default_output_dir()
        self.json_extension = _json_extension()

    # ------------------------------------------------------------------
    # Template-method hooks (see DocumentExtractor)
    # ------------------------------------------------------------------

    def _run_backend(self, path: Path) -> None:
        """Run MinerU on the PDF unless OCR is bypassed (test mode)."""
        if not self.bypass_ocr:
            self.extract_pdf_content(path, self.mineru_folder)

    def _build_document(self, path: Path) -> DocumentExtract:
        """Build the DocumentExtract from MinerU output + PyMuPDF data."""
        return self.construct_pdf_content(path, self.mineru_folder)

    # ------------------------------------------------------------------
    # MinerU invocation
    # ------------------------------------------------------------------

    @staticmethod
    def extract_pdf_content(input_file: Path, output_folder: Path) -> None:
        """OCRize a PDF and extract its content using the layout analyzer.

        Raises:
            OSError: MinerU exited with a non-zero code.
        """
        logger.info("Running MinerU with file: '%s' and result in '%s'", input_file, output_folder)
        mineru_return = subprocess.call(
            [
                "mineru", "-p", str(input_file), "-o", str(output_folder),
                "-b", "pipeline", "--ocr", "--effort", "high",
            ],
            shell=True,
        )
        if mineru_return != 0:
            raise OSError(f"MinerU stopped with {mineru_return} error code.")

    # ------------------------------------------------------------------
    # Document assembly
    # ------------------------------------------------------------------

    def construct_pdf_content(self, file_path: Path, output_folder: Path) -> DocumentExtract:
        """Read the MinerU extraction and build the document structure.

        Raises:
            FileNotFoundError: MinerU output JSON or source PDF is missing.
        """
        filename = file_path.stem
        output_path = output_folder / filename / "auto"

        json_file_name = filename + self.json_extension
        json_path = output_path / json_file_name
        logger.debug("Looking for '%s' into '%s'!", json_file_name, output_path)

        if not json_path.exists():
            raise FileNotFoundError(
                f"MinerU output not found: {json_path}. "
                f"Run the extraction (or bypass_ocr with existing artifacts)."
            )

        with open(json_path, "r", encoding="utf-8") as json_file:
            pdf_data = json.load(json_file)

        with fitz.open(str(file_path)) as doc:
            toc = self.extract_toc(doc)
            metadata = self.extract_metadata(doc, file_path)
            # Page geometry (fitz points): MinerU's JSON carries no width/
            # height, so we read it from the PDF itself while it is open.
            geometry = self.page_geometry(doc)

        pages = self.extract_pages(pdf_data, geometry)
        chapters, orphans = self.build_chapters(toc, pages)

        logger.info(
            "Extraction done — %d pages, %d chapters, %d orphan pages",
            len(pages), len(chapters), len(orphans),
        )

        return DocumentExtract(
            source_path=str(file_path),
            title=metadata.get("title", file_path.stem),
            author=metadata.get("author", ""),
            subject=metadata.get("subject", ""),
            total_pages=len(pages),
            toc=toc,
            chapters=chapters,
            orphan_pages=orphans,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Pages and sections
    # ------------------------------------------------------------------

    def extract_pages(
        self, json_file: list[dict], geometry: Optional[dict[int, tuple[float, float]]] = None
    ) -> list[PageContent]:
        """Group MinerU blocks per page and build the page list.

        ``geometry`` maps 0-based page index → (width, height) in points;
        pages absent from it (e.g. geometry read failure) keep None.
        """
        # Sorted copy: groupby only groups *consecutive* identical keys.
        ordered = sorted(json_file, key=itemgetter("page_idx"))
        pages: dict[int, list[dict]] = {
            page_index: [self.transform_pagegroup(elem) for elem in group]
            for page_index, group in groupby(ordered, key=itemgetter("page_idx"))
        }

        sections = self.construct_sections(pages)

        pages_content = [
            self.extract_page(page_groups, sections, geometry or {})
            for page_groups in pages.values()
        ]

        logger.debug("Pages found: %d, sections found: %d", len(pages_content), len(sections))
        return pages_content

    @staticmethod
    def transform_pagegroup(page_json: dict) -> dict:
        """Hook invoked for each raw MinerU block (version shims go here).

        Text hygiene happens here so every downstream consumer — sections,
        pages, chapters, summaries, vector chunks — derives from clean text:
        MinerU's line-break hyphenation markers (STX, ``\x02``) are rejoined
        (``cre\x02 vacier`` → ``crevacier``) and stripped.
        """
        if "text" in page_json:
            page_json["text"] = strip_mineru_hyphenation(page_json["text"])
        if "table_body" in page_json:
            page_json["table_body"] = strip_mineru_hyphenation(page_json["table_body"])
        return page_json

    @staticmethod
    def construct_sections(pages_groups: dict[int, list[dict]]) -> list[Section]:
        """Group blocks into sections, page by page.

        A block carrying a ``text_level`` starts a new section on its page
        (it becomes the section title). A page that begins without any
        titled section continues the previous page's section and is flagged
        ``is_orphan``.
        """
        page_sections: list[Section] = []
        section_index = 0

        for page_index, page_groups in pages_groups.items():
            first_page_section = section_index
            section_id = 0
            block_index = 0
            new_section = Section(
                section_id=section_id, blocks=[], page_number=page_index + 1,
                bbox=(0, 0, 0, 0), section_title=None, section_level=0, raw_text="",
            )

            for group in page_groups:
                if group.get("type") not in ("text", "table"):
                    continue

                if group["type"] == "text" and "text" not in group:
                    logger.warning("TEXT group without text: %s", group)
                if group["type"] == "table" and "table_body" not in group:
                    logger.warning("TABLE group without body: %s", group)

                block = TextBlock(
                    block_id=block_index,
                    page_number=page_index + 1,
                    bbox=tuple(group["bbox"]),
                    raw_text=group.get("text", group.get("table_body", "<<ERROR:table/text>>")),
                    block_type=BlockType.TEXT if group["type"] == "text" else BlockType.TABLE,
                    text_level=0,
                )

                if "text_level" in group:
                    block.text_level = group["text_level"]
                    if len(new_section.blocks) != 0:
                        # Close the current section and open a titled one.
                        section_id += 1
                        page_sections.append(new_section)
                        section_index += 1
                        new_section = Section(
                            section_id=section_id, blocks=[], page_number=page_index + 1,
                            bbox=block.bbox, section_title=block.raw_text,
                            section_level=block.text_level, raw_text=block.raw_text + "\n",
                        )
                    else:
                        # First block of the page carries the section title.
                        new_section.raw_text = block.raw_text + "\n"
                        new_section.section_title = block.raw_text
                else:
                    new_section.raw_text = new_section.raw_text + block.raw_text + "\n"
                    new_section.blocks.append(block)
                    new_section.bbox = tuple(fitz.IRect(new_section.bbox) | fitz.IRect(block.bbox))

                block_index += 1

            page_sections.append(new_section)
            section_index += 1

            # A page starting with an untitled section continues the
            # previous page's section: flag it as orphan and inherit title.
            if page_sections[first_page_section].section_title is None and first_page_section > 0:
                page_sections[first_page_section].is_orphan = True
                page_sections[first_page_section].section_title = (
                    page_sections[first_page_section - 1].section_title
                )

        return page_sections

    @staticmethod
    def page_geometry(doc: fitz.Document) -> dict[int, tuple[float, float]]:
        """Read each page's (width, height) in points from the opened PDF.

        Fills the geometry MinerU's JSON lacks. Per-page failures are
        logged and skipped (the page keeps ``width=None``): geometry is
        advisory data, never worth failing an extraction over.
        """
        geometry: dict[int, tuple[float, float]] = {}
        for index, page in enumerate(doc):
            try:
                rect = page.rect
            except Exception as exc:  # noqa: BLE001 — per-page, advisory only
                logger.warning("No geometry for page %d: %s", index, exc)
                continue
            geometry[index] = (rect.width, rect.height)
        return geometry

    @staticmethod
    def extract_page(
        page_groups: list[dict],
        sections: list[Section],
        geometry: Optional[dict[int, tuple[float, float]]] = None,
    ) -> PageContent:
        """Build one PageContent and attach the sections belonging to it."""
        page_index = page_groups[0]["page_idx"] + 1 if len(page_groups) > 0 else 0
        width, height = (geometry or {}).get(page_index - 1, (None, None))
        page = PageContent(
            page_number=page_index,
            width=width,
            height=height,
            raw_text="\n".join(block["text"] for block in page_groups if block["type"] == "text"),
        )

        for section in sections:
            if section.page_number == page.page_number:
                page.sections.append(section)
            if section.page_number > page.page_number:
                break

        return page

    # ------------------------------------------------------------------
    # TOC, chapters, metadata
    # ------------------------------------------------------------------

    @staticmethod
    def build_chapters(
        toc: list[TocEntry], pages: list[PageContent]
    ) -> tuple[list[Chapter], list[PageContent]]:
        """Associate each page with its TOC chapter.

        Strategy: chapter ``i`` owns the pages in
        ``[toc[i].page_number, toc[i+1].page_number)``. Pages before the
        first TOC entry are orphans, grouped in a fallback chapter.
        """
        chapters: list[Chapter] = []
        page_map = {p.page_number: p for p in pages}
        total_pages = pages[-1].page_number if pages else 0

        for i, entry in enumerate(toc):
            next_start = toc[i + 1].page_number if i + 1 < len(toc) else total_pages + 1
            logger.debug("Entry p.%d: '%s', up to %d", entry.page_number, entry.title, next_start)

            chapter_pages = []
            for pnum in range(entry.page_number, next_start):
                if pnum in page_map:
                    pc = copy.copy(page_map[pnum])
                    pc.chapter_title = entry.title
                    chapter_pages.append(pc)
                else:
                    logger.warning("Looking for page %d but page_map doesn't have it.", pnum)

            full_text = "\n".join(p.raw_text for p in chapter_pages if p.raw_text)
            chapters.append(
                Chapter(
                    toc_entry=entry,
                    pages=chapter_pages,
                    full_text=full_text,
                    metadata={
                        "start_page": chapter_pages[0].page_number if chapter_pages else entry.page_number,
                        "end_page": chapter_pages[-1].page_number if chapter_pages else entry.page_number,
                        "page_count": len(chapter_pages),
                    },
                )
            )

        # Orphan pages: before the first TOC entry (or all pages without TOC).
        if toc:
            first_toc_page = toc[0].page_number
            orphans = [p for p in pages if p.page_number < first_toc_page]
        else:
            logger.warning("No TOC detected — all pages are orphans.")
            orphans = list(pages)

        # Orphans should not be copied in chapters as it creates issues with duplicates later.
        '''if orphans:
            chapters.append(
                Chapter(
                    toc_entry=TocEntry(
                        level=0,
                        title="Orphans",
                        page_number=orphans[0].page_number,
                        page_index=orphans[0].page_number - 1,
                    ),
                    pages=orphans,
                    full_text="\n\n".join(p.raw_text for p in orphans if p.raw_text),
                    metadata={
                        "start_page": orphans[0].page_number,
                        "end_page": orphans[-1].page_number,
                        "page_count": len(orphans),
                    },
                )
            )'''

        return chapters, orphans

    @staticmethod
    def extract_toc(doc: fitz.Document) -> list[TocEntry]:
        """Extract the table of contents via PyMuPDF.

        PyMuPDF returns the TOC as a list of triplets ``[level, title, page]``
        (page in 1-based index).
        """
        raw_toc = doc.get_toc(simple=True)  # [[level, title, page], ...]

        if not raw_toc:
            logger.warning("No TOC detected in the document.")
            return []

        entries = [
            TocEntry(
                level=level,
                title=title.strip(),
                page_number=page,
                page_index=page - 1,  # 0-based conversion for PyMuPDF
            )
            for level, title, page in raw_toc
        ]

        logger.debug("TOC extracted: %d entries", len(entries))
        return entries

    @staticmethod
    def extract_metadata(doc: fitz.Document, path: Path) -> dict:
        """Extract document metadata from the PDF."""
        raw = doc.metadata or {}
        return {
            "title": raw.get("title") or path.stem,
            "author": raw.get("author", ""),
            "subject": raw.get("subject", ""),
            "creator": raw.get("creator", ""),
            "producer": raw.get("producer", ""),
            "creation_date": raw.get("creationDate", ""),
            "modification_date": raw.get("modDate", ""),
            "page_count": doc.page_count,
            "encrypted": doc.is_encrypted,
        }
