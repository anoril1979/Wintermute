"""
PDF Extractor with Page Layout Analysis

This tool tries to inspect and elaborate on the page layout of a PDF file for text organisation
in a natural reader order.

"""
import os
import copy
import json
from pathlib import Path
import subprocess
from itertools import groupby
from operator import itemgetter
from typing import Optional, Protocol, runtime_checkable

import fitz  # PyMuPDF
from src.ingestion.old import rag_config

from src.ingestion.old.rag_models import (
	BlockType,
	DocumentExtract,
	PageContent,
	Chapter,
	Section,
	TextBlock,
	TocEntry,
)

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Protocole : définit le contrat API de cette étape du pipeline
# ---------------------------------------------------------------------------

@runtime_checkable
class PDFExtractorProtocol(Protocol):
	"""
	Interface que tout extracteur PDF du pipeline doit respecter.
	Permet de swapper l'implémentation sans toucher aux étapes aval.
	"""

	def extract(self, pdf_path: str | Path) -> DocumentExtract:
		"""
		Lit un fichier PDF et retourne un DocumentExtract structuré.

		Args:
			pdf_path: Chemin absolu ou relatif vers le fichier .pdf

		Returns:
			DocumentExtract: objet structuré consommable par le pipeline

		Raises:
			FileNotFoundError: si le fichier est introuvable
			ValueError: si le fichier n'est pas un PDF valide
			RuntimeError: pour toute erreur d'extraction inattendue
		"""
		...

# ---------------------------------------------------------------------------
# Implémentation Mineru
# ---------------------------------------------------------------------------

class MineruPDFExtractor:
	"""
	Implémentation concrète de PDFExtractorProtocol basée sur MinerU.

	Paramètres de configuration :
		mineru_folder : chemin de travail pour MinerU
		bypass_ocr    : travail avec les fichiers disponibles sans phase d'OCR
	"""

	def __init__(
		self,
		mineru_folder: Path = None,
		bypass_ocr: bool = False,
	) -> None:
		self.bypass_ocr = bypass_ocr
		self.mineru_folder = mineru_folder

	# ------------------------------------------------------------------
	# Point d'entrée public (implémente le protocole)
	# ------------------------------------------------------------------

	def extract(self, input_file: str | Path) -> DocumentExtract:
		"""Analyze PDF and extract its data. Then return JSON/Markdown content for later use."""

		pdf_path = Path(input_file)
		if not pdf_path.exists():
			raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

		extraction_OK = self.extract_pdf_content(pdf_path, self.mineru_folder) if not self.bypass_ocr else 0
		if extraction_OK != 0:
			logger.error(f"Extraction of {pdf_path} failed. Check traces for more info (error code: {extraction_OK}.")
			#return False

		return self.construct_pdf_content(pdf_path, self.mineru_folder)


	# ------------------------------------------------------------------
	# Étapes internes
	# ------------------------------------------------------------------

	@staticmethod
	def extract_pdf_content(input_file: Path, output_folder: Path) -> bool:
		"""OCRize a PDF and extract its content using Page Layout Analyzer."""

		logger.info(f"Running MinerU with file: '{input_file}' and result in '{output_folder}'")
		mineru_return = subprocess.call(["mineru", "-p", input_file, "-o", output_folder, "-b", "pipeline", "--ocr", "--effort", "high"], shell=True)
		if mineru_return != 0:
			raise OSError(f"Mineru stopped with {mineru_return} error code.")

		return mineru_return;

	def construct_pdf_content(self, file_path: Path, output_folder: Path) -> DocumentExtract:
		"""Read PDF extraction and construct a PDF content structure"""

		filename = file_path.stem
		output_path = output_folder / filename / "auto"
		output_path.resolve()

		jsonFileName = filename + rag_config.MINERU_JSON_FILE_EXTENSION
		jsonPath = output_path / jsonFileName
		logger.debug(f"Looking for '{jsonFileName}' into '{output_path}' !")

		if not jsonPath.exists():
			logger.error(f"Unable to find file {jsonPath}.")
			return None

		with open(jsonPath, 'r', encoding='utf-8') as jsonFile:
			pdf_data = json.load(jsonFile)

		if not file_path.exists():
			logger.error(f"Unable to find file {file_path}.")
			return None

		with fitz.open(str(file_path)) as doc:
			toc = self.extract_toc(doc)
			metadata = self.extract_metadata(doc, file_path)
		pages = self.extract_pages(pdf_data)
		chapters, orphans = self.build_chapters(toc, pages)

		logger.info(
			"Extraction terminée — %d pages, %d chapters, %d pages orphelines",
			len(pages),
			len(chapters) if chapters is not None else 0,
			len(orphans) if orphans is not None else 0,
		)

		return DocumentExtract(
			source_path=str(filename),
			title=metadata.get("title", file_path.stem),
			author=metadata.get("author", ""),
			subject=metadata.get("subject", ""),
			total_pages=len(pages),
			toc=toc,
			chapters=chapters,
			orphan_pages=orphans,
			metadata=metadata,
		)

	@staticmethod
	def extract_page(page_groups: list[dict], sections: list[Section]) -> PageContent:
		page = PageContent(
			page_number=page_groups[0]["page_idx"]+1 if len(page_groups)>0 else 0,
			width=None,
			height=None,
			raw_text='\n'.join(block["text"] for block in page_groups if block["type"]=="text"),
		)

		for section in sections:
			if section.page_number == page.page_number:
				page.sections.append(section)
			if section.page_number > page.page_number:
				break

		return page

	@staticmethod
	def transform_pagegroup(page_json: dict) -> dict:
		return page_json

	def extract_pages(self, jsonFile: dict) -> list[PageContent]:

		pages = {
			page_index: [self.transform_pagegroup(elem) for elem in group]
			for page_index, group in groupby(jsonFile, key=itemgetter("page_idx"))
		}

		sections = self.construct_sections(pages)

		pages_content = []
		for page in pages.values():
			pages_content.append(self.extract_page(page, sections))

		logger.debug(f"Pages found: {len(pages_content)}, section found: {len(sections)}")
		return pages_content

	@staticmethod
	def construct_sections(pages_groups: dict) -> list[Section]:
		previousPageSectionIndex = None
		pageSections = []
		sectionIndex = 0

		for pageIndex, pageGroups in pages_groups.items():
			# Navigate a page
			firstPageSection = sectionIndex
			sectionId = 0
			blockIdx = 0
			newSection = Section(section_id=sectionId,blocks=[],page_number=pageIndex+1,bbox=[0,0,0,0],section_title=None,section_level=0,raw_text="")
			for group in pageGroups:
				if group["type"] != "text" and group["type"] != "table":
					continue

				if group["type"] == "text" and "text" not in group:
					logger.warning(f"Groupe TEXT sans texte: {group}")

				if group["type"] == "table" and "table_body" not in group:
					logger.warning(f"Groupe TABLE sans body: {group}")


				block = TextBlock(
					block_id=blockIdx,
					page_number=pageIndex+1,
					bbox=group["bbox"],
					raw_text=group["text"] if "text" in group else (group["table_body"] if "table_body" in group else "<<ERROR:table/text>>"),
					block_type=BlockType.TEXT if group["type"] == "text" else BlockType.TABLE,
					text_level=0
				)

				if "text_level" in group:
					block.text_level = group["text_level"]
					if len(newSection.blocks) != 0:
						sectionId = sectionId +1
						pageSections.append(newSection);
						sectionIndex = sectionIndex+1
						newSection = Section(section_id=sectionId,blocks=[],page_number=pageIndex+1,bbox=block.bbox,section_title=block.raw_text,section_level=block.text_level,raw_text=block.raw_text+"\n")
					else:
						newSection.raw_text = block.raw_text+"\n"
						newSection.section_title = block.raw_text
				else:
					newSection.raw_text = newSection.raw_text+block.raw_text+"\n"
					newSection.blocks.append(block)
					newSection.bbox = tuple(fitz.IRect(newSection.bbox) | fitz.IRect(block.bbox))

				blockIdx = blockIdx+1

			pageSections.append(newSection)

			sectionIndex = sectionIndex +1

			if pageSections[firstPageSection].section_title is None and firstPageSection > 0:
				pageSections[firstPageSection].is_orphan = True # Attached to previous page/chapter
				pageSections[firstPageSection].section_title = pageSections[firstPageSection-1].section_title

		return pageSections

	@staticmethod
	def build_chapters(toc: list[TocEntry], pages: list[PageContent]) -> tuple[list[Chapter], list[PageContent]]:
		"""
		Associe chaque page à son chapter TOC.

		Stratégie : pour chaque entrée TOC[i], les pages comprises entre
		toc[i].page_number et toc[i+1].page_number - 1 (inclus) lui appartiennent.
		Les pages antérieures à la première entrée TOC deviennent des orphelins.
		"""
		#if not toc:
			# Pas de TOC : toutes les pages sont orphelines
			#logger.warning("Pas de TOC — toutes les pages sont orphelines.")
			#return [], list(pages)

		chapters: list[Chapter] = []
		page_map = {p.page_number: p for p in pages}
		total_pages = pages[-1].page_number if pages else 0

		for i, entry in enumerate(toc):
			next_start = toc[i + 1].page_number if i + 1 < len(toc) else total_pages + 1
			logger.debug(f"Entry p.{entry.page_number}: '{entry.title}', up to {next_start+1}")

			chapter_pages = []
			for pnum in range(entry.page_number, next_start+1):
				if pnum in page_map:
					pc = copy.copy(page_map[pnum])
					if pnum <= next_start or pc.sections[0].is_orphan:
						pc.chapter_title = entry.title
						chapter_pages.append(pc)
					else:
						logger.debug(f"Page {pnum} is out-of-range [{next_start}] and it's first section is not orphan ({pc.sections[0].section_title})")
				else:
					logger.warning(f"Looking for page {pnum} but page_map don't have it.")

			full_text = "\n".join(p.raw_text for p in chapter_pages if p.raw_text)
			chapters.append(
				Chapter(
					toc_entry=entry,
					pages=chapter_pages,
					full_text=full_text,
					metadata={
						"start_page": chapter_pages[0].page_number if len(chapter_pages) > 0 else entry.page_number,
						"end_page": chapter_pages[-1].page_number if len(chapter_pages) > 0 else entry.page_number,
						"page_count": len(chapter_pages),
					},
				)
			)

		# Pages orphelines : avant la première entrée TOC
		if len(toc) > 0:
			first_toc_page = toc[0].page_number
			orphans = [p for p in pages if p.page_number < first_toc_page]
		else:
			orphans = list(pages)
		if len(orphans) > 0:
			# Add orphans as a Chapter
			chapters.append(
				Chapter(
					toc_entry=TocEntry(
						level=0,
    					title="Orphelines",
    					page_number=orphans[0].page_number if len(orphans) > 0 else 1,
    					page_index=orphans[0].page_number-1 if len(orphans) > 0 else 0,
    				),
					pages=orphans,
					full_text="\n\n".join(p.raw_text for p in orphans if p.raw_text),
					metadata={
						"start_page": orphans[0].page_number if len(orphans) > 0 else 1,
						"end_page": orphans[-1].page_number if len(orphans) > 0 else 1,
						"page_count": len(orphans),
					},
				)
			)

		return chapters, orphans

	@staticmethod
	def extract_toc(doc: fitz.Document) -> list[TocEntry]:
		"""
		PyMuPDF retourne le TOC sous forme de liste de triplets :
		[level, title, page]  (page en index 1)
		"""
		raw_toc = doc.get_toc(simple=True)  # [[level, title, page], ...]

		if not raw_toc:
			logger.warning("Aucun TOC détecté dans le document.")
			return []

		entries = []
		for level, title, page in raw_toc:
			entries.append(
				TocEntry(
					level=level,
					title=title.strip(),
					page_number=page,
					page_index=page - 1,  # conversion 0-indexed pour PyMuPDF
				)
			)

		logger.debug("TOC extrait : %d entrées", len(entries))
		return entries

	@staticmethod
	def extract_metadata(doc: fitz.Document, path: Path) -> dict:
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


# ---------------------------------------------------------------------------
# Sérialisation utilitaire (pour debug / persistance inter-étapes)
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

	def chapter_to_dict(s: Chapter) -> dict:
		return {
			"title": s.toc_entry.title,
			"level": s.toc_entry.level,
			"start_page": s.start_page,
			"end_page": s.end_page,
			"page_count": len(s.pages),
			"full_text_length": len(s.full_text),
			"summary": s.summary,
			"pages": [page_to_dict(p) for p in s.pages],
			"metadata": s.metadata,
		}

	return {
		"source_path": doc.source_path,
		"title": doc.title,
		"author": doc.author,
		"subject": doc.subject,
		"total_pages": doc.total_pages,
		"toc": [toc_to_dict(t) for t in doc.toc],
		"chapters": [chapter_to_dict(s) for s in doc.chapters] if doc.chapters is not None else None,
		"orphan_page_count": len(doc.orphan_pages) if doc.orphan_pages is not None else 0,
		"orphans": [page_to_dict(p) for p in doc.orphan_pages] if doc.orphan_pages is not None else None,
		"metadata": doc.metadata,
	}



if __name__ == "__main__":
	from argparse import ArgumentParser

	parser = ArgumentParser()
	parser.add_argument("-i", "--input", dest="input_file", required=True,
						help="The file to load", metavar="FILE")
	parser.add_argument("-o", "--output", dest="output_folder", default="mineru",
						help="Folder to save output", metavar="FOLDER")
	parser.add_argument("-j", "--json", dest="output_file", default=None,
						help="Output JSON file", metavar="FILE")
	parser.add_argument("-t", "--test", dest="test", action="store_true",
						help="Test mode: do not perform OCR")

	args = parser.parse_args()

	extractor = MineruPDFExtractor(Path(args.output_folder), args.test)

	result = extractor.extract(Path(args.input_file))

	serialized = document_extract_to_dict(result)

	if args.output_file is not None:
		with open(args.output_file, "w", encoding="utf-8") as f:
			json.dump(serialized, f, ensure_ascii=False, indent=4)
			print(f"Résultat écrit dans : {args.output_file}")
	else:
		print(json.dumps(serialized, ensure_ascii=False, indent=4))
