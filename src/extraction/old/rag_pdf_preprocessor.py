"""
pdf_ocr_preprocessor.py
-----------------------
Étape 0 du pipeline RAG :
  - Analyse un PDF pour déterminer s'il est scanné (pas/peu de texte)
  - Si oui, applique OCR via ocrmypdf et produit un PDF searchable
  - Retourne le chemin du PDF à utiliser dans la suite du pipeline
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
from argparse import ArgumentParser

import fitz          # pymupdf  → pip install pymupdf
import ocrmypdf      # pip install ocrmypdf
from ocrmypdf import OcrOptions

from PIL import Image
Image.MAX_IMAGE_PIXELS = 110000000 #100610624

DOSSIER_ARCHIVE = "./Sources/Archives"

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
logger = logging.getLogger(__name__)

# Seuil : si moins de N caractères en moyenne par page → considéré comme scan
CHARS_PER_PAGE_THRESHOLD = 50

# Langue(s) OCR Tesseract (ex: ["fra"], ["eng"], ["fra","eng"])
OCR_LANGUAGE = ["fra"]


# ─────────────────────────────────────────────
# 1. Analyse du PDF
# ─────────────────────────────────────────────
def analyze_pdf(pdf_path: Path) -> dict:
    """
    Analyse le contenu texte d'un PDF page par page.
    Retourne un dictionnaire avec :
      - total_pages       : nombre de pages
      - total_chars       : total de caractères extraits
      - avg_chars_per_page: moyenne par page
      - is_scanned        : bool — True si le PDF est probablement un scan
    """
    doc = fitz.open(str(pdf_path))
    total_chars = 0
    total_pages = len(doc)

    for page in doc:
        text = page.get_text("text")
        total_chars += len(text.strip())

    doc.close()

    avg = total_chars / total_pages if total_pages > 0 else 0

    result = {
        "total_pages": total_pages,
        "total_chars": total_chars,
        "avg_chars_per_page": round(avg, 1),
        "is_scanned": avg < CHARS_PER_PAGE_THRESHOLD,
    }

    logger.info(
        f"Analyse PDF : {total_pages} pages │ "
        f"{total_chars} caractères │ "
        f"~{avg:.1f} chars/page │ "
        f"{'⚠ SCAN détecté' if result['is_scanned'] else '✓ Texte présent'}"
    )
    return result

# ─────────────────────────────────────────────
# 2. Déplacement de fichier transformés
# ─────────────────────────────────────────────
import shutil
import os

def move_and_rename(file_a: Path, file_b: Path, folder_d: str) -> None:
    """
    Déplace le fichier A dans le dossier D, puis renomme le fichier B en A.

    Args:
        file_a   : Chemin complet vers le fichier A.
        file_b   : Chemin complet vers le fichier B.
        folder_d : Chemin vers le dossier de destination D.
    """
    print(f"moving {file_a.name} vers {folder_d}")

    # Résolution en chemins absolus dès l'entrée
    file_a      = file_a.resolve()
    file_b      = file_b.resolve()
    folder_d_path = Path(folder_d).resolve()

    if not file_a.is_file():
        print(f"Fichier A introuvable : {file_a}")
    if not file_b.is_file():
        print(f"Fichier B introuvable : {file_b}")

    folder_d_path.mkdir(parents=True, exist_ok=True)  # crée D si besoin

    print(f"miaou")
    if not file_a.is_file():
        print(f"Fichier A introuvable : {file_a}")
    print(f"ronron")
    if not file_b.is_file():
        print(f"Fichier B introuvable : {file_b}")
    print(f"pouet")
    if not folder_d_path.is_dir():
        os.makedirs(folder_d)
    print(f"grunt")
    print(f"on est passé partout")
    
    # --- Étape 1 : déplacer A dans D ---
    destination_a = folder_d_path / file_a.name
    shutil.copy2(file_a, destination_a)  # copie avec métadonnées
    file_a.unlink()                      # suppression explicite de la source

    # --- Étape 2 : renommer B en A ---
    file_b.rename(file_a)

    print(f"renamed")


# ─────────────────────────────────────────────
# 3. Application de l'OCR
# ─────────────────────────────────────────────
def apply_ocr(
    input_path: Path,
    output_path: Path,
    languages: str = OCR_LANGUAGE,
    deskew: bool = True,
) -> Path:
    """
    Applique ocrmypdf sur le PDF scanné.
    Produit un PDF visuellement identique avec couche texte invisible.

    Options notables :
      --skip-text     : ignore les pages qui ont déjà du texte
      --deskew        : corrige l'inclinaison des pages scannées
      --clean         : nettoie le bruit avant OCR (sans modifier l'image finale)
      --optimize 1    : compression légère sans perte visuelle
    """
    logger.info(f"OCR en cours : {input_path.name} → {output_path.name}")

    try:
        options = OcrOptions(
            input_file=str(input_path),
            output_file=str(output_path),
            languages=languages,
            deskew=deskew,
            skip_text=False,        # ré-OCRise pas les pages déjà textuelles
            #clean=True,            # améliore la qualité avant OCR (image inchangée) nécessite UNPAPER
            optimize=1,            # compression minimale, visuel intact
            progress_bar=True,
        )
        ocrmypdf.ocr(options)
        logger.info(f"✓ PDF searchable créé : {output_path}")
        move_and_rename(input_path, output_path, DOSSIER_ARCHIVE)
        return output_path

    except ocrmypdf.exceptions.PriorOcrFoundError:
        logger.warning("OCR déjà présent sur certaines pages — utilisation de --redo-ocr")
        options = OcrOptions(
            input_file=str(input_path),
            output_file=str(output_path),
            languages=languages,
            redo_ocr=True,        # ré-OCRise pas les pages déjà textuelles
            optimize=1,            # compression minimale, visuel intact
            progress_bar=True,
        )
        ocrmypdf.ocr(options)
        logger.info(f"✓ PDF searchable créé : {output_path}")
        move_and_rename(input_path, output_path, DOSSIER_ARCHIVE)
        return output_path

    except Exception as e:
        logger.error(f"Échec OCR : {e}")
        raise


# ─────────────────────────────────────────────
# 4. Point d'entrée du pipeline
# ─────────────────────────────────────────────
def preprocess_pdf_for_rag(
    pdf_path: str | Path,
    output_dir: Optional[str | Path] = None,
    force_ocr: bool = False,
) -> Path:
    """
    Fonction principale à appeler en étape 0 du pipeline RAG.

    Paramètres :
      pdf_path   : chemin du PDF source
      output_dir : dossier de sortie (défaut : même dossier que la source)
      force_ocr  : forcer l'OCR même si du texte est déjà présent

    Retourne :
      Path vers le PDF à utiliser dans la suite du pipeline
      (soit le PDF OCRisé, soit le PDF original si déjà textuel)
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    output_dir = Path(output_dir) if output_dir else pdf_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{pdf_path.stem}_searchable.pdf"

    # Analyse
    analysis = analyze_pdf(pdf_path)

    # Décision
    if not force_ocr and not analysis["is_scanned"]:
        logger.info("PDF déjà textuel — aucune OCR nécessaire, passage direct au pipeline.")
        return pdf_path

    # OCR
    logger.info(
        f"Lancement OCR "
        f"({'forcé' if force_ocr else f'scan détecté : ~{analysis[avg_chars_per_page]} chars/page'})"
    )
    return apply_ocr(pdf_path, output_path)


# ─────────────────────────────────────────────
# 5. Utilisation directe en CLI ou import
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", dest="input_file", required=True,
                        help="The file to load", metavar="FILE")
    parser.add_argument("-o", "--output", dest="output_folder", default=None,
                        help="Folder to save output", metavar="FOLDER")
    parser.add_argument("-f", "--force",  dest="force_ocr", action="store_true",
                        help="Force OCR for already OCRed files.")

    args = parser.parse_args()

    ready_pdf = preprocess_pdf_for_rag(args.input_file, args.output_folder, args.force_ocr)
    print(f"\nPDF prêt pour le RAG : {ready_pdf}")