"""ingest_tool.py — Tool exposed to the LLM orchestrator: ingest a document.

Contract (for the future orchestrator):
    1. The orchestrator receives an intent such as
       "Please ingest the new 'meow.pdf'" and calls
       ``ingest_document("meow.pdf")``.
    2. The tool resolves the document SANDBOXED inside the documents root
       (config/ingestion.yaml: ``documents_root``), in the subfolder
       mapped to the file's extension (``extensions`` mapping). The LLM
       can only reference file names, never arbitrary paths.
    3. It returns a structured result the orchestrator can forward to the
       LLM so it phrases the final user-facing reply:
           - {"status": "ready", ...}   -> found, ingestion not wired yet
           - {"status": "ingested", ...}-> success (not wired yet)
           - {"status": "no_file", ...} -> orchestrator answers "no file";
              the LLM then tells the user e.g. "I'm sorry dude, but I
              can't find that file for ingestion."
           - {"status": "invalid_reference", ...} -> unusable input.
           - {"status": "unsupported_extension", ...} -> extension not in
              the ingestion.yaml mapping.

NOTE: the actual ingestion execution is intentionally NOT wired yet —
see ``_execute_ingestion``. The import of the ingestion workflow is
already in place but unused, as requested.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from src.tools import config_loader
from src.tools.config_loader import PROJECT_ROOT

# Master switch: flip to True (and implement _execute_ingestion) once the
# ingestion workflow is ready to be triggered by the orchestrator.
# The actual execution entry point now lives in
# src/ingestion/ingestion_orchestrator.py (run_ingestion / run_ingestion_file).
EXECUTE_INGESTION = False

# Result statuses (stable strings the orchestrator can route on)
STATUS_READY = "ready"                          # document found but not ingested
STATUS_INGESTED = "ingested"                    # document found and ingestion ran
STATUS_NO_FILE = "no_file"                      # document not found -> "nope"
STATUS_INVALID_REFERENCE = "invalid_reference"  # unusable document argument
STATUS_UNSUPPORTED_EXTENSION = "unsupported_extension"  # extension not mapped


def _load_config() -> Tuple[Path, Dict[str, str]]:
    """Return (documents_root, extensions) from config/ingestion.yaml.

    documents_root is resolved against the project root when relative.
    """
    config = config_loader.load_ingestion_config()
    root = Path(config["documents_root"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    extensions = {
        str(ext).lower(): str(folder)
        for ext, folder in config["extensions"].items()
    }
    return root, extensions


def _sanitize(document: str) -> Optional[str]:
    """Reduce a document reference to a bare file name, or None.

    Rejects anything that smells like a path traversal or an absolute
    path: the LLM must only ever pass a file *name*.
    """
    if not isinstance(document, str):
        return None
    name = document.strip().strip("\"'`")
    if not name:
        return None
    if "/" in name or "\\" in name or ".." in name or ":" in name:
        return None
    return name


# ---------------------------------------------------------------------------
# Candidate promotion (fuzzy matching against the extension's subfolder)
# ---------------------------------------------------------------------------

# Word tokens for similarity: letters, digits and '#' (issue numbers), so
# "(Gazette #1).pdf" tokenizes like "gazette #1".
_WORD_RE = re.compile(r"[\w#]+", re.UNICODE)


def _similar_names(
    reference: str,
    available: List[str],
    *,
    max_candidates: int = 5,
    min_score: float = 0.5,
) -> List[str]:
    """Rank ``available`` file names by similarity to ``reference``.

    Tolerates the two nomenclature failures an LLM (or a user) produces:

    * **fuzzy distance** — truncations, typos, missing characters (the
      analyzer asking for ``... (Gazette.pdf`` when the file is
      ``... (Gazette #1).pdf``);
    * **word-level recall** — a name that shares whole words with the
      reference (``Gazette-17.pdf`` matching "Dark Earth Gazette") even
      when character distance is high.

    The score is ``max(fuzzy_ratio, word_overlap, containment)``; results
    below ``min_score`` are dropped, ties keep alphabetical order.
    Deterministic: no randomness, so a given request always proposes the
    same list. Containment (fraction of the reference's words found in the
    candidate) is weighted slightly below 1.0 so a title-style partial
    reference ("Gazette #1") matches files containing those words even
    though the reference is much shorter than the file name. Word tokens
    keep ``#`` (issue numbers) but drop other punctuation, so ``(Gazette
    #1).pdf`` tokenizes like ``Gazette #1``.
    """
    ref = reference.lower()
    ref_words = set(_WORD_RE.findall(ref))
    if not ref_words:
        return []
    scored: List[tuple] = []
    for name in available:
        candidate = name.lower()
        fuzzy = difflib.SequenceMatcher(None, " ".join(ref.split()),
                                        " ".join(candidate.split())).ratio()
        words = set(_WORD_RE.findall(candidate))
        overlap = (
            len(ref_words & words) / len(ref_words | words)
            if ref_words | words else 0.0
        )
        containment = (
            len(ref_words & words) / len(ref_words)
            if ref_words else 0.0
        )
        score = max(fuzzy, overlap, 0.9 * containment)
        if score >= min_score:
            scored.append((-score, name))
    scored.sort()
    return [name for _, name in scored[:max_candidates]]


def _find_in_documents(name: str, documents_root: Path, extensions: Dict[str, str]) -> Optional[Path]:
    """Locate ``name`` inside the documents root, in the subfolder mapped
    to its extension. Returns the resolved path or None."""
    suffix = Path(name).suffix.lower()
    folder_name = extensions.get(suffix)
    if folder_name is None:
        return None  # extension not mapped -> unsupported

    target_dir = documents_root / folder_name
    if not target_dir.is_dir():
        return None

    # 1. exact file name
    exact = target_dir / name
    if exact.is_file():
        return exact
    # 2. case-insensitive file name
    lowered = name.lower()
    for path in sorted(target_dir.iterdir()):
        if path.is_file() and path.name.lower() == lowered:
            return path
    # 3. name without extension (e.g. "meow" -> "meow.pdf")
    if "." not in lowered:
        for path in sorted(target_dir.iterdir()):
            if path.is_file() and path.stem.lower() == lowered:
                return path
    return None


def _execute_ingestion(path: Path) -> dict:
    """Trigger the real ingestion workflow — INTENTIONALLY NOT WIRED YET.

    The actual execution now lives in the ingestion orchestrator; when this
    tool is finally allowed to trigger ingestion, the stub becomes::

        from src.ingestion.ingestion_orchestrator import run_ingestion_file
        return run_ingestion_file(path)
    """
    raise NotImplementedError(
        "Ingestion execution is not wired yet; only file resolution is active."
    )


# TODO: ingest_document is supposed to be a straightforward tool but it is
# reduced to a file existence checked. Would be renamed accordingly.
def ingest_document(
    document: str,
    *,
    documents_root: Optional[Union[str, Path]] = None,
    extensions: Optional[Dict[str, str]] = None,
) -> dict:
    """Resolve a document by name and trigger its ingestion.

    Args:
        document: File name as referenced by the LLM, e.g. ``"meow.pdf"``
            (quotes tolerated). Paths and traversal are rejected.
        documents_root: Override of the documents root (tests / tooling
            only); defaults to ingestion.yaml ``documents_root``.
        extensions: Override of the extension->subfolder mapping (tests /
            tooling only); defaults to ingestion.yaml ``extensions``.

    Returns:
        A dict the orchestrator can hand back to the LLM::

            {
                "status": "ready" | "ingested" | "no_file"
                          | "invalid_reference" | "unsupported_extension",
                "document": <requested name>,
                "path": <resolved path, when found>,
                "message": <human-readable hint>,
                "candidates": <similar file names, when not found>,
            }
    """
    name = _sanitize(document)
    if name is None:
        return {
            "status": STATUS_INVALID_REFERENCE,
            "document": document,
            "message": "invalid document reference; pass a bare file name.",
        }

    if documents_root is None or extensions is None:
        cfg_root, cfg_ext = _load_config()  # partial override: fill the gaps
        root = Path(documents_root) if documents_root is not None else cfg_root
        ext_map = extensions if extensions is not None else cfg_ext
    else:
        root, ext_map = Path(documents_root), extensions

    suffix = Path(name).suffix.lower()
    if suffix not in ext_map:
        known = sorted(ext_map)
        return {
            "status": STATUS_UNSUPPORTED_EXTENSION,
            "document": name,
            "message": f"extension '{suffix or '(none)'}' is not ingestable "
                       f"(supported: {', '.join(known)})",
        }

    path = _find_in_documents(name, root, ext_map)

    if path is None:
        result = {
            "status": STATUS_NO_FILE,
            "document": name,
            "message": "No file with that exact name found. Check candidates.",
        }
        # Offer similar names from the extension's subfolder.
        folder = root / ext_map[suffix]
        if folder.is_dir():
            candidates = _similar_names(
                name, [p.name for p in folder.iterdir() if p.is_file()]
            )
            if candidates:
                result["candidates"] = candidates
        return result

    if not EXECUTE_INGESTION:
        # Dry-run: the document exists, ingestion is stubbed.
        return {
            "status": STATUS_READY,
            "document": name,
            "path": str(path),
            "message": "document found; ingestion is not wired yet (dry-run)",
        }

    result = _execute_ingestion(path)
    result.setdefault("document", name)
    result.setdefault("path", str(path))
    return result
