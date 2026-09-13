"""ingest_tool.py — sandboxed document resolution for the documents tree.

The single place that knows how a document reference maps onto the
documents root (config/ingestion.yaml): the extension→subfolder mapping,
exact/case-insensitive/stem lookup, and the fuzzy candidate promotion for
near-miss names.

Since the paradigm change, ingestion is NOT triggered from the chat: the
routing layer never ingests, and this module no longer executes anything —
it only **resolves** references. The ingestion itself runs through
scripts/ingest.py → src/ingestion/ingestion_orchestrator (deterministic,
non-zero exit code on failure), and corpus removal through
scripts/remove.py → src/ingestion/remove_from_corpus.

Callers: the routing graph (origin gate fact-check), the corpus removal
engine, and the CLI scripts.

Resolution result (a plain dict)::

    {"status": "found", "document": ..., "path": ...}
    {"status": "not_found", "document": ..., "candidates": [...], "message": ...}
    {"status": "invalid_reference", "document": ..., "message": ...}
    {"status": "unsupported_extension", "document": ..., "message": ...}
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from src.tools import config_loader
from src.tools.config_loader import PROJECT_ROOT

# Result statuses (stable strings callers route on)
STATUS_FOUND = "found"                          # document resolved
STATUS_NO_FILE = "not_found"                    # document not found
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


def resolve_document(
    document: str,
    *,
    documents_root: Optional[Union[str, Path]] = None,
    extensions: Optional[Dict[str, str]] = None,
) -> dict:
    """Resolve a document reference to its path in the documents tree.

    Args:
        document: file name, e.g. ``"meow.pdf"`` (quotes tolerated). Paths
            and traversal are rejected — references are bare names.
        documents_root: Override of the documents root (tests / tooling
            only); defaults to ingestion.yaml ``documents_root``.
        extensions: Override of the extension->subfolder mapping (tests /
            tooling only); defaults to ingestion.yaml ``extensions``.

    Returns:
        A dict::

            {
                "status": "found" | "not_found"
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

    return {
        "status": STATUS_FOUND,
        "document": name,
        "path": str(path),
        "message": "document found",
    }
