"""Knowledge cache — per-document character extractions.

First persistence layer of the knowledge ingestion: after the
CharacterExtractionAgent's LLM passes, the extracted characters are stored
under ``knowledge_output_dir`` (config/ingestion.yaml, default
``data/cache/knowledge``) as ``<document stem>.json``.

The file is deliberately **plain** — a single
``{"characters": [...]}`` object, pretty-printed UTF-8, no envelope, no
fingerprint: it is a working cache of what the LLM found, meant to be
read (and hand-fixed) by the human before the knowledge validation,
check-n-merge and storage layers consume it. Identity and staleness are
the job of those later layers (via the extraction/summarized stores and
the unified doc ids), not of this cache.

Design mirrors the other project stores:

* config-driven folder (relative paths anchored to the project root,
  absolute passed through), fail-open to the default on a broken config —
  an extraction must not die because ingestion.yaml is unreadable;
* atomic writes (tempfile + os.replace) so a crash never leaves a
  half-written cache behind;
* defensive loading — malformed JSON raises
  :class:`KnowledgeJsonError` (a ``ValueError``) with an explicit path
  locator, so a hand-edit gone wrong is reported, not silently ignored.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

from src.tools.config_loader import PROJECT_ROOT

logger = logging.getLogger(__name__)

#: Fallback when ingestion.yaml does not set ``knowledge_output_dir``.
DEFAULT_KNOWLEDGE_OUTPUT_DIR = "data/cache/knowledge"

#: Default content unit fed to the knowledge LLM (see
#: ``knowledge_unit_granularity`` in ingestion.yaml). "section" is the
#: smallest unit still carrying heading context.
DEFAULT_GRANULARITY = "section"


class KnowledgeJsonError(ValueError):
    """A knowledge-cache JSON file is malformed."""


# ---------------------------------------------------------------------------
# Path resolution (config-driven, fail-open like every other store)
# ---------------------------------------------------------------------------

def knowledge_output_dir() -> Path:
    """Knowledge-cache folder from ingestion.yaml
    (``knowledge_output_dir``; default ``data/cache/knowledge``).

    Relative values are resolved against the project root; absolute values
    are used as-is. Fails open to the default folder on a broken config
    (knowledge extraction must keep working).
    """
    raw = DEFAULT_KNOWLEDGE_OUTPUT_DIR
    try:
        from src.tools.config_loader import load_ingestion_config

        value = load_ingestion_config().get("knowledge_output_dir")
        if isinstance(value, str) and value.strip():
            raw = value
    except Exception as exc:  # noqa: BLE001 — fail-open to the default folder
        logger.warning(
            "Could not read ingestion.yaml for knowledge_output_dir; "
            "using default %s (%s)", DEFAULT_KNOWLEDGE_OUTPUT_DIR, exc,
        )
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def knowledge_cache_path_for(source_path: Path) -> Path:
    """Knowledge-cache JSON path for a source document:
    ``<dir>/<stem>.json``."""
    return knowledge_output_dir() / f"{Path(source_path).stem}.json"


def knowledge_unit_granularity() -> str:
    """Content unit fed to the knowledge LLM from ingestion.yaml
    (``knowledge_unit_granularity``; default ``section``).

    The loader validates the value at load time, so a present value is one
    of section/page/chapter; the fail-open path only covers a broken or
    unreadable config.
    """
    try:
        from src.tools.config_loader import load_ingestion_config

        value = load_ingestion_config().get("knowledge_unit_granularity")
        if isinstance(value, str) and value.strip().lower() in ("section", "page", "chapter"):
            return value.strip().lower()
    except Exception as exc:  # noqa: BLE001 — fail-open to the default
        logger.warning(
            "Could not read ingestion.yaml for knowledge_unit_granularity; "
            "using default %r (%s)", DEFAULT_GRANULARITY, exc,
        )
    return DEFAULT_GRANULARITY


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_knowledge(characters: list, path: Path) -> Path:
    """Serialize the character payload to ``path`` (pretty UTF-8, atomic).

    Args:
        characters: list of character dicts as produced by the extraction
            agent (``full_name`` / ``short_name`` / ``aliases``).
        path: target ``<stem>.json`` path.

    Returns the written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"characters": list(characters)}
    tmp_name: str = ""
    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.stem}_", suffix=".tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(tmp_name, path)
    except OSError:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
        raise
    logger.debug("Knowledge cache saved: %s", path)
    return path


def load_knowledge(path: Path) -> Dict[str, Any]:
    """Load a knowledge-cache JSON file.

    Returns the raw payload dict (``{"characters": [...]}``).

    Raises:
        FileNotFoundError: the file does not exist.
        KnowledgeJsonError: the file is not valid JSON, is not an object,
            or its ``characters`` entry is not a list of objects — a
            hand-edit gone wrong must be reported, not ignored.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge cache not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise KnowledgeJsonError(
            f"{path}: invalid JSON (line {exc.lineno}, column {exc.colno}) — "
            "the file may have been edited by hand; fix it or delete it to "
            "trigger a fresh knowledge extraction."
        ) from exc
    if not isinstance(data, dict):
        raise KnowledgeJsonError(
            f"{path}: root must be an object, got {type(data).__name__}"
        )
    characters = data.get("characters")
    if not isinstance(characters, list) or any(
        not isinstance(entry, dict) for entry in characters
    ):
        raise KnowledgeJsonError(
            f"{path}: 'characters' must be a list of objects"
        )
    return data
