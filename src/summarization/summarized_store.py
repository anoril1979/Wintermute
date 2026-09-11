"""Summarized-content store — the LLM summaries persistence layer.

Home: ``src/summarization`` — everything owned by the summarization
process lives here (the store today; summarization agents/prompts may join
later).

After a successful summarization, the fully-summarized ``DocumentExtract``
is serialized under ``summarization_output_dir`` (config/ingestion.yaml,
default ``data/summarized``) as ``<document stem>.json``. LLM
summarization is expensive: when an ingestion fails at a later step and is
re-run, the summarizer resumes from this file and performs **zero LLM
calls**.

Staleness guard — the **content fingerprint**: the envelope records a
SHA-256 of the *content* (the serialized document with every ``summary``
field stripped). On resume the fingerprint of the current canonical
extraction is compared with the stored one:

* identical  → the summaries were computed from exactly this content:
  resume, no LLM call;
* different  → the canonical content changed (user fix, re-extraction,
  forced run...): the stored summaries are stale, they are discarded and
  the document is summarized again. Never served stale.

Design choices mirror the canonical extraction store
(src/helpers/document_extract_json_store.py):

* **Full fidelity** — the document body uses the same full-fidelity
  serialization, so the summarized file is a complete ``DocumentExtract``;
* **Human-readable** — pretty-printed UTF-8 JSON;
* **Defensive loading** — a malformed envelope raises
  :class:`SummarizedJsonError` (a ``ValueError``) with an explicit path
  locator.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

from src.helpers.document_extract_json_store import (
    document_extract_from_json_dict,
    document_extract_to_json_dict,
)
from src.extraction.models import DocumentExtract
from src.tools.config_loader import PROJECT_ROOT

logger = logging.getLogger(__name__)

#: Marker written into summarized files; tolerated absent on load (forward
#: compatibility), rejected when it names another schema.
SCHEMA_MARKER = "wintermute-summarized/1"

#: Fallback when ingestion.yaml does not set ``summarization_output_dir``.
DEFAULT_SUMMARIZATION_OUTPUT_DIR = "data/summarized"


class SummarizedJsonError(ValueError):
    """A summarized-content JSON file is malformed."""


# ---------------------------------------------------------------------------
# Content fingerprint (staleness guard)
# ---------------------------------------------------------------------------

def _strip_summaries(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive removal of ``summary`` and ``id`` keys from a serialized document.

    The fingerprint must describe the *content* only: identical content
    yields an identical fingerprint whether or not summaries were computed
    yet. ``id`` keys are stripped for the same reason — ids are identity,
    not content, so assigning ids (new extraction, or a file touched before
    the id scheme existed) must not invalidate already-computed summaries.
    """

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: clean(v) for k, v in node.items()
                    if k not in ("summary", "id")}
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node

    return clean(payload)


def content_fingerprint(doc: DocumentExtract) -> str:
    """SHA-256 (hex) of the document's content, summaries AND ids excluded.

    Stable across the summarization step itself: computed on the
    pre-summarization document or on the summarized one, it yields the
    same value — that is what makes the resume check reliable. Ids are
    excluded too (identity, not content): id assignment must never
    invalidate stored summaries.
    """
    payload = _strip_summaries(document_extract_to_json_dict(doc))
    digest_input = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


# ---------------------------------------------------------------------------
# Envelope (de)serialization
# ---------------------------------------------------------------------------

def summarized_document_to_json_dict(
    doc: DocumentExtract, fingerprint: str
) -> Dict[str, Any]:
    """Envelope for a summarized document: marker, fingerprint, full body."""
    return {
        "schema": SCHEMA_MARKER,
        "source_stem": Path(doc.source_path).stem if doc.source_path else "",
        "content_fingerprint": fingerprint,
        "document": document_extract_to_json_dict(doc),
    }


def summarized_document_from_json_dict(data: Any) -> Tuple[DocumentExtract, str]:
    """Rebuild ``(document, fingerprint)`` from :func:`summarized_document_to_json_dict` output.

    Raises:
        SummarizedJsonError: with an explicit path locator for every
            malformed entry (the file may have been edited by hand).
    """
    if not isinstance(data, dict):
        raise SummarizedJsonError(
            f"root: expected an object, got {type(data).__name__}"
        )
    marker = data.get("schema")
    if marker is not None and marker != SCHEMA_MARKER:
        raise SummarizedJsonError(
            f"root.schema: unexpected marker {marker!r} (expected {SCHEMA_MARKER!r})"
        )
    fingerprint = data.get("content_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise SummarizedJsonError(
            "root.content_fingerprint: missing or not a string — the file "
            "does not look like a summarized-content file"
        )
    if "document" not in data:
        raise SummarizedJsonError("root: missing required 'document'")
    try:
        document = document_extract_from_json_dict(data["document"])
    except ValueError as exc:
        # Re-raise under the summarized-store error type with a locator.
        raise SummarizedJsonError(f"root.document: {exc}") from exc
    return document, fingerprint


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_summarized(doc: DocumentExtract, path: Path) -> Path:
    """Serialize the summarized ``doc`` to ``path`` (pretty UTF-8, atomic).

    The content fingerprint is computed from the document itself. Returns
    the written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = summarized_document_to_json_dict(doc, content_fingerprint(doc))
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
    logger.debug("Summarized extraction saved: %s", path)
    return path


def load_summarized(path: Path) -> Tuple[DocumentExtract, str]:
    """Load a summarized-content JSON file into ``(document, fingerprint)``.

    Raises:
        FileNotFoundError: the file does not exist.
        SummarizedJsonError: the file is not valid JSON or violates the
            envelope schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Summarized content not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SummarizedJsonError(
            f"{path}: invalid JSON (line {exc.lineno}, column {exc.colno}) — "
            "the file may have been edited by hand; fix it or delete it to "
            "force re-summarization."
        ) from exc
    return summarized_document_from_json_dict(data)


# ---------------------------------------------------------------------------
# Path resolution (config-driven)
# ---------------------------------------------------------------------------

def summarization_output_dir() -> Path:
    """Summarized-content folder from ingestion.yaml
    (``summarization_output_dir``; default ``data/summarized``).

    Relative values are resolved against the project root; absolute values
    are used as-is. Fails open to the default folder on a broken config
    (summarization must keep working).
    """
    from src.tools.config_loader import load_ingestion_config

    raw = DEFAULT_SUMMARIZATION_OUTPUT_DIR
    try:
        config = load_ingestion_config()
        value = config.get("summarization_output_dir")
        if isinstance(value, str) and value.strip():
            raw = value
    except Exception as exc:  # noqa: BLE001 — fail-open to the default folder
        logger.warning(
            "Could not read ingestion.yaml for summarization_output_dir; "
            "using default %s (%s)", DEFAULT_SUMMARIZATION_OUTPUT_DIR, exc,
        )
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def summarized_path_for(source_path: Path) -> Path:
    """Summarized JSON path for a source document: ``<dir>/<stem>.json``."""
    return summarization_output_dir() / f"{Path(source_path).stem}.json"
