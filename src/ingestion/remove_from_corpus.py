"""Corpus removal — take one document out of the ingested corpus.

The ingestion pipeline projects one source document onto several stores:

    data/sources/<ext>/<file>      the SOURCE (user scope — never touched)
    data/vector/...                the vector projection (chunks)
    data/cache/*_jobs.json         the checkpoint entries (extraction,
                                   summarization)
    data/extracted/<stem>.json     the canonical extracted content
    data/summarized/<stem>.json    the LLM summaries
    data/extracted/mineru/<stem>/  MinerU's own sandbox (PDFs only)

Removing a document from the corpus means cleaning **every projection**
while keeping the source file in the user's documents tree (data/sources
belongs to the user: Wintermute indexes it, it does not manage it).

Removal is driven by the unified identity scheme: the document id
``doc:<8hex>`` (SHA-256 of the lowercased filename with extension) keys the
vector deletion, and the file name keys the job-file entries and the
per-stem JSON files. Everything is id-derived, nothing is searched.

Design:

* **Partial-failure tolerant.** Every step deletes an independent
  projection. A failure in one (store missing, file locked...) is
  reported in the per-step report and the next step still runs — the user
  fixes one store without losing the report of the others. The final
  status is ``removed`` only when every step succeeded; otherwise it is
  ``partial`` with the failing steps and reasons listed.
* **Idempotent.** Removing a document that was never ingested (no chunks,
  no jobs, no JSONs) is a successful no-op: every step reports its
  zero-work and the status is ``removed``. A remove script re-run after a
  partial failure finishes the cleanup without erroring on the missing
  pieces.
* **The source file is never touched.** data/sources is the user's scope;
  this engine only ever *reads* it (to resolve stem/doc_id).

Entry point: :func:`remove_document` returns a structured report dict
(the scripts layer phrases it for the user; tests assert on it).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from src.extraction.ids import doc_id_from_filename
from src.tools.extraction_job_file import (
    ExtractionJobFile,
    SummarizationJobFile,
)

logger = logging.getLogger(__name__)

# Stable statuses of a removal report (scripts/tests route on these).
STATUS_REMOVED = "removed"    # every projection cleaned
STATUS_PARTIAL = "partial"    # at least one step failed; report says which
STATUS_REJECTED = "rejected"  # unusable request (bad reference)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def _remove_vector_projection(doc_id: str) -> Dict[str, Any]:
    """Delete every chunk of the document from both vector collections.

    Returns ``{"ok": bool, "deleted": <n>, "reason": str?}``.
    """
    deleted = 0
    for key in ("source_chunks", "knowledge_chunks"):
        try:
            from src.indexing.chroma_client import ChromaVectorClient

            client = ChromaVectorClient(key)
            deleted += client.delete_document(doc_id)
        except Exception as exc:  # noqa: BLE001 — reported, not raised
            return {
                "ok": False,
                "deleted": deleted,
                "reason": f"vector collection '{key}': {exc}",
            }
    return {"ok": True, "deleted": deleted}


def _remove_job_entries(file_name: str) -> Dict[str, Any]:
    """Remove the checkpoint entries (extraction + summarization job files)."""
    removed = []
    reasons = []
    for label, store in (
        ("extraction", ExtractionJobFile()),
        ("summarization", SummarizationJobFile()),
    ):
        try:
            if store.remove(file_name):
                removed.append(label)
        except Exception as exc:  # noqa: BLE001 — reported, not raised
            reasons.append(f"{label} job file: {exc}")
    if reasons:
        return {"ok": False, "removed": removed, "reason": "; ".join(reasons)}
    return {"ok": True, "removed": removed}


def _remove_json_files(source_path: Optional[Path], stem: str) -> Dict[str, Any]:
    """Remove the canonical and summarized JSONs for the document stem."""
    removed: list[str] = []
    reasons: list[str] = []

    targets: list[tuple[str, Path]] = []
    try:
        from src.helpers.document_extract_json_store import canonical_path_for

        targets.append(("extraction", canonical_path_for(Path(f"{stem}.pdf"))))
    except Exception as exc:  # noqa: BLE001 — reported, not raised
        reasons.append(f"canonical JSON path: {exc}")
    try:
        from src.summarization.summarized_store import summarized_path_for

        targets.append(("summarized", summarized_path_for(Path(f"{stem}.pdf"))))
    except Exception:
        pass  # config problem already reported via the canonical path

    for label, path in targets:
        try:
            if path.exists():
                path.unlink()
                removed.append(str(path))
        except OSError as exc:
            reasons.append(f"{label} JSON '{path.name}': {exc}")

    if reasons:
        return {"ok": False, "removed": removed, "reason": "; ".join(reasons)}
    return {"ok": True, "removed": removed}


def _remove_mineru_sandbox(source_path: Optional[Path], stem: str) -> Dict[str, Any]:
    """Remove MinerU's own working folder for the document, when present.

    Only PDFs have a MinerU sandbox; other formats simply have no folder
    here — absence is normal, not an error.
    """
    try:
        from src.extraction.mineru_pdf_extractor import _default_output_dir

        mineru_root = _default_output_dir()
    except Exception as exc:  # noqa: BLE001 — reported, not raised
        return {"ok": False, "removed": None, "reason": f"config: {exc}"}

    folder = mineru_root / stem
    if not folder.is_dir():
        return {"ok": True, "removed": None}
    try:
        shutil.rmtree(folder)
        return {"ok": True, "removed": str(folder)}
    except OSError as exc:
        return {"ok": False, "removed": None, "reason": f"MinerU folder '{folder}': {exc}"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def remove_document(
    reference: str,
    *,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Remove one document's whole footprint from the corpus.

    Args:
        reference: file name (``meow.pdf``) or absolute/relative path.
        source_path: optional pre-resolved source path (skips the
            documents-tree resolution); a caller that already located the
            file passes it here.

    Returns:
        A report dict::

            {
              "status": "removed" | "partial" | "rejected",
              "document": <name>,
              "doc_id": "doc:<8hex>",
              "steps": {
                 "vector":     {"ok": bool, "deleted": n, ...},
                 "job_files":  {"ok": bool, "removed": [...], ...},
                 "json_files": {"ok": bool, "removed": [...], ...},
                 "mineru":     {"ok": bool, "removed": str|None, ...},
              },
              "reason": str,   # only when partial/rejected
              "source_kept": True   # data/sources is the user's scope
            }
    """
    name = (reference or "").strip().strip("\"'")
    if not name or "/" in name or "\\" in name or ".." in name:
        # A bare name is expected; a path-like reference is resolved only
        # through source_path (the scripts may pass one explicitly).
        if source_path is None:
            return {
                "status": STATUS_REJECTED,
                "document": reference,
                "reason": "pass a bare file name (e.g. 'meow.pdf')",
            }

    resolved: Optional[Path] = None
    if source_path is not None:
        resolved = Path(source_path)
    else:
        try:
            from src.tools.ingest_tool import resolve_document

            resolution = resolve_document(name)
            if resolution.get("status") not in ("found",):
                return {
                    "status": STATUS_REJECTED,
                    "document": name,
                    "reason": (
                        f"document '{name}' not found in the documents tree "
                        f"({resolution.get('message', 'no such file')})"
                    ),
                }
            resolved = Path(str(resolution["path"]))
        except Exception as exc:  # noqa: BLE001 — reported, not raised
            return {
                "status": STATUS_REJECTED,
                "document": name,
                "reason": f"could not resolve the document: {exc}",
            }

    file_name = resolved.name
    doc_id = doc_id_from_filename(file_name)

    steps: Dict[str, Any] = {
        "vector": _remove_vector_projection(doc_id),
        "job_files": _remove_job_entries(file_name),
        "json_files": _remove_json_files(resolved, resolved.stem),
        "mineru": _remove_mineru_sandbox(resolved, resolved.stem),
    }

    failed = [label for label, report in steps.items() if not report.get("ok")]
    status = STATUS_REMOVED if not failed else STATUS_PARTIAL

    report: Dict[str, Any] = {
        "status": status,
        "document": file_name,
        "doc_id": doc_id,
        "steps": steps,
        "source_kept": True,
    }
    if failed:
        report["failed_steps"] = failed
        report["reason"] = "; ".join(
            f"{label}: {steps[label].get('reason', 'failed')}" for label in failed
        )
    logger.info(
        "Corpus removal of '%s' (%s): %s%s",
        file_name, doc_id, status,
        f" — {report['reason']}" if failed else "",
    )
    return report
