"""Corpus clearing — erase every GENERATED artifact, keep the sources.

The one-shot reset behind ``scripts/clear.py``. Where
:mod:`src.ingestion.remove_from_corpus` cleans one document's projections,
this module wipes the whole generated surface in one pass:

    data/vector/                        the vector store (file-level erase:
                                        the whole folder is removed)
    data/extracted/                     canonical extracted JSONs
    data/extracted/mineru/              MinerU's working sandbox
    data/summarized/                    the LLM summaries
    data/cache/consolidation/           the consolidation cache
    data/cache/knowledge/               the knowledge cache JSONs
    data/cache/*_jobs.json              the checkpoint job files
    data/knowledge/                     the markdown knowledge base
                                        (entities, indexers, source
                                        registrations)

What is NEVER touched: the project code, the configuration, the source
documents (``documents_root`` — the user's scope) and the logs.

Design (same contract as :mod:`remove_from_corpus`):

* **Config-driven.** Every path comes from ``ingestion.yaml`` /
  ``setup.yaml`` through :func:`resolve_clear_paths`; the module never
  hardcodes a location.
* **Root-safe.** A resolved target must sit strictly INSIDE the project
  root (or, for an absolute configured path, outside the project root is
  refused too — the engine never deletes above its station). The
  documents root is additionally protected by name.
* **Partial-failure tolerant.** Each target is independent; a locked
  folder or an unreadable config yields a failed step in the report, the
  other steps still run, and the final status is ``cleared`` only when
  every step succeeded.
* **Idempotent.** Clearing an already-clear corpus is a successful
  no-op — a re-run after a partial failure finishes the job.

Entry point: :func:`clear_corpus` returns a structured report dict (the
script layer phrases it for the user; tests assert on it).
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Stable statuses of a clear report (scripts/tests route on these).
STATUS_CLEARED = "cleared"  # every generated surface erased
STATUS_PARTIAL = "partial"  # at least one step failed; report says which


@dataclass(frozen=True)
class ClearPaths:
    """Every generated surface the clear targets, already resolved.

    ``project_root`` is the safety anchor of the root-s guards: the real
    project root in production, a temp root in the hermetic tests.
    """

    project_root: Path
    vector_store: Path
    extraction_store: Path
    mineru_store: Path
    summarized_store: Path
    consolidation_cache: Path
    knowledge_cache: Path
    extraction_job_file: Path
    summarization_job_file: Path
    knowledge_base: Path
    documents_root: Path  # protected — reported, never deleted


def resolve_clear_paths() -> ClearPaths:
    """Resolve every clear target from the configs; raise ConfigError on breakage.

    Ingestion.yaml carries all the store/job/knowledge paths; setup.yaml
    carries the vector store's. A broken config must REFUSE to clear
    (wiping half-mapped targets on a misread yaml would be a disaster).
    """
    from src.tools.config_loader import (
        PROJECT_ROOT,
        load_ingestion_config,
        load_vector_config,
    )

    def _anchor(raw: str) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else PROJECT_ROOT / path

    ingestion = load_ingestion_config()
    vector = load_vector_config()
    return ClearPaths(
        project_root=Path(PROJECT_ROOT),
        vector_store=_anchor(str(vector["path"])),
        extraction_store=_anchor(str(ingestion["extraction_output_dir"])),
        mineru_store=_anchor(str(ingestion["extraction_mineru_output_dir"])),
        summarized_store=_anchor(str(ingestion["summarization_output_dir"])),
        consolidation_cache=_anchor(str(ingestion["consolidation_output_dir"])),
        knowledge_cache=_anchor(str(ingestion["knowledge_output_dir"])),
        extraction_job_file=_anchor(str(ingestion["extraction_job_file"])),
        summarization_job_file=_anchor(str(ingestion["summarization_job_file"])),
        knowledge_base=_anchor(str(ingestion["knowledge_base_dir"])),
        documents_root=_anchor(str(ingestion["documents_root"])),
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _rmtree_step(label: str, target: Path, project_root: Path) -> Dict[str, Any]:
    """Remove a whole directory tree, with the root-safety guards."""
    if not target.exists():
        return {"ok": True, "removed": None}  # already clear — no-op
    resolved = target.resolve()
    if resolved == project_root or project_root not in resolved.parents:
        return {
            "ok": False,
            "removed": None,
            "reason": f"refusing to remove '{resolved}' (outside the project root)",
        }
    try:
        shutil.rmtree(resolved)
        logger.info("Cleared %s: %s", label, resolved)
        return {"ok": True, "removed": str(resolved)}
    except OSError as exc:
        return {"ok": False, "removed": None, "reason": str(exc)}


def _unlink_step(label: str, target: Path, project_root: Path) -> Dict[str, Any]:
    """Remove a single file (the job files), with the same guards."""
    if not target.exists():
        return {"ok": True, "removed": None}
    resolved = target.resolve()
    if resolved == project_root or project_root not in resolved.parents:
        return {
            "ok": False,
            "removed": None,
            "reason": f"refusing to remove '{resolved}' (outside the project root)",
        }
    try:
        resolved.unlink()
        logger.info("Cleared %s: %s", label, resolved)
        return {"ok": True, "removed": str(resolved)}
    except OSError as exc:
        return {"ok": False, "removed": None, "reason": str(exc)}


def clear_corpus(paths: ClearPaths) -> Dict[str, Any]:
    """Erase every generated surface; return a structured report.

    Raises nothing for store-level issues (they are reported per step);
    only an unusable CONFIG raises (via :func:`resolve_clear_paths`) —
    the caller must never wipe on a misread configuration. The
    root-safety anchor is ``paths.project_root``.
    """
    project_root = Path(paths.project_root).resolve()
    if paths.documents_root.exists():
        # Belt-and-braces: the documents root is the user's scope. It is
        # never a clear target; assert the config has not collapsed it
        # onto one (e.g. documents_root == extraction store).
        for label in ("extraction_store", "summarized_store", "knowledge_base",
                      "vector_store", "mineru_store", "consolidation_cache",
                      "knowledge_cache"):
            if getattr(paths, label).resolve() == paths.documents_root.resolve():
                return {
                    "status": STATUS_PARTIAL,
                    "steps": {},
                    "reason": (
                        f"configuration collision: {label} resolves onto the "
                        f"documents root '{paths.documents_root}' — refusing "
                        "to clear (fix ingestion.yaml first)"
                    ),
                }

    steps: Dict[str, Any] = {
        "vector_store": _rmtree_step("vector store", paths.vector_store, project_root),
        "extraction_store": _rmtree_step(
            "extraction store", paths.extraction_store, project_root,
        ),
        "mineru_store": _rmtree_step("MinerU sandbox", paths.mineru_store, project_root),
        "summarized_store": _rmtree_step(
            "summarized store", paths.summarized_store, project_root,
        ),
        "consolidation_cache": _rmtree_step(
            "consolidation cache", paths.consolidation_cache, project_root,
        ),
        "knowledge_cache": _rmtree_step(
            "knowledge cache", paths.knowledge_cache, project_root,
        ),
        "extraction_job_file": _unlink_step(
            "extraction job file", paths.extraction_job_file, project_root,
        ),
        "summarization_job_file": _unlink_step(
            "summarization job file", paths.summarization_job_file, project_root,
        ),
        "knowledge_base": _rmtree_step(
            "knowledge base", paths.knowledge_base, project_root,
        ),
    }

    failed: List[str] = [label for label, report in steps.items() if not report["ok"]]
    status = STATUS_CLEARED if not failed else STATUS_PARTIAL
    report: Dict[str, Any] = {
        "status": status,
        "steps": steps,
        "documents_root_kept": str(paths.documents_root),
    }
    if failed:
        report["failed_steps"] = failed
        report["reason"] = "; ".join(
            f"{label}: {steps[label].get('reason', 'failed')}" for label in failed
        )
    logger.info("Corpus clear: %s%s", status, f" — {report['reason']}" if failed else "")
    return report
