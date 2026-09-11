"""Extraction job file — the extraction checkpoint store.

A simple, human-editable JSON file remembering which documents the
extraction agent has already worked out. It is the ingestion system's
first checkpoint: a completed extraction is recorded so the work is never
done twice, and an entry can be removed by hand (or by the orchestrator's
force routing) to re-run a document after it was changed.

Design choices:

* **JSON, one entry per extracted document.** Human-readable and
  human-editable: the user fixes a "stuck" state by deleting a line.
* **Keyed by file name** (not full path): the ingestion sandbox is rooted
  at ``documents_root`` and the LLM/user reference documents by name; a
  job file entry must stay meaningful if the project moves.
* **Each entry stores the source mtime/size at extraction time.** When a
  document changes on disk, the agent can report it as ``stale`` instead of
  ``already_done`` — the user then knows a force re-extraction is genuinely
  useful, not just a re-run.
* **Fail-open.** A corrupted or missing job file never blocks ingestion:
  the store reports empty and the extraction simply runs again. A manual
  edit that broke the JSON must not wedge the pipeline.

Location: ``extraction_job_file`` in config/ingestion.yaml (validated by
``validate_ingestion_config``); default ``data/cache/extraction_jobs.json``
when the key is absent.

The :class:`SummarizationJobFile` (same file layout, own config key
``summarization_job_file``) subclasses this store: the summarizer records
the same checkpoint contract so ``status_of`` / ``remove`` semantics stay
identical across steps.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from src.tools.config_loader import PROJECT_ROOT

logger = logging.getLogger(__name__)

#: Default location of the job file when ingestion.yaml does not set
#: ``extraction_job_file`` (overridable at construction).
DEFAULT_JOB_FILE = PROJECT_ROOT / "data" / "cache" / "extraction_jobs.json"

#: Default location of the summarization job file when ingestion.yaml does
#: not set ``summarization_job_file``.
DEFAULT_SUMMARIZATION_JOB_FILE = (
    PROJECT_ROOT / "data" / "cache" / "summarization_jobs.json"
)

#: Result of a checkpoint lookup.
STATUS_ALREADY_DONE = "already_done"   # extracted, artifacts up to date
STATUS_STALE = "stale"                 # extracted, but the source changed since
STATUS_NEW = "new"                     # never extracted


class ExtractionJobFile:
    """Checkpoint store for the extraction step (JSON job file).

    Args:
        job_file: path of the JSON job file; defaults to
            ``data/ingestion/extraction_jobs.json`` under the project root.
    """

    def __init__(self, job_file: Optional[Path] = None) -> None:
        self.job_file = Path(job_file) if job_file else _configured_job_file()

    # -- Reading -------------------------------------------------------------

    def load(self) -> Dict[str, dict]:
        """Load the job file, returning ``{}`` when absent or corrupted.

        Fail-open: a broken job file (bad JSON, wrong shape) is logged and
        treated as empty so ingestion never blocks on a manual edit gone
        wrong.
        """
        if not self.job_file.exists():
            return {}
        try:
            with open(self.job_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Job file unreadable (%s); treating as empty: %s", self.job_file, exc)
            return {}
        if not isinstance(data, dict):
            logger.warning("Job file has unexpected shape (%s); treating as empty.", self.job_file)
            return {}
        return data

    def status_of(self, file_name: str) -> str:
        """Checkpoint status of a document: already_done / stale / new."""
        entry = self.load().get(file_name)
        if entry is None:
            return STATUS_NEW

        source = Path(str(entry.get("source_path", "")))
        try:
            stat = source.stat()
        except OSError:
            # Source moved/deleted since extraction: still "done" — the
            # recorded extraction is what we have; resume will use it.
            return STATUS_ALREADY_DONE

        if stat.st_mtime != entry.get("source_mtime") or stat.st_size != entry.get("source_size"):
            return STATUS_STALE
        return STATUS_ALREADY_DONE

    # -- Writing ---------------------------------------------------------------

    def record(self, file_name: str, source_path: Path) -> None:
        """Record a document as extracted (mtime/size snapshot included)."""
        entries = self.load()
        try:
            stat = source_path.stat()
            mtime, size = stat.st_mtime, stat.st_size
        except OSError:
            mtime, size = None, None
        entries[file_name] = {
            "source_path": str(source_path),
            "source_mtime": mtime,
            "source_size": size,
        }
        self._write(entries)

    def remove(self, file_name: str) -> bool:
        """Remove a document from the job file (force re-extraction).

        Returns True when an entry was removed, False when it was absent.
        """
        entries = self.load()
        if file_name not in entries:
            return False
        del entries[file_name]
        self._write(entries)
        return True

    def _write(self, entries: Dict[str, dict]) -> None:
        """Persist entries atomically (tmp file + os.replace)."""
        self.job_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=str(self.job_file.parent), prefix=".extraction_jobs_", suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(entries, handle, indent=2, ensure_ascii=False)
            os.replace(tmp_name, self.job_file)
        except OSError:
            # Best effort cleanup of the tmp file, then re-raise.
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    # -- Convenience ------------------------------------------------------------

    def entries(self) -> List[str]:
        """Names of all extracted documents, sorted."""
        return sorted(self.load())

    def clear(self) -> None:
        """Empty the job file (admin/debug helper)."""
        self._write({})


def _configured_job_file() -> Path:
    """Resolve the job file location from config/ingestion.yaml.

    Reads ``extraction_job_file`` when set (absolute paths used as-is,
    relative resolved against the project root) and falls back to
    ``DEFAULT_JOB_FILE`` (``data/cache/extraction_jobs.json``) when the key
    is absent, when the yaml cannot be read, or when the value is not a
    string. Fail-open by design: a configuration problem must not prevent
    the checkpoint store from working at its default location.
    """
    path = _configured_job_file_for("extraction_job_file", DEFAULT_JOB_FILE)
    if path is not None:
        return path
    return DEFAULT_JOB_FILE


def _configured_job_file_for(config_key: str, default: Path) -> Optional[Path]:
    """Resolve a ``*_job_file`` config key; ``None`` means "use ``default``".

    Shared by the extraction and summarization stores. Fail-open: a missing,
    blank or unreadable value resolves to the default location.
    """
    try:
        from src.tools.config_loader import load_ingestion_config

        value = load_ingestion_config().get(config_key)
    except Exception as exc:  # noqa: BLE001 — fail-open, see docstring
        logger.warning("Could not read ingestion.yaml for %s; using default %s (%s)",
                       config_key, default, exc)
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

# ---------------------------------------------------------------------------
# Summarization job file — same checkpoint contract, own store
# ---------------------------------------------------------------------------

class SummarizationJobFile(ExtractionJobFile):
    """Checkpoint store for the summarization step.

    Same file layout and semantics as the extraction job file (an entry per
    summarized document, mtime/size snapshot, fail-open reads, atomic
    writes) — the summarizer records the same contract so checkpoint
    behavior is uniform across steps. Own config key
    ``summarization_job_file``; default ``data/cache/summarization_jobs.json``.
    """

    def __init__(self, job_file: Optional[Path] = None) -> None:
        self.job_file = Path(job_file) if job_file else (
            _configured_job_file_for(
                "summarization_job_file", DEFAULT_SUMMARIZATION_JOB_FILE
            )
            or DEFAULT_SUMMARIZATION_JOB_FILE
        )
