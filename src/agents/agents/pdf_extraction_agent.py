"""First concrete agent: PDF content extraction.

``PDFExtractionAgent`` implements the ``ContentExtractorAgent`` protocol
(src/agents/protocols.py): it runs a ``DocumentExtractor`` (MinerU by
default) on the document referenced by the ingestion context and stores the
resulting ``DocumentExtract`` in ``context.outputs["content_extraction"]``.

Checkpoint / resume — three stores, consulted in order:

1. the canonical extracted-content JSON (``data/extracted/<stem>.json``,
   src/extraction/json_store.py) — the fixable persistence
   layer: when it exists and is valid the extraction resumes from it,
   bypassing MinerU entirely (this is the resume path for user-corrected
   content);
2. the extraction job file (src/tools/extraction_job_file.py) — a
   human-editable JSON list of already-extracted documents (checkpoint);
3. MinerU's own artifacts (``extraction_mineru_output_dir``), rebuilt via
   ``bypass_ocr`` when the job file remembers the document but no canonical
   JSON exists yet (legacy checkpoint from before the two-store split).

On a checkpoint hit the agent does NOT redo the expensive MinerU pass:

* canonical JSON present and valid       → resumed from it (fast path);
* job file hit + MinerU artifacts usable → rebuilt from artifacts, then
  the canonical JSON is (re)written so the store self-heals;
* job file hit + nothing usable          → ``already_done`` status, no
  re-extraction (the user is expected to clean the stale entries or use
  force);
* a canonical JSON that exists but is *malformed* (user edit gone wrong)
  → ``resume_failed`` with an actionable hint, then a fresh extraction
  which will overwrite the broken file;
* source changed on disk                 → reported as ``stale`` (force
  re-extraction is the real answer there).

On a fresh extraction, the validated ``DocumentExtract`` is saved to the
canonical JSON store before the checkpoint is recorded.

The force flag (``context.metadata["force_extraction"]``, set by the
orchestrator's keyword routing) bypasses the checkpoint entirely and
re-runs the extraction, then re-records the job file entry.

Failure mapping (agents never raise for expected failures — they report a
``FailureDomain`` so the graph can decide restart vs rejection):

* missing/invalid document        → ``INPUT_DATA``  (not retryable)
* extraction backend failure      → ``EXTERNAL``    (MinerU missing/crashed)
* anything unexpected             → ``UNKNOWN``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.agents.contexts import IngestionContext
from src.agents.protocols import (
    AgentResult,
    AgentStatus,
    FailureDomain,
)
from src.tools.extraction_job_file import (
    STATUS_ALREADY_DONE,
    STATUS_NEW,
    STATUS_STALE,
    ExtractionJobFile,
)
from src.extraction.document_extractor import DocumentExtractor
from src.extraction.json_store import (
    ExtractJsonError,
    canonical_path_for,
    load_extract,
    save_extract,
)
from src.extraction.ids import assign_extract_ids
from src.extraction.models import DocumentExtract
from src.tools.config_loader import coerce_origin
from src.extraction.mineru_pdf_extractor import MineruPDFExtractor
from src.extraction.validation import (
    SEVERITY_WARNING,
    structural_errors,
    structural_issues,
)

logger = logging.getLogger(__name__)

# Context key where the extraction result is stored (graph step name).
OUTPUT_KEY = "content_extraction"

# Context metadata flag: bypass the checkpoint and re-extract (set by the
# orchestrator's force-keyword routing).
FORCE_EXTRACTION_KEY = "force_extraction"


class PDFExtractionAgent:
    """Extracts structured content from a PDF document.

    Args:
        extractor: the DocumentExtractor to run; defaults to a
            ``MineruPDFExtractor`` (MinerU backend, ``data/extracted``).
        output_key: context key for the result; defaults to
            ``content_extraction`` (the graph step name).
    """

    name = "content_extractor"

    def __init__(
        self,
        extractor: Optional[DocumentExtractor] = None,
        output_key: str = OUTPUT_KEY,
        job_file: Optional[ExtractionJobFile] = None,
        canonical_dir: Optional[Path] = None,
    ) -> None:
        self._extractor = extractor
        self._output_key = output_key
        self._job_file = job_file
        # Canonical extracted-content folder override (tests); None uses the
        # config-driven default (extraction_output_dir in ingestion.yaml).
        self._canonical_dir = Path(canonical_dir) if canonical_dir else None

    @property
    def job_file(self) -> ExtractionJobFile:
        """Checkpoint store (lazily-built default: the project job file)."""
        if self._job_file is None:
            self._job_file = ExtractionJobFile()
        return self._job_file

    @property
    def extractor(self) -> DocumentExtractor:
        """Lazily-built default extractor (MinerU)."""
        if self._extractor is None:
            self._extractor = MineruPDFExtractor()
        return self._extractor

    # -- Canonical JSON store helpers -----------------------------------------

    def _canonical_path(self, document_path: Path) -> Path:
        """Canonical JSON path for the document being ingested."""
        if self._canonical_dir is not None:
            return self._canonical_dir / f"{document_path.stem}.json"
        return canonical_path_for(document_path)

    # -- IngestionAgent contract ---------------------------------------------

    def run(self, context: IngestionContext) -> AgentResult:
        """Extract the document referenced by the context.

        Honors the extraction checkpoint (job file) unless
        ``context.metadata[FORCE_EXTRACTION_KEY]`` is set.
        """
        if context.document_path is None:
            context.emit("task", "extraction_failed",
                         "no document path in ingestion context")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail="no document path in ingestion context",
            )

        # -- canonical JSON resume (user-fixable content, bypasses MinerU) ------
        # Consulted before the checkpoint: when the canonical extraction of
        # this document exists, it IS the current content (the user may have
        # fixed it by hand) — MinerU artifacts are irrelevant. Force mode
        # bypasses it like the checkpoint: everything is reworked from the
        # source and the canonical JSON is overwritten at the end.
        if not context.metadata.get(FORCE_EXTRACTION_KEY, False):
            resume_result = self._resume_from_canonical(context)
            if resume_result is not None:
                return resume_result

        # -- checkpoint ---------------------------------------------------------
        if not context.metadata.get(FORCE_EXTRACTION_KEY, False):
            checkpoint = self._check_checkpoint(context)
            if checkpoint is not None:
                return checkpoint
        else:
            context.emit("task", "extraction_forced",
                         f"forced extraction of '{context.document_path.name}' "
                         "(checkpoint bypassed)",
                         document=context.document_path.name)

        # -- fresh extraction ---------------------------------------------------
        context.emit("task", "extracting",
                     f"running MinerU extraction on '{context.document_path.name}' "
                     "(this can take a while)",
                     document=context.document_path.name)
        try:
            document = self.extractor.extract(context.document_path)
        except (FileNotFoundError, ValueError) as exc:
            context.emit("task", "extraction_failed", str(exc))
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail=str(exc),
            )
        except OSError as exc:
            # MinerU is an external tool: missing binary or non-zero exit.
            context.emit("task", "extraction_failed",
                         f"extraction backend failure: {exc}")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.EXTERNAL,
                detail=f"extraction backend failure: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 — last resort, reported not raised
            context.emit("task", "extraction_failed",
                         f"unexpected extraction error: {exc}")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.UNKNOWN,
                detail=f"unexpected extraction error: {exc}",
            )

        if document is None:
            # Out-of-contract extractor (protocol promises a DocumentExtract);
            # defend anyway — agents must never crash the graph.
            context.emit("task", "extraction_failed",
                         "extractor returned no document")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail="extractor returned no document",
            )

        # Document origin (governance metadata) — decided by the CLI
        # BEFORE the graph runs; the agent only applies it. A missing value
        # takes the configured default (first entry of setup.yaml
        # documents.origins) but is reported: an unverified origin must be
        # visible (the extraction validator warns too). A value OUTSIDE the
        # user-defined vocabulary is rejected, never guessed.
        origin_raw = context.metadata.get("document_origin")
        if origin_raw is not None:
            coerced = coerce_origin(str(origin_raw))
            if coerced is not None:
                document.origin = coerced
            else:
                context.emit("task", "origin_warning",
                             f"unknown document origin {origin_raw!r}; "
                             f"defaulting to '{document.origin}'")
        else:
            context.emit("task", "origin_missing",
                         f"no document origin provided; defaulting to "
                         f"'{document.origin}' (unverified)")
        context.metadata["document_origin"] = document.origin

        # Unified ids (src/extraction/ids.py) — assigned before any
        # persistence so the canonical JSON carries them from birth. A
        # document with no derivable name is an input problem, not a
        # backend one.
        try:
            assign_extract_ids(document)
        except ValueError as exc:
            context.emit("task", "extraction_failed", str(exc))
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail=str(exc),
            )

        context.outputs[self._output_key] = document
        context.metadata.update(
            {
                "document_id": document.id,
                "document_title": document.title,
                "total_pages": document.total_pages,
                "chapter_count": len(document.chapters),
            }
        )

        # Persist the canonical extracted content BEFORE recording the
        # checkpoint — the job file entry must only exist once the fixable
        # JSON exists too (otherwise a resume could find nothing to load).
        canonical_path = self._canonical_path(context.document_path)
        try:
            save_extract(document, canonical_path)
        except OSError as exc:
            # The extraction succeeded; report the persistence failure
            # without discarding the work. The checkpoint is deliberately
            # NOT recorded: the next run must redo both (a job file entry
            # without its canonical JSON would resume into nothing).
            logger.warning("Could not save canonical extraction: %s", exc)
            context.errors[self._output_key] = f"canonical extraction not saved: {exc}"
            context.emit("task", "canonical_save_failed",
                         f"could not save canonical extraction: {exc}")
        else:
            context.emit(
                "task", "canonical_saved",
                f"canonical extraction saved: {canonical_path.name}",
                path=str(canonical_path),
            )

            # Record the checkpoint only when the canonical JSON is on disk
            # — a failed extraction (or an unsavable one) must never mark
            # the document as done.
            try:
                self.job_file.record(context.document_path.name, context.document_path)
            except OSError as exc:
                # The extraction and the save both succeeded: report the
                # checkpoint failure without discarding the work (next run
                # would just redo the recording — and the canonical resume
                # path still works since the JSON exists).
                logger.warning("Could not update the extraction job file: %s", exc)
                context.errors[self._output_key] = f"checkpoint not recorded: {exc}"

        context.emit(
            "task",
            "extracted",
            f"extracted '{document.title}': {document.total_pages} page(s), "
            f"{len(document.chapters)} chapter(s), checkpoint recorded",
            document=context.document_path.name,
            pages=document.total_pages,
        )
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.OK,
            payload={
                "title": document.title,
                "total_pages": document.total_pages,
                "chapters": len(document.chapters),
                "toc_entries": len(document.toc),
                "checkpoint": "recorded",
            },
        )

    # -- Checkpoint helpers ----------------------------------------------------

    def _resume_from_canonical(self, context: IngestionContext) -> Optional[AgentResult]:
        """Resume from the canonical extracted-content JSON when it exists.

        The canonical store (``extraction_output_dir/<stem>.json``) is the
        fixable persistence layer: a user may have edited it to correct a
        malformed source. When present and valid it is authoritative — the
        extraction resumes from it and MinerU is never invoked.

        Returns None when there is nothing to resume (no file), or when the
        run should fall through to a fresh extraction (malformed file — the
        fresh extraction will overwrite it).
        """
        canonical_path = self._canonical_path(context.document_path)
        if not canonical_path.exists():
            return None

        name = context.document_path.name
        context.emit("task", "canonical_found",
                     f"canonical extraction found: {canonical_path.name}; loading",
                     document=name, path=str(canonical_path))
        try:
            document = load_extract(canonical_path)
            assign_extract_ids(document)  # id-less legacy file: fill ids in
        except ExtractJsonError as exc:
            # A hand-edit gone wrong must not wedge the pipeline: report,
            # then fall through to a fresh extraction that overwrites it.
            context.emit("task", "resume_failed",
                         f"canonical extraction unusable ({exc}); "
                         "falling back to a fresh extraction",
                         document=name, path=str(canonical_path))
            logger.warning("Unusable canonical extraction for '%s': %s", name, exc)
            return None

        # Origin: an explicit request-level origin wins (the user may be
        # correcting the record); otherwise the file's stored origin stands
        # — it was decided at its extraction and persisted with it.
        origin_raw = context.metadata.get("document_origin")
        if origin_raw is not None:
            coerced = coerce_origin(str(origin_raw))
            if coerced is not None:
                document.origin = coerced
            else:
                context.emit("task", "origin_warning",
                             f"unknown document origin {origin_raw!r}; "
                             f"keeping the stored '{document.origin}'")
        context.metadata["document_origin"] = document.origin

        context.outputs[self._output_key] = document
        context.metadata.update(
            {
                "document_id": document.id,
                "document_title": document.title,
                "total_pages": document.total_pages,
                "chapter_count": len(document.chapters),
                "resume_source": "canonical_json",
            }
        )
        context.emit(
            "task", "resumed",
            f"resumed '{document.title}' from canonical JSON: "
            f"{document.total_pages} page(s), no re-extraction",
            document=name, source="canonical_json", pages=document.total_pages,
        )
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.OK,
            payload={
                "title": document.title,
                "total_pages": document.total_pages,
                "chapters": len(document.chapters),
                "toc_entries": len(document.toc),
                "resume": "canonical_json",
            },
        )

    def _check_checkpoint(self, context: IngestionContext) -> Optional[AgentResult]:
        """Consult the job file; resume or short-circuit on a hit.

        Returns None when the document must be extracted (no checkpoint or
        artifacts unusable), or an OK/FAILED result when the checkpoint
        decides the run.
        """
        name = context.document_path.name
        status = self.job_file.status_of(name)
        if status == STATUS_NEW:
            return None  # never extracted: run the extractor

        context.emit("task", "checkpoint_hit",
                     f"'{name}' already extracted ({status}); checking artifacts",
                     document=name, checkpoint=status)
        context.metadata["checkpoint_status"] = status

        # Resume: rebuild the DocumentExtract from the artifacts already on
        # disk (bypass_ocr mode) — no expensive MinerU pass.
        if isinstance(self.extractor, MineruPDFExtractor):
            try:
                saved_bypass = self.extractor.bypass_ocr
                self.extractor.bypass_ocr = True
                document = self.extractor.extract(context.document_path)
                assign_extract_ids(document)  # unified ids before the self-heal save
            except (FileNotFoundError, ValueError, OSError) as exc:
                logger.info(
                    "Checkpoint hit but artifacts unusable for '%s' (%s); "
                    "re-extracting.", name, exc,
                )
                context.emit("task", "resume_failed",
                             f"artifacts unusable for '{name}' ({exc}); "
                             "falling back to a fresh extraction", document=name)
                return None  # fall through to a fresh extraction
            finally:
                self.extractor.bypass_ocr = saved_bypass

            context.emit("task", "resumed",
                         f"resumed '{document.title}' from artifacts: "
                         f"{document.total_pages} page(s), no OCR re-run",
                         document=name, checkpoint=status, pages=document.total_pages)
            # Origin: artifact rebuilds carry no stored origin — apply the
            # request-level one, or default with a visible warning.
            origin_raw = context.metadata.get("document_origin")
            if origin_raw is not None:
                coerced = coerce_origin(str(origin_raw))
                if coerced is not None:
                    document.origin = coerced
                else:
                    context.emit("task", "origin_warning",
                                 f"unknown document origin {origin_raw!r}; "
                                 f"defaulting to '{document.origin}'")
            else:
                context.emit("task", "origin_missing",
                             f"no document origin provided; defaulting to "
                             f"'{document.origin}' (unverified)")
            context.metadata["document_origin"] = document.origin

            context.outputs[self._output_key] = document
            context.metadata.update(
                {
                    "document_id": document.id,
                    "document_title": document.title,
                    "total_pages": document.total_pages,
                    "chapter_count": len(document.chapters),
                    "resume_source": "mineru_artifacts",
                }
            )
            # Self-heal the canonical store: rebuild the fixable JSON from
            # the artifacts when it is missing (pre-split checkpoints).
            canonical_path = self._canonical_path(context.document_path)
            try:
                save_extract(document, canonical_path)
                context.emit("task", "canonical_saved",
                             f"canonical extraction saved: {canonical_path.name}",
                             path=str(canonical_path))
            except OSError as exc:
                logger.warning("Could not save canonical extraction: %s", exc)
                context.errors[self._output_key] = f"canonical extraction not saved: {exc}"
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.OK,
                payload={
                    "title": document.title,
                    "total_pages": document.total_pages,
                    "chapters": len(document.chapters),
                    "toc_entries": len(document.toc),
                    "checkpoint": status,
                    "resume": "mineru_artifacts",
                },
            )

        # Non-MinerU (stub/test) extractor: no artifact-rebuild path — the
        # checkpoint short-circuits the step.
        context.emit("task", "already_done",
                     f"'{name}' already extracted ({status}); skipping "
                     "(use force to re-extract)", document=name, checkpoint=status)
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.OK,
            payload={
                "title": name,
                "checkpoint": status,
                "message": (
                    "document already extracted; skipping extraction "
                    "(use force to re-extract)"
                ),
            },
        )

    def validate(self, context: IngestionContext) -> Optional[AgentResult]:
        """Pedantic structural validation of the produced DocumentExtract.

        Runs the shape checker (src/extraction/validation.py) and fails the
        step only on *errors* (types, hierarchy, counts, identifiers).
        *Warnings* — optional metadata a backend legitimately omits, like
        MinerU's page geometry or a 0-based fallback TOC level — are traced
        and carried in the payload but never stop the ingestion.
        The data-consistency pass (ordering, empty-text/empty-container
        pruning) is the *next graph step's* job (extraction_validator), not
        this one. A structural failure fails the extraction step itself
        (INPUT_DATA domain — a malformed extraction is not retryable).
        """
        document: Optional[DocumentExtract] = context.outputs.get(self._output_key)
        if document is None:
            return None  # nothing to check (run did not execute)

        issues = structural_issues(document)
        warnings = [i for i in issues if i.severity == SEVERITY_WARNING]
        errors = [i for i in issues if i.severity != SEVERITY_WARNING]

        # Warnings: visible in the thinking panel, never fatal.
        for issue in warnings[:5]:
            context.emit("task", "structural_warning",
                         f"{issue.path}: {issue.message}",
                         code=issue.code, path=issue.path)
        if len(warnings) > 5:
            context.emit("task", "structural_warning",
                         f"... and {len(warnings) - 5} more structural warning(s)")
        if warnings:
            context.emit("task", "structural_warnings",
                         f"{len(warnings)} structural warning(s) in the extraction "
                         "(non-fatal)")

        if not errors:
            return None  # structurally sound; warnings do not fail the step

        # One trace per first few errors, plus a summary — the thinking
        # panel should show what is wrong without drowning in repeats.
        for issue in errors[:5]:
            context.emit("task", "structural_issue",
                         f"{issue.path}: {issue.message}",
                         code=issue.code, path=issue.path)
        if len(errors) > 5:
            context.emit("task", "structural_issue",
                         f"... and {len(errors) - 5} more structural issue(s)")
        context.emit("task", "structural_validation_failed",
                     f"{len(errors)} structural issue(s) in the extraction")

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.FAILED,
            failure_domain=FailureDomain.INPUT_DATA,
            detail=(
                f"structurally invalid extraction: {len(errors)} issue(s), "
                f"first: {errors[0].path} ({errors[0].message})"
            ),
            payload={
                "structural_issues": [
                    {"code": i.code, "path": i.path, "message": i.message,
                     "severity": i.severity}
                    for i in issues
                ],
            },
        )
