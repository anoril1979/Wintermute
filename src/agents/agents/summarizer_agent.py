"""SummarizerAgent — hierarchical summarization of the extraction.

The ``hierarchical_summarization`` graph step (agent key ``summarizer``).
Runs after the extraction validation: reads the cleaned ``DocumentExtract``
from the context and fills every ``summary`` field, bottom-up so each level
summarizes what the level below produced:

    blocks -> sections -> pages -> chapters -> document

Input text per unit:

* ``TextBlock`` / ``Section`` / ``PageContent`` — the unit's ``raw_text``;
* ``Chapter`` — its ``full_text``, or the concatenation of its pages'
  summaries when the aggregated text is empty;
* ``DocumentExtract`` — the concatenation of the chapters' summaries
  (orphan pages included).

Size rules (config/ingestion.yaml, validated by the config loader):

* ``summary_min_chars`` — content shorter than the limit is copied verbatim
  into ``summary``: no LLM call, a short text IS its own summary. Applied
  at every level, chapters and document included (a doc can already be
  very short);
* ``summary_max_chars`` — the size target the LLM is asked to stay under.
  A longer answer is **kept** but reported as a warning (we want to know,
  it is not a failure).

Feedback: every action is traced (``task`` phase, visible in the thinking
panel) — ``summarized`` per unit, ``summarization_warning`` for oversize
summaries, ``summarization_skipped`` for empty units, ``copied`` counts in
the closing ``summarization_done`` event. LLM-shaped failures (call error,
empty or unreadable answer) fail the step with ``FailureDomain.LLM_RESPONSE``
so the graph's retry policy can restart it.

Persistence (src/summarization/summarized_store.py): on success the fully
summarized document is saved to ``summarization_output_dir``
(config/ingestion.yaml, default ``data/summarized``) as ``<stem>.json``,
and the summarization job file (``data/cache/summarization_jobs.json``)
records the checkpoint. When an ingestion fails at a later step and is
re-run, the agent resumes from that file — its content fingerprint must
match the current canonical extraction — and performs ZERO LLM calls
(traces ``summarized_found`` → ``summarized_resume``; ``summarized_stale``
when the content changed and re-summarization is required).

Checkpoint flow (mirrors the extraction agent):

* no checkpoint hit → normal summarization, then store + record;
* checkpoint hit + fingerprint matches → resume with zero LLM calls
  (``already_done``-like fast path);
* checkpoint hit + fingerprint differs (content changed) → re-summarize;
* ``context.metadata["force_summarization"]`` (router option
  ``force_summarization``, keyword routing "re-summarize the file") →
  both stores bypassed, LLM re-runs, store + checkpoint refreshed.

There is no "skip summarization" mode by design: summaries follow the
content. If the extraction changed, the stored summaries are stale
(content fingerprint) and are re-run; if it did not, the resume path
reuses them with zero LLM calls — there is no state in which skipping is
the consistent choice.

The prompt (``prompts/summerization/hierarchical_summary.md``, role
``summarizer``) enforces a plain-text answer — no JSON fences: the summary
must be storable and readable as-is.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.agents.contexts import IngestionContext
from src.agents.llm_roles import LLMRoleAgent
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
from src.extraction.models import Chapter, DocumentExtract, PageContent, Section, TextBlock
from src.summarization.summarized_store import (
    SummarizedJsonError,
    content_fingerprint,
    load_summarized,
    save_summarized,
    summarized_path_for,
)
from src.tools.config_loader import load_ingestion_config
from src.tools.extraction_job_file import (
    STATUS_NEW,
    SummarizationJobFile,
)

logger = logging.getLogger(__name__)

#: Context key this agent reads from (the validated extraction) and writes
#: back to (the same document, summaries filled in place).
INPUT_KEY = "content_extraction"

AGENT_NAME = "summarizer"

#: Context metadata flag set by the orchestrator's routing (or the LLM
#: router options) and the ingestion task agent.
FORCE_SUMMARY_KEY = "force_summarization"

#: Where the summarization prompt lives (loaded lazily, once).
SUMMARY_PROMPT_PATH = Path("prompts/summerization/hierarchical_summary.md")

#: The marker the prompt defines for an unusable input (LLM protocol).
UNREADABLE_MARKER = "UNREADABLE"

#: Defaults when ingestion.yaml does not set the limits (must stay in sync
#: with config/ingestion.yaml and the config loader's validation).
DEFAULT_SUMMARY_MIN_CHARS = 1000
DEFAULT_SUMMARY_MAX_CHARS = 2500


def summary_limits() -> Tuple[int, int]:
    """``(min_chars, max_chars)`` from ingestion.yaml, fail-open to defaults.

    The yaml is validated at load time by the config loader, so a present
    value is a positive number; the fail-open path only covers a broken or
    unreadable config, in which case summarization still runs on defaults.
    """
    try:
        config = load_ingestion_config()
        min_chars = int(config.get("summary_min_chars", DEFAULT_SUMMARY_MIN_CHARS))
        max_chars = int(config.get("summary_max_chars", DEFAULT_SUMMARY_MAX_CHARS))
    except Exception as exc:  # noqa: BLE001 — fail-open to the defaults
        logger.warning(
            "Could not read ingestion.yaml for summary limits; using defaults "
            "(%d, %d): %s", DEFAULT_SUMMARY_MIN_CHARS, DEFAULT_SUMMARY_MAX_CHARS, exc,
        )
        return DEFAULT_SUMMARY_MIN_CHARS, DEFAULT_SUMMARY_MAX_CHARS
    if min_chars <= 0 or max_chars <= 0:
        return DEFAULT_SUMMARY_MIN_CHARS, DEFAULT_SUMMARY_MAX_CHARS
    return min_chars, max_chars


def combined_text(parts: List[Optional[str]], separator: str = "\n\n") -> str:
    """Concatenate non-empty child summaries, skipping blank parts.

    Shared helper for the chapter and document levels: their input is the
    text produced by the level below, not raw content.
    """
    return separator.join(p for p in parts if p and p.strip())


def _blank(text: Optional[str]) -> bool:
    """True when a summary is absent or whitespace-only."""
    return not text or not text.strip()


class SummarizerAgent(LLMRoleAgent):
    """Fills every ``summary`` field of the extraction, bottom-up."""

    name = AGENT_NAME
    llm_role = "summarizer"

    def __init__(self, *args, prompt_path: Optional[Path] = None,
                 summarized_dir: Optional[Path] = None,
                 job_file: Optional[SummarizationJobFile] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._prompt_path = Path(prompt_path) if prompt_path else SUMMARY_PROMPT_PATH
        self._prompt_template: Optional[str] = None
        # Summarized-content folder override (tests); None uses the
        # config-driven default (summarization_output_dir in ingestion.yaml).
        self._summarized_dir = Path(summarized_dir) if summarized_dir else None
        self._job_file = job_file

    @property
    def job_file(self) -> SummarizationJobFile:
        """Checkpoint store (lazily-built default: the project job file)."""
        if self._job_file is None:
            self._job_file = SummarizationJobFile()
        return self._job_file

    def _summarized_path(self, document_path: Path) -> Path:
        """Summarized JSON path for the document being ingested."""
        if self._summarized_dir is not None:
            return self._summarized_dir / f"{document_path.stem}.json"
        return summarized_path_for(document_path)

    # -- public API ----------------------------------------------------------

    def run(self, context: IngestionContext) -> AgentResult:
        document: Optional[DocumentExtract] = context.outputs.get(INPUT_KEY)

        if document is None:
            # The extraction step did not produce anything to summarize —
            # either it never ran or it failed; nothing to do here.
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SKIPPED,
                detail="no extraction output to summarize",
            )

        # -- router flag: force --------------------------------------------------
        forced = bool(context.metadata.get(FORCE_SUMMARY_KEY, False))
        if forced:
            document_path = getattr(context, "document_path", None)
            context.emit("task", "summarization_forced",
                         "forced re-summarization: summarized store and "
                         "checkpoint bypassed, the LLM will re-run",
                         document=document_path.name if document_path else None)

        # -- summarized-store resume (expensive LLM work is never redone) ------
        if not forced:
            resumed = self._resume_from_summarized(context, document)
            if resumed is not None:
                return resumed

        min_chars, max_chars = summary_limits()
        context.emit(
            "task", "summarization_start",
            f"summarizing '{document.title}' "
            f"(copy below {min_chars} chars, target {max_chars} chars, "
            f"role '{self._llm_role}')",
        )

        stats: Dict[str, int] = {
            "blocks": 0, "blocks_llm": 0, "blocks_copied": 0,
            "sections": 0, "sections_llm": 0, "sections_copied": 0,
            "pages": 0, "pages_llm": 0, "pages_copied": 0,
            "chapters": 0, "chapters_llm": 0, "chapters_copied": 0,
            "llm_calls": 0, "failures": 0, "oversize": 0, "copied": 0,
        }
        warnings: List[str] = []
        for chapter_index, chapter in enumerate(document.chapters, 1):
            chapter_summaries: List[Optional[str]] = []
            for page in chapter.pages:
                for section in page.sections:
                    section_summaries: List[Optional[str]] = []
                    for block in section.blocks:
                        stats["blocks"] += 1
                        if block.summary:
                            continue  # already summarized (resume) — keep it
                        summary = self._summarize_unit(
                            block.raw_text, label=f"block {block.block_id} (page {page.page_number})",
                            min_chars=min_chars, max_chars=max_chars,
                            context=context, stats=stats, warnings=warnings,
                            copied_key="blocks_copied",
                        )
                        if summary is None:
                            return self._failed_result(context, document, stats, warnings)
                        block.summary = summary
                        section_summaries.append(summary)
                    self._summarize_container(
                        section, section_summaries,
                        label=f"chapter {chapter_index} · page {page.page_number} · section {section.section_id}",
                        min_chars=min_chars, max_chars=max_chars,
                        context=context, stats=stats, warnings=warnings,
                        level_key="sections", copied_key="sections_copied",
                    )
                stats["pages"] += 1
                self._summarize_container(
                    page, self._page_parts(page),
                    label=f"page {page.page_number}",
                    min_chars=min_chars, max_chars=max_chars,
                    context=context, stats=stats, warnings=warnings,
                    level_key="pages", copied_key="pages_copied",
                )
                chapter_summaries.append(page.summary)
            stats["chapters"] += 1
            self._summarize_container(
                chapter, self._chapter_parts(chapter, chapter_summaries),
                label=f"chapter {chapter_index} ('{chapter.toc_entry.title}')",
                min_chars=min_chars, max_chars=max_chars,
                context=context, stats=stats, warnings=warnings,
                level_key="chapters", copied_key="chapters_copied",
            )

        for orphan in document.orphan_pages:
            for section in orphan.sections:
                section_summaries: List[Optional[str]] = []
                for block in section.blocks:
                    stats["blocks"] += 1
                    if block.summary:
                        continue
                    summary = self._summarize_unit(
                        block.raw_text, label=f"block {block.block_id} (orphan page {orphan.page_number})",
                        min_chars=min_chars, max_chars=max_chars,
                        context=context, stats=stats, warnings=warnings,
                        copied_key="blocks_copied",
                    )
                    if summary is None:
                        return self._failed_result(context, document, stats, warnings)
                    block.summary = summary
                    section_summaries.append(summary)
                self._summarize_container(
                    section, section_summaries,
                    label=f"orphan page {orphan.page_number} · section {section.section_id}",
                    min_chars=min_chars, max_chars=max_chars,
                    context=context, stats=stats, warnings=warnings,
                    level_key="sections", copied_key="sections_copied",
                )
            stats["pages"] += 1
            self._summarize_container(
                orphan, self._page_parts(orphan),
                label=f"orphan page {orphan.page_number}",
                min_chars=min_chars, max_chars=max_chars,
                context=context, stats=stats, warnings=warnings,
                level_key="pages", copied_key="pages_copied",
            )

        # Document level: the chapters' summaries (orphans included).
        doc_parts = self._document_parts(document)
        if doc_parts:
            summary = self._summarize_unit(
                combined_text(doc_parts), label=f"document '{document.title}'",
                min_chars=min_chars, max_chars=max_chars,
                context=context, stats=stats, warnings=warnings,
                copied_key=None,
            )
            if summary is None:
                return self._failed_result(context, document, stats, warnings)
            document.summary = summary
        else:
            context.emit("task", "summarization_skipped",
                         "document has no summarizable content")

        if stats["failures"]:
            return self._failed_result(context, document, stats, warnings)

        copied = (stats["blocks_copied"] + stats["sections_copied"]
                  + stats["pages_copied"] + stats["chapters_copied"])
        stats["copied"] = copied
        context.emit(
            "task", "summarization_done",
            f"summaries done: {stats['llm_calls']} LLM call(s), "
            f"{copied} copied as-is (< {min_chars} chars)",
            **stats,
        )

        self._persist_summarized(context, document)

        # Record the checkpoint only when the summarized store is on disk —
        # same invariant as extraction: an entry never exists without its
        # resumable content.
        document_path = getattr(context, "document_path", None)
        if document_path is not None:
            try:
                self.job_file.record(document_path.name, document_path)
            except OSError as exc:
                logger.warning("Could not update the summarization job file: %s", exc)
                context.errors[INPUT_KEY] = f"checkpoint not recorded: {exc}"

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.OK,
            payload={
                "llm_calls": stats["llm_calls"],
                "copied": copied,
                "oversize": stats["oversize"],
                "warnings": warnings,
            },
        )

    def validate(self, context: IngestionContext) -> Optional[AgentResult]:
        """Every unit must carry a non-empty summary after a successful run."""
        document: Optional[DocumentExtract] = context.outputs.get(INPUT_KEY)
        if document is None:
            return None

        missing: List[str] = []
        for chapter in document.chapters:
            chapter_owes = False
            for page in chapter.pages:
                page_owes = False
                for section in page.sections:
                    # Content flows bottom-up: a unit owes a summary only
                    # when it actually holds text below it (blank units are
                    # legitimately skipped without one).
                    section_owes = False
                    for block in section.blocks:
                        if block.raw_text.strip() and _blank(block.summary):
                            missing.append(
                                f"block {block.block_id} (page {page.page_number})"
                            )
                        section_owes = section_owes or bool(block.raw_text.strip())
                    if section_owes and _blank(section.summary):
                        missing.append(
                            f"section {section.section_id} (page {page.page_number})"
                        )
                    page_owes = page_owes or section_owes
                if page_owes and _blank(page.summary):
                    missing.append(f"page {page.page_number}")
                chapter_owes = chapter_owes or page_owes
            if chapter_owes and _blank(chapter.summary):
                missing.append(f"chapter {chapter.start_page}")
        doc_owes = bool(document.chapters)
        for orphan in document.orphan_pages:
            orphan_owes = any(
                any(b.raw_text.strip() for b in s.blocks) for s in orphan.sections
            )
            if orphan_owes and _blank(orphan.summary):
                missing.append(f"orphan page {orphan.page_number}")
            doc_owes = doc_owes or orphan_owes
        if doc_owes and _blank(document.summary):
            missing.append("document")

        if missing:
            shown = ", ".join(missing[:5]) + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else "")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.LLM_RESPONSE,
                detail=f"missing summaries after the summarization step: {shown}",
                payload={"missing": missing},
            )
        return None

    # -- Summarized-store helpers ---------------------------------------------

    @staticmethod
    def _iter_units(document: DocumentExtract):
        """Yield every summarizable unit of the document (structure order)."""
        for chapter in document.chapters:
            for page in chapter.pages:
                for section in page.sections:
                    yield from section.blocks
                    yield section
                yield page
            yield chapter
        for orphan in document.orphan_pages:
            for section in orphan.sections:
                yield from section.blocks
                yield section
            yield orphan

    def _resume_from_summarized(
        self, context: IngestionContext, document: DocumentExtract
    ) -> Optional[AgentResult]:
        """Resume from the summarized-content JSON when it matches.

        LLM summarization is expensive: when the summarized file exists and
        its content fingerprint equals the current (pre-summarization)
        document's fingerprint, the stored summaries were computed from
        exactly this content — they are applied and the step succeeds with
        ZERO LLM calls. A different fingerprint (user fix, re-extraction,
        forced run) means stale summaries: they are discarded and normal
        summarization proceeds.

        Returns None when there is nothing usable (no file, malformed file
        or stale content — summarization then runs normally).

        Note: a resume does NOT re-record the job file (same contract as
        the extraction agent's canonical resume). A user who deleted the
        checkpoint entry gets the resume anyway as long as the summarized
        JSON exists; only a fresh or forced summarization records it.
        """
        document_path = getattr(context, "document_path", None)
        if document_path is None:
            return None  # no source identity — nothing to look up
        summarized_path = self._summarized_path(document_path)
        if not summarized_path.exists():
            # No resumable content. A job-file entry without its summarized
            # JSON means the store was cleaned by hand — nothing to resume,
            # summarization will run and re-record the checkpoint.
            return None

        name = document_path.name
        # Checkpoint observability: the job file is the human-editable
        # witness that this document went through summarization; its status
        # rides the traces and metadata (the fingerprint below remains the
        # staleness oracle).
        checkpoint_status = self.job_file.status_of(name)
        context.metadata["summarization_checkpoint_status"] = checkpoint_status
        if checkpoint_status != STATUS_NEW:
            context.emit("task", "summarization_checkpoint_hit",
                         f"'{name}' already summarized per the job file "
                         f"({checkpoint_status}); checking the summarized store",
                         document=name, checkpoint=checkpoint_status)

        context.emit("task", "summarized_found",
                     f"summarized content found: {summarized_path.name}; "
                     "checking content fingerprint",
                     document=name, path=str(summarized_path))
        try:
            stored_document, fingerprint = load_summarized(summarized_path)
        except (SummarizedJsonError, OSError) as exc:
            # A hand-edit gone wrong must not wedge the pipeline: report,
            # then summarize normally (the save at the end overwrites it).
            context.emit("task", "summarized_unusable",
                         f"summarized content unusable ({exc}); re-summarizing",
                         document=name, path=str(summarized_path))
            logger.warning("Unusable summarized extraction for '%s': %s", name, exc)
            return None

        current_fingerprint = self._pre_summarization_fingerprint(document)
        if fingerprint != current_fingerprint:
            context.emit("task", "summarized_stale",
                         f"summarized content does not match the current "
                         "extraction (content changed since it was computed); "
                         "re-summarizing",
                         document=name,
                         stored_fingerprint=fingerprint,
                         current_fingerprint=current_fingerprint)
            return None

        # Fingerprints match: apply the stored summaries in place.
        stored_units = list(self._iter_units(stored_document))
        current_units = list(self._iter_units(document))
        applied = 0
        for stored_unit, current_unit in zip(stored_units, current_units):
            if _blank(getattr(stored_unit, "summary", None)):
                continue
            current_unit.summary = stored_unit.summary
            applied += 1
        # The document level is not a unit of _iter_units: apply it too.
        if not _blank(stored_document.summary):
            document.summary = stored_document.summary
            applied += 1

        context.metadata["resume_source"] = "summarized_json"
        context.emit(
            "task", "summarized_resume",
            f"resumed summaries for '{document.title}' from the summarized "
            f"store: {applied} unit(s), 0 LLM call",
            document=name, applied=applied, source="summarized_json",
        )
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.OK,
            payload={
                "llm_calls": 0,
                "copied": 0,
                "oversize": 0,
                "warnings": [],
                "resume": "summarized_json",
                "applied_summaries": applied,
            },
        )

    @staticmethod
    def _pre_summarization_fingerprint(document: DocumentExtract) -> str:
        """Content fingerprint of the document as it enters the step.

        The resume check compares fingerprints of *content*, so any
        summaries carried by the incoming document (none in the normal
        flow, or partial ones after a graph retry) are stripped before
        hashing — only the pre-summarization state counts.
        """
        stripped_units = list(SummarizerAgent._iter_units(document))
        saved = [unit.summary for unit in stripped_units]
        try:
            for unit in stripped_units:
                unit.summary = None
            if document.summary is not None:
                saved.append(document.summary)
                document.summary = None
            return content_fingerprint(document)
        finally:
            for unit, previous in zip(stripped_units, saved):
                unit.summary = previous
            if len(saved) > len(stripped_units):
                document.summary = saved[len(stripped_units)]

    def _persist_summarized(
        self, context: IngestionContext, document: DocumentExtract
    ) -> None:
        """Save the fully-summarized document to the summarized store.

        Non-fatal on failure: the summaries live in the context either way,
        only the future resume would miss them.
        """
        document_path = getattr(context, "document_path", None)
        if document_path is None:
            return
        summarized_path = self._summarized_path(document_path)
        try:
            save_summarized(document, summarized_path)
        except OSError as exc:
            logger.warning("Could not save summarized extraction: %s", exc)
            context.errors[INPUT_KEY] = f"summarized extraction not saved: {exc}"
            context.emit("task", "summarized_save_failed",
                         f"could not save summarized content: {exc}")
        else:
            context.emit("task", "summarized_saved",
                         f"summarized content saved: {summarized_path.name} "
                         "(future re-ingestions resume from it)",
                         path=str(summarized_path))

    # -- internals -------------------------------------------------------------

    def _prompt_template_text(self) -> str:
        """Load (and cache) the summarization prompt."""
        if self._prompt_template is None:
            self._prompt_template = self._prompt_path.read_text(encoding="utf-8")
        return self._prompt_template

    def _build_prompt(self, text: str, max_chars: int) -> str:
        """Full prompt: instructions + size target + the delimited text."""
        return (
            f"{self._prompt_template_text().strip()}\n"
            "\n---\n\n"
            f"Instruction: keep the summary under {max_chars} characters.\n\n"
            "Text to summarize:\n"
            "<<<<TEXT>>>>\n"
            f"{text}\n"
            "<<<<TEXT>>>>\n"
        )

    def _summarize_unit(
        self,
        text: str,
        *,
        label: str,
        min_chars: int,
        max_chars: int,
        context: IngestionContext,
        stats: Dict[str, int],
        warnings: List[str],
        copied_key: Optional[str],
    ) -> Optional[str]:
        """Summarize one unit's text; ``None`` on an LLM-shaped failure.

        Short content (< ``min_chars``) is copied verbatim — no LLM call.
        An oversize answer is kept but warned about (never a failure).
        """
        if not text.strip():
            # A blank unit has nothing to summarize; leave it empty without
            # failing (the consistency pass normally prunes those, but a
            # hand-edited canonical JSON can still carry one).
            context.emit("task", "summarization_skipped",
                         f"{label}: blank content, nothing to summarize")
            return text.strip()

        if len(text) < min_chars:
            stats["copied"] += 1
            if copied_key is not None:
                stats[copied_key] += 1
            context.emit("task", "summarized",
                         f"{label}: copied as-is ({len(text)} chars < {min_chars})",
                         label=label, chars=len(text), copied=True)
            return text.strip()

        try:
            # No explicit budget: the role's ``max_response_tokens``
            # (config/llm.yaml) is the generation ceiling — tune it there.
            raw = self.llm_client().complete(prompt=self._build_prompt(text, max_chars))
        except Exception as exc:  # LLMClientError / ValueError / transport
            stats["failures"] += 1
            logger.warning("Summarizer LLM call failed for %s: %s", label, exc)
            context.emit("task", "summarization_failed",
                         f"LLM call failed for {label}: {exc}", label=label)
            return None

        stats["llm_calls"] += 1
        summary = (raw or "").strip()
        if not summary:
            stats["failures"] += 1
            context.emit("task", "summarization_failed",
                         f"LLM returned an empty summary for {label}", label=label)
            return None
        if summary == UNREADABLE_MARKER:
            stats["failures"] += 1
            context.emit("task", "summarization_failed",
                         f"unit reported unreadable by the LLM: {label}", label=label)
            return None
        if len(summary) > max_chars:
            stats["oversize"] += 1
            message = (f"{label}: summary is {len(summary)} chars, expected "
                       f"<= {max_chars} (kept as is)")
            warnings.append(message)
            context.emit("task", "summarization_warning", message,
                         label=label, length=len(summary), max_chars=max_chars)

        context.emit("task", "summarized",
                     f"{label}: summarized ({len(summary)} chars)",
                     label=label, chars=len(summary), copied=False)
        return summary

    def _summarize_container(
        self,
        unit,
        parts: List[Optional[str]],
        *,
        label: str,
        min_chars: int,
        max_chars: int,
        context: IngestionContext,
        stats: Dict[str, int],
        warnings: List[str],
        level_key: str,
        copied_key: str,
    ) -> None:
        """Summarize a container (section/page/chapter) from its children."""
        if getattr(unit, "summary", None):
            return  # already summarized (resume) — keep it, the parent reads it
        text = combined_text(parts)
        if not text:
            context.emit("task", "summarization_skipped",
                         f"{label}: nothing to summarize (no content)")
            return
        stats[level_key] += 1
        summary = self._summarize_unit(
            text, label=label, min_chars=min_chars, max_chars=max_chars,
            context=context, stats=stats, warnings=warnings,
            copied_key=copied_key,
        )
        if summary is None:
            # Mark the failure by leaving unit.summary unset; run() returns
            # a failed result right after seeing it in the stats.
            return
        unit.summary = summary

    def _page_parts(self, page: PageContent) -> List[Optional[str]]:
        return [section.summary for section in page.sections]

    def _chapter_parts(self, chapter: Chapter, page_summaries: List[Optional[str]]) -> List[Optional[str]]:
        # Prefer the aggregated full_text when present, else the pages' summaries.
        if chapter.full_text and chapter.full_text.strip():
            return [chapter.full_text]
        return page_summaries

    def _document_parts(self, document: DocumentExtract) -> List[Optional[str]]:
        parts = [chapter.summary for chapter in document.chapters]
        # Orphan pages only add their summary when they belong to no chapter.
        chapter_page_ids = {id(p) for c in document.chapters for p in c.pages}
        for orphan in document.orphan_pages:
            if id(orphan) not in chapter_page_ids:
                parts.append(orphan.summary)
        return [p for p in parts if p]

    def _failed_result(
        self,
        context: IngestionContext,
        document: DocumentExtract,
        stats: Dict[str, int],
        warnings: List[str],
    ) -> AgentResult:
        """Build the step failure after an LLM-shaped problem."""
        context.emit(
            "task", "summarization_failed",
            f"summarization of '{document.title}' failed after "
            f"{stats['failures']} failure(s); "
            f"{stats['llm_calls']} call(s) succeeded before the failure",
        )
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.FAILED,
            failure_domain=FailureDomain.LLM_RESPONSE,
            detail=(
                f"summarization failed: {stats['failures']} unit(s) could not "
                "be summarized (LLM call error, empty or unreadable answer)"
            ),
            payload={"warnings": warnings, **stats},
        )
