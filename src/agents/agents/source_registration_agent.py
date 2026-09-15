"""SourceRegistrationAgent — registers the ingested document in the base.

The ``source_registration`` graph step (agent key ``source_registrar``),
wired right after ``check_and_merge``: once the document's entities live
in the knowledge base, the document itself is registered as an entity of
its own — a markdown file named by the document's unified id holding the
metadata the extraction produced (title, file, path, origin, chapter and
page counts), plus the user's free ``Notes:`` section.

Deterministic and idempotent — no LLM anywhere:

* the metadata comes from the ``DocumentExtract`` the pipeline carries
  (``context.outputs["content_extraction"]``) — never re-derived, never
  asked to the LLM;
* the file name is the unified id (``doc:<8hex>`` → ``doc_<8hex>.md``),
  so a re-ingestion REGENERATES the same file in place: system fields are
  refreshed, the user's ``Notes:`` section is preserved verbatim (the
  user may have annotated the source between two ingestions);
* the ``sources.md`` listing is rebuilt from the files at the end of the
  step (single-writer rule: files are the truth, the listing their
  projection) — a hand-deleted registration disappears from the listing,
  a hand-created one is picked up.

Failure handling: the context carries no extraction (an orchestrator bug
or an unplugged upstream) → ``INPUT_DATA`` failure, visible, not silent.
A write/OSError fails the step the same way — the knowledge base must be
writable for the step to claim success.

Traces (``task`` phase): ``source_registration_start``, ``registered``
(created or refreshed), ``source_registration_done`` with the counts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.agents.contexts import IngestionContext
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
from src.extraction.models import DocumentExtract
from src.knowledge.source_markdown_store import (
    SourceMarkdownError,
    document_type_for,
    listing_path_for,
    rebuild_listing,
    source_path_for,
    write_source,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "source_registrar"

#: Context key the extraction agent wrote the DocumentExtract under
#: (the whole pipeline's canonical input).
INPUT_KEY = "content_extraction"

#: Payload key this agent writes its report under.
OUTPUT_KEY = "source_registration"


class SourceRegistrationAgent:
    """Registers the ingested document into ``<base>/sources/<doc_id>.md``."""

    name = AGENT_NAME

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        """Args:
        base_dir: knowledge-base override (tests); the config-driven
            folder is used otherwise.
        """
        self._base_dir_override = Path(base_dir) if base_dir else None

    # -- graph step -----------------------------------------------------------

    def run(self, context: IngestionContext) -> AgentResult:
        document: Optional[DocumentExtract] = context.outputs.get(INPUT_KEY)
        if not isinstance(document, DocumentExtract):
            detail = f"no DocumentExtract in context.outputs[{INPUT_KEY!r}]"
            context.emit("task", "source_registration_failed", detail)
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail=detail,
            )

        doc_id = (document.id or "").strip()
        if not doc_id:
            detail = "DocumentExtract has no id (assign_extract_ids was not run)"
            context.emit("task", "source_registration_failed", detail)
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail=detail,
            )

        base_dir = self._base_dir()
        registration_path = source_path_for(doc_id, base_dir)
        created = not registration_path.is_file()

        context.emit(
            "task", "source_registration_start",
            f"registering source {doc_id!r} in the knowledge base "
            f"({registration_path.name})",
            doc_id=doc_id,
        )

        try:
            file_name = Path(document.source_path).name if document.source_path else ""
            write_source(
                doc_id=doc_id,
                # Readable document type, extension-derived (SourceType
                # vocabulary; 'other' when the extension says nothing).
                doc_type=document_type_for(document.source_path),
                title=document.title,
                file_name=file_name,
                path=document.source_path,
                origin=document.origin,
                chapters=len(document.chapters),
                pages=int(document.total_pages or 0),
                registration_path=registration_path,
            )
            # Single-writer rule: the listing is the files' projection.
            listing_path = rebuild_listing(base_dir)
        except (SourceMarkdownError, OSError, ValueError) as exc:
            detail = f"source registration failed: {exc}"
            logger.warning("%s", detail)
            context.emit("task", "source_registration_failed", detail)
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail=detail,
            )

        payload = {
            "doc_id": doc_id,
            "file": registration_path.name,
            "type": document_type_for(document.source_path),
            "created": created,
            "listing": str(listing_path),
            "chapters": len(document.chapters),
            "pages": int(document.total_pages or 0),
            "origin": document.origin,
        }
        context.emit(
            "task", "registered",
            ("source registered: " if created else "source refreshed: ")
            + f"{registration_path.name} — {len(document.chapters)} chapter(s), "
            f"{int(document.total_pages or 0)} page(s), origin {document.origin!r}",
            **payload,
        )
        context.emit(
            "task", "source_registration_done",
            f"source registration done ({'created' if created else 'refreshed'})",
            **payload,
        )
        context.outputs[OUTPUT_KEY] = payload
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.OK,
            detail=f"source {doc_id} {'created' if created else 'refreshed'}",
            payload=payload,
        )

    def validate(self, context: IngestionContext) -> Optional[AgentResult]:
        """After a successful run, the file and the listing must exist."""
        if context.outputs.get(OUTPUT_KEY) is None:
            return None
        base_dir = self._base_dir()
        payload = context.outputs[OUTPUT_KEY]
        registration_path = source_path_for(str(payload.get("doc_id")), base_dir)
        if not registration_path.is_file():
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail=f"source registration missing after the step: {registration_path}",
            )
        if not listing_path_for(base_dir).is_file():
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail="source listing missing after the step: sources.md",
            )
        return None

    # -- internals ------------------------------------------------------------

    def _base_dir(self) -> Path:
        """Knowledge-base folder (constructor override > config)."""
        if self._base_dir_override is not None:
            return self._base_dir_override
        from src.knowledge.character_markdown_store import knowledge_base_dir

        return knowledge_base_dir()
