"""ExtractionValidationAgent — data-consistency pass on the extraction.

The second graph step (``extraction_validation``, agent key
``extraction_validator``). The *extraction* agent already gated the shape
(pedantic structural validation, ``src/extraction/validation.py``); this
agent checks the extraction's **logical coherence** and cleans it up:

* page numbers coherent with ``total_pages`` (out-of-bounds pages removed,
  unsorted lists warned, never silently reordered);
* blank-text objects (spaces/tabs/newlines only) pruned, with a warning —
  this lightens the final ``DocumentExtract``;
* empty containers (sections without blocks, pages without sections,
  chapters without pages) pruned, with a warning;
* section/page-number realignment to the containing page;
* a document left with **no content at all** is a failure (INPUT_DATA —
  there is nothing to ingest), not a cleanup.

Warnings are non-fatal by design (they never stop the ingestion): they are
emitted as ``task``-phase traces (visible in the thinking panel) and ride
the result payload. The pass itself lives in
``src/extraction/consistency.py`` (pure logic, idempotent); this agent
only wires it into the graph with tracing and status mapping.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.agents.contexts import IngestionContext
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
from src.extraction.consistency import check_and_prune, has_content
from src.extraction.models import DocumentExtract
from src.tools.config_loader import get_default_origin

logger = logging.getLogger(__name__)

#: Context key this agent reads from (the extraction step's output) and
#: writes to (the cleaned document replaces it in place).
INPUT_KEY = "content_extraction"

AGENT_NAME = "extraction_validator"


class ExtractionValidationAgent:
    """Validates and cleans the extraction's data consistency in place."""

    name = AGENT_NAME

    def run(self, context: IngestionContext) -> AgentResult:
        document: Optional[DocumentExtract] = context.outputs.get(INPUT_KEY)

        if document is None:
            # The extraction step did not produce anything to validate —
            # either it never ran or it failed; nothing to do here.
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SKIPPED,
                detail="no extraction output to validate",
            )

        title = getattr(document, "title", "?")
        context.emit(
            "task", "consistency_check",
            f"checking data consistency of '{title}' "
            f"({document.total_pages} declared page(s))",
        )

        # Governance check — origin must be present. The model defaults to
        # the configured default (first entry of setup.yaml
        # documents.origins), so the *field* is never empty; the point of
        # this check is to make an UNVERIFIED default visible (the CLI
        # should always provide document_origin in the metadata).
        if not context.metadata.get("document_origin"):
            context.emit(
                "task", "origin_missing",
                f"document origin unknown at validation time ('{get_default_origin()}' "
                "assumed by default, unverified)",
            )
        report = check_and_prune(document)

        # Warnings never stop the ingestion: trace each one (the thinking
        # panel shows what was cleaned) and cap the spam for huge reports.
        for warning in report.warnings[:20]:
            context.emit("task", "consistency_warning", warning)
        if len(report.warnings) > 20:
            context.emit(
                "task", "consistency_warning",
                f"... and {len(report.warnings) - 20} more warning(s)",
            )
        for warning in report.warnings:
            logger.info("Consistency: %s", warning)

        if not has_content(document):
            context.emit(
                "task", "consistency_failed",
                "no content left after consistency pruning — nothing to ingest",
                removals=report.removals,
            )
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail=(
                    "extraction is empty after consistency checks "
                    f"({report.removals} object(s) removed)"
                ),
                payload={"report": report.as_payload()},
            )

        context.emit(
            "task", "consistency_done",
            _summary_line(report),
            removals=report.removals,
            warnings=len(report.warnings),
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.OK,
            payload={"report": report.as_payload()},
        )

    def validate(self, context: IngestionContext) -> Optional[AgentResult]:
        """Nothing beyond the run's own checks (the pass is idempotent)."""
        return None


def _summary_line(report) -> str:
    """One-line human summary of the report for the traces."""
    parts = []
    if report.removals:
        details = [
            f"{count} {noun}" for noun, count in (
                ("block(s)", report.removed_blocks),
                ("section(s)", report.removed_sections),
                ("page(s)", report.removed_pages),
                ("chapter(s)", report.removed_chapters),
            ) if count
        ]
        parts.append("removed " + ", ".join(details))
    if report.realigned:
        parts.append(f"realigned {report.realigned} section(s)")
    if report.out_of_bounds_pages:
        parts.append(f"{report.out_of_bounds_pages} out-of-bounds page(s)")
    if report.unsorted_lists:
        parts.append(f"{report.unsorted_lists} unsorted list(s) (warned, kept)")
    if not parts:
        return "extraction is coherent — nothing to clean"
    return "consistency pass: " + "; ".join(parts)
