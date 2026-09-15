"""ConsolidationAgent — the pre-summarization merge pass.

The graph step between ``extraction_validation`` and
``hierarchical_summarization`` (agent key ``consolidator``): rebuilds every
section's block list so no narrative fragment shorter than
``paragraph_min_length`` is ever summarized, embedded or fed to knowledge
extraction, while nothing exceeds about ``paragraph_max_length``. Running
BEFORE summarization means every summary describes the final block
structure — no summary is computed for text about to be merged away.

Two rules govern the design:

* **Non-destructive to the stores.** The extraction JSON and the
  summarized JSON are *not* rewritten — the consolidated view lives
  only in the context (and, for human inspection / later resume, in
  ``<cache>/consolidation/<stem>.json``).
* **Deterministic.** No LLM: pure list surgery on the DocumentExtract.
  Same document, same parameters, same output.

Config (config/ingestion.yaml, validated by
``validate_ingestion_config``)::

    consolidation_output_dir: "data/cache/consolidation"   # optional
    paragraph_min_length: 100       # hard low limit (required)
    paragraph_max_length: 1000      # soft high limit (required)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.agents.contexts import IngestionContext
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
from src.consolidation.consolidator import (
    ConsolidationError,
    consolidate_document,
)
from src.extraction.json_store import (
    ExtractJsonError,
    document_extract_from_json_dict,
    document_extract_to_json_dict,
)
from src.tools.config_loader import (
    ConfigError,
    PROJECT_ROOT,
    load_ingestion_config,
)

logger = logging.getLogger("wintermute.consolidation")

#: Context key read and written: the step name IS the key (graph
#: convention). Downstream consumers (source_indexing, knowledge
#: extraction, source registration) keep reading ``content_extraction``
#: and transparently receive the consolidated view.
INPUT_KEY = "content_extraction"

__all__ = [
    "ConsolidationAgent",
    "consolidation_output_dir",
    "consolidation_path_for",
    "load_consolidated_extract",
    "save_consolidated_extract",
]

#: Fallback when ingestion.yaml does not set ``consolidation_output_dir``.
DEFAULT_CONSOLIDATION_OUTPUT_DIR = "data/cache/consolidation"


# --------------------------------------------------------------------------
# Storage tool — the consolidation cache (data/cache/consolidation/<stem>.json)
# --------------------------------------------------------------------------

def consolidation_output_dir() -> Path:
    """Consolidation cache folder from ingestion.yaml
    (``consolidation_output_dir``; default ``data/cache/consolidation``).

    Relative values are resolved against the project root; absolute values
    are used as-is. Fails open to the default folder on a broken config —
    same tolerance as the extraction and summarization stores: the cache is
    an inspection convenience, never a pipeline blocker.
    """
    from src.tools.config_loader import load_ingestion_config  # noqa: F811

    raw = DEFAULT_CONSOLIDATION_OUTPUT_DIR
    try:
        config = load_ingestion_config()
        value = config.get("consolidation_output_dir")
        if isinstance(value, str) and value.strip():
            raw = value
    except Exception as exc:  # noqa: BLE001 — fail-open to the default folder
        logger.warning(
            "Could not read ingestion.yaml for consolidation_output_dir; "
            "using default %s (%s)", DEFAULT_CONSOLIDATION_OUTPUT_DIR, exc,
        )
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def consolidation_path_for(source_path: Path) -> Path:
    """Consolidation cache path for a source document: ``<dir>/<stem>.json``."""
    return consolidation_output_dir() / f"{Path(source_path).stem}.json"


def save_consolidated_extract(document, path: Path) -> Path:
    """Serialize the consolidated DocumentExtract (same canonical schema)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = document_extract_to_json_dict(document)
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ExtractJsonError(f"Écriture impossible ({path}) : {exc}") from exc
    logger.debug("Consolidation cache saved: %s", path)
    return path


def load_consolidated_extract(path: Path):
    """Load a consolidation cache file back into a DocumentExtract.

    Raises:
        FileNotFoundError: the file does not exist.
        ExtractJsonError: the file is not valid JSON or violates the schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExtractJsonError(f"{path} : JSON invalide ({exc}).") from exc
    return document_extract_from_json_dict(payload)


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------


class ConsolidationAgent:
    """Ingestion step: merge too-small text blocks before summarization."""

    name = "consolidator"

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """``output_dir``: consolidation cache folder override (tests);
        None uses the config-driven default (``consolidation_output_dir``
        in ingestion.yaml)."""
        self._output_dir = Path(output_dir) if output_dir else None

    def validate(self, context: IngestionContext) -> Optional[AgentResult]:
        """Nothing beyond the run's own guard (the pass is idempotent)."""
        return None

    def run(self, context: IngestionContext) -> AgentResult:
        document = context.outputs.get(INPUT_KEY)
        if document is None:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail=f"no document_extract in context.outputs[{INPUT_KEY!r}] "
                       "— run content_extraction first",
            )

        try:
            min_length, max_length = self._read_limits()
        except ConfigError as exc:
            context.emit("task", "consolidation_config_error", detail=str(exc))
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail=f"configuration de consolidation invalide : {exc}",
            )

        try:
            consolidated, stats = consolidate_document(
                document, min_length, max_length,
            )
        except ConsolidationError as exc:
            context.emit("task", "consolidation_error", detail=str(exc))
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail=str(exc),
            )

        cache_path: Optional[Path] = None
        try:
            source = context.document_path
            stem = Path(source).stem if source is not None else "document"
            out_dir = self._output_dir or consolidation_output_dir()
            cache_path = save_consolidated_extract(
                consolidated, out_dir / f"{stem}.json",
            )
        except ExtractJsonError as exc:
            # The cache is an inspection/resume convenience, not a pipeline
            # product: a write failure degrades to a warning.
            context.emit("task", "consolidation_cache_failed", detail=str(exc))
            context.errors[self.name] = f"consolidation cache write failed: {exc}"

        context.outputs[INPUT_KEY] = consolidated
        context.outputs[self.name] = stats.as_dict()
        if cache_path is not None:
            context.outputs[self.name]["cache"] = str(cache_path)

        payload: Dict[str, Any] = {**stats.as_dict(), "cache": str(cache_path)}
        context.emit("task", "consolidation_done", data=payload)
        logger.info(
            "Consolidation : %d bloc(s) avant, %d après (%d fusion(s)).",
            stats.blocks_before, stats.blocks_after, stats.merged_away,
        )
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.OK,
            payload=payload,
        )

    # -- internals ----------------------------------------------------------

    def _read_limits(self) -> tuple:
        """Strict config read: min/max are REQUIRED, max >= min."""
        config = load_ingestion_config()
        min_length = config.get("paragraph_min_length")
        if not isinstance(min_length, int) or isinstance(min_length, bool) \
                or min_length <= 0:
            raise ConfigError(
                "'paragraph_min_length' doit être un entier strictement "
                f"positif (reçu : {min_length!r}) — ajoutez la clé à "
                "config/ingestion.yaml."
            )
        max_length = config.get("paragraph_max_length")
        if not isinstance(max_length, int) or isinstance(max_length, bool) \
                or max_length < min_length:
            raise ConfigError(
                "'paragraph_max_length' doit être un entier >= "
                f"paragraph_min_length ({min_length}) (reçu : {max_length!r})."
            )
        return min_length, max_length
