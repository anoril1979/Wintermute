"""EntityExtractionAgent — LLM knowledge extraction, one pass per entity type.

The ``knowledge_extraction`` graph step (agent key ``knowledge_extractor``).
Runs after source indexing: reads the extracted ``DocumentExtract`` from the
context, walks its content units (sections by default — the smallest unit
still carrying heading context; page or chapter granularity is a
``knowledge_unit_granularity`` setting in ingestion.yaml) and prompts the
LLM once per unit with the dedicated extraction prompt.

This is the ONLY LLM-based step of the knowledge layer: the LLM identifies
and structures content; everything downstream (validation, check-n-merge,
storage) is deterministic Python.

Architecture: the reusable walk-and-extract loop lives in
:class:`EntityExtractionAgent`; each entity type derives it with its own
prompt (``prompts/knowledge/<type>_extraction.md``) and payload semantics.
The first derivation is :class:`CharacterExtractionAgent` (characters with
``full_name`` / ``short_name`` / ``aliases``); future ones (places,
organizations, objects, events, claims) follow the same shape.

Output: the collected entries are stored as a plain, human-editable JSON
file under ``knowledge_output_dir`` (config/ingestion.yaml, default
``data/cache/knowledge``) as ``<document stem>.json`` — see
``src/knowledge/character_cache.py``. The step output also lands in
``context.outputs[OUTPUT_KEY]`` for the in-memory consumers (the future
knowledge-validation step reads the stores, not the context).

Failure handling: the graph's retry policy applies — an LLM call error or
a malformed answer fails the step with ``FailureDomain.LLM_RESPONSE``
(retryable). An empty extraction is a legitimate ``characters: []``
result — never a failure. A unit the LLM reports unreadable via the
prompt's ``error`` marker only skips THAT unit (with a warning): retrying
the whole step could not fix it and would re-call the LLM for every
already-analyzed unit.

Traces (``task`` phase, visible in the thinking panel):
``knowledge_start`` (granularity + unit count), ``knowledge_unit`` per
unit, ``knowledge_skipped`` for blank units, ``knowledge_deduped`` when
cross-unit duplicates are merged, ``knowledge_done`` / ``knowledge_failed``
at the end.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from src.agents.contexts import IngestionContext
from src.agents.llm_roles import LLMRoleAgent
from src.agents.protocols import AgentResult, AgentStatus, FailureDomain
from src.extraction.models import (
    Chapter,
    DocumentExtract,
    PageContent,
    Section,
)
from src.knowledge.character_cache import (
    DEFAULT_GRANULARITY,
    KnowledgeJsonError,
    knowledge_cache_path_for,
    save_knowledge,
)

logger = logging.getLogger(__name__)


class UnitUnreadableError(ValueError):
    """The LLM reported the unit unreadable via the prompt's ``error`` marker.

    Deliberately NOT a step failure: one unit without extractable content
    (e.g. a purely technical section) is normal. The unit is skipped with a
    warning and the walk continues — a step-level retry could not change the
    outcome and would re-call the LLM for every unit already analyzed."""

#: Context key this agent reads (the extraction produced by content_extraction).
INPUT_KEY = "content_extraction"

#: Context key / payload key this agent writes its result under.
OUTPUT_KEY = "knowledge_characters"

#: Marker the prompt defines for an unusable input (LLM protocol): the
#: answer then carries an ``error`` field instead of entries.
ERROR_MARKER_KEY = "error"

#: Valid unit granularities (mirrors the config loader's validation of
#: ``knowledge_unit_granularity``).
GRANULARITIES = ("section", "page", "chapter")

#: Where the characters prompt lives (CharacterExtractionAgent default).
CHARACTER_PROMPT_PATH = Path("prompts/knowledge/character_extraction.md")


class EntityExtractionAgent(LLMRoleAgent):
    """Walks the document's content units and extracts entities via the LLM.

    Subclasses declare the prompt (``prompt_path`` or ``PROMPT_PATH`` class
    attribute) and the role; the base class owns the unit walk, the
    prompt/call/parse cycle, cross-unit dedup and the cache persistence.
    """

    #: Prompt file for this entity type (subclass MUST set it).
    PROMPT_PATH: Path = CHARACTER_PROMPT_PATH

    #: Key of the entry list inside the LLM's JSON answer (subclass MUST
    #: set it, e.g. "characters" for characters, "places" for places).
    OUTPUT_ENTRY_KEY: str = ""

    def __init__(self, *args: Any, prompt_path: Optional[Path] = None,
                 granularity: Optional[str] = None,
                 output_dir: Optional[Path] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._prompt_path = Path(prompt_path) if prompt_path else Path(self.PROMPT_PATH)
        self._prompt_template: Optional[str] = None
        # Unit granularity override (tests); None reads ingestion.yaml.
        if granularity is not None and granularity not in GRANULARITIES:
            raise ValueError(
                f"granularity must be one of {', '.join(GRANULARITIES)}, "
                f"got {granularity!r}"
            )
        self._granularity = granularity
        # Cache folder override (tests); None uses the config-driven default.
        self._output_dir = Path(output_dir) if output_dir else None

    # -- Configuration ---------------------------------------------------------

    @property
    def granularity(self) -> str:
        """Content unit fed to the LLM (constructor override > config > default)."""
        if self._granularity is not None:
            return self._granularity
        try:
            from src.knowledge.character_cache import knowledge_unit_granularity

            return knowledge_unit_granularity()
        except Exception:  # noqa: BLE001 — fail-open to the default
            return DEFAULT_GRANULARITY

    def _cache_path(self, document_path: Optional[Path]) -> Optional[Path]:
        """Knowledge-cache JSON path for the document being ingested."""
        if document_path is None:
            return None
        if self._output_dir is not None:
            return self._output_dir / f"{document_path.stem}.json"
        return knowledge_cache_path_for(document_path)

    def _prompt_template_text(self) -> str:
        """Load (and cache) the extraction prompt."""
        if self._prompt_template is None:
            self._prompt_template = self._prompt_path.read_text(encoding="utf-8")
        return self._prompt_template

    def _build_prompt(self, text: str, label: str) -> str:
        """Full prompt: instructions + the delimited content unit."""
        return (
            f"{self._prompt_template_text().strip()}\n"
            "\n---\n\n"
            "Content to analyze:\n"
            "<<<<TEXT>>>>\n"
            f"{text}\n"
            "<<<<TEXT>>>>\n"
        )

    # -- Unit walk ---------------------------------------------------------------

    def _iter_units(self, document: DocumentExtract) -> Iterator[Any]:
        """Yield the content units of the document, in reading order, at the
        configured granularity (sections / pages / chapters)."""
        g = self.granularity
        for chapter in document.chapters:
            if g == "chapter":
                yield chapter
                continue
            for page in chapter.pages:
                if g == "page":
                    yield page
                    continue
                for section in page.sections:
                    yield section
        for orphan in document.orphan_pages:
            if g == "page":
                yield orphan
                continue
            for section in orphan.sections:
                yield section

    @staticmethod
    def unit_text(unit: Any) -> str:
        """The text fed to the LLM for one unit.

        Sections and pages use their ``raw_text``; chapters use their
        aggregated ``full_text`` (falling back to the concatenation of their
        pages' raw text when the aggregate is empty).
        """
        text = getattr(unit, "full_text", None) or getattr(unit, "raw_text", "") or ""
        if isinstance(unit, Chapter) and not text.strip():
            text = "\n".join(p.raw_text for p in unit.pages if p.raw_text)
        return text.strip()

    @staticmethod
    def unit_label(unit: Any) -> str:
        """Human-readable unit label for traces and errors."""
        if isinstance(unit, Chapter):
            return f"chapter '{unit.toc_entry.title}' (page {unit.toc_entry.page_number})"
        if isinstance(unit, PageContent):
            return f"page {unit.page_number}"
        if isinstance(unit, Section):
            title = f" '{unit.section_title}'" if unit.section_title else ""
            return f"section {unit.section_id}{title} (page {unit.page_number})"
        return repr(unit)

    # -- LLM cycle -----------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """One LLM call. Raises on transport/config problems (the caller
        maps them to a retryable step failure, like the summarizer)."""
        return self.llm_client().complete(prompt=prompt)

    def _parse_answer(self, raw: str, label: str) -> List[Dict[str, Any]]:
        """Parse the LLM answer into this entity type's entries.

        Returns the list of entries; raises ``ValueError`` when the answer
        is unusable (not JSON, wrong shape) and
        :class:`UnitUnreadableError` when the answer carries the
        prompt-defined ``error`` marker (unit skipped, not a failure).
        Subclasses may narrow the entry validation.
        """
        from src.routing.models import _extract_json_payload

        payload = _extract_json_payload(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"expected a JSON object, got {type(payload).__name__}")
        error = payload.get(ERROR_MARKER_KEY)
        if error:
            raise UnitUnreadableError(f"unit reported unreadable by the LLM: {error}")
        entries = payload.get(self.OUTPUT_ENTRY_KEY)
        if not isinstance(entries, list):
            raise ValueError(
                f"answer must carry a '{self.OUTPUT_ENTRY_KEY}' list, "
                f"got {type(entries).__name__}"
            )
        cleaned: List[Dict[str, Any]] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"{self.OUTPUT_ENTRY_KEY}[{index}] must be an object")
            cleaned.append(self._clean_entry(entry, label, index))
        return cleaned

    def _clean_entry(self, entry: Dict[str, Any], label: str, index: int) -> Dict[str, Any]:
        """Validate/normalize one raw entry. Subclasses override to check
        their own fields; the base class only enforces a non-empty shape."""
        return entry

    # -- Dedup ----------------------------------------------------------------------

    def _entry_identity(self, entry: Dict[str, Any]) -> Optional[tuple]:
        """Dedup key of an entry across units, or None when it cannot be
        identified (it is then kept as-is). Subclasses override."""
        return None

    def _merge_entries(self, kept: Dict[str, Any], duplicate: Dict[str, Any]) -> None:
        """Merge a duplicate into the already-kept entry (in place).
        Default: nothing (identity-less entries are never merged)."""

    # -- Template ---------------------------------------------------------------------

    def run(self, context: IngestionContext) -> AgentResult:
        document: Optional[DocumentExtract] = context.outputs.get(INPUT_KEY)

        if document is None:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SKIPPED,
                detail="no extraction output to analyze",
            )

        units = [u for u in self._iter_units(document) if self.unit_text(u)]
        g = self.granularity
        context.emit(
            "task", "knowledge_start",
            f"extracting {self.OUTPUT_ENTRY_KEY} from '{document.title}' "
            f"({len(units)} {g}(s), role '{self._llm_role}')",
            document=document.title, units=len(units), granularity=g,
        )

        entries: List[Dict[str, Any]] = []
        seen: Dict[tuple, int] = {}  # identity -> index in entries
        merged = 0
        llm_calls = 0

        for unit in units:
            label = self.unit_label(unit)
            text = self.unit_text(unit)
            try:
                raw = self._call_llm(self._build_prompt(text, label))
            except Exception as exc:  # LLMClientError / ValueError / transport
                logger.warning("Knowledge LLM call failed for %s: %s", label, exc)
                context.emit("task", "knowledge_failed",
                             f"LLM call failed for {label}: {exc}", label=label)
                return self._failed_result(context, document, entries, llm_calls,
                                           f"LLM call failed for {label}: {exc}")
            llm_calls += 1
            try:
                found = self._parse_answer(raw, label)
            except UnitUnreadableError as exc:
                # One unit the LLM could not read is normal (e.g. a purely
                # technical section): warn, skip the unit, keep walking.
                # A step-level retry could not fix this unit and would
                # re-call the LLM for every unit already analyzed.
                logger.warning("Unit skipped (unreadable) %s: %s", label, exc)
                context.emit("task", "knowledge_unit_skipped",
                             f"{label}: skipped — {exc}", label=label)
                continue
            except ValueError as exc:
                logger.warning("Unusable knowledge answer for %s: %s", label, exc)
                context.emit("task", "knowledge_failed",
                             f"unusable LLM answer for {label}: {exc}", label=label)
                return self._failed_result(context, document, entries, llm_calls,
                                           f"unusable LLM answer for {label}: {exc}")

            context.emit("task", "knowledge_unit",
                         f"{label}: {len(found)} {self.OUTPUT_ENTRY_KEY} found",
                         label=label, found=len(found))
            for entry in found:
                identity = self._entry_identity(entry)
                if identity is not None and identity in seen:
                    merged += 1
                    self._merge_entries(entries[seen[identity]], entry)
                    continue
                if identity is not None:
                    seen[identity] = len(entries)
                entries.append(entry)

        if merged:
            context.emit("task", "knowledge_deduped",
                         f"{merged} cross-unit duplicate(s) merged "
                         "(final merge belongs to the check-n-merge step)",
                         merged=merged)

        # Persistence: the plain, human-editable cache file. Non-fatal on
        # failure — the entries live in the context payload either way.
        saved_path: Optional[Path] = None
        document_path = getattr(context, "document_path", None)
        cache_path = self._cache_path(document_path)
        if cache_path is not None:
            try:
                saved_path = save_knowledge(entries, cache_path)
            except (OSError, KnowledgeJsonError) as exc:
                logger.warning("Could not save the knowledge cache: %s", exc)
                context.errors[OUTPUT_KEY] = f"knowledge cache not saved: {exc}"
                context.emit("task", "knowledge_save_failed",
                             f"could not save the knowledge cache: {exc}")
            else:
                context.emit("task", "knowledge_saved",
                             f"knowledge cached: {saved_path.name}",
                             path=str(saved_path))

        context.emit(
            "task", "knowledge_done",
            f"knowledge extraction done: {len(entries)} {self.OUTPUT_ENTRY_KEY} "
            f"from {len(units)} {g}(s), {llm_calls} LLM call(s)",
            **{self.OUTPUT_ENTRY_KEY: len(entries), "units": len(units),
               "llm_calls": llm_calls, "merged": merged, "granularity": g},
        )

        payload: Dict[str, Any] = {
            self.OUTPUT_ENTRY_KEY: entries,
            "llm_calls": llm_calls,
            "units": len(units),
            "merged": merged,
            "granularity": g,
            "cache_path": str(saved_path) if saved_path else None,
        }
        context.outputs[OUTPUT_KEY] = payload
        return AgentResult(
            agent_name=self.name, status=AgentStatus.OK, payload=payload,
        )

    def validate(self, context: IngestionContext) -> Optional[AgentResult]:
        """After a successful run, the cache file must exist on disk when the
        step had a document to work on (a missing file means the store
        failed silently — a data-shaped problem, not an LLM one)."""
        if context.outputs.get(OUTPUT_KEY) is None:
            return None
        document_path = getattr(context, "document_path", None)
        cache_path = self._cache_path(document_path)
        if cache_path is not None and not cache_path.exists():
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                failure_domain=FailureDomain.INPUT_DATA,
                detail=f"knowledge cache missing after the step: {cache_path}",
            )
        return None

    # -- internals ---------------------------------------------------------------

    def _failed_result(
        self,
        context: IngestionContext,
        document: DocumentExtract,
        entries: List[Dict[str, Any]],
        llm_calls: int,
        detail: str,
    ) -> AgentResult:
        """Build the step failure after an LLM-shaped problem."""
        context.emit(
            "task", "knowledge_failed",
            f"knowledge extraction of '{document.title}' failed: {detail} "
            f"({llm_calls} call(s) succeeded before the failure)",
        )
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.FAILED,
            failure_domain=FailureDomain.LLM_RESPONSE,
            detail=f"knowledge extraction failed: {detail}",
            payload={self.OUTPUT_ENTRY_KEY: entries, "llm_calls": llm_calls},
        )


class CharacterExtractionAgent(EntityExtractionAgent):
    """Extracts the characters of the document (first knowledge pass).

    Per content unit, the LLM returns
    ``{"characters": [{"full_name", "short_name", "aliases"}, ...]}``:
    ``full_name`` is the most precise name found in the unit, ``aliases``
    every other way the source refers to the character (short names,
    titles, pseudonyms — pronouns excluded). Cross-unit duplicates (same
    full name + short name) are merged by union of aliases here; the real
    identity resolution belongs to the later check-n-merge step.
    """

    name = "character_extractor"
    llm_role = "knowledge_extractor"
    PROMPT_PATH = CHARACTER_PROMPT_PATH

    #: Key of the entry list inside the LLM's JSON answer.
    OUTPUT_ENTRY_KEY = "characters"

    def _clean_entry(self, entry: Dict[str, Any], label: str, index: int) -> Dict[str, Any]:
        """Enforce the character shape: non-empty string names, list aliases."""
        full_name = entry.get("full_name")
        if not isinstance(full_name, str) or not full_name.strip():
            raise ValueError(
                f"characters[{index}] ({label}): 'full_name' must be a "
                "non-empty string"
            )
        aliases = entry.get("aliases", [])
        if aliases is None:
            aliases = []
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) for alias in aliases
        ):
            raise ValueError(
                f"characters[{index}] ({label}): 'aliases' must be a list "
                "of strings"
            )
        short_name = entry.get("short_name")
        if short_name is not None and not isinstance(short_name, str):
            raise ValueError(
                f"characters[{index}] ({label}): 'short_name' must be a string"
            )
        return {
            "full_name": full_name.strip(),
            "short_name": (short_name or "").strip() or None,
            "aliases": [alias.strip() for alias in aliases if alias.strip()],
        }

    def _entry_identity(self, entry: Dict[str, Any]) -> Optional[tuple]:
        full_name = (entry.get("full_name") or "").strip().lower()
        if not full_name:
            return None
        short_name = (entry.get("short_name") or "").strip().lower()
        return (full_name, short_name)

    def _merge_entries(self, kept: Dict[str, Any], duplicate: Dict[str, Any]) -> None:
        """Union of aliases (order-preserving); first full/short names win."""
        kept_aliases = kept.setdefault("aliases", [])
        for alias in duplicate.get("aliases", []):
            if alias and alias not in kept_aliases:
                kept_aliases.append(alias)
