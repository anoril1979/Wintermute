"""Metadata filters for fine-grained vector retrieval.

The chunk metadata written by ``build_source_chunks`` is the filter
vocabulary: doc identity (``doc_id``, ``doc_title``, ``source_path``),
hierarchical position (``level``, ``elem_id``, ``full_id``, ``chapter_id``,
``chapter_title``, ``page_id``, ``page_number``, ``section_id``), kind
(``content`` vs ``summary``) and governance (``origin``).

Retrieval filters are built by **Python only** — the router LLM may extract
a document or chapter name from the user's words, but the ChromaDB ``where``
clause itself is assembled here, deterministically, from validated values.

ChromaDB where-clause dialect (subset used here):

* ``{"field": value}``                     — equality;
* ``{"field": {"$eq": v}}`` etc.           — operators ($eq, $ne, $gte,
  $lte, $in, $nin);
* ``{"$and": [clause, ...]}``              — conjunction (a bare dict with
  several keys is ambiguous across versions, so multi-condition filters
  are ALWAYS wrapped in ``$and``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.indexing.chunks import (
    KIND_CONTENT,
    KIND_SUMMARY,
    LEVEL_BLOCK,
    LEVEL_CHAPTER,
    LEVEL_DOCUMENT,
    LEVEL_PAGE,
    LEVEL_SECTION,
)


class InvalidFilterError(ValueError):
    """A filter value is unusable (empty, path-like, wrong type)."""


#: Metadata fields a filter may target (everything ``build_source_chunks``
#: writes). Anything else is refused — a typo must surface, not silently
#: match nothing.
ALLOWED_FIELDS = frozenset({
    "doc_id", "doc_title", "source_path", "total_pages", "origin",
    "level", "kind", "elem_id", "full_id",
    "chapter_id", "chapter_title", "page_id", "page_number", "section_id",
    "block_type", "section_title", "start_page", "end_page",
})

_LEVELS = frozenset({LEVEL_BLOCK, LEVEL_SECTION, LEVEL_PAGE, LEVEL_CHAPTER,
                     LEVEL_DOCUMENT})
_KINDS = frozenset({KIND_CONTENT, KIND_SUMMARY})
_ORIGINS = frozenset({"canon", "community", "rpg"})

#: Guard against path-like values in free-text scopes ("data/sources/x.pdf"
#: is not a doc_title; bare names are).
_PATHISH_RE = re.compile(r"[/\\]|\.\.")


def _clean_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidFilterError(f"{name} must be a non-empty string, got {value!r}")
    cleaned = value.strip()
    if _PATHISH_RE.search(cleaned):
        raise InvalidFilterError(
            f"{name} must be a bare name, got path-like {value!r}"
        )
    return cleaned


@dataclass
class RetrievalFilters:
    """Validated, user-visible filter facets for one retrieval request.

    Attributes:
        document:     bare document name (matched against ``doc_title``,
                      case-insensitively via a ChromaDB $in on lowercased
                      candidates is NOT possible — ChromaDB has no case
                      folding, so the exact spelling is required; the
                      router copies it from the user's words and the
                      matcher below also tries common casings).
        chapter_title: chapter title scope (exact match on the stored
                       metadata, same case caveat as above).
        origins:      governance filter (subset of canon/community/rpg).
        kinds:        content vs summary chunks; None = both.
        levels:       structural levels to keep; None = all.
        doc_ids:      explicit document ids (unified scheme) — used by
                      prompt-local resolution (a preceding ingestion), not
                      by the router.
        page_number:  exact page scope (rare, fine-grained).
    """

    document: Optional[str] = None
    chapter_title: Optional[str] = None
    origins: Optional[List[str]] = None
    kinds: Optional[List[str]] = None
    levels: Optional[List[str]] = None
    doc_ids: Optional[List[str]] = None
    page_number: Optional[int] = None

    def is_empty(self) -> bool:
        """True when no facet is set (no where-clause needed)."""
        return not any([
            self.document, self.chapter_title, self.origins, self.kinds,
            self.levels, self.doc_ids,
            self.page_number is not None,
        ])

    def summary(self) -> Dict[str, Any]:
        """Compact dict for traces/payloads (None facets omitted)."""
        data: Dict[str, Any] = {}
        if self.document:
            data["document"] = self.document
        if self.chapter_title:
            data["chapter_title"] = self.chapter_title
        if self.origins:
            data["origins"] = list(self.origins)
        if self.kinds:
            data["kinds"] = list(self.kinds)
        if self.levels:
            data["levels"] = list(self.levels)
        if self.doc_ids:
            data["doc_ids"] = list(self.doc_ids)
        if self.page_number is not None:
            data["page_number"] = self.page_number
        return data

    # -- validation ------------------------------------------------------------

    def _validate(self) -> None:
        if self.document is not None:
            _clean_text("document", self.document)
        if self.chapter_title is not None:
            _clean_text("chapter_title", self.chapter_title)
        for name, values, allowed in (
            ("origins", self.origins, _ORIGINS),
            ("kinds", self.kinds, _KINDS),
            ("levels", self.levels, _LEVELS),
        ):
            if values is None:
                continue
            if not isinstance(values, list) or not values:
                raise InvalidFilterError(
                    f"{name} must be a non-empty list when set"
                )
            for value in values:
                if value not in allowed:
                    raise InvalidFilterError(
                        f"{name}: unknown value {value!r} "
                        f"(allowed: {', '.join(sorted(allowed))})"
                    )
        if self.doc_ids is not None:
            if not isinstance(self.doc_ids, list) or not self.doc_ids:
                raise InvalidFilterError("doc_ids must be a non-empty list when set")
            for doc_id in self.doc_ids:
                _clean_text("doc_ids", doc_id)
                if not doc_id.startswith("doc:"):
                    raise InvalidFilterError(
                        f"doc_ids entries must be unified ids (doc:<hex>), got {doc_id!r}"
                    )
        if self.page_number is not None:
            if isinstance(self.page_number, bool) or not isinstance(self.page_number, int) \
                    or self.page_number <= 0:
                raise InvalidFilterError(
                    f"page_number must be a positive int, got {self.page_number!r}"
                )

    # -- ChromaDB where-clause ---------------------------------------------------

    @staticmethod
    def _eq_candidates(text: str) -> List[str]:
        """Spelling candidates for a case-sensitive equality match.

        ChromaDB metadata matching is case-sensitive; the user may type
        "dumas" for a "Dumas" document. Candidates: as-given, title-cased,
        upper, lower — deduplicated. (A fold-insensitive $regex does not
        exist in the where dialect; candidate expansion is the cheap,
        deterministic answer.)
        """
        seen: List[str] = []
        for candidate in (text, text.title(), text.upper(), text.lower()):
            if candidate not in seen:
                seen.append(candidate)
        return seen

    def build_where(self) -> Optional[Dict[str, Any]]:
        """Build the ChromaDB ``where`` clause — None when no facet is set.

        Deterministic assembly from validated values only; conditions are
        always combined under a single ``$and`` (never a bare multi-key
        dict, whose semantics vary across ChromaDB versions).

        Raises:
            InvalidFilterError: a facet value is unusable (validated first,
                so callers may rely on build_where() alone).
        """
        self._validate()
        if self.is_empty():
            return None

        conditions: List[Dict[str, Any]] = []

        if self.document:
            conditions.append(
                {"doc_title": {"$in": self._eq_candidates(self.document)}}
            )
        if self.chapter_title:
            conditions.append(
                {"chapter_title": {"$in": self._eq_candidates(self.chapter_title)}}
            )
        if self.origins:
            conditions.append({"origin": {"$in": list(self.origins)}})
        if self.kinds:
            conditions.append({"kind": {"$in": list(self.kinds)}})
        if self.levels:
            conditions.append({"level": {"$in": list(self.levels)}})
        if self.doc_ids:
            conditions.append({"doc_id": {"$in": list(self.doc_ids)}})
        if self.page_number is not None:
            conditions.append({"page_number": {"$eq": self.page_number}})

        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


def build_where(filters: Optional[RetrievalFilters] = None) -> Optional[Dict[str, Any]]:
    """Module-level convenience: filters -> where clause (None when empty)."""
    if filters is None:
        return None
    return filters.build_where()
