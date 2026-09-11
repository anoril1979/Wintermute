"""Pydantic model definitions for the knowledge layer."""

import os
import re
import unicodedata
from enum import Enum
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# -------------------------------------------------------------------
# Source types
# -------------------------------------------------------------------


class SourceType(str, Enum):
    """Type of a documentary source."""

    PDF = "pdf"
    ONENOTE = "onenote"
    HTML = "html"
    WORD = "word"
    OPENOFFICE = "openoffice"
    TEXT = "text"
    MARKDOWN = "markdown"
    URL = "url"
    OTHER = "other"


# -------------------------------------------------------------------
# Unique IDs
# -------------------------------------------------------------------

# Process-local counter used to auto-generate sequential claim ids.

def _make_id_factory(prefix: str):
    """Return a factory generating sequential ids like ``<prefix>:001``."""

    count = 0

    def factory() -> str:
        nonlocal count
        count += 1
        return f"{prefix}:{count:03d}"

    return factory


_next_claim_id = _make_id_factory("claim")

# NOTE — Source ids are NOT generated here anymore: a source id is the
# document's extraction id (``doc:<8hex>``, src/extraction/ids.py) — one
# unified identity across the extraction stores, the vector chunks and the
# knowledge layer. SourceLocator.id is therefore required at construction;
# producers build locators from a DocumentExtract's id.


# -------------------------------------------------------------------
# Source locator
# -------------------------------------------------------------------

# Entity references look like "<kind>:<identifier>", e.g. "source:001",
# "char:jean", "place:001" — numeric ids remain valid.
_ENTITY_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*:[A-Za-z0-9][A-Za-z0-9_\-]*$")


class Page(BaseModel):
    """A page inside a section, holding a number of paragraphs.

    Attributes:
        page_index: The page's DOCUMENT-GLOBAL 0-based number (e.g. 15 and
            17 for two pages collected from a 300-page book) — documents
            number their pages continuously across sections. Required:
            collected structure is always tagged explicitly.
        num_paragraphs: Number of paragraphs on the page (0 for an empty
            page).
    """

    page_index: int = Field(ge=0)
    num_paragraphs: int = Field(default=0, ge=0)


class Section(BaseModel):
    """A section (chapter, heading, internal book, ...) of a document.

    Attributes:
        section_index: The section's DOCUMENT-GLOBAL 0-based number (e.g. 2
            for chapter 3 of a book) — the first section collected is not
            necessarily the document's first section. Required: collected
            structure is always tagged explicitly.
        pages: The pages collected for this section (at least one; possibly
            a sparse subset of the section's real pages, each tagged with
            its document-global ``page_index``).
    """

    section_index: int = Field(ge=0)
    pages: List[Page] = Field(default_factory=lambda: [Page(page_index=0)], min_length=1)


class SourceLocator(BaseModel):
    """Describes where a documentary-assistant element comes from.

    Attributes:
        source_type: Type of the source — PDF, OneNote, HTML, Word or
            OpenOffice document, raw or Markdown text file, ...
        title: Title of the document — file name, book/PDF title, website
            name for a URL, ...
        path: Path to the document — a local relative path or a valid URL.
        id: Unique identifier of the document — the extraction layer's
            unified id (``doc:<8hex>``, src/extraction/ids.py), shared by
            the extraction JSON stores, the vector chunks and the knowledge
            layer. Required at construction (no auto-generation: identity
            must come from the extraction, never be invented here).
        page_count: Total number of pages in the document — a GLOBAL,
            continuous count (a book is made of N pages, numbered 1..N
            across all sections). Always >= 1: a document has at least one
            page, otherwise it is not a document. Required, and NOT derived
            from the collected sections: a partially-collected 300-page
            book has page_count=300 even if only 2 pages were parsed.
        sections: Ordered list of the sections COLLECTED so far, each
            tagged with its document-global ``section_index`` and holding
            the pages collected for it (each tagged with its document-
            global ``page_index``) — possibly a sparse subset of the real
            document (e.g. a 300-page book where only chapter 3, pages 15
            and 17, were parsed). Sections may not exist in the document
            itself (a single-page document) but the model still represents
            that as one section — "no section" is a section. For formats
            without sections or pages (e.g. an HTML page), the locator
            carries a single global section containing a single global
            page.
    """

    source_type: SourceType
    title: str
    path: Optional[str] = Field(default=None)
    id: str
    page_count: int = Field(ge=1)
    sections: List[Section] = Field(
        default_factory=lambda: [Section(section_index=0, pages=[Page(page_index=0)])],
        min_length=1,
    )

    @model_validator(mode="after")
    def _validate_structure(self) -> "SourceLocator":
        """Integrity of the collected structure: unique section/page
        indices, and no collected page beyond ``page_count``."""
        seen_sections: set = set()
        for section in self.sections:
            if section.section_index in seen_sections:
                raise ValueError(
                    f"duplicate section_index {section.section_index} in sections"
                )
            seen_sections.add(section.section_index)
            seen_pages: set = set()
            for page in section.pages:
                if page.page_index in seen_pages:
                    raise ValueError(
                        f"duplicate page_index {page.page_index} "
                        f"in section {section.section_index}"
                    )
                seen_pages.add(page.page_index)
                if page.page_index >= self.page_count:
                    raise ValueError(
                        f"page_index {page.page_index} is beyond page_count "
                        f"{self.page_count}"
                    )
        return self

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: Optional[str]) -> Optional[str]:
        """Accept only a non-empty local relative path or a valid URL."""
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("path must be a non-empty relative path or a valid URL")
        if "\x00" in value:
            raise ValueError("path must not contain null bytes")
        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc:
            return value  # absolute URL
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", value):
            # Has a scheme prefix ("https://...") but no host — malformed URL
            raise ValueError("path looks like a URL but is malformed")
        # Windows drive-absolute paths ("C:\\...", "c:/...") are not relative
        if os.path.isabs(value) or re.match(r"^[a-zA-Z]:[\\/]", value):
            raise ValueError("path must be a relative path or a valid URL")
        return value


# -------------------------------------------------------------------
# Source reference (lightweight pointer, embedded in claims)
# -------------------------------------------------------------------

class SourceRef(BaseModel):
    """Lightweight reference to a precise point in a document.

    Claims carry these instead of full model instances to keep the data
    lean: locator existence and index bounds are checked by the
    SourceValidator (src/validation/source_validator.py).

    Attributes:
        locator_id: Id of an existing SourceLocator (e.g. ``source:001``)
            describing the document the information comes from.
        section_index: The section's DOCUMENT-GLOBAL 0-based number — it
            must match a collected ``Section.section_index`` of the
            locator. None means the document's first section (index 0),
            which is only valid when that section was collected — a
            partially-collected locator requires an explicit index.
        page_index: The page's DOCUMENT-GLOBAL 0-based number — documents
            number their pages continuously across sections (they
            essentially never restart numbering at each section). It must
            be lower than the locator's ``page_count`` and match a page
            collected in the targeted section. None means page 0, which
            is only valid when page 0 was collected in the targeted
            section.
        paragraph_index: 0-based paragraph on the page, when pinpointed.
    """

    locator_id: str
    section_index: Optional[int] = Field(default=None, ge=0)
    page_index: Optional[int] = Field(default=None, ge=0)
    paragraph_index: Optional[int] = Field(default=None, ge=0)

    @field_validator("locator_id")
    @classmethod
    def _validate_locator_ref(cls, value: str) -> str:
        """The locator reference must look like ``<kind>:<identifier>``."""
        value = value.strip()
        if not _ENTITY_ID_PATTERN.match(value):
            raise ValueError(
                "locator_id must look like '<kind>:<identifier>', e.g. 'source:001'"
            )
        return value

    def __str__(self) -> str:
        """Compact citation, e.g. ``source:001, ch.1 p.2 §5`` (indices
        rendered 1-based, unset ones omitted)."""
        parts = []
        if self.section_index is not None:
            parts.append(f"ch.{self.section_index + 1}")
        if self.page_index is not None:
            parts.append(f"p.{self.page_index + 1}")
        if self.paragraph_index is not None:
            parts.append(f"§{self.paragraph_index + 1}")
        return f"{self.locator_id}, {' '.join(parts)}" if parts else self.locator_id


# -------------------------------------------------------------------
# Entities
# -------------------------------------------------------------------

class EntityType(str, Enum):
    """Kind of entity a claim can refer to.

    Extend this list (and ``_ENTITY_TYPE_PREFIXES`` below) when adding new
    entity models.
    """

    CHARACTER = "character"
    PLACE = "place"
    OBJECT = "object"
    EVENT = "event"
    ORGANIZATION = "organization"


# Maps an entity-id prefix to its EntityType, e.g. "char:jean" -> CHARACTER.
# Used by the Entity consistency check below and by the validation/ingestion
# layers (e.g. creating entities on the fly from an id prefix).
# Update this mapping when adding a new EntityType.
_ENTITY_TYPE_PREFIXES = {
    "char": EntityType.CHARACTER,
    "character": EntityType.CHARACTER,
    "place": EntityType.PLACE,
    "object": EntityType.OBJECT,
    "obj": EntityType.OBJECT,
    "event": EntityType.EVENT,
    "organization": EntityType.ORGANIZATION,
    "org": EntityType.ORGANIZATION,
}


def _entity_type_of(entity_id: str) -> Optional[EntityType]:
    """Return the EntityType matching an entity id prefix, or None."""
    prefix = entity_id.split(":", 1)[0].lower()
    return _ENTITY_TYPE_PREFIXES.get(prefix)


class Entity(BaseModel):
    """Base for every knowledge entity (Character, and future Place,
    Object, Event, Organization models).

    Attributes:
        id: Unique identifier of the form ``<kind>:<identifier>`` (e.g.
            ``char:jean``, ``place:paris``) — readable, and stable across
            runs so entities can be created on the fly without checking
            for existing ids; knowledge validators merge duplicates from
            these ids.
        type: EntityType of the entity; the validation layer checks it
            against the claim's predicate definition (see
            src/validation.ClaimValidator).
    """

    id: str
    type: EntityType

    @model_validator(mode="after")
    def _check_type_matches_prefix(self) -> "Entity":
        """The id prefix, when recognized, must agree with ``type``."""
        expected = _entity_type_of(self.id)
        if expected is not None and expected is not self.type:
            raise ValueError(
                f"id '{self.id}' looks like a {expected.value} "
                f"but type is '{self.type.value}'"
            )
        return self


# -------------------------------------------------------------------
# Characters
# -------------------------------------------------------------------

def _slugify(name: str) -> str:
    """Slugify a name for use in an entity id, e.g. 'Édmond Dantès' -> 'edmond_dantes'."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text


class Character(Entity):
    """A character appearing across the documentary sources.

    Attributes:
        full_name: The name most often used for the character.
        short_name: Short name extracted by the dedicated LLM; the id
            derives from it when not given explicitly.
        aliases: Other names the character may appear under in the sources —
            short names, hypocoristics, actual aliases, pseudonyms, ...
        id: Unique identifier of the form ``char:<short_name>`` — slug of
            the LLM-extracted short name (e.g. ``char:jean``) — keeping
            extracted data readable and letting characters be created on
            the fly without checking for existing ids; the knowledge
            validators check, create or merge characters from these ids.
            An explicitly provided id always wins; falls back to a slug of
            ``full_name`` when no short name is available.
        type: Entity type, always CHARACTER (default — no need to pass it).
    """

    type: EntityType = EntityType.CHARACTER
    full_name: str
    short_name: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _derive_id(cls, data):
        """Derive ``char:<short_name>`` when no explicit id is given."""
        if isinstance(data, dict) and not data.get("id"):
            base = data.get("short_name") or data.get("full_name") or ""
            slug = _slugify(base)
            if slug:
                data["id"] = f"char:{slug}"
        return data


# -------------------------------------------------------------------
# Claims
# -------------------------------------------------------------------

class Predicate(str, Enum):
    """Predicates usable in claims (not restricted to this list)."""

    # --- Between characters -------------------------------------------
    SPOUSE_OF = "spouse_of"
    PARENT_OF = "parent_of"
    CHILD_OF = "child_of"
    SIBLING_OF = "sibling_of"
    FRIEND_OF = "friend_of"
    ENEMY_OF = "enemy_of"
    ALLY_OF = "ally_of"
    KNOWS = "knows"
    WORKS_WITH = "works_with"
    COMMANDS = "commands"
    FOLLOWS = "follows"
    BETRAYED = "betrayed"
    TRUSTS = "trusts"
    LOVES = "loves"

    # --- Character / Place --------------------------------------------
    LIVES_IN = "lives_in"
    BORN_IN = "born_in"
    DIED_IN = "died_in"
    VISITED = "visited"
    TRAVELLED_TO = "travelled_to"

    # --- Place / Character --------------------------------------------
    HAS_INHABITANT = "has_inhabitant"

    # --- Character / Object -------------------------------------------
    OWNS = "owns"
    USES = "uses"
    CARRIES = "carries"
    CREATED = "created"
    DESTROYED = "destroyed"

    # --- Object / Character -------------------------------------------
    IS_OWNED = "is_owned"
    IS_USED = "is_used"
    IS_CARRIED = "is_carried"
    IS_CREATED = "is_created"
    IS_DESTROYED = "is_destroyed"

    # --- Character / Event --------------------------------------------
    PARTICIPATED_IN = "participated_in"
    WITNESSED = "witnessed"
    CAUSED = "caused"
    PREVENTED = "prevented"
    SURVIVED = "survived"

    # --- Object / Place -------------------------------------------------
    LOCATED_IN = "located_in"
    CREATED_AT = "created_at"
    DESTROYED_AT = "destroyed_at"

    # --- Object / Event -------------------------------------------------
    LOST_DURING = "lost_during"
    CREATED_DURING = "created_during"
    DESTROYED_DURING = "destroyed_during"

    # --- Value-only (no entity object) ----------------------------------
    OCCUPATION = "occupation"


class PredicateDefinition(BaseModel):
    """Defines a predicate usable in a claim.

    Attributes:
        predicate: The predicate itself, e.g. ``Predicate.LOVES``.
        subject_types: Set of EntityTypes accepted for the claim's subject.
        object_types: Set of EntityTypes accepted for the claim's object —
            None when the predicate takes no entity object and the claim
            carries a simple ``value`` instead (e.g. an occupation).
        value_type: Hint describing what the simple ``value`` holds when
            ``object_types`` is None (e.g. "profession", "activity").
        symmetric: True when the claim is still valid with subject and
            object swapped (e.g. spouse_of, sibling_of) — no inverse is
            needed.
        inverse: Predicate to use when the relation is read the other way
            round (e.g. owns ↔ is_owned); the inverse's definition must
            swap subject and object types. Unused when ``symmetric`` is
            True.
    """

    # Reject unknown fields so registry typos (e.g. "invsere=") fail fast.
    model_config = ConfigDict(extra="forbid")

    predicate: Predicate
    subject_types: set[EntityType]
    object_types: set[EntityType] | None = None
    value_type: str | None = None
    symmetric: bool = False
    inverse: Predicate | None = None


class ClaimStatus(str, Enum):
    """What a claim actually is."""

    ASSERTED = "asserted"
    CONTRADICTED = "contradicted"
    INFERRED = "inferred"
    FALSE = "false"
    UNKNOWN = "unknown"


class Claim(BaseModel):
    """A piece of information found in the sources — a fact or a relation.

    A claim either points to an entity (``object_id``, e.g. "This Guy lives
    in that City") or carries a simple literal value (``value``, e.g.
    "soldier") when the target is not referenced as an entity.

    Attributes:
        subject_id: Id of the entity the claim is about, e.g. ``char:003``
            or ``char:jean``.
        predicate: Predicate naming the relation; its definition (accepted
            entity types, symmetric/inverse) lives in the PREDICATES registry
            (see predicates.py) and is enforced by the validation layer.
        object_id: Id of the target entity, e.g. ``place:001`` or
            ``char:002`` — None when the claim carries a simple value.
        value: Simple literal value (e.g. ``"soldier"``) when the object is
            not referenced as an entity.
        status: ClaimStatus telling what the claim actually is — asserted,
            contradicted, inferred, a lie, ...
        confidence: Extractor confidence between 0.0 and 1.0 — the text may
            be ambiguous and the claim misunderstood.
        sources: Where the claim comes from — lightweight SourceRef
            pointers (a SourceLocator id plus optional section/page/
            paragraph indices; existence and bounds checked by
            SourceValidator).
        id: Unique identifier based on incrementation, e.g. ``claim:001``.
            Auto-generated on creation unless provided explicitly.
    """

    subject_id: str
    predicate: Predicate
    object_id: Optional[str] = None
    value: Optional[str] = None
    status: ClaimStatus
    confidence: float = Field(..., ge=0.0, le=1.0)
    asserted_by: Optional[str] = None
    sources: List[SourceRef] = Field(default_factory=list)
    id: str = Field(default_factory=_next_claim_id)

    @field_validator("subject_id", "object_id")
    @classmethod
    def _validate_entity_ref(cls, value: Optional[str]) -> Optional[str]:
        """Entity references must look like ``<kind>:<number>``."""
        if value is None:
            return value
        value = value.strip()
        if not _ENTITY_ID_PATTERN.match(value):
            raise ValueError(
                "entity id must look like '<kind>:<identifier>', e.g. 'char:jean' or 'place:001'"
            )
        return value

    @model_validator(mode="after")
    def _validate_object(self) -> "Claim":
        """A claim must target an entity or carry a simple value.

        Entity existence and type consistency with the predicate are
        deliberately NOT checked here — they need the registered entities
        and the PREDICATES registry, so they belong to the validation
        layer (src/validation.ClaimValidator).
        """
        if self.object_id is None and (self.value is None or not self.value.strip()):
            if self.object_id is None and self.value is not None:
                raise ValueError("value must be a non-empty string when object_id is None")
            raise ValueError("a claim must have an object_id or a value")
        return self