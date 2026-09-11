"""Validation layer for the knowledge models.

Structural checks (entity-id format, object-or-value) live on the Pydantic
models themselves (src/knowledge/models.py). This package adds everything
that needs external state: the PREDICATES registry, the registered entities
and the registered SourceLocators.
"""

from src.validation.claim_validator import (
    ClaimValidator,
    ClaimValidationError,
    InvalidObjectTypeError,
    InvalidSubjectTypeError,
    MissingValueError,
    PredicateRequiresObjectError,
    UnknownObjectError,
    UnknownPredicateError,
    UnknownSubjectError,
)
from src.validation.source_validator import (
    SourceValidator,
    SourceValidationError,
    InvalidParagraphIndexError,
    UnknownLocatorError,
    UnknownPageError,
    UnknownSectionError,
)

__all__ = [
    "ClaimValidator",
    "ClaimValidationError",
    "InvalidObjectTypeError",
    "InvalidSubjectTypeError",
    "MissingValueError",
    "PredicateRequiresObjectError",
    "UnknownObjectError",
    "UnknownPredicateError",
    "UnknownSubjectError",
    "SourceValidator",
    "SourceValidationError",
    "InvalidParagraphIndexError",
    "UnknownLocatorError",
    "UnknownPageError",
    "UnknownSectionError",
]
