"""Claim validation against registered entities, predicate definitions and sources."""

from typing import Dict, Optional, Tuple

from src.knowledge.models import Claim, Entity, Predicate, SourceRef, SourceLocator
from src.knowledge.predicates import PREDICATES
from src.validation.source_validator import (
    SourceValidationError,
    SourceValidator,
)


class ClaimValidationError(ValueError):
    """Base class for claim validation errors."""


class UnknownPredicateError(ClaimValidationError):
    """The claim's predicate has no definition in the registry."""


class UnknownSubjectError(ClaimValidationError):
    """The claim's subject_id is not among the registered entities."""


class UnknownObjectError(ClaimValidationError):
    """The claim's object_id is not among the registered entities."""


class InvalidSubjectTypeError(ClaimValidationError):
    """The subject entity's type is not accepted by the predicate."""


class InvalidObjectTypeError(ClaimValidationError):
    """The object entity's type is not accepted by the predicate."""


class PredicateRequiresObjectError(ClaimValidationError):
    """An entity object is required but the claim carries only a value."""


class MissingValueError(ClaimValidationError):
    """A value-only predicate got neither an object nor a value."""


class ClaimValidator:
    """Validates claims against the PREDICATES registry and registered entities.

    Structural checks (id format, object-or-value) are already enforced by
    the Pydantic models; this validator adds everything that needs external
    state. Each claim's sources are checked too: locator existence and
    section/page/paragraph bounds via SourceValidator. ``validate`` returns
    (subject, object-or-None) or raises a ClaimValidationError subclass.
    """

    def __init__(
        self,
        registry: Optional[dict] = None,
        source_validator: Optional[SourceValidator] = None,
        check_sources: bool = True,
    ):
        # Defaults to the canonical registry; injectable for tests/variants.
        self.registry = registry if registry is not None else PREDICATES
        self.source_validator = source_validator or SourceValidator()
        # Set False to skip source checks (e.g. claims validated before
        # their locators are registered).
        self.check_sources = check_sources

    def validate(
        self,
        claim: Claim,
        entities: Dict[str, Entity],
        locators: Optional[Dict[str, SourceLocator]] = None,
    ) -> "Tuple[Entity, Optional[Entity]]":
        """Validate ``claim`` and return (subject, object-or-None).

        When ``locators`` is provided (and ``check_sources`` is True), each
        of the claim's sources is validated through SourceValidator.
        Raises a ClaimValidationError subclass on the first problem found.
        """
        definition = self.registry.get(claim.predicate)
        if definition is None:
            raise UnknownPredicateError(f"Unknown predicate: {claim.predicate}")

        subject = entities.get(claim.subject_id)
        if subject is None:
            raise UnknownSubjectError(f"Unknown subject: {claim.subject_id}")

        if subject.type not in definition.subject_types:
            raise InvalidSubjectTypeError(
                f"{claim.predicate.value}: invalid subject type "
                f"'{subject.type.value}' (expected one of "
                f"{sorted(t.value for t in definition.subject_types)})"
            )

        obj: Optional[Entity] = None
        if claim.object_id is not None:
            obj = entities.get(claim.object_id)
            if obj is None:
                raise UnknownObjectError(f"Unknown object: {claim.object_id}")

            if definition.object_types is None:
                raise InvalidObjectTypeError(
                    f"{claim.predicate.value} does not accept an object"
                )

            if obj.type not in definition.object_types:
                raise InvalidObjectTypeError(
                    f"{claim.predicate.value}: invalid object type "
                    f"'{obj.type.value}' (expected one of "
                    f"{sorted(t.value for t in definition.object_types)})"
                )
        elif not (claim.value and claim.value.strip()):
            if definition.object_types is not None:
                raise PredicateRequiresObjectError(
                    f"{claim.predicate.value} requires an entity object"
                )
            raise MissingValueError(
                f"{claim.predicate.value} requires a value when there is no object"
            )

        if self.check_sources and locators is not None:
            for source in claim.sources:
                self.source_validator.validate(source, locators)

        return subject, obj

    def can_swap(self, claim: Claim) -> bool:
        """True when the claim's predicate is symmetric (subject/object
        can be switched without changing validity)."""
        definition = self.registry.get(claim.predicate)
        return bool(definition and definition.symmetric)

    def swapped(self, claim: Claim) -> Claim:
        """Return the subject/object-swapped variant of a symmetric claim."""
        if not self.can_swap(claim):
            raise ClaimValidationError(
                f"{claim.predicate.value} is not symmetric; cannot swap"
            )
        return claim.model_copy(update={"subject_id": claim.object_id, "object_id": claim.subject_id})
