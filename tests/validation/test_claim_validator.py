"""Global use-case, side-effect and error checks for ClaimValidator."""

import unittest

from src.knowledge.models import (
    Character,
    Claim,
    ClaimStatus,
    Entity,
    EntityType,
    Page,
    Predicate,
    Section,
    SourceLocator,
    SourceRef,
    SourceType,
)
from src.validation import (
    ClaimValidator,
    ClaimValidationError,
    InvalidObjectTypeError,
    InvalidSubjectTypeError,
    MissingValueError,
    PredicateRequiresObjectError,
    UnknownObjectError,
    UnknownPredicateError,
    UnknownSubjectError,
    InvalidParagraphIndexError,
    SourceValidationError,
    SourceValidator,
    UnknownLocatorError,
    UnknownPageError,
    UnknownSectionError,
)


def make_claim(**kwargs):
    defaults = dict(status=ClaimStatus.ASSERTED, confidence=0.9)
    defaults.update(kwargs)
    return Claim(**defaults)


def build_entities():
    jean = Character(full_name="Jean Valjean", short_name="Jean")
    marie = Character(full_name="Marie", short_name="Marie")
    paris = Entity(id="place:paris", type=EntityType.PLACE)
    sword = Entity(id="object:sword", type=EntityType.OBJECT)
    return jean, marie, paris, sword, {
        jean.id: jean,
        marie.id: marie,
        paris.id: paris,
        sword.id: sword,
    }


def build_locators():
    """A fully-collected book, a page-like HTML doc, and a sparse partial book."""
    book = SourceLocator(
        source_type=SourceType.PDF,
        title="Dumas.pdf",
        path="data/dumas.pdf",
        id="doc:dumas001",
        page_count=3,
        sections=[
            Section(
                section_index=0,
                title="Ch. 1",
                pages=[Page(page_index=0, num_paragraphs=12), Page(page_index=1, num_paragraphs=8)],
            ),
            Section(
                section_index=1,
                title="Ch. 2",
                pages=[Page(page_index=2, num_paragraphs=5)],
            ),
        ],
    )
    webpage = SourceLocator(
        source_type=SourceType.HTML,
        title="Example",
        id="doc:webpage1",
        path="https://example.org",
        page_count=1,
        # default: one global section (section_index=0) with one page (page_index=0)
    )
    return book, webpage, {book.id: book, webpage.id: webpage}


class ClaimValidatorUseCaseTestCase(unittest.TestCase):
    """Global use-cases: end-to-end valid claims."""

    def setUp(self):
        self.jean, self.marie, self.paris, self.sword, self.entities = build_entities()
        self.validator = ClaimValidator()

    def test_character_to_character_claim(self):
        claim = make_claim(
            subject_id="character:jean", predicate=Predicate.SPOUSE_OF, object_id="character:marie"
        )
        subject, obj = self.validator.validate(claim, self.entities)
        self.assertIs(subject, self.jean)
        self.assertIs(obj, self.marie)

    def test_character_to_place_claim(self):
        claim = make_claim(
            subject_id="character:jean", predicate=Predicate.LIVES_IN, object_id="place:paris"
        )
        _, obj = self.validator.validate(claim, self.entities)
        self.assertIs(obj, self.paris)

    def test_character_to_object_claim(self):
        claim = make_claim(
            subject_id="character:jean", predicate=Predicate.OWNS, object_id="object:sword"
        )
        _, obj = self.validator.validate(claim, self.entities)
        self.assertIs(obj, self.sword)

    def test_value_only_claim(self):
        claim = make_claim(
            subject_id="character:jean", predicate=Predicate.OCCUPATION, value="mayor"
        )
        subject, obj = self.validator.validate(claim, self.entities)
        self.assertIs(subject, self.jean)
        self.assertIsNone(obj)

    def test_symmetric_swap_round_trip(self):
        claim = make_claim(
            subject_id="character:jean", predicate=Predicate.SPOUSE_OF, object_id="character:marie"
        )
        swapped = self.validator.swapped(claim)
        subject, obj = self.validator.validate(swapped, self.entities)
        self.assertIs(subject, self.marie)
        self.assertIs(obj, self.jean)

    def test_swapped_rejects_non_symmetric(self):
        claim = make_claim(
            subject_id="character:jean", predicate=Predicate.OWNS, object_id="object:sword"
        )
        with self.assertRaises(ClaimValidationError):
            self.validator.swapped(claim)

    def test_can_swap_flag(self):
        symmetric = make_claim(
            subject_id="character:jean", predicate=Predicate.SPOUSE_OF, object_id="character:marie"
        )
        directional = make_claim(
            subject_id="character:jean", predicate=Predicate.OWNS, object_id="object:sword"
        )
        self.assertTrue(self.validator.can_swap(symmetric))
        self.assertFalse(self.validator.can_swap(directional))


class SourceValidatorTestCase(unittest.TestCase):
    """SourceValidator: locator existence + document-global index resolution."""

    def setUp(self):
        self.book, self.webpage, self.locators = build_locators()
        self.validator = SourceValidator()

    def test_resolves_known_locator(self):
        # Global page 2 of the book lives in section 1 (ch.2), 5 paragraphs.
        source = SourceRef(locator_id=self.book.id, section_index=1, page_index=2, paragraph_index=4)
        locator = self.validator.validate(source, self.locators)
        self.assertIs(locator, self.book)

    def test_global_defaults_resolve_on_dense_locator(self):
        # webpage is dense: section 0 / page 0 are collected, so None works.
        source = SourceRef(locator_id=self.webpage.id)
        self.validator.validate(source, self.locators)  # must not raise

    def test_unknown_locator(self):
        source = SourceRef(locator_id="source:999")
        with self.assertRaises(UnknownLocatorError):
            self.validator.validate(source, self.locators)

    def test_uncollected_section_rejected(self):
        source = SourceRef(locator_id=self.webpage.id, section_index=1)  # only section 0
        with self.assertRaises(UnknownSectionError):
            self.validator.validate(source, self.locators)

    def test_page_index_must_be_below_page_count_and_collected(self):
        # Page 2 is collected, in section 1 (ch.2); a page below page_count
        # but never collected (sparse locator) is rejected as unknown.
        self.validator.validate(
            SourceRef(locator_id=self.book.id, section_index=1, page_index=2), self.locators
        )
        sparse = SourceLocator(
            source_type=SourceType.PDF,
            title="Sparse",
            id="doc:sparse01",
            page_count=300,
            sections=[Section(section_index=0, pages=[Page(page_index=0)])],
        )
        with self.assertRaises(UnknownPageError):
            self.validator.validate(
                SourceRef(locator_id=sparse.id, section_index=0, page_index=100),
                {sparse.id: sparse},
            )

    def test_page_must_be_collected_in_the_targeted_section(self):
        # Global page 0 exists, but in section 0 — not section 1.
        source = SourceRef(locator_id=self.book.id, section_index=1, page_index=0)
        with self.assertRaises(UnknownPageError):
            self.validator.validate(source, self.locators)

    def test_sparse_locator_with_uncollected_default_indices(self):
        # 300-page book, only chapter 3 parsed: sections [2], pages [14, 16].
        partial = SourceLocator(
            source_type=SourceType.PDF,                title="Dumas.pdf",
                path="data/dumas.pdf",
                id="doc:partial1",
                page_count=300,
            sections=[
                Section(
                    section_index=2,
                    title="Ch. 3",
                    pages=[Page(page_index=14, num_paragraphs=9), Page(page_index=16, num_paragraphs=7)],
                ),
            ],
        )
        locators = {partial.id: partial}
        # Explicit global indices resolve into the sparse structure.
        self.validator.validate(
            SourceRef(locator_id=partial.id, section_index=2, page_index=14, paragraph_index=8),
            locators,
        )
        self.validator.validate(
            SourceRef(locator_id=partial.id, section_index=2, page_index=16), locators
        )
        # None defaults to section 0 / page 0, which were NOT collected.
        for bad in (
            SourceRef(locator_id=partial.id, page_index=14),  # section 0 not collected
            SourceRef(locator_id=partial.id, section_index=2),  # page 0 not collected
            SourceRef(locator_id=partial.id, section_index=2, page_index=15),  # never parsed
        ):
            with self.assertRaises(SourceValidationError):
                self.validator.validate(bad, locators)

    def test_paragraph_index_out_of_range(self):
        source = SourceRef(locator_id=self.book.id, section_index=1, page_index=2, paragraph_index=5)
        with self.assertRaises(InvalidParagraphIndexError):
            self.validator.validate(source, self.locators)

    def test_page_index_bound_is_enforced_against_page_count(self):
        # page 100 < page_count 300, but was never collected in ch.3.
        partial = SourceLocator(
            source_type=SourceType.PDF,
            title="Partial.pdf",
            id="doc:partial2",
            page_count=300,
            sections=[
                Section(section_index=2, pages=[Page(page_index=14), Page(page_index=16)]),
            ],
        )
        with self.assertRaises(UnknownPageError):
            self.validator.validate(
                SourceRef(locator_id=partial.id, section_index=2, page_index=100),
                {partial.id: partial},
            )

    def test_errors_are_value_errors(self):
        source = SourceRef(locator_id="source:999")
        with self.assertRaises(ValueError):
            self.validator.validate(source, self.locators)


class ClaimValidatorSourceWiringTestCase(unittest.TestCase):
    """ClaimValidator calls SourceValidator on each claim source."""

    def setUp(self):
        self.jean, self.marie, self.paris, self.sword, self.entities = build_entities()
        self.book, self.webpage, self.locators = build_locators()
        self.validator = ClaimValidator()

    def _claim(self, **source_kw):
        return make_claim(
            subject_id="character:jean",
            predicate=Predicate.OCCUPATION,
            value="mayor",
            sources=[SourceRef(locator_id=self.book.id, **source_kw)],
        )

    def test_valid_sources_pass(self):
        claim = self._claim(section_index=0, page_index=1, paragraph_index=7)
        subject, _ = self.validator.validate(claim, self.entities, self.locators)
        self.assertIs(subject, self.jean)
    def test_invalid_source_rejects_claim(self):
        claim = self._claim(section_index=5)  # book has sections 0 and 1 only
        with self.assertRaises(UnknownSectionError):
            self.validator.validate(claim, self.entities, self.locators)

    def test_invalid_locator_rejects_claim(self):
        claim = make_claim(
            subject_id="character:jean",
            predicate=Predicate.OCCUPATION,
            value="mayor",
            sources=[SourceRef(locator_id="source:999")],
        )
        with self.assertRaises(UnknownLocatorError):
            self.validator.validate(claim, self.entities, self.locators)

    def test_source_checks_skipped_without_locators(self):
        claim = self._claim(section_index=5)  # would fail, but no locators given
        self.validator.validate(claim, self.entities)  # must not raise

    def test_source_checks_disabled_by_flag(self):
        validator = ClaimValidator(check_sources=False)
        claim = self._claim(section_index=5)
        validator.validate(claim, self.entities, self.locators)  # must not raise


class ClaimValidatorErrorTestCase(unittest.TestCase):
    """Error mapping: each failure raises its specific exception."""

    def setUp(self):
        self.jean, self.marie, self.paris, self.sword, self.entities = build_entities()
        self.validator = ClaimValidator()

    def test_unknown_predicate(self):
        validator = ClaimValidator(registry={})  # injected empty registry
        claim = make_claim(
            subject_id="character:jean", predicate=Predicate.KNOWS, object_id="character:marie"
        )
        with self.assertRaises(UnknownPredicateError):
            validator.validate(claim, self.entities)

    def test_unknown_subject(self):
        claim = make_claim(
            subject_id="character:nobody", predicate=Predicate.SPOUSE_OF, object_id="character:marie"
        )
        with self.assertRaises(UnknownSubjectError):
            self.validator.validate(claim, self.entities)

    def test_invalid_subject_type(self):
        claim = make_claim(
            subject_id="place:paris", predicate=Predicate.SPOUSE_OF, object_id="character:marie"
        )
        with self.assertRaises(InvalidSubjectTypeError):
            self.validator.validate(claim, self.entities)

    def test_unknown_object(self):
        claim = make_claim(
            subject_id="character:jean", predicate=Predicate.LIVES_IN, object_id="place:nowhere"
        )
        with self.assertRaises(UnknownObjectError):
            self.validator.validate(claim, self.entities)

    def test_object_on_value_only_predicate(self):
        claim = make_claim(
            subject_id="character:jean", predicate=Predicate.OCCUPATION, object_id="character:marie"
        )
        with self.assertRaises(InvalidObjectTypeError):
            self.validator.validate(claim, self.entities)

    def test_invalid_object_type(self):
        claim = make_claim(
            subject_id="character:jean", predicate=Predicate.OWNS, object_id="place:paris"
        )
        with self.assertRaises(InvalidObjectTypeError):
            self.validator.validate(claim, self.entities)

    def test_errors_are_value_errors(self):
        # ClaimValidationError subclasses ValueError: legacy except ValueError keeps working.
        claim = make_claim(
            subject_id="character:nobody", predicate=Predicate.SPOUSE_OF, object_id="character:marie"
        )
        with self.assertRaises(ValueError):
            self.validator.validate(claim, self.entities)

    def test_multi_type_definition_accepts_any_listed_type(self):
        # lives_in accepts CHARACTER subjects; entity ids agree.
        claim = make_claim(
            subject_id="character:jean", predicate=Predicate.LIVES_IN, object_id="place:paris"
        )
        self.validator.validate(claim, self.entities)  # must not raise


class EntityRegistrySideEffectsTestCase(unittest.TestCase):
    """Validator must not mutate the caller's data."""

    def setUp(self):
        self.jean, self.marie, self.paris, self.sword, self.entities = build_entities()
        self.validator = ClaimValidator()

    def test_validate_does_not_mutate_claim_or_entities(self):
        claim = make_claim(
            subject_id="character:jean", predicate=Predicate.SPOUSE_OF, object_id="character:marie"
        )
        before_claim = claim.model_dump()
        before_subject = self.jean.model_dump()
        self.validator.validate(claim, self.entities)
        self.assertEqual(claim.model_dump(), before_claim)
        self.assertEqual(self.jean.model_dump(), before_subject)

    def test_swapped_returns_a_copy_not_the_original(self):
        claim = make_claim(
            subject_id="character:jean", predicate=Predicate.SPOUSE_OF, object_id="character:marie"
        )
        swapped = self.validator.swapped(claim)
        self.assertIsNot(swapped, claim)
        self.assertEqual(claim.subject_id, "character:jean")  # original untouched
        self.assertEqual(swapped.subject_id, "character:marie")


if __name__ == "__main__":
    unittest.main()
