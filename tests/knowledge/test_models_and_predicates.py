"""Global use-case, side-effect and error checks for the knowledge layer."""

import unittest

from src.knowledge.models import (
    Character,
    Claim,
    ClaimStatus,
    Entity,
    EntityType,
    Page,
    Predicate,
    PredicateDefinition,
    Section,
    SourceLocator,
    SourceRef,
    SourceType,
)
from src.knowledge.predicates import PREDICATES, get_definition


def make_claim(**kwargs):
    """Claim factory with sensible defaults."""
    defaults = dict(status=ClaimStatus.ASSERTED, confidence=0.9)
    defaults.update(kwargs)
    return Claim(**defaults)


class EntityTestCase(unittest.TestCase):
    """Character ids, slugs and Entity base checks."""

    def test_character_id_from_short_name(self):
        c = Character(full_name="Jean Valjean", short_name="Jean")
        self.assertEqual(c.id, "char:jean")
        self.assertEqual(c.type, EntityType.CHARACTER)

    def test_character_id_slugifies_accents_and_spaces(self):
        c = Character(full_name="Édmond Dantès", short_name="Édmond Dantès")
        self.assertEqual(c.id, "char:edmond_dantes")

    def test_character_id_falls_back_to_full_name(self):
        c = Character(full_name="The Bishop")
        self.assertEqual(c.id, "char:the_bishop")

    def test_character_explicit_id_wins(self):
        c = Character(full_name="X", short_name="Y", id="char:custom")
        self.assertEqual(c.id, "char:custom")

    def test_entity_rejects_prefix_type_mismatch(self):
        with self.assertRaises(Exception):
            Entity(id="place:paris", type=EntityType.CHARACTER)

    def test_entity_accepts_matching_prefix(self):
        e = Entity(id="place:paris", type=EntityType.PLACE)
        self.assertEqual(e.type, EntityType.PLACE)


class ClaimStructuralTestCase(unittest.TestCase):
    """Structural rules enforced by the model itself."""

    def test_valid_entity_claim(self):
        claim = make_claim(
            subject_id="char:jean", predicate=Predicate.SPOUSE_OF, object_id="char:marie"
        )
        self.assertEqual(claim.predicate, Predicate.SPOUSE_OF)

    def test_valid_value_claim(self):
        claim = make_claim(
            subject_id="char:jean", predicate=Predicate.OCCUPATION, value="mayor"
        )
        self.assertEqual(claim.value, "mayor")

    def test_rejects_missing_object_and_value(self):
        with self.assertRaises(Exception):
            make_claim(subject_id="char:jean", predicate=Predicate.LIVES_IN)

    def test_rejects_blank_value(self):
        with self.assertRaises(Exception):
            make_claim(
                subject_id="char:jean", predicate=Predicate.OCCUPATION, value="   "
            )

    def test_rejects_malformed_entity_id(self):
        with self.assertRaises(Exception):
            make_claim(
                subject_id="Jean", predicate=Predicate.SPOUSE_OF, object_id="char:marie"
            )

    def test_rejects_confidence_out_of_bounds(self):
        for bad in (-0.1, 1.5):
            with self.assertRaises(Exception):
                make_claim(
                    subject_id="char:jean",
                    predicate=Predicate.KNOWS,
                    object_id="char:marie",
                    confidence=bad,
                )

    def test_auto_id_increments(self):
        a = make_claim(subject_id="char:jean", predicate=Predicate.KNOWS, object_id="char:x")
        b = make_claim(subject_id="char:jean", predicate=Predicate.KNOWS, object_id="char:x")
        self.assertNotEqual(a.id, b.id)
        self.assertTrue(a.id.startswith("claim:"))

    def test_sources_are_validated(self):
        # A bad nested SourceLocator (invalid path) must fail claim construction.
        with self.assertRaises(Exception):
            make_claim(
                subject_id="char:jean",
                predicate=Predicate.OCCUPATION,
                value="mayor",
                sources=[SourceLocator(source_type=SourceType.PDF, title="t", id="doc:abc12345", path="/abs/x.pdf")],
            )

    def test_sources_accept_source_refs(self):
        claim = make_claim(
            subject_id="char:jean",
            predicate=Predicate.OCCUPATION,
            value="mayor",
            sources=[SourceRef(locator_id="source:001", page_index=2)],
        )
        self.assertEqual(claim.sources[0].locator_id, "source:001")


class SourceLocatorHierarchyTestCase(unittest.TestCase):
    """sections -> pages -> paragraphs rework, and the SourceRef pointer."""

    def test_default_locator_has_one_global_section_and_page(self):
        locator = SourceLocator(
            source_type=SourceType.HTML,
            title="Example",
            id="doc:webpage1",
            path="https://example.org",
            page_count=1,
        )
        self.assertEqual(len(locator.sections), 1)
        self.assertEqual(locator.sections[0].section_index, 0)
        self.assertEqual(len(locator.sections[0].pages), 1)
        self.assertEqual(locator.sections[0].pages[0].page_index, 0)

    def test_locator_with_explicit_structure(self):
        locator = SourceLocator(
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
                Section(section_index=1, title="Ch. 2", pages=[Page(page_index=2, num_paragraphs=5)]),
            ],
        )
        self.assertEqual(len(locator.sections), 2)
        self.assertEqual(locator.sections[0].pages[1].num_paragraphs, 8)

    def test_page_rejects_negative_paragraphs(self):
        with self.assertRaises(Exception):
            Page(page_index=0, num_paragraphs=-1)

    def test_page_and_section_require_global_indices(self):
        with self.assertRaises(Exception):
            Page()  # page_index is required
        with self.assertRaises(Exception):
            Section(pages=[Page(page_index=0)])  # section_index is required

    def test_locator_structure_integrity(self):
        def make(**overrides):
            data = dict(
                source_type=SourceType.PDF,
                title="Book",
                id="doc:abcdef01",
                page_count=3,
                sections=[
                    Section(
                        section_index=0,
                        pages=[Page(page_index=0), Page(page_index=1)],
                    ),
                ],
            )
            data.update(overrides)
            return SourceLocator(**data)

        make()  # valid
        with self.assertRaises(Exception):
            make(page_count=0)  # a document has at least one page
        with self.assertRaises(Exception):
            make(sections=[])  # "no section" is still a section
        with self.assertRaises(Exception):
            make(page_count=1)  # collected page 1 beyond page_count=1
        with self.assertRaises(Exception):
            make(
                sections=[
                    Section(section_index=0, pages=[Page(page_index=3)]),
                ],
                page_count=3,
            )  # page beyond page_count
        with self.assertRaises(Exception):
            make(
                sections=[
                    Section(section_index=0, pages=[Page(page_index=0)]),
                    Section(section_index=0, pages=[Page(page_index=2)]),
                ]
            )  # duplicate section_index
        with self.assertRaises(Exception):
            make(
                sections=[
                    Section(
                        section_index=0,
                        pages=[Page(page_index=0), Page(page_index=0)],
                    ),
                ]
            )  # duplicate page_index within a section

    def test_source_ref_default_indices_are_none(self):
        source = SourceRef(locator_id="source:001")
        self.assertIsNone(source.section_index)
        self.assertIsNone(source.page_index)
        self.assertIsNone(source.paragraph_index)

    def test_source_ref_rejects_malformed_locator_id(self):
        with self.assertRaises(Exception):
            SourceRef(locator_id="not-an-id")

    def test_source_ref_rejects_negative_indices(self):
        for kw in (dict(section_index=-1), dict(page_index=-1), dict(paragraph_index=-1)):
            with self.assertRaises(Exception):
                SourceRef(locator_id="source:001", **kw)

    def test_page_count_is_always_explicit(self):
        # The global count is NOT derived from collected sections: a sparse
        # 300-page book with 2 parsed pages still declares page_count=300.
        sparse = SourceLocator(
            source_type=SourceType.PDF,
            title="Dumas.pdf",
            id="doc:sparse01",
            page_count=300,
            sections=[
                Section(
                    section_index=2,
                    title="Ch. 3",
                    pages=[Page(page_index=14), Page(page_index=16)],
                ),
            ],
        )
        self.assertEqual(sparse.page_count, 300)
        with self.assertRaises(Exception):
            SourceLocator(source_type=SourceType.PDF, title="x")  # page_count required


class SourceRefStrTestCase(unittest.TestCase):
    """Compact human-readable citations on SourceRef."""

    def test_str_with_all_indices(self):
        source = SourceRef(
            locator_id="source:001", section_index=0, page_index=1, paragraph_index=4
        )
        self.assertEqual(str(source), "source:001, ch.1 p.2 §5")

    def test_indices_are_displayed_one_based(self):
        source = SourceRef(locator_id="source:001", section_index=0, page_index=0, paragraph_index=0)
        self.assertEqual(str(source), "source:001, ch.1 p.1 §1")

    def test_unset_indices_are_omitted(self):
        self.assertEqual(str(SourceRef(locator_id="source:001")), "source:001")
        self.assertEqual(
            str(SourceRef(locator_id="source:001", paragraph_index=3)),
            "source:001, §4",
        )


class PredicateRegistryTestCase(unittest.TestCase):
    """PREDICATES registry integrity — side-effect-free global invariants."""

    def test_registry_covers_every_predicate(self):
        self.assertEqual(set(PREDICATES), set(Predicate))

    def test_inverse_targets_have_definitions(self):
        for predicate, definition in PREDICATES.items():
            if definition.inverse is not None:
                self.assertIn(definition.inverse, PREDICATES)

    def test_inverse_swaps_subject_and_object_types(self):
        for predicate, definition in PREDICATES.items():
            inverse = PREDICATES.get(definition.inverse) if definition.inverse else None
            if inverse is None:
                continue
            self.assertEqual(
                definition.subject_types,
                inverse.object_types,
                f"{predicate.value}: inverse does not swap subject types",
            )
            self.assertEqual(
                definition.object_types,
                inverse.subject_types,
                f"{predicate.value}: inverse does not swap object types",
            )

    def test_symmetric_predicates_match_their_own_types(self):
        for predicate, definition in PREDICATES.items():
            if definition.symmetric:
                self.assertEqual(definition.subject_types, definition.object_types)

    def test_get_definition_raises_for_unknown(self):
        with self.assertRaises(ValueError):
            get_definition("not-a-predicate")

    def test_definitions_reject_unknown_fields(self):
        # extra="forbid" catches registry typos like "invsere=".
        with self.assertRaises(Exception):
            PredicateDefinition(
                predicate=Predicate.KNOWS,
                subject_types={EntityType.CHARACTER},
                object_types={EntityType.CHARACTER},
                invsere=Predicate.CHILD_OF,
            )


if __name__ == "__main__":
    unittest.main()
