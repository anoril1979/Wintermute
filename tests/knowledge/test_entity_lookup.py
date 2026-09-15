"""Tests for the knowledge-base entity lookup tools (read side).

Hermetic: a temp knowledge base written through the real store
primitives — no config dependency, no LLM.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from src.knowledge.character_markdown_store import (
    character_path_for,
    index_path_for,
    load_index,
    rebuild_index,
    write_character,
    write_index,
)
from src.knowledge.entity_lookup import (
    DEFAULT_MAX_CONTENT_HITS,
    EntityMatch,
    close_candidates,
    fetch_unit_content,
    fold_name,
    known_entity_count,
    resolve_entity,
)


def _entry(alias: str, *source_ids: str) -> dict:
    return {"alias": alias, "source_ids": list(source_ids)}


class EntityLookupTest(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="wm_entity_lookup_"))
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)

        # Two characters, written with the real store primitives.
        write_character(
            "Joe le Clodo",
            [
                _entry("Joe le Clodo", "doc:aaaaaaaa::chp:1::pg:1::sec:1"),
                _entry("Bobby", "doc:aaaaaaaa::chp:1::pg:1::sec:2",
                       "doc:aaaaaaaa::chp:2::pg:3::sec:1"),
            ],
            character_path_for("Joe le Clodo", self.base),
        )
        write_character(
            "Épée de vif-argent",
            [_entry("Épée de vif-argent",
                    "doc:bbbbbbbb::chp:1::pg:2::sec:1")],
            character_path_for("Épée de vif-argent", self.base),
        )
        rebuild_index(self.base)

    # -- fold_name ----------------------------------------------------------

    def test_fold_name_is_insensitive_to_case_accents_punctuation(self):
        self.assertEqual(fold_name("Épée de Vif-Argent"),
                         fold_name("epee de vif argent"))
        self.assertEqual(fold_name("  JOE  "), fold_name("joe"))

    # -- resolve_entity: direct slug -----------------------------------------

    def test_exact_name_resolves_directly(self):
        match = resolve_entity("Joe le Clodo", base_dir=self.base)
        self.assertIsInstance(match, EntityMatch)
        self.assertEqual(match.full_name, "Joe le Clodo")
        # The union of every name's source ids (deduped, in file order).
        self.assertEqual(match.source_ids, [
            "doc:aaaaaaaa::chp:1::pg:1::sec:1",
            "doc:aaaaaaaa::chp:1::pg:1::sec:2",
            "doc:aaaaaaaa::chp:2::pg:3::sec:1",
        ])

    def test_case_accent_insensitive_resolution(self):
        match = resolve_entity("epee de vif-argent", base_dir=self.base)
        self.assertIsNotNone(match)
        self.assertEqual(match.full_name, "Épée de vif-argent")

    # -- resolve_entity: index scan (full names, then aliases) ---------------

    def test_alias_resolves_to_the_character_file(self):
        match = resolve_entity("bobby", base_dir=self.base)
        self.assertIsNotNone(match, "an alias-only discovery must find Joe")
        self.assertEqual(match.full_name, "Joe le Clodo")
        ids = match.source_ids
        self.assertIn("doc:aaaaaaaa::chp:1::pg:1::sec:2", ids)
        self.assertIn("doc:aaaaaaaa::chp:2::pg:3::sec:1", ids)

    def test_unknown_entity_is_none(self):
        self.assertIsNone(resolve_entity("Inconnu du Far West",
                                         base_dir=self.base))

    def test_empty_query_is_none(self):
        self.assertIsNone(resolve_entity("   ", base_dir=self.base))

    def test_stale_index_entry_is_repaired_and_retried(self):
        """A hand-deleted character file leaves a stale index line: the
        resolver rebuilds the index (files are the truth) and reports
        the entity unknown — instead of crashing or 'finding' a ghost."""
        index_path = index_path_for(self.base)
        write_index(index_path, [
            {"full_name": "Joe le Clodo", "aliases": ["Bobby"]},
            {"full_name": "Ghost",
             "aliases": ["Fantôme"]},  # file deleted behind the index
        ])
        match = resolve_entity("Fantôme", base_dir=self.base)
        self.assertIsNone(match)
        # The rebuild swept the stale line (the real files survive).
        names = sorted(
            str(c["full_name"]) for c in load_index(index_path)
        )
        self.assertEqual(
            names, ["Joe le Clodo", "Épée de vif-argent"],
            "the ghost line is gone, the real characters stay",
        )
        # And a known entity still resolves after the repair.
        self.assertIsNotNone(resolve_entity("Bobby", base_dir=self.base))

    def test_malformed_file_degrades_to_none(self):
        """A hand-mangled character file: the resolver's one repair pass
        fails (the store raises on a malformed file — pinned elsewhere)
        and the resolution degrades to None instead of crashing."""
        path = character_path_for("Broken", self.base)
        path.write_text("no title here\n", encoding="utf-8")
        self.assertIsNone(resolve_entity("Broken", base_dir=self.base))
        # A healthy entity still resolves — the broken file only
        # degrades its own lookups.
        self.assertIsNotNone(resolve_entity("Joe le Clodo",
                                            base_dir=self.base))

    # -- close_candidates ------------------------------------------------------

    def test_fragment_of_a_name_is_a_candidate(self):
        candidates = close_candidates("vif-argent", base_dir=self.base)
        self.assertIn("Épée de vif-argent", candidates)

    def test_name_containing_the_query_is_a_candidate(self):
        candidates = close_candidates("je cherche Joe le Clodo merci",
                                      base_dir=self.base)
        self.assertIn("Joe le Clodo", candidates)

    def test_full_name_hits_rank_above_alias_hits(self):
        candidates = close_candidates("bobby", base_dir=self.base)
        # "Bobby" is an alias of Joe — but a containment on the full
        # name "Joe le Clodo"? No. So the alias hit surfaces.
        self.assertIn("Joe le Clodo", candidates)

    def test_no_candidates_for_an_unrelated_name(self):
        self.assertEqual(close_candidates("zzzqqq", base_dir=self.base), [])

    def test_candidates_respect_the_limit(self):
        candidates = close_candidates("e", base_dir=self.base, limit=1)
        self.assertLessEqual(len(candidates), 1)

    # -- known_entity_count ------------------------------------------------------

    def test_known_entity_count_reads_the_index(self):
        self.assertEqual(known_entity_count(self.base), 2)
        empty = Path(tempfile.mkdtemp(prefix="wm_entity_empty_"))
        self.addCleanup(shutil.rmtree, empty, ignore_errors=True)
        self.assertEqual(known_entity_count(empty), 0)


class FetchUnitContentTest(unittest.TestCase):
    """The vector companion: opaque source ids -> actual corpus content."""

    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="wm_fetch_unit_"))
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)

    def test_dedupes_and_keeps_first_seen_order(self):
        class _Store:
            def __init__(self):
                self.asked: list[str] = []

            def get_unit_chunks(self, prefix, *, limit):
                self.asked.append(prefix)
                return []

        store = _Store()
        fetch_unit_content(
            ["doc:aa::chp:1::sec:2", "doc:aa::chp:1::sec:1",
             "doc:aa::chp:1::sec:2", "  ", "doc:aa::chp:1::sec:2"],
            store=store,
        )
        self.assertEqual(store.asked,
                         ["doc:aa::chp:1::sec:2", "doc:aa::chp:1::sec:1"])

    def test_unknown_units_are_skipped_not_failed(self):
        class _Store:
            def get_unit_chunks(self, prefix, *, limit):
                return []

        self.assertEqual(fetch_unit_content(
            ["doc:zzzzzzzz::chp:9::sec:9"], store=_Store(), max_hits=4,
        ), [])

    def test_store_failure_is_swallowed(self):
        class _Store:
            def get_unit_chunks(self, prefix, *, limit):
                raise RuntimeError("store down")

        self.assertEqual(fetch_unit_content(
            ["doc:aa::chp:1::sec:1"], store=_Store(), max_hits=4,
        ), [])

    def test_cap_applies_across_units(self):
        class _Store:
            def get_unit_chunks(self, prefix, *, limit):
                self.last_limit = limit
                return [f"chunk-of-{prefix}-{i}" for i in range(4)]

        store = _Store()
        chunks = fetch_unit_content(
            ["doc:aa::chp:1::sec:1", "doc:aa::chp:1::sec:2"],
            store=store, max_hits=6,
        )
        self.assertEqual(store.last_limit, 6, "fetches are capped per unit")
        self.assertEqual(len(chunks), 6, "the global cap holds")
        self.assertEqual(chunks[0], "chunk-of-doc:aa::chp:1::sec:1-0")
        self.assertEqual(chunks[-1], "chunk-of-doc:aa::chp:1::sec:2-1")

    def test_empty_ids_fetch_nothing(self):
        self.assertEqual(fetch_unit_content([], store=object()), [])

    def test_config_cap_failure_fails_open_to_default(self):
        import unittest.mock

        from src.knowledge import entity_lookup

        with unittest.mock.patch(
            "src.tools.config_loader.load_retrieval_config",
            side_effect=RuntimeError("no config"),
        ):
            self.assertEqual(
                entity_lookup._config_max_content_hits(),
                DEFAULT_MAX_CONTENT_HITS,
            )


if __name__ == "__main__":
    unittest.main()
