"""Tests for the EntityResolverAgent / CharacterResolver (check_and_merge).

Hermetic: every knowledge base lives in a throwaway temp dir (base_dir
constructor override), no LLM (the resolver is deterministic), no writes
to the real data/knowledge folder.

Covers: create path (new character file + index line), merge path
(idempotent id union, new aliases), skip path (shapeless entries), index
rebuild after the pass, validate() hook, and the skip/failed status
mapping.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.agents.agents.entity_resolver_agent import (
    CharacterResolver,
    EntityResolverAgent,
)
from src.agents.contexts import IngestionContext
from src.agents.protocols import AgentStatus
from src.knowledge.character_markdown_store import (
    INDEX_FILENAME,
    characters_dir,
    read_character,
)


def make_context(entries):
    context = IngestionContext(document_path=Path("gazette.pdf"))
    context.outputs["knowledge_characters"] = {"characters": entries}
    return context


class ResolverBaseTest(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.chars = characters_dir(self.base)  # <base>/characters
        self.resolver = CharacterResolver(base_dir=self.base)
        self.base_missing = Path(tempfile.mkdtemp())

    def run_resolver(self, entries):
        context = make_context(entries)
        result = self.resolver.run(context)
        return result, context


class StatusMappingTest(ResolverBaseTest):
    def test_no_extraction_output_is_skipped(self):
        context = IngestionContext(document_path=Path("gazette.pdf"))
        result = self.resolver.run(context)
        self.assertEqual(result.status, AgentStatus.SKIPPED)

    def test_create_new_character(self):
        result, _ = self.run_resolver([
            {"full_name": "Joe le Clodo", "short_name": "Joe",
             "aliases": ["Bobby"],
             "source_ids": ["doc:36a911e2::chp:1::pg:1::sec:2"]},
        ])
        self.assertEqual(result.status, AgentStatus.OK)
        self.assertEqual(result.payload["created"], 1)
        self.assertEqual(result.payload["merged"], 0)

        char_file = self.chars / "joe-le-clodo.md"
        self.assertTrue(char_file.exists())
        data = read_character(char_file)
        self.assertEqual(data["full_name"], "Joe le Clodo")
        names = data["names"]
        self.assertEqual(names[0]["alias"], "Joe le Clodo")
        self.assertEqual(
            names[0]["source_ids"], ["doc:36a911e2::chp:1::pg:1::sec:2"]
        )
        self.assertEqual(names[1]["alias"], "Bobby")
        # The alias shares the unit's provenance.
        self.assertEqual(
            names[1]["source_ids"], ["doc:36a911e2::chp:1::pg:1::sec:2"]
        )

    def test_merge_appends_only_new_source_ids(self):
        # First pass creates the character.
        self.run_resolver([
            {"full_name": "Joe le Clodo", "aliases": ["Bobby"],
             "source_ids": ["doc:36a911e2::chp:1::pg:1::sec:2"]},
        ])
        # Second pass: same character, same unit + a NEW unit.
        result, _ = self.run_resolver([
            {"full_name": "Joe le Clodo", "aliases": ["Bobby"],
             "source_ids": [
                 "doc:36a911e2::chp:1::pg:1::sec:2",
                 "doc:36a911e2::chp:1::pg:2::sec:1",
             ]},
        ])
        self.assertEqual(result.payload["created"], 0)
        self.assertEqual(result.payload["merged"], 1)

        data = read_character(self.chars / "joe-le-clodo.md")
        by_alias = {n["alias"]: n["source_ids"] for n in data["names"]}
        self.assertEqual(
            by_alias["Joe le Clodo"],
            ["doc:36a911e2::chp:1::pg:1::sec:2",
             "doc:36a911e2::chp:1::pg:2::sec:1"],
        )
        self.assertEqual(
            by_alias["Bobby"],
            ["doc:36a911e2::chp:1::pg:1::sec:2",
             "doc:36a911e2::chp:1::pg:2::sec:1"],
        )

    def test_merge_is_idempotent(self):
        entry = [{"full_name": "Joe le Clodo", "aliases": ["Bobby"],
                  "source_ids": ["doc:36a911e2::chp:1::pg:1::sec:2"]}]
        self.run_resolver(entry)
        before = (self.chars / "joe-le-clodo.md").read_text(encoding="utf-8")
        result, _ = self.run_resolver(entry)
        after = (self.chars / "joe-le-clodo.md").read_text(encoding="utf-8")
        self.assertEqual(before, after)  # byte-identical: nothing re-added
        self.assertEqual(result.payload["merged"], 1)

    def test_merge_adds_brand_new_alias(self):
        self.run_resolver([
            {"full_name": "Joe le Clodo", "aliases": [],
             "source_ids": ["doc:a::sec:1"]},
        ])
        self.run_resolver([
            {"full_name": "Joe le Clodo", "aliases": ["le Clodo"],
             "source_ids": ["doc:a::sec:1"]},
        ])
        data = read_character(self.chars / "joe-le-clodo.md")
        self.assertEqual(
            [n["alias"] for n in data["names"]],
            ["Joe le Clodo", "le Clodo"],
        )

    def test_alias_hit_finds_existing_file(self):
        # The 'existing character' path keys on ANY name of the entry —
        # here the discovery names the character only by its alias, and
        # the index scan routes it into Joe's file.
        self.run_resolver([
            {"full_name": "Joe le Clodo", "aliases": ["Bobby"],
             "source_ids": ["doc:a::sec:1"]},
        ])
        result, _ = self.run_resolver([
            {"full_name": "Bobby", "aliases": [],
             "source_ids": ["doc:b::sec:9"]},
        ])
        self.assertEqual(result.payload["created"], 0)
        self.assertEqual(result.payload["merged"], 1)
        # 'Bobby' gained its own bullet inside JOE's file — no bobby.md.
        data = read_character(self.chars / "joe-le-clodo.md")
        by_alias = {n["alias"]: n["source_ids"] for n in data["names"]}
        self.assertEqual(by_alias["Bobby"], ["doc:a::sec:1", "doc:b::sec:9"])
        self.assertFalse((self.chars / "bobby.md").exists())

    def test_shapeless_entries_are_skipped(self):
        result, context = self.run_resolver([
            "not a dict",
            {"aliases": ["no name"]},
        ])
        self.assertEqual(result.status, AgentStatus.OK)
        self.assertEqual(result.payload["skipped"], 2)
        self.assertEqual(result.payload["created"], 0)
        kinds = [e["kind"] for e in context.events]
        self.assertIn("resolver_skipped", kinds)

    def test_index_line_written_on_create_and_merge(self):
        self.run_resolver([
            {"full_name": "Joe le Clodo", "aliases": ["Bobby"],
             "source_ids": ["doc:a::sec:1"]},
        ])
        index = self.chars / INDEX_FILENAME
        self.assertTrue(index.exists())
        text = index.read_text(encoding="utf-8")
        self.assertIn("- Joe le Clodo (aka: Bobby)", text)

        self.run_resolver([
            {"full_name": "Mercedes", "aliases": [],
             "source_ids": ["doc:a::sec:2"]},
        ])
        text = index.read_text(encoding="utf-8")
        self.assertIn("- Mercedes", text)
        self.assertIn("- Joe le Clodo (aka: Bobby)", text)


class IndexScanTest(unittest.TestCase):
    """The sidecar index is the resolver's batch-scan surface."""

    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.chars = characters_dir(self.base)
        self.resolver = CharacterResolver(base_dir=self.base)

    def test_index_rebuild_after_pass(self):
        context = make_context([
            {"full_name": "A", "aliases": ["a1"], "source_ids": ["doc:a::sec:1"]},
            {"full_name": "B", "aliases": [], "source_ids": ["doc:a::sec:2"]},
        ])
        self.resolver.run(context)
        from src.knowledge.character_markdown_store import load_index

        characters = load_index(self.chars / INDEX_FILENAME)
        self.assertEqual(
            [c["full_name"] for c in characters], ["A", "B"]
        )

    def test_hand_created_file_is_indexed_on_next_pass(self):
        # The user hand-writes a character file between runs: the rebuild
        # picks it up (files are the truth, the index their projection).
        from src.knowledge.character_markdown_store import write_character

        write_character(
            "Hand Made",
            [{"alias": "Hand Made", "source_ids": []}],
            self.chars / "hand-made.md",
        )
        context = make_context([
            {"full_name": "A", "aliases": [], "source_ids": ["doc:a::sec:1"]},
        ])
        self.resolver.run(context)
        from src.knowledge.character_markdown_store import load_index

        characters = load_index(self.chars / INDEX_FILENAME)
        self.assertEqual(
            [c["full_name"] for c in characters], ["A", "Hand Made"]
        )


class ValidateHookTest(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.chars = characters_dir(self.base)
        self.resolver = CharacterResolver(base_dir=self.base)

    def test_validate_ok_when_index_written(self):
        context = make_context([
            {"full_name": "A", "aliases": [], "source_ids": ["doc:a::1"]},
        ])
        self.resolver.run(context)
        self.assertIsNone(self.resolver.validate(context))

    def test_validate_fails_when_index_missing(self):
        context = make_context([
            {"full_name": "A", "aliases": [], "source_ids": ["doc:a::1"]},
        ])
        self.resolver.run(context)
        (self.chars / INDEX_FILENAME).unlink()
        validation = self.resolver.validate(context)
        self.assertIsNotNone(validation)
        self.assertEqual(validation.failure_domain.value, "input_data")

    def test_validate_none_before_any_run(self):
        context = IngestionContext(document_path=Path("gazette.pdf"))
        self.assertIsNone(self.resolver.validate(context))


class RegistryAndGraphWiringTest(unittest.TestCase):
    def test_registry_has_entity_resolver(self):
        from src.agents.registry import build_default_agents

        registry = build_default_agents()
        self.assertIn("entity_resolver", registry)
        self.assertIsInstance(registry["entity_resolver"], CharacterResolver)

    def test_graph_step_targets_entity_resolver(self):
        from src.graphs.ingestion_graph import IngestionGraph

        steps = {s.name: s.agent_key for s in IngestionGraph.DEFAULT_STEPS}
        self.assertEqual(steps["check_and_merge"], "entity_resolver")


if __name__ == "__main__":
    unittest.main()
