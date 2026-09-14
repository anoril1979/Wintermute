"""Tests for the markdown knowledge-base store (character files + index).

Hermetic: every path lives in a throwaway temp dir. Covers path helpers,
character-file roundtrip (write/read, hand-edit errors), the sidecar index
(upsert, merge-on-resave, removal) and the atomic-write hygiene.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.knowledge.character_markdown_store import (
    INDEX_FILENAME,
    CharacterMarkdownError,
    character_path_for,
    characters_dir,
    format_index_line,
    index_path_for,
    knowledge_base_dir,
    load_index,
    read_character,
    remove_from_index,
    slugify_filename,
    upsert_index,
    write_character,
)


def tmp_dir() -> Path:
    return Path(tempfile.mkdtemp())


class PathHelpersTest(unittest.TestCase):
    def test_slugify_filename(self):
        self.assertEqual(slugify_filename("Édmond Dantès"), "edmond-dantes")
        self.assertEqual(slugify_filename("Joe  le   Clodo!"), "joe-le-clodo")
        self.assertEqual(slugify_filename("###"), "unnamed")

    def test_character_path_for_uses_slug_in_characters_subdir(self):
        base = tmp_dir()
        path = character_path_for("Édmond Dantès", base)
        self.assertEqual(path, base / "characters" / "edmond-dantes.md")

    def test_index_path_lives_in_characters_subdir(self):
        base = tmp_dir()
        self.assertEqual(index_path_for(base), base / "characters" / "characters.md")
        self.assertEqual(characters_dir(base), base / "characters")


class CharacterFileTest(unittest.TestCase):
    def test_write_then_read_roundtrip(self):
        base = tmp_dir()
        path = character_path_for("Joe le Clodo", base)
        write_character(
            "Joe le Clodo",
            [
                {"alias": "Joe le Clodo", "source_ids": ["doc:36a911e2::chp:1::pg:1::sec:2"]},
                {"alias": "Bobby", "source_ids": [
                    "doc:36a911e2::chp:1::pg:1::sec:4",
                    "doc:36a911e2::chp:1::pg:2::sec:1",
                ]},
            ],
            path,
        )
        text = path.read_text(encoding="utf-8")
        self.assertIn("# Character : Joe le Clodo", text)
        self.assertIn("Known names:", text)

        data = read_character(path)
        self.assertEqual(data["full_name"], "Joe le Clodo")
        names = data["names"]
        self.assertEqual(len(names), 2)
        self.assertEqual(names[0]["alias"], "Joe le Clodo")
        self.assertEqual(names[0]["source_ids"], ["doc:36a911e2::chp:1::pg:1::sec:2"])
        self.assertEqual(len(names[1]["source_ids"]), 2)

    def test_name_without_source_gets_placeholder(self):
        base = tmp_dir()
        path = character_path_for("Solo", base)
        write_character("Solo", [{"alias": "Solo", "source_ids": []}], path)
        data = read_character(path)
        self.assertEqual(data["names"][0]["source_ids"], [])
        self.assertIn("(no source yet)", path.read_text(encoding="utf-8"))

    def test_read_missing_title_is_an_error(self):
        base = tmp_dir()
        path = base / "broken.md"
        path.write_text("no title here\n", encoding="utf-8")
        with self.assertRaises(CharacterMarkdownError):
            read_character(path)

    def test_read_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            read_character(tmp_dir() / "nope.md")


class IndexTest(unittest.TestCase):
    def test_format_index_line(self):
        self.assertEqual(format_index_line("Joe", ["Joe", "Bobby"]), "- Joe (aka: Bobby)")
        self.assertEqual(format_index_line("Joe", []), "- Joe")

    def test_upsert_then_load(self):
        base = tmp_dir()
        index = base / INDEX_FILENAME
        upsert_index(index, "Joe le Clodo", ["Bobby", "le Clodo"])
        upsert_index(index, "Mercedes", [])
        characters = load_index(index)
        self.assertEqual([c["full_name"] for c in characters], ["Joe le Clodo", "Mercedes"])
        self.assertEqual(characters[0]["aliases"], ["Bobby", "le Clodo"])

    def test_upsert_merges_aliases_across_saves(self):
        base = tmp_dir()
        index = base / INDEX_FILENAME
        upsert_index(index, "Joe", ["Bobby"])
        upsert_index(index, "Joe", ["the Clodo"])  # re-save with new alias
        characters = load_index(index)
        self.assertEqual(len(characters), 1)
        self.assertEqual(characters[0]["aliases"], ["the Clodo", "Bobby"])

    def test_upsert_replaces_line_keeps_order(self):
        base = tmp_dir()
        index = base / INDEX_FILENAME
        upsert_index(index, "A", [])
        upsert_index(index, "B", [])
        upsert_index(index, "A", ["alias-a"])  # replacement keeps position
        characters = load_index(index)
        self.assertEqual([c["full_name"] for c in characters], ["A", "B"])

    def test_remove_from_index(self):
        base = tmp_dir()
        index = base / INDEX_FILENAME
        upsert_index(index, "Joe", [])
        upsert_index(index, "Mercedes", [])
        self.assertTrue(remove_from_index(index, "Joe"))
        self.assertFalse(remove_from_index(index, "Joe"))  # already gone
        self.assertEqual(
            [c["full_name"] for c in load_index(index)], ["Mercedes"]
        )

    def test_load_missing_index_is_empty(self):
        self.assertEqual(load_index(tmp_dir() / "nope.md"), [])


class knowledgeBaseDirDefaultsTest(unittest.TestCase):
    def test_default_dir_fails_open(self):
        # Without touching config: the default folder resolves under the
        # project root (config unreadable -> default, never an exception).
        path = knowledge_base_dir()
        self.assertTrue(path.is_absolute())


if __name__ == "__main__":
    unittest.main()
