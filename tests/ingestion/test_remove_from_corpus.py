"""Tests for the corpus-removal engine's knowledge-base purge.

Hermetic: temp knowledge base, other removal steps mocked (vector/Chroma,
job files, JSON stores, MinerU are covered by their own modules). Focuses
on the user-specified semantics:

* source ids of the removed document are dropped from every alias;
* an alias left with no source is removed (file AND index);
* a character left with no alias at all loses its file and index line;
* other documents' provenance is untouched; idempotent re-run is safe.
"""

from __future__ import annotations

import tempfile
import unittest
import unittest.mock
from pathlib import Path

from src.ingestion.remove_from_corpus import (
    STATUS_REMOVED,
    _purge_knowledge_base,
    remove_document,
)
from src.knowledge.character_markdown_store import (
    character_path_for,
    characters_dir,
    index_path_for,
    load_index,
    read_character,
    write_character,
    write_index,
)

DOC1 = "doc:aaaaaaaa"   # the document being removed
DOC2 = "doc:bbbbbbbb"   # a document that stays

ID1 = f"{DOC1}::chp:1::pg:1::sec:2"
ID2 = f"{DOC2}::chp:2::pg:3::sec:1"


class PurgeTest(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.chars = characters_dir(self.base)
        self.chars.mkdir(parents=True, exist_ok=True)
        # The purge resolves the REAL config base dir (one folder per
        # entity type): redirect the ROOT to the temp base for every
        # test of this class.
        patcher = unittest.mock.patch(
            "src.knowledge.character_markdown_store.knowledge_base_dir",
            return_value=self.base,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_purge_drops_removed_doc_ids_keeps_others(self):
        path = character_path_for("Joe", self.base)
        write_character("Joe", [
            {"alias": "Joe", "source_ids": [ID1, ID2]},
            {"alias": "Bobby", "source_ids": [ID2]},
        ], path)
        report = _purge_knowledge_base(DOC1)
        self.assertTrue(report["ok"])
        self.assertEqual(report["purged_files"], 1)
        self.assertEqual(report["deleted_files"], 0)

        data = read_character(path)
        self.assertEqual(data["names"][0]["source_ids"], [ID2])
        self.assertEqual(data["names"][1]["source_ids"], [ID2])

    def test_sourceless_alias_is_removed_from_file(self):
        # 'Bobby' was ONLY ever seen in the removed document.
        path = character_path_for("Joe", self.base)
        write_character("Joe", [
            {"alias": "Joe", "source_ids": [ID1, ID2]},
            {"alias": "Bobby", "source_ids": [ID1]},
        ], path)
        report = _purge_knowledge_base(DOC1)
        self.assertTrue(report["ok"])

        data = read_character(path)
        self.assertEqual([n["alias"] for n in data["names"]], ["Joe"])
        # The index loses the pruned alias too.
        index = load_index(index_path_for(self.base))
        self.assertEqual(index[0]["aliases"], [])

    def test_character_with_nothing_left_is_deleted(self):
        path = character_path_for("Ghost", self.base)
        write_character("Ghost", [
            {"alias": "Ghost", "source_ids": [ID1]},
            {"alias": "Spook", "source_ids": [ID1]},
        ], path)
        # A survivor must remain so the base itself is not emptied.
        survivor = character_path_for("Joe", self.base)
        write_character("Joe", [{"alias": "Joe", "source_ids": [ID2]}], survivor)

        report = _purge_knowledge_base(DOC1)
        self.assertTrue(report["ok"])
        self.assertEqual(report["deleted_files"], 1)
        self.assertFalse(path.exists())

        index = load_index(index_path_for(self.base))
        self.assertEqual([c["full_name"] for c in index], ["Joe"])

    def test_untouched_file_is_byte_identical(self):
        path = character_path_for("Joe", self.base)
        write_character("Joe", [{"alias": "Joe", "source_ids": [ID2]}], path)
        before = path.read_text(encoding="utf-8")
        report = _purge_knowledge_base(DOC1)
        self.assertTrue(report["ok"])
        self.assertEqual(report["purged_files"], 0)
        self.assertEqual(path.read_text(encoding="utf-8"), before)

    def test_index_line_removed_for_deleted_character(self):
        path = character_path_for("Ghost", self.base)
        write_character("Ghost", [{"alias": "Ghost", "source_ids": [ID1]}], path)
        _purge_knowledge_base(DOC1)
        self.assertEqual(load_index(index_path_for(self.base)), [])

    def test_full_deletion_rebuilds_the_listing_at_the_base_root(self):
        # Regression: the purge used to rebuild the index at
        # <characters>/characters/characters.md (index_path_for fed the
        # characters FOLDER instead of the base dir), leaving the real
        # listing stale with entries of deleted characters.
        gone = character_path_for("Ghost", self.base)
        write_character("Ghost", [{"alias": "Ghost", "source_ids": [ID1]}], gone)
        survivor = character_path_for("Joe", self.base)
        write_character("Joe", [{"alias": "Joe", "source_ids": [ID2]}], survivor)

        report = _purge_knowledge_base(DOC1)
        self.assertTrue(report["ok"])

        # No bogus nested folder may appear next to the real listing.
        nested = self.chars / "characters"
        self.assertFalse(nested.exists(), "index rebuilt at the wrong depth")
        # The real listing shows exactly the surviving files.
        index = load_index(index_path_for(self.base))
        self.assertEqual([c["full_name"] for c in index], ["Joe"])

    def test_stale_index_lines_are_swept_by_the_rebuild(self):
        # The user's live failure mode: listing kept entries whose files
        # are gone (whatever the reason — the rebuild sweeps them).
        index_path = index_path_for(self.base)
        survivor = character_path_for("Joe", self.base)
        write_character("Joe", [{"alias": "Joe", "source_ids": [ID2]}], survivor)
        write_index(index_path, [
            {"full_name": "Harmandy", "aliases": []},       # file missing
            {"full_name": "Rorg Yanhalas", "aliases": []},  # file missing
            {"full_name": "Joe", "aliases": []},
        ])

        report = _purge_knowledge_base(DOC1)   # purges nothing of DOC1
        self.assertTrue(report["ok"])

        index = load_index(index_path)
        self.assertEqual([c["full_name"] for c in index], ["Joe"])

    def test_malformed_file_fails_the_step(self):
        (self.chars / "broken.md").write_text("garbage, no title\n", encoding="utf-8")
        report = _purge_knowledge_base(DOC1)
        self.assertFalse(report["ok"])
        self.assertIn("broken.md", report["reason"])

    def test_missing_base_dir_is_a_successful_noop(self):
        with unittest.mock.patch(
            "src.knowledge.character_markdown_store.characters_dir",
            return_value=Path(tempfile.mkdtemp()) / "absent",
        ):
            report = _purge_knowledge_base(DOC1)
        self.assertTrue(report["ok"])
        self.assertEqual(report["purged_files"], 0)
        self.assertEqual(report["deleted_files"], 0)


class RemovalReportTest(unittest.TestCase):
    def test_report_carries_the_knowledge_base_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "gaz.pdf"
            source.write_bytes(b"%PDF-fake")
            with unittest.mock.patch(
                "src.ingestion.remove_from_corpus._remove_vector_projection",
                return_value={"ok": True, "deleted": 0, "remaining": 0},
            ), unittest.mock.patch(
                "src.ingestion.remove_from_corpus._remove_job_entries",
                return_value={"ok": True, "removed": []},
            ), unittest.mock.patch(
                "src.ingestion.remove_from_corpus._remove_json_files",
                return_value={"ok": True, "removed": []},
            ), unittest.mock.patch(
                "src.ingestion.remove_from_corpus._purge_knowledge_base",
                return_value={"ok": True, "purged_files": 0, "deleted_files": 0},
            ), unittest.mock.patch(
                "src.ingestion.remove_from_corpus._remove_mineru_sandbox",
                return_value={"ok": True, "removed": None},
            ):
                report = remove_document("gaz.pdf", source_path=source)

            self.assertEqual(report["status"], STATUS_REMOVED)
            self.assertIn("knowledge_base", report["steps"])
            self.assertTrue(report["steps"]["knowledge_base"]["ok"])


if __name__ == "__main__":
    unittest.main()
