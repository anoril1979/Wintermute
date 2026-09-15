"""Tests for the source markdown knowledge store (per-document
registrations + sidecar listing) and the SourceRegistrationAgent that
writes them.

Hermetic: temp knowledge bases written through the real store primitives —
no config dependency (base_dir overrides everywhere), no LLM, no vector
store.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from src.knowledge.source_markdown_store import (
    LISTING_FILENAME,
    SOURCE_TITLE_PREFIX,
    SourceMarkdownError,
    document_type_for,
    filename_for_doc_id,
    format_listing_line,
    listing_path_for,
    load_listing,
    read_notes,
    read_source,
    rebuild_listing,
    remove_registration,
    source_path_for,
    sources_dir,
    write_source,
)
from src.knowledge.character_markdown_store import knowledge_base_dir


def _write(base: Path, doc_id: str = "doc:3fa2b81c", **overrides) -> Path:
    kwargs = dict(
        doc_id=doc_id,
        title="Gazette #1",
        file_name="Dark Earth - Gazette #1.pdf",
        path="data/sources/pdf/Dark Earth - Gazette #1.pdf",
        origin="community",
        chapters=5,
        pages=17,
        registration_path=source_path_for(doc_id, base),
    )
    kwargs.update(overrides)
    # Like the agent: the readable type derives from the source path.
    kwargs.setdefault("doc_type", document_type_for(kwargs["path"]))
    return write_source(**kwargs)


class FilenameTest(unittest.TestCase):
    def test_doc_id_maps_to_a_file_name(self):
        self.assertEqual(filename_for_doc_id("doc:3fa2b81c"), "doc_3fa2b81c.md")

    def test_invalid_doc_id_is_rejected(self):
        for bad in ("doc:XYZ", "doc:3fa2b81", "source:001", "", "doc:3fa2b81c::chp:1"):
            with self.assertRaises(SourceMarkdownError, msg=bad):
                filename_for_doc_id(bad)


class DocumentTypeTest(unittest.TestCase):
    def test_extension_derives_the_type(self):
        for path, expected in (
            ("data/sources/pdf/Gazette.pdf", "pdf"),
            ("data/sources/text/notes.txt", "text"),
            ("data/sources/text/notes.md", "markdown"),
            ("data/sources/html/page.htm", "html"),
            ("data/sources/word/report.docx", "word"),
            ("data/sources/openoffice/thesis.odt", "openoffice"),
        ):
            self.assertEqual(document_type_for(path), expected, path)

    def test_unknown_or_missing_extension_falls_back_to_other(self):
        self.assertEqual(document_type_for("data/sources/pdf/Gazette.weird"), "other")
        self.assertEqual(document_type_for("no-extension"), "other")
        self.assertEqual(document_type_for(""), "other")

    def test_extension_case_does_not_matter(self):
        self.assertEqual(document_type_for("Gazette.PDF"), "pdf")


class SourceFileTest(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="wm_source_store_"))
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)

    def test_write_creates_the_file_with_all_metadata(self):
        path = _write(self.base)
        self.assertTrue(path.is_file())
        self.assertEqual(path.parent, sources_dir(self.base))
        text = path.read_text(encoding="utf-8")
        self.assertIn(f"{SOURCE_TITLE_PREFIX}Gazette #1", text)
        self.assertIn("- Id: doc:3fa2b81c", text)
        self.assertIn("- Type: pdf", text)
        self.assertIn("- File: Dark Earth - Gazette #1.pdf", text)
        self.assertIn("- Origin: community", text)
        self.assertIn("- Chapters: 5", text)
        self.assertIn("- Pages: 17", text)
        self.assertIn("Notes:", text)

    def test_roundtrip(self):
        path = _write(self.base)
        data = read_source(path)
        self.assertEqual(data["doc_id"], "doc:3fa2b81c")
        self.assertEqual(data["type"], "pdf")
        self.assertEqual(data["title"], "Gazette #1")
        self.assertEqual(data["file"], "Dark Earth - Gazette #1.pdf")
        self.assertEqual(data["origin"], "community")
        self.assertEqual(data["chapters"], 5)
        self.assertEqual(data["pages"], 17)
        self.assertEqual(data["notes"], "")

    def test_notes_are_preserved_on_rewrite(self):
        path = _write(self.base)
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\nUser note: the author confirmed the map is canon.\n",
            encoding="utf-8",
        )
        # Re-ingestion: same id, changed counts.
        _write(self.base, chapters=6, pages=18)
        data = read_source(path)
        self.assertEqual(data["chapters"], 6, "system fields are refreshed")
        self.assertIn(
            "User note: the author confirmed the map is canon.",
            data["notes"],
            "the user's Notes section survives re-ingestion",
        )

    def test_read_notes_on_missing_file_is_empty(self):
        self.assertEqual(read_notes(self.base / "nope.md"), "")

    def test_missing_id_line_is_an_error(self):
        path = self.base / "sources" / "doc_11111111.md"
        path.parent.mkdir(parents=True)
        path.write_text("# Source : Broken\n\n- Origin: canon\n", encoding="utf-8")
        with self.assertRaises(SourceMarkdownError):
            read_source(path)

    def test_missing_file_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            read_source(self.base / "sources" / "doc_11111111.md")


class SourceListingTest(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="wm_source_listing_"))
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)

    def test_format_line(self):
        self.assertEqual(
            format_listing_line("doc:3fa2b81c", "Gazette #1"),
            "- doc:3fa2b81c — Gazette #1",
        )

    def test_rebuild_lists_all_registrations(self):
        _write(self.base, doc_id="doc:bbbbbbbb", title="Beta Doc")
        _write(self.base, doc_id="doc:aaaaaaaa", title="Alpha Doc")
        listing = rebuild_listing(self.base)
        self.assertEqual(listing, listing_path_for(self.base))
        entries = load_listing(listing)
        # Files are read in id order — stable listing.
        self.assertEqual(
            [(e["doc_id"], e["stem"]) for e in entries],
            [("doc:aaaaaaaa", "Alpha Doc"), ("doc:bbbbbbbb", "Beta Doc")],
        )

    def test_removed_registration_disappears_from_the_listing(self):
        _write(self.base)
        _write(self.base, doc_id="doc:bbbbbbbb", title="Beta Doc")
        rebuild_listing(self.base)
        self.assertTrue(remove_registration("doc:3fa2b81c", self.base))
        self.assertEqual(
            [e["doc_id"] for e in load_listing(listing_path_for(self.base))],
            ["doc:bbbbbbbb"],
        )
        self.assertFalse(source_path_for("doc:3fa2b81c", self.base).is_file())

    def test_remove_unknown_registration_is_a_noop(self):
        self.assertFalse(remove_registration("doc:00000000", self.base))

    def test_missing_listing_is_empty(self):
        self.assertEqual(load_listing(listing_path_for(self.base)), [])

    def test_malformed_listing_line_is_an_error(self):
        folder = sources_dir(self.base)
        folder.mkdir(parents=True)
        (folder / LISTING_FILENAME).write_text(
            "- doc:nothex — broken\n", encoding="utf-8"
        )
        with self.assertRaises(SourceMarkdownError):
            load_listing(listing_path_for(self.base))


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

from src.agents.agents.source_registration_agent import SourceRegistrationAgent
from src.agents.contexts import IngestionContext
from src.agents.protocols import AgentStatus
from src.extraction.ids import assign_extract_ids
from src.extraction.models import (
    Chapter,
    DocumentExtract,
    PageContent,
    Section,
    TextBlock,
    TocEntry,
)


def _document(stem: str = "Gazette", title: str = "Gazette") -> DocumentExtract:
    """A minimal id-assigned DocumentExtract: one chapter, one page."""
    doc = DocumentExtract(
        source_path=f"data/sources/pdf/{stem}.pdf",
        title=title,
        author="",
        subject="",
        total_pages=2,
        chapters=[Chapter(
            toc_entry=TocEntry(level=1, title=title, page_number=1, page_index=0),
            pages=[PageContent(
                page_number=1, width=595, height=842, raw_text="x",
                sections=[Section(
                    section_id=0,
                    blocks=[TextBlock(block_id=0, page_number=1,
                                      bbox=(0, 0, 1, 1), raw_text="x")],
                    page_number=1, bbox=(0, 0, 1, 1), raw_text="x",
                )],
            )],
            full_text="x",
        )],
        origin="canon",
    )
    return assign_extract_ids(doc)


class SourceRegistrationAgentTest(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="wm_source_agent_"))
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        self.agent = SourceRegistrationAgent(base_dir=self.base)

    def _context(self, document) -> IngestionContext:
        context = IngestionContext(document_path=Path(document.source_path))
        context.outputs["content_extraction"] = document
        return context

    def test_registers_the_document_from_the_context(self):
        document = _document()
        result = self.agent.run(self._context(document))
        self.assertEqual(result.status.value, "ok")
        self.assertTrue(result.payload["created"])

        registration = source_path_for(document.id, self.base)
        self.assertTrue(registration.is_file())
        data = read_source(registration)
        self.assertEqual(data["doc_id"], document.id)
        self.assertEqual(data["type"], "pdf", "extension-derived, readable")
        self.assertEqual(data["title"], "Gazette")
        self.assertEqual(data["file"], "Gazette.pdf")
        self.assertEqual(data["origin"], "canon")
        self.assertEqual(data["chapters"], 1)
        self.assertEqual(data["pages"], 2)
        # The listing was rebuilt.
        self.assertEqual(
            [e["doc_id"] for e in load_listing(listing_path_for(self.base))],
            [document.id],
        )

    def test_rerun_refreshes_in_place(self):
        document = _document()
        self.agent.run(self._context(document))
        # Second ingestion: more pages, and a user note appeared meanwhile.
        registration = source_path_for(document.id, self.base)
        registration.write_text(
            registration.read_text(encoding="utf-8") + "\nHand-written note.\n",
            encoding="utf-8",
        )
        document.total_pages = 9
        result = self.agent.run(self._context(document))
        self.assertEqual(result.status.value, "ok")
        self.assertFalse(result.payload["created"], "same id = refresh, not create")
        self.assertEqual(result.payload["type"], "pdf")
        data = read_source(registration)
        self.assertEqual(data["pages"], 9)
        self.assertIn("Hand-written note.", data["notes"])

    def test_no_document_fails_input_data(self):
        result = self.agent.run(IngestionContext())
        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.failure_domain.value, "input_data")

    def test_validate_requires_file_and_listing(self):
        context = self._context(_document())
        self.agent.run(context)
        self.assertIsNone(self.agent.validate(context))
        source_path_for(context.outputs["source_registration"]["doc_id"],
                        self.base).unlink()
        validation = self.agent.validate(context)
        self.assertIsNotNone(validation)


class SourceRegistrationWiringTest(unittest.TestCase):
    def test_graph_step_after_check_and_merge(self):
        from src.graphs.ingestion_graph import IngestionGraph

        names = [s.name for s in IngestionGraph.DEFAULT_STEPS]
        self.assertIn("source_registration", names)
        self.assertEqual(
            names.index("source_registration"),
            names.index("check_and_merge") + 1,
            "the registration follows the entity resolution",
        )

    def test_registry_includes_the_agent(self):
        from src.agents.registry import build_default_agents

        registry = build_default_agents()
        self.assertIn("source_registrar", registry)
        self.assertEqual(registry["source_registrar"].name, "source_registrar")


class SourceRegistrationPurgeTest(unittest.TestCase):
    """The removal engine deletes the registration like any projection."""

    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="wm_source_purge_"))
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)

    def test_remove_document_purges_the_registration(self):
        import unittest.mock

        from src.ingestion.remove_from_corpus import remove_document

        document = _document(stem="PurgeMe")
        _write(self.base, doc_id=document.id, title="PurgeMe")
        rebuild_listing(self.base)

        with unittest.mock.patch(
            "src.knowledge.source_markdown_store.knowledge_base_dir",
            return_value=self.base,
        ), unittest.mock.patch(
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
            report = remove_document(
                "PurgeMe.pdf", source_path=Path("data/sources/pdf/PurgeMe.pdf"))

        self.assertEqual(report["status"], "removed")
        step = report["steps"]["source_registration"]
        self.assertTrue(step["ok"])
        self.assertTrue(step["removed"], "the registration was deleted")
        self.assertFalse(source_path_for(document.id, self.base).is_file())
        self.assertEqual(load_listing(listing_path_for(self.base)), [])

    def test_never_registered_is_a_noop_step(self):
        import unittest.mock

        from src.ingestion.remove_from_corpus import remove_document

        document = _document(stem="Ghost")
        with unittest.mock.patch(
            "src.knowledge.source_markdown_store.knowledge_base_dir",
            return_value=self.base,
        ), unittest.mock.patch(
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
            report = remove_document(
                "Ghost.pdf", source_path=Path("data/sources/pdf/Ghost.pdf"))

        self.assertEqual(report["status"], "removed")
        step = report["steps"]["source_registration"]
        self.assertTrue(step["ok"])
        self.assertFalse(step["removed"], "nothing to delete: idempotent no-op")


if __name__ == "__main__":
    unittest.main()
