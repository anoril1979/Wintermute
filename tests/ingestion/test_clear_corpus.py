"""Tests for the corpus-clear engine and the clear.py CLI.

Hermetic: a temp project tree stands in for the real one — generated
stores filled with fake artifacts, a protected documents root with a
source file, and a config side mocked through the loaders' seams. No
real data path is ever touched.
"""

from __future__ import annotations

import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.clear_corpus import (  # noqa: E402
    STATUS_CLEARED,
    STATUS_PARTIAL,
    ClearPaths,
    clear_corpus,
    resolve_clear_paths,
)
from scripts.clear import CONFIRM_WORD, main as clear_main  # noqa: E402


class TempCorpus:
    """A fake project tree with every generated surface populated."""

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()

        def mk(rel: str, *files: str) -> Path:
            path = self.root / rel
            path.mkdir(parents=True, exist_ok=True)
            for name in files:
                (path / name).write_text("generated", encoding="utf-8")
            return path

        self.vector = mk("data/vector", "chroma.sqlite3")
        self.extracted = mk("data/extracted", "doc.json")
        self.mineru = mk("data/extracted/mineru/doc", "out.json")
        self.summarized = mk("data/summarized", "doc.json")
        self.consolidation = mk("data/cache/consolidation", "doc.json")
        self.knowledge_cache = mk("data/cache/knowledge", "doc.json")
        self.extraction_jobs = self.root / "data/cache/extraction_jobs.json"
        self.extraction_jobs.write_text("{}", encoding="utf-8")
        self.summarization_jobs = self.root / "data/cache/summarization_jobs.json"
        self.summarization_jobs.write_text("{}", encoding="utf-8")
        self.knowledge = mk("data/knowledge/characters", "joe.md",
                            "characters.md")
        self.sources = mk("data/sources/pdf", "KEEP ME.pdf")
        self.code = mk("src", "keep.py")

    def paths(self) -> ClearPaths:
        return ClearPaths(
            project_root=self.root,
            vector_store=self.vector,
            extraction_store=self.extracted,
            mineru_store=self.mineru,
            summarized_store=self.summarized,
            consolidation_cache=self.consolidation,
            knowledge_cache=self.knowledge_cache,
            extraction_job_file=self.extraction_jobs,
            summarization_job_file=self.summarization_jobs,
            knowledge_base=self.knowledge,
            documents_root=self.sources,
        )

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class ClearCorpusEngineTest(unittest.TestCase):
    def setUp(self):
        self.corpus = TempCorpus()

    def tearDown(self):
        self.corpus.cleanup()

    def test_every_generated_surface_is_erased(self):
        report = clear_corpus(self.corpus.paths())
        self.assertEqual(report["status"], STATUS_CLEARED, report)
        for target in (
            self.corpus.vector, self.corpus.extracted, self.corpus.mineru,
            self.corpus.summarized, self.corpus.consolidation,
            self.corpus.knowledge_cache, self.corpus.knowledge,
        ):
            self.assertFalse(target.exists(), target)
        self.assertFalse(self.corpus.extraction_jobs.exists())
        self.assertFalse(self.corpus.summarization_jobs.exists())

    def test_sources_and_code_are_preserved(self):
        report = clear_corpus(self.corpus.paths())
        self.assertEqual(report["status"], STATUS_CLEARED)
        self.assertTrue((self.corpus.sources / "KEEP ME.pdf").exists())
        self.assertTrue((self.corpus.code / "keep.py").exists())
        self.assertEqual(report["documents_root_kept"],
                         str(self.corpus.sources))

    def test_idempotent_second_run_is_a_clean_noop(self):
        self.assertEqual(clear_corpus(self.corpus.paths())["status"],
                         STATUS_CLEARED)
        second = clear_corpus(self.corpus.paths())
        self.assertEqual(second["status"], STATUS_CLEARED)
        self.assertTrue(all(step["removed"] is None
                            for step in second["steps"].values()))

    def test_partial_failure_is_reported_per_step(self):
        # A locked store (simulated): the step fails, the others still run.
        paths = self.corpus.paths()
        real_rmtree = shutil.rmtree

        def fake_rmtree(path, **kwargs):
            if Path(path).resolve() == paths.vector_store.resolve():
                raise OSError("store locked")
            return real_rmtree(path, **kwargs)

        with mock.patch("src.ingestion.clear_corpus.shutil") as fake_shutil:
            fake_shutil.rmtree = fake_rmtree
            report = clear_corpus(paths)
        self.assertEqual(report["status"], STATUS_PARTIAL)
        self.assertIn("vector_store", report["failed_steps"])
        self.assertIn("store locked", report["reason"])
        # Every other surface WAS erased despite the failure.
        self.assertFalse(self.corpus.extracted.exists())
        self.assertTrue(self.corpus.vector.exists())

    def test_refuses_to_remove_outside_project_root(self):
        outside = Path(tempfile.mkdtemp())  # NOT inside the project root
        try:
            (outside / "victim.txt").write_text("x", encoding="utf-8")
            paths = self.corpus.paths()
            hijacked = ClearPaths(**{
                **vars(paths),
                "knowledge_base": outside,
            })
            report = clear_corpus(hijacked)
            self.assertEqual(report["status"], STATUS_PARTIAL)
            self.assertIn("knowledge_base", report["failed_steps"])
            self.assertIn("outside the project root",
                          report["steps"]["knowledge_base"]["reason"])
            self.assertTrue((outside / "victim.txt").exists())
        finally:
            shutil.rmtree(outside, ignore_errors=True)

    def test_collision_with_documents_root_refuses_everything(self):
        paths = self.corpus.paths()
        hijacked = ClearPaths(**{**vars(paths), "extraction_store": paths.documents_root})
        report = clear_corpus(hijacked)
        self.assertEqual(report["status"], STATUS_PARTIAL)
        self.assertIn("collision", report["reason"])
        # Nothing was erased at all.
        self.assertTrue(self.corpus.vector.exists())
        self.assertTrue((self.corpus.sources / "KEEP ME.pdf").exists())


class ResolveClearPathsTest(unittest.TestCase):
    def test_resolves_from_the_real_configs(self):
        paths = resolve_clear_paths()  # the real yaml — read-only access
        self.assertTrue(paths.vector_store.is_absolute())
        self.assertTrue(paths.documents_root.is_absolute())
        self.assertNotEqual(paths.vector_store, paths.documents_root)

    def test_broken_config_refuses_to_clear(self):
        with mock.patch("src.tools.config_loader.load_ingestion_config",
                        side_effect=RuntimeError("broken yaml")):
            with self.assertRaises(Exception):
                resolve_clear_paths()


class ClearScriptTest(unittest.TestCase):
    """The CLI: confirmation gate, dry-run, automation flag, exit codes."""

    def setUp(self):
        self.corpus = TempCorpus()
        paths = self.corpus.paths()
        patcher = mock.patch("scripts.clear.resolve_clear_paths",
                             return_value=paths)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.corpus.cleanup)

    def test_dry_run_deletes_nothing(self):
        code = clear_main(["--dry-run"])
        self.assertEqual(code, 0)
        self.assertTrue(self.corpus.vector.exists())
        self.assertTrue((self.corpus.sources / "KEEP ME.pdf").exists())

    def test_non_tty_without_yes_refuses(self):
        with mock.patch("sys.stdin", new=io.StringIO()) as fake_stdin:
            fake_stdin.isatty = lambda: False
            code = clear_main([])
        self.assertEqual(code, 1)  # refused, not cleared
        self.assertTrue(self.corpus.vector.exists())

    def test_wrong_confirmation_aborts_cleanly(self):
        stdin = io.StringIO(f"yes\n")  # a reflex 'yes' is NOT accepted
        stdin.isatty = lambda: True
        with mock.patch("sys.stdin", new=stdin):
            code = clear_main([])
        self.assertEqual(code, 0)
        self.assertTrue(self.corpus.vector.exists())

    def test_typed_confirmation_clears(self):
        stdin = io.StringIO(f"{CONFIRM_WORD}\n")
        stdin.isatty = lambda: True
        with mock.patch("sys.stdin", new=stdin):
            code = clear_main([])
        self.assertEqual(code, 0)
        self.assertFalse(self.corpus.vector.exists())
        self.assertTrue((self.corpus.sources / "KEEP ME.pdf").exists())

    def test_yes_flag_clears_without_prompt(self):
        code = clear_main(["--yes"])
        self.assertEqual(code, 0)
        self.assertFalse(self.corpus.vector.exists())
        self.assertTrue((self.corpus.sources / "KEEP ME.pdf").exists())

    def test_config_error_exit_code(self):
        with mock.patch("scripts.clear.resolve_clear_paths",
                        side_effect=RuntimeError("broken")):
            code = clear_main(["--yes"])
        self.assertEqual(code, 2)  # EXIT_CONFIG

    def test_yes_with_dry_run_still_deletes_nothing(self):
        code = clear_main(["--yes", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertTrue(self.corpus.vector.exists())


if __name__ == "__main__":
    unittest.main()
