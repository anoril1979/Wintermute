"""Tests for the ingestion CLI scripts.

Focused on the config-driven seam: the durable log path comes from
ingestion.yaml (``ingestion_log``), with a fail-open default when the
config is broken — the CLI must stay able to log the very error that
broke its configuration.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest import DEFAULT_INGESTION_LOG, ingestion_log_path


class IngestionLogPathTest(unittest.TestCase):
    def test_default_when_config_has_no_key(self):
        config = {"documents_root": "data/sources"}
        with mock.patch(
            "src.tools.config_loader.load_ingestion_config",
            return_value=config,
        ):
            self.assertEqual(ingestion_log_path(), DEFAULT_INGESTION_LOG)

    def test_relative_path_resolved_against_project_root(self):
        config = {"ingestion_log": "data/logs/custom.log"}
        with mock.patch(
            "src.tools.config_loader.load_ingestion_config",
            return_value=config,
        ):
            path = ingestion_log_path()
        self.assertTrue(path.is_absolute())
        self.assertEqual(path, PROJECT_ROOT / "data" / "logs" / "custom.log")
        self.assertNotEqual(path, DEFAULT_INGESTION_LOG)

    def test_absolute_path_used_verbatim(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"ingestion_log": str(Path(tmp) / "ingestion.log")}
            with mock.patch(
                "src.tools.config_loader.load_ingestion_config",
                return_value=config,
            ):
                self.assertEqual(
                    ingestion_log_path(), Path(tmp) / "ingestion.log"
                )

    def test_falls_back_when_config_loader_raises(self):
        with mock.patch(
            "src.tools.config_loader.load_ingestion_config",
            side_effect=RuntimeError("broken yaml"),
        ):
            self.assertEqual(ingestion_log_path(), DEFAULT_INGESTION_LOG)

    def test_falls_back_on_bad_values(self):
        for bad in (None, 7, "  ", "../escape/ingestion.log"):
            config = {"ingestion_log": bad}
            with mock.patch(
                "src.tools.config_loader.load_ingestion_config",
                return_value=config,
            ):
                with self.subTest(bad=bad):
                    self.assertEqual(ingestion_log_path(), DEFAULT_INGESTION_LOG)


if __name__ == "__main__":
    unittest.main()
