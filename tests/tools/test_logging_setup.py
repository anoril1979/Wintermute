"""Tests for the central logging setup and trace-event mirroring.

Covers:

* ``configure_logging`` — file creation, level/format application,
  idempotence, ``force=True`` reconfiguration, file logging disabled.
* Trace mirroring — every ``emit()`` on a graph context also reaches the
  ``wintermute.traces`` logger, so the "thinking" flow lands in the log
  file.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from src.agents.contexts import RoutingContext
import src.logging_setup as logging_setup
from src.logging_setup import (
    _INSTALLED,
    configure_logging,
)


class _LoggingTestCase(unittest.TestCase):
    """Base: isolate every test from the real setup.yaml and root handlers.

    Cleanup order matters on Windows: ``_restore`` (closes our file
    handlers) runs first, then the config patch stops, then the temp dir
    is removed — an open handler would make ``rmtree`` fail with
    ``PermissionError``.
    """

    def setUp(self) -> None:
        super().setUp()
        self.tmp = Path(tempfile.mkdtemp())
        # Registered first → runs last, after handlers are closed.
        self.addCleanup(shutil.rmtree, self.tmp, True)

        # Hermetic setup.yaml: no real file logging side effects unless a
        # test opts in via _configure_with_file().
        patcher = unittest.mock.patch(
            "src.tools.config_loader.load_setup_config",
            return_value={"logging": {"level": "INFO", "file": ""}},
        )
        self.config_mock = patcher.start()
        self.addCleanup(patcher.stop)

        self._prior_handlers = logging.getLogger().handlers[:]
        self._prior_level = logging.getLogger().level
        # Run with a clean root: importing app.api (other test modules do)
        # installs real handlers and a later force-reconfiguration closes
        # them — re-adding closed handlers would pollute every subsequent
        # configure() call with zombie FileHandlers.
        logging.getLogger().handlers[:] = []
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        root = logging.getLogger()
        for handler in list(_INSTALLED):
            if handler in root.handlers:
                root.removeHandler(handler)
            handler.close()
        _INSTALLED.clear()
        root.handlers[:] = self._prior_handlers
        root.setLevel(self._prior_level)

    def _configure_with_file(self, name: str = "wintermute.log") -> Path:
        """Configure logging with a file target inside self.tmp; return path."""
        log_path = self.tmp / name
        self.config_mock.return_value = {
            "logging": {
                "level": "INFO",
                "file": str(log_path),
                "max_bytes": 100000,
                "backup_count": 2,
            }
        }
        configure_logging(force=True)
        return log_path


class ConfigureLoggingTest(_LoggingTestCase):
    def test_creates_the_log_file_and_logs_into_it(self) -> None:
        log_path = self._configure_with_file()

        self.assertTrue(log_path.exists())
        logging.getLogger("test.probe").info("hello from the probe")

        content = log_path.read_text(encoding="utf-8")
        self.assertIn("hello from the probe", content)

    def test_active_log_file_reports_the_target(self) -> None:
        log_path = self._configure_with_file()
        # Read through the module: the global is reassigned on each call.
        self.assertEqual(logging_setup.active_log_file, log_path)

    def test_idempotent_call_does_not_stack_handlers(self) -> None:
        self._configure_with_file()
        configure_logging()  # no force: no-op

        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.FileHandler)
        ]
        self.assertEqual(len(file_handlers), 1)

    def test_force_reconfiguration_closes_the_old_file(self) -> None:
        first = self._configure_with_file("first.log")
        self.config_mock.return_value = {"logging": {"level": "INFO", "file": ""}}
        configure_logging(force=True)  # drop to file-less config

        self.assertFalse(any(
            isinstance(h, logging.FileHandler) for h in _INSTALLED
        ))
        self.assertTrue(first.exists())  # old file still on disk

    def test_file_logging_can_be_disabled(self) -> None:
        self.config_mock.return_value = {"logging": {"level": "INFO", "file": ""}}
        configure_logging(force=True)
        self.assertIsNone(logging_setup.active_log_file)

    def test_setup_yaml_failure_falls_back_to_defaults(self) -> None:
        self.config_mock.side_effect = RuntimeError("broken yaml")
        with self.assertLogs("src.logging_setup", level="WARNING"):
            configure_logging(force=True)
        # Defaults apply — and the default config *includes* the log file,
        # resolved against the project root.
        import src.tools.config_loader as config_loader
        self.assertEqual(
            logging_setup.active_log_file,
            config_loader.PROJECT_ROOT / "data" / "logs" / "wintermute.log",
        )

    def test_level_comes_from_setup_yaml(self) -> None:
        self.config_mock.return_value = {"logging": {"level": "DEBUG", "file": ""}}
        configure_logging(force=True)
        self.assertEqual(logging.getLogger().level, logging.DEBUG)


class TraceMirroringTest(_LoggingTestCase):
    """assertLogs() sets propagate=False during capture, which would block
    the root file handler — so these tests attach their own capture handler
    and keep the root wiring intact.
    """

    class _Capture(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    def test_emit_is_mirrored_into_the_traces_logger(self) -> None:
        log_path = self._configure_with_file()
        context = RoutingContext()
        capture = self._Capture()

        traces = logging.getLogger("wintermute.traces")
        old_level = traces.level
        traces.addHandler(capture)
        traces.setLevel(logging.DEBUG)
        try:
            context.emit(
                "task", "general_start", "handling request",
                request_kind="general",
            )
        finally:
            traces.removeHandler(capture)
            traces.setLevel(old_level)

        messages = [r.getMessage() for r in capture.records]
        self.assertTrue(
            any("[task] general_start" in m and "handling request" in m
                for m in messages),
            messages,
        )

        # And the mirrored record survives in the log file (root handler
        # receives it by propagation):
        content = log_path.read_text(encoding="utf-8")
        self.assertIn("[task] general_start", content)

    def test_event_data_goes_to_debug_level(self) -> None:
        context = RoutingContext()
        capture = self._Capture()

        traces = logging.getLogger("wintermute.traces")
        old_level = traces.level
        traces.addHandler(capture)
        traces.setLevel(logging.DEBUG)
        try:
            context.emit("analysis", "understood", data_key="value")
        finally:
            traces.removeHandler(capture)
            traces.setLevel(old_level)

        self.assertTrue(
            any(r.levelno == logging.DEBUG and "data_key" in r.getMessage()
                for r in capture.records),
            [r.getMessage() for r in capture.records],
        )


if __name__ == "__main__":
    unittest.main()
