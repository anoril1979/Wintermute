"""Tests for the central logging setup and trace-event mirroring.

Covers:

* ``configure_logging`` — file creation, level/format application,
  idempotence, ``force=True`` reconfiguration, file logging disabled.
* Trace mirroring — every ``emit()`` on a graph context also reaches the
  ``wintermute.traces`` logger, so the "thinking" flow lands in the log
  file.
* Correlation ids — one id per chat request stamped on every log line it
  produces (endpoint propagation, streaming worker, unbound placeholder),
  so multi-request sessions read as grouped blocks in the log file.
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


class CorrelationIdTest(_LoggingTestCase):
    """One id per chat request, on every line that request produces.

    The end-to-end test drives the real endpoint through TestClient with
    routing stubbed (hermetic) and reads the log FILE — the property the
    feature exists for: a multi-request session must read as grouped
    blocks in data/logs/wintermute.log.
    """

    def test_new_correlation_id_shape(self) -> None:
        first = logging_setup.new_correlation_id()
        second = logging_setup.new_correlation_id()
        self.assertRegex(first, r"^[0-9a-f]{8}$")
        self.assertNotEqual(first, second)

    def test_bound_id_is_stamped_on_file_lines(self) -> None:
        log_path = self._configure_with_file()
        cid = logging_setup.new_correlation_id()
        logging_setup.bind_correlation_id(cid)
        try:
            logging.getLogger("test.probe").info("grouped line")
        finally:
            logging_setup.bind_correlation_id(None)
        self.assertIn(f"[{cid}]", log_path.read_text(encoding="utf-8"))
        self.assertIn("grouped line", log_path.read_text(encoding="utf-8"))

    def test_unbound_records_carry_the_placeholder(self) -> None:
        log_path = self._configure_with_file()
        logging.getLogger("test.probe").info("background line")
        content = log_path.read_text(encoding="utf-8")
        self.assertIn("[-]", content)  # placeholder, not a missing attribute
        self.assertIn("background line", content)

    def test_custom_format_without_placeholder_still_works(self) -> None:
        # A setup.yaml format without %(correlation_id)s must not crash:
        # the filter only stamps the attribute; the format decides.
        self.config_mock.return_value = {
            "logging": {
                "level": "INFO",
                "file": str(self.tmp / "custom.log"),
                "format": "%(name)s: %(message)s",
            }
        }
        configure_logging(force=True)
        logging.getLogger("test.probe").info("old style")
        content = (self.tmp / "custom.log").read_text(encoding="utf-8")
        self.assertIn("test.probe: old style", content)

    def test_streaming_worker_carries_the_cid(self) -> None:
        # The stream worker is a manually spawned thread: threadpool
        # inheritance does not apply, the id must be handed over.
        import unittest.mock

        import app.api as api
        from src.logging_setup import current_correlation_id

        seen = {}

        def fake_routing(question, on_event=None):
            seen["cid"] = current_correlation_id()
            return {"status": "handled", "results": [], "traces": []}

        with unittest.mock.patch(
            "src.routing.routing_orchestrator.run_routing", fake_routing
        ):
            items = list(api._routing_stream("hello", cid="feedface"))

        self.assertEqual(seen["cid"], "feedface")
        self.assertEqual(items[-1][0], "final")

    def test_endpoint_lines_share_one_correlation_id(self) -> None:
        # End to end: middleware mints the id, the sync endpoint's
        # threadpool thread inherits it, every wintermute line of the
        # request carries the SAME id in the log file.
        import unittest.mock

        from fastapi.testclient import TestClient

        import app.api as api
        from src.logging_setup import current_correlation_id

        log_path = self._configure_with_file()

        def fake_routing(question, on_event=None):
            return {"status": "handled", "results": [], "traces": []}

        with unittest.mock.patch(
            "src.routing.routing_orchestrator.run_routing", fake_routing
        ):
            client = TestClient(api.app)
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello there"}]},
            )
        self.assertEqual(response.status_code, 200)
        # No leak into the caller's context after the request.
        self.assertIsNone(current_correlation_id())

        lines = log_path.read_text(encoding="utf-8").splitlines()
        request_lines = [
            line for line in lines
            if ">>> POST /v1/chat/completions" in line
            or "Incoming chat:" in line
            or "<<< 200 /v1/chat/completions" in line
        ]
        self.assertEqual(len(request_lines), 3, request_lines)
        ids = {line.split("[")[1].split("]")[0] for line in request_lines}
        self.assertEqual(len(ids), 1, request_lines)  # one group, one id
        self.assertNotIn("-", ids)  # a real id, not the placeholder


if __name__ == "__main__":
    unittest.main()
