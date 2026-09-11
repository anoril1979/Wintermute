"""Central logging configuration for Wintermute.

Every module logs through ``logging.getLogger(__name__)``; what was missing
was *where the records go*. Until now each entry point ran a bare
``logging.basicConfig(level=INFO)`` — console only, nothing durable. This
module installs, once per process, two handlers on the root logger:

* a console ``StreamHandler`` — same human-readable output as before;
* a rotating ``FileHandler`` — the durable log file
  (``data/logs/wintermute.log`` by default), so an ingestion can be
  investigated after the fact, outside the live "thinking" panel.

Settings come from ``config/setup.yaml`` (``logging:`` section). ``level``
and ``format`` were already declared there but never applied; ``file``
(optional — empty string disables the file handler), ``max_bytes`` and
``backup_count`` complete the section.

``configure_logging()`` is idempotent: the API, the CLI entry points and the
tests may all call it, only the first call installs handlers (unless
``force=True``).
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional, Union

from src.tools import config_loader

#: Default destination of the log file, relative to the project root.
DEFAULT_LOG_FILE = Path("data/logs/wintermute.log")

#: Defaults for rotation (only meaningful when a file handler is installed).
DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 3

_FORMAT_CONSOLE = "%(asctime)s %(levelname)-8s [%(correlation_id)s] %(name)s: %(message)s"

#: Log format used when setup.yaml does not declare one. The correlation
#: id groups every line of one chat request together in the durable log —
#: fix D's answer to interleaved multi-request sessions.
_DEFAULT_FORMAT = (
    "%(asctime)s - [%(correlation_id)s] - %(name)s - %(levelname)s - %(message)s"
)

#: Handlers installed by the last :func:`configure_logging` call — removed
#: (and closed) by a subsequent call, so reconfiguration never stacks.
_INSTALLED: List[logging.Handler] = []

#: Path of the log file installed by the last call (``None`` = file logging
#: disabled in setup.yaml). Useful for tests and for "where are the logs?"
#: answers.
active_log_file: Optional[Path] = None


_module_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Correlation ids — one id per incoming chat request, stamped on every log
# record that request produces, however deep in the pipeline the line comes
# from (middleware, analyzer, traces, agents). Python's logging has no
# built-in per-record context, so this is the standard two-part pattern:
#
#   * a ContextVar holding the current request's id — per *async task*, so
#     two concurrent requests never cross-stamp each other on the event
#     loop, and automatically copied into FastAPI's threadpool threads, so
#     sync endpoints inherit the middleware's id; a manually spawned thread
#     (the streaming worker) must be handed the id explicitly;
#   * a logging.Filter stamping ``record.correlation_id`` on every record
#     passing through our handlers — "-" when no id is bound (startup,
#     /health, background jobs), so the format string never raises.
# ---------------------------------------------------------------------------

_CID_VAR: ContextVar[Optional[str]] = ContextVar(
    "wintermute_correlation_id", default=None
)

#: Value stamped when no correlation id is bound to the current context.
NO_CORRELATION_ID = "-"


def current_correlation_id() -> Optional[str]:
    """The correlation id bound to the current context (``None`` if none)."""
    return _CID_VAR.get()


def bind_correlation_id(cid: Optional[str]) -> None:
    """Bind ``cid`` to the current context's log records (``None`` clears)."""
    _CID_VAR.set(cid)


def new_correlation_id() -> str:
    """A fresh short correlation id (8 hex chars — enough to eyeball groups)."""
    return uuid.uuid4().hex[:8]


class CorrelationIdFilter(logging.Filter):
    """Stamp ``record.correlation_id`` on every record passing through."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.correlation_id = current_correlation_id() or NO_CORRELATION_ID
        return True


def _logging_settings() -> dict:
    """Return the ``logging:`` section of setup.yaml ({} on any failure).

    Logging setup must never take the whole application down because
    setup.yaml is malformed: fall back to defaults and let the app run.
    """
    try:
        settings = config_loader.load_setup_config().get("logging")
    except Exception as exc:  # noqa: BLE001 — config problems must not kill logging
        # Use the module logger *before* configuration: with no handler
        # installed, the record goes to lastResort (stderr) — visible enough.
        _module_logger.warning(
            "setup.yaml logging section unreadable (%s); using defaults", exc
        )
        return {}
    return settings if isinstance(settings, dict) else {}


def _resolve_log_path(raw: Union[str, Path]) -> Path:
    """Resolve the configured log path against the project root."""
    path = Path(raw)
    if not path.is_absolute():
        path = config_loader.PROJECT_ROOT / path
    return path


def configure_logging(force: bool = False) -> Optional[Path]:
    """Install console + rotating-file handlers on the root logger.

    Idempotent: subsequent calls are no-ops unless ``force=True``, which
    removes the previously installed handlers first (used by tests).

    Returns:
        The resolved log file path, or ``None`` when file logging is
        disabled (``logging.file: ""`` in setup.yaml).
    """
    global active_log_file
    if _INSTALLED and not force:
        return active_log_file

    # Drop the handlers a previous call installed (never uvicorn's or the
    # tests' own handlers — only ours are tracked).
    for handler in _INSTALLED:
        root = logging.getLogger()
        if handler in root.handlers:
            root.removeHandler(handler)
        handler.close()
    _INSTALLED.clear()

    settings = _logging_settings()

    level_name = str(settings.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    format_string = str(settings.get("format") or _DEFAULT_FORMAT)
    formatter = logging.Formatter(format_string)
    # Stamp the per-request correlation id on every record crossing our
    # handlers; threads without a bound id (startup, background jobs) get
    # the "-" placeholder.
    cid_filter = CorrelationIdFilter()

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(cid_filter)
    root.addHandler(console)
    _INSTALLED.append(console)

    active_log_file = None
    raw_file = settings.get("file", str(DEFAULT_LOG_FILE))
    if raw_file:
        log_path = _resolve_log_path(raw_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=int(settings.get("max_bytes", DEFAULT_MAX_BYTES)),
            backupCount=int(settings.get("backup_count", DEFAULT_BACKUP_COUNT)),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(cid_filter)
        root.addHandler(file_handler)
        _INSTALLED.append(file_handler)
        active_log_file = log_path

    return active_log_file
