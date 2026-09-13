"""ingest.py — the user-facing ingestion CLI.

The paradigm-change entry point: document ingestion is a command-line
operation, not a chat request. Wintermute the assistant never ingests;
this script does — deterministically, with a durable log and explicit
exit codes.

Usage::

    python scripts/ingest.py -i "meow.pdf" -o canon
    python scripts/ingest.py -i "meow.pdf" -o community -f -s
    python scripts/ingest.py -i "meow.pdf" -d

    (``-o`` accepts any origin declared in setup.yaml
    ``documents.origins`` — the shipped default defines canon, community
    and rpg; the vocabulary is yours to adapt.)

Arguments:

    -i / --input       FILE   the document to ingest or delete (mandatory)
    -o / --origin      KIND   document origin: canon | community | rpg
                              (mandatory unless --delete)
    -f / --force              force complete re-ingestion (re-extraction;
                              implies re-summarization)
    -s / --summarize          force re-summarization
    -d / --delete              remove the document from the corpus (the
                              source file in data/sources is kept)
    -q / --quiet               print only errors (results go to the log)

Every issue and ingestion error is reported to ``data/logs/ingestion.log``
(in addition to the console), and the script returns a **non-zero exit
code on any error** — usable in batch files and automation:

    0  success (ingested, or removed)
    1  ingestion/removal failure (rejected, partial, failed step)
    2  configuration error (fix config/ingestion.yaml or llm.yaml)
    64 usage error (bad arguments)

The ``-d`` mode delegates to scripts/remove.py's ``main`` — the same
engine, kept user-facing under the verb it reads as.
"""

from __future__ import annotations

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Make `src` importable when the script is run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import ingestion_orchestrator  # noqa: E402
from src.logging_setup import configure_logging  # noqa: E402
from src.tools.config_loader import get_valid_origins  # noqa: E402
from src.tools.ingest_tool import resolve_document  # noqa: E402

logger = logging.getLogger("scripts.ingest")

#: Exit codes (stable contract for automation).
EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG = 2
EXIT_USAGE = 64

#: Default durable log, used only when ingestion.yaml cannot provide one
#: (missing key, or the yaml itself is broken — the CLI must stay able to
#: report THAT problem into a file before exiting with the config code).
DEFAULT_INGESTION_LOG = PROJECT_ROOT / "data" / "logs" / "ingestion.log"

# User-defined origin vocabulary (setup.yaml ``documents.origins``), read
# at CLI start so --help and argparse's choices always match the CURRENT
# configuration. Fails open to the loader's documented default trio when
# the yaml is broken (the orchestrator's config gate then reports it).
ORIGINS = get_valid_origins()


def ingestion_log_path() -> Path:
    """Resolve the CLI's durable log path from config/ingestion.yaml.

    Fails open to the default when the config is missing, malformed or
    lacks the key: the log must stay available to report the very error
    that broke the config.
    """
    try:
        from src.tools.config_loader import load_ingestion_config

        value = load_ingestion_config().get("ingestion_log")
        if isinstance(value, str) and value.strip() and ".." not in Path(value).parts:
            path = Path(value)
            return path if path.is_absolute() else PROJECT_ROOT / path
    except Exception:  # noqa: BLE001 — the CLI must keep its log reachable
        pass
    return DEFAULT_INGESTION_LOG


def _configure_ingestion_logging(quiet: bool, log_path: Path | None = None) -> None:
    """Install console + file handlers for the CLI run.

    The file handler writes to the configured ``ingestion_log`` (default:
    ``data/logs/ingestion.log``) so every issue, warning and error survives
    the console session. With ``--quiet``, the console prints WARNING and
    above only.
    """
    log_file = log_path if log_path is not None else ingestion_log_path()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    level = logging.WARNING if quiet else logging.INFO
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def _resolve_or_report(document: str, delete: bool) -> int:
    """Sandboxed resolution of the -i reference (bare file name)."""
    resolution = resolve_document(document)
    status = resolution.get("status")
    if status == "found":
        return EXIT_OK  # path used by the caller
    if status == "unsupported_extension":
        logger.error(
            "Cannot handle '%s': %s", document, resolution.get("message")
        )
        return EXIT_FAILURE
    if status == "not_found":
        message = resolution.get("message", "no such file")
        logger.error("Cannot handle '%s': %s", document, message)
        candidates = resolution.get("candidates") or []
        if candidates:
            logger.error("Did you mean one of these?")
            for name in candidates:
                logger.error("  + %s", name)
        return EXIT_FAILURE
    logger.error(
        "Cannot handle '%s': %s", document, resolution.get("message", status)
    )
    return EXIT_FAILURE


def main(argv: list | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="ingest.py",
        description=(
            "Ingest a document into the Wintermute corpus (or remove it "
            "with --delete). Deterministic; the chat assistant never "
            "ingests."
        ),
    )
    parser.add_argument(
        "-i", "--input", dest="input", required=True,
        help="The document to ingest or delete (file name in the documents tree)",
        metavar="FILE",
    )
    parser.add_argument(
        "-o", "--origin", dest="origin", choices=ORIGINS, default=None,
        help="Document origin — one of the configured vocabulary "
             "(setup.yaml documents.origins: %s). Required for ingestion."
             % ", ".join(ORIGINS),
    )
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="Force complete re-ingestion (bypass the extraction checkpoint)",
    )
    parser.add_argument(
        "-s", "--summarize", action="store_true", dest="summarize",
        help="Force re-summarization (bypass the summarization checkpoints)",
    )
    parser.add_argument(
        "-d", "--delete", action="store_true",
        help="Remove the document from the corpus (the source file is kept)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Console prints errors only; details go to data/logs/ingestion.log",
    )
    args = parser.parse_args(argv)

    if args.delete:
        if args.origin or args.force or args.summarize:
            parser.error("--delete cannot be combined with -o/-f/-s")
        # Delegate to the dedicated removal CLI (user-facing semantics).
        from scripts.remove import main as remove_main

        return remove_main(["-i", args.input] + (["-q"] if args.quiet else []))

    if not args.origin:
        parser.error(
            "-o/--origin is required for ingestion "
            f"({' | '.join(ORIGINS)}); use --delete to remove a document"
        )

    _configure_ingestion_logging(args.quiet)
    logger.info("=== ingestion requested: '%s' (origin=%s, force=%s, summarize=%s)",
                args.input, args.origin, args.force, args.summarize)

    exit_code = _resolve_or_report(args.input, args.delete)
    if exit_code != EXIT_OK:
        return exit_code

    resolution = resolve_document(args.input)
    result = ingestion_orchestrator.run_ingestion_file(
        Path(str(resolution["path"])),
        force=args.force,
        force_summarization=args.summarize,
        origin=args.origin,
    )

    status = result.get("status")
    if status == ingestion_orchestrator.STATUS_CONFIG_ERROR:
        logger.error("[CONFIG ERROR] %s", result.get("message"))
        logger.error("Hint: %s", result.get("fix_hint"))
        return EXIT_CONFIG
    if status == ingestion_orchestrator.STATUS_ACCEPTED:
        completed = ", ".join(result.get("completed_steps", [])) or "no step"
        skipped = result.get("skipped_steps") or []
        logger.info("Ingestion of '%s' completed (steps: %s).",
                    result.get("document"), completed)
        if skipped:
            logger.info("Skipped steps (checkpoints/resume): %s",
                        ", ".join(skipped))
        return EXIT_OK
    if status == ingestion_orchestrator.STATUS_NOT_IMPLEMENTED:
        logger.warning(
            "Ingestion of '%s' accepted but the pipeline is not fully "
            "wired yet (failed step: %s — %s)",
            result.get("document"), result.get("failed_step"),
            result.get("failure_detail"),
        )
        return EXIT_FAILURE
    logger.error(
        "Ingestion of '%s' failed: %s",
        result.get("document"),
        result.get("failure_detail") or result.get("reason") or status,
    )
    return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
