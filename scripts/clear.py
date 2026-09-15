"""clear.py — the one-shot corpus reset (CRITICAL operation).

Erases every GENERATED artifact of the corpus:

    data/vector/                     the vector store (file-level erase)
    data/extracted/                  canonical extracted JSONs
    data/extracted/mineru/           MinerU's working sandbox
    data/summarized/                 the LLM summaries
    data/cache/consolidation/        the consolidation cache
    data/cache/knowledge/            the knowledge cache JSONs
    data/cache/*_jobs.json           the checkpoint job files
    data/knowledge/                  the markdown knowledge base

What is NEVER touched: the code, the configuration, the source documents
(``documents_root`` — the user's scope) and the logs.

Because this is destructive and corpus-wide, the script REFUSES to run
without an explicit typed confirmation (``DELETE``); automation opts out
with ``--yes``. ``--dry-run`` prints the plan and deletes nothing.

Exit codes: 0 cleared · 1 partial failure (see data/logs/ingestion.log) ·
2 configuration error (the engine refuses to clear on a broken config) ·
64 usage error.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make `src` importable when the script is run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.clear_corpus import (  # noqa: E402
    STATUS_CLEARED,
    STATUS_PARTIAL,
    clear_corpus,
    resolve_clear_paths,
)
from scripts.ingest import (  # noqa: E402
    EXIT_CONFIG,
    EXIT_FAILURE,
    EXIT_OK,
    _configure_ingestion_logging,
)

logger = logging.getLogger("scripts.clear")

#: The confirmation word — typed, not a y/n: a reflex "yes" must not do.
CONFIRM_WORD = "DELETE"


def _print_plan(paths) -> None:
    """The critical-operation warning: exact targets, exact keeps."""
    print("=" * 72)
    print("  CORPUS CLEAR — CRITICAL OPERATION")
    print("=" * 72)
    print()
    print("  The following GENERATED data will be ERASED:")
    for label, path in (
        ("vector store", paths.vector_store),
        ("extracted JSONs", paths.extraction_store),
        ("MinerU sandbox", paths.mineru_store),
        ("summarized JSONs", paths.summarized_store),
        ("consolidation cache", paths.consolidation_cache),
        ("knowledge cache", paths.knowledge_cache),
        ("extraction job file", paths.extraction_job_file),
        ("summarization job file", paths.summarization_job_file),
        ("knowledge base (markdowns)", paths.knowledge_base),
    ):
        print(f"    - {path}")
    print()
    print("  KEPT: the source documents, the code, the configuration")
    print(f"    (documents root: {paths.documents_root})")
    print("  and the logs (data/logs).")
    print()


def _confirm() -> bool:
    """Typed confirmation; a bare 'yes' is deliberately NOT accepted."""
    try:
        answer = input(f"Type {CONFIRM_WORD} to erase the corpus, anything else to abort: ")
    except EOFError:
        return False
    return answer.strip() == CONFIRM_WORD


def main(argv: list | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="clear.py",
        description=(
            "Erase every generated artifact (vector store, extracted/"
            "summarized/consolidation/knowledge caches, job files, markdown "
            "knowledge base). Source documents and code are KEPT."
        ),
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", dest="yes",
        help="Skip the interactive confirmation (for automation)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Print the plan and exit without deleting anything",
    )
    args = parser.parse_args(argv)

    _configure_ingestion_logging(False)  # console INFO + durable file log
    logger.info("=== corpus clear requested (dry_run=%s, yes=%s)",
                args.dry_run, args.yes)

    # The plan must resolve BEFORE anything else: a broken configuration
    # refuses to clear (exit code 2) instead of wiping half-mapped targets.
    try:
        paths = resolve_clear_paths()
    except Exception as exc:  # noqa: BLE001 — any config breakage refuses
        logger.error("[CONFIG ERROR] cannot resolve the clear targets: %s", exc)
        return EXIT_CONFIG

    _print_plan(paths)

    if args.dry_run:
        print("Dry run: nothing was deleted.")
        return EXIT_OK

    if not args.yes:
        if not sys.stdin.isatty():
            # Refuse to clear from a pipe/CI without explicit --yes: an
            # accidental redirect must never wipe the corpus.
            logger.error(
                "Refusing to clear without a terminal confirmation; "
                "use --yes to authorize (automation)."
            )
            return EXIT_FAILURE
        if not _confirm():
            print("Aborted — nothing was deleted.")
            logger.info("Corpus clear aborted by the user.")
            return EXIT_OK

    report = clear_corpus(paths)

    status = report.get("status")
    if status == STATUS_CLEARED:
        print("Corpus cleared:")
        for label, step in report["steps"].items():
            removed = step.get("removed")
            print(f"    [x] {label}: {removed if removed else 'already empty'}")
        print(f"\nSource documents kept: {report['documents_root_kept']}")
        logger.info("Corpus cleared — every generated surface erased.")
        return EXIT_OK

    print("Corpus clear PARTIALLY failed:")
    for label in report.get("failed_steps", []):
        print(f"    [!] {label}: {report['steps'][label].get('reason', 'failed')}")
    print("Fix the issue and re-run: clearing is idempotent.")
    logger.error("Corpus clear partial: %s", report.get("reason"))
    return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
