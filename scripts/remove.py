"""remove.py — user-facing corpus-removal CLI.

User-side semantic entry point: "ingest.py -i meow.pdf -d" reads wrong —
ingesting is the opposite of removing. This script performs the same
deletion, spelled the way a human thinks it:

    python scripts/remove.py -i "meow.pdf"

What is removed (everything the ingestion projected):

* the vector chunks of the document (filtered by its unified ``doc_id``);
* the checkpoint entries (extraction + summarization job files);
* ``data/extracted/<stem>.json`` and ``data/summarized/<stem>.json``;
* MinerU's working folder for the document, when it exists.

What is NOT removed: the source file itself — ``data/sources`` is the
user's scope; Wintermute indexes it, it does not manage it.

Exit codes: 0 removed · 1 failure/partial (see data/logs/ingestion.log) ·
2 config error · 64 usage error.
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

from src.ingestion.remove_from_corpus import (  # noqa: E402
    STATUS_PARTIAL,
    STATUS_REMOVED,
    STATUS_REJECTED,
    remove_document,
)
from scripts.ingest import (  # noqa: E402
    EXIT_CONFIG,
    EXIT_FAILURE,
    EXIT_OK,
    EXIT_USAGE,
    _configure_ingestion_logging,
)

logger = logging.getLogger("scripts.remove")


def main(argv: list | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="remove.py",
        description=(
            "Remove a document from the Wintermute corpus (vector chunks, "
            "checkpoints, extracted and summarized JSONs, MinerU folder). "
            "The source file in data/sources is KEPT."
        ),
    )
    parser.add_argument(
        "-i", "--input", dest="input", required=True,
        help="The document to remove (file name in the documents tree)",
        metavar="FILE",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Console prints errors only; details go to data/logs/ingestion.log",
    )
    args = parser.parse_args(argv)

    _configure_ingestion_logging(args.quiet)
    logger.info("=== corpus removal requested: '%s'", args.input)

    try:
        report = remove_document(args.input)
    except Exception as exc:  # noqa: BLE001 — a crash is still a failure
        logger.exception("Corpus removal of '%s' crashed: %s", args.input, exc)
        return EXIT_FAILURE

    status = report.get("status")
    if status == STATUS_REMOVED:
        deleted = report["steps"]["vector"].get("deleted", 0)
        logger.info(
            "'%s' (%s) removed from the corpus — %d vector chunk(s) deleted, "
            "0 remaining in both collections, checkpoints and stores cleaned. "
            "The source file is kept.",
            report.get("document"), report.get("doc_id"), deleted,
        )
        return EXIT_OK
    if status == STATUS_PARTIAL:
        logger.error(
            "Removal of '%s' partially failed: %s",
            report.get("document"), report.get("reason"),
        )
        logger.error("Re-run the same command once the issue is fixed: "
                     "removal is idempotent.")
        return EXIT_FAILURE

    logger.error("Cannot remove '%s': %s", args.input, report.get("reason"))
    return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
