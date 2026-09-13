"""Wintermute CLI scripts.

The user-facing entry points of the corpus lifecycle (the paradigm
change): ingestion and removal are command-line operations, never chat
requests.

* ``ingest.py``  — ingest one document (-i FILE -o ORIGIN [-f] [-s]),
                   or remove it (-i FILE -d);
* ``remove.py``  — remove one document from the corpus (-i FILE), the
                   human-verb spelling of the same operation.

Both log to ``data/logs/ingestion.log`` and return non-zero exit codes on
failure.
"""
