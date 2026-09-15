"""String helpers — the ONE place for name normalization.

Name slugification lived in three near-identical copies across the
knowledge layer (entity ids, knowledge-base filenames, lookup matching),
all repeating the same accent-stripping dance. It now lives here once;
the call sites below are thin, intent-named wrappers:

    src/knowledge/models.py                 _slugify   (ids, ``_``)
    src/knowledge/character_markdown_store  slugify_filename (files, ``-``)
    src/knowledge/entity_lookup.py          fold_name  (match keys)
"""

from __future__ import annotations

import re
import unicodedata

#: Any run of characters that is not a letter/digit collapses to the
#: joiner (``_`` or ``-`` depending on the flavor).
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _ascii_fold(text: str) -> str:
    """Lowercase ASCII form of ``text``: NFKD-normalized, combining marks
    (accents, cedillas...) stripped — 'Édmond' -> 'edmond'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def slugify(text: str, joiner: str = "_") -> str:
    """Slug of ``name`` for ids and file names: accent-stripped, lowercase,
    every non-alphanumeric run collapsed to ``joiner``.

    'Édmond Dantès' -> 'edmond_dantes' (default) or 'edmond-dantes'
    (``joiner="-"``). Empty/degenerate input gives '' — callers decide
    their own fallback (the knowledge store uses 'unnamed').
    """
    if not isinstance(text, str):
        text = str(text)
    stripped = _ascii_fold(text)
    return re.sub(_NON_ALNUM, joiner, stripped).strip(joiner)


def fold(text: str) -> str:
    """Match key of ``text``: accent-stripped lowercase, non-alphanumeric
    runs collapsed to single spaces, trimmed.

    'Épée de Vif-Argent' and 'épée de vif argent' fold to the same key —
    comparing user input against corpus names that are rarely typed
    identically must not depend on case, accents or punctuation.
    """
    if not isinstance(text, str):
        text = str(text)
    return re.sub(_NON_ALNUM, " ", _ascii_fold(text)).strip()


__all__ = ["fold", "slugify"]
