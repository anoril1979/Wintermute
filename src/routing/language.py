"""Reply-language management — detect once, enforce at the reply.

The language problem: the user asks in French, the corpus is French, and
the reply agents (AnswerAgent, GeneralTaskAgent) answered in English —
each LLM free-floating on "answer in the user's language" while reading
mixed-language material. The fix is structural: the FIRST LLM of the flow
(the request analyzer) already reads the whole prompt, so it extracts the
language once (``language`` top-level field of the analysis) and that
value is carried — unchanged — to the last LLM, which receives it as an
explicit, authoritative prompt key ("Reply language"). The prompt-level
"answer in the user's language" guidance stays as a belt; the injected
key is the braces.

This module holds the two seams of that flow:

* :func:`normalize_language` — the analyzer's ``language`` string is
  LLM output, so it is normalized fail-open: a recognized value survives
  (canonical two-letter code), anything else falls back to the English
  default. Language detection must never be able to break routing: the
  worst case is an English reply, not a crashed pipeline.
* The localized deterministic fallbacks (:func:`nothing_found_reply`,
  :func:`dormant_corpus_reply`, :func:`unserved_kind_reply`) — the
  non-LLM user-facing texts (AnswerAgent's no-hits reply, the retrieval
  task agent's dormant/unserved statuses) must honor the same language
  the LLM replies are forced into.

The normalization table covers the common cases generously (native names,
endonyms, common LLM spellings — "French", "français", "fr", "fr-FR");
an unknown label is mapped to ``en`` with a logged warning. The analyzer
prompt asks for a two-letter code, which keeps this path boring.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

#: Default reply language when nothing (or nothing usable) was detected.
DEFAULT_LANGUAGE = "en"

#: The analyzer's value, normalized: canonical two-letter code -> itself.
#: Every alias (endonym, English name, regional code...) maps onto the
#: canonical code. Keep lowercase keys; lookup normalizes first.
_LANGUAGE_ALIASES: Dict[str, str] = {
    # canonical codes
    "en": "en", "fr": "fr", "de": "de", "es": "es", "it": "it",
    "pt": "pt", "nl": "nl", "ru": "ru", "ja": "ja", "zh": "zh",
    "ko": "ko", "ar": "ar", "pl": "pl", "sv": "sv", "no": "no",
    "da": "da", "fi": "fi", "cs": "cs", "el": "el", "tr": "tr",
    "hu": "hu", "ro": "ro", "he": "he", "uk": "uk",
    # English names
    "english": "en", "french": "fr", "german": "de", "spanish": "es",
    "italian": "it", "portuguese": "pt", "dutch": "nl", "russian": "ru",
    "japanese": "ja", "chinese": "zh", "korean": "ko", "arabic": "ar",
    "polish": "pl", "swedish": "sv", "norwegian": "no", "danish": "da",
    "finnish": "fi", "czech": "cs", "greek": "el", "turkish": "tr",
    "hungarian": "hu", "romanian": "ro", "hebrew": "he", "ukrainian": "uk",
    # endonyms the LLM may answer with
    "français": "fr", "francais": "fr", "anglais": "en", "allemand": "de",
    "espagnol": "es", "italien": "it", "portugais": "pt", "néerlandais": "nl",
    "neerlandais": "nl", "russe": "ru", "deutsch": "de", "english (us)": "en",
    # regional / BCP-47 shaped codes: normalize on the primary subtag
    "fr-fr": "fr", "fr-ca": "fr", "en-us": "en", "en-gb": "en",
    "de-de": "de", "es-es": "es", "pt-br": "pt", "pt-pt": "pt",
    "zh-cn": "zh", "zh-tw": "zh",
}


def normalize_language(value: Optional[str]) -> str:
    """Normalize the analyzer's ``language`` value, fail-open to ``en``.

    Accepted shapes: a canonical code (``fr``), a language name
    (``French``, ``français``), a regional code (``fr-FR`` — normalized on
    the primary subtag). Anything else (None, blank, unknown label, a
    whole sentence) yields the default, with a warning for non-blank
    unknown values so a drifting prompt stays visible in the logs.
    """
    if value is None:
        return DEFAULT_LANGUAGE
    text = str(value).strip().lower()
    if not text:
        return DEFAULT_LANGUAGE
    if text in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[text]
    # BCP-47-shaped ("fr_CA", "en-US-x-..."): try the primary subtag.
    primary = text.replace("_", "-").split("-", 1)[0]
    if primary in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[primary]
    logger.warning(
        "Unknown reply language %r from the analyzer; falling back to %r",
        value, DEFAULT_LANGUAGE,
    )
    return DEFAULT_LANGUAGE


# ---------------------------------------------------------------------------
# Localized deterministic fallbacks (the non-LLM user-facing replies)
# ---------------------------------------------------------------------------

# One template per language; missing languages fall back to the English
# template. Keys mirror the canonical codes of the alias table.
_NOTHING_FOUND: Dict[str, str] = {
    "en": (
        "The sources I hold say nothing about that. Ingest a document "
        "covering it, or ask me something else my memory contains."
    ),
    "fr": (
        "Les sources que je détiens ne disent rien à ce sujet. Ingérez un "
        "document qui en traite, ou interrogez ma mémoire autrement."
    ),
    "de": (
        "Die Quellen, die ich halte, sagen dazu nichts. Ingestieren Sie ein "
        "Dokument dazu, oder fragen Sie mein Gedächtnis nach etwas anderem."
    ),
    "es": (
        "Las fuentes que guardo no dicen nada sobre eso. Ingiera un "
        "documento que lo trate, o pregunte otra cosa a mi memoria."
    ),
    "it": (
        "Le fonti che custodisco non dicono nulla al riguardo. Ingerisci un "
        "documento che lo tratta, o chiedi altro alla mia memoria."
    ),
}

_DORMANT: Dict[str, str] = {
    "en": (
        "Wintermute's memory is dormant: nothing indexed yet. "
        "Ingest a document first."
    ),
    "fr": (
        "La mémoire de Wintermute est en sommeil : rien n'est indexé pour "
        "l'instant. Ingérez d'abord un document."
    ),
    "de": (
        "Wintermutes Gedächtnis ruht: noch nichts indexiert. Ingestieren "
        "Sie zuerst ein Dokument."
    ),
    "es": (
        "La memoria de Wintermute está inactiva: nada indexado todavía. "
        "Ingiera primero un documento."
    ),
    "it": (
        "La memoria di Wintermute è dormiente: nessun documento indicizzato. "
        "Ingerisci prima un documento."
    ),
}


def _template(table: Dict[str, str], language: Optional[str]) -> str:
    """Pick ``table[language]``, falling back to the English entry."""
    code = normalize_language(language)
    return table.get(code) or table[DEFAULT_LANGUAGE]


def nothing_found_reply(language: Optional[str]) -> str:
    """The deterministic 'no hit to ground an answer on' reply, localized."""
    return _template(_NOTHING_FOUND, language)


def dormant_corpus_reply(language: Optional[str]) -> str:
    """The deterministic 'no indexed document yet' reply, localized."""
    return _template(_DORMANT, language)


def unserved_kind_reply(kind: str, language: Optional[str]) -> str:
    """The deterministic 'lookup kind not served yet' reply, localized.

    ``kind`` is the retrieval lookup kind (``index``, ``relation``,
    ``summary``, ``listing``...) interpolated into the sentence.
    """
    code = normalize_language(language)
    templates = {
        "en": (
            "This kind of lookup ('{kind}') is not served yet — the storage "
            "layer behind it is still under construction."
        ),
        "fr": (
            "Ce type de recherche (« {kind} ») n'est pas encore servi — la "
            "couche de stockage qui le porte est encore en construction."
        ),
        "de": (
            "Diese Art der Suche ('{kind}') wird noch nicht angeboten — die "
            "dahinterliegende Speicherschicht ist im Aufbau."
        ),
        "es": (
            "Este tipo de búsqueda ('{kind}') aún no está disponible — la "
            "capa de almacenamiento que la soporta está en construcción."
        ),
        "it": (
            "Questo tipo di ricerca ('{kind}') non è ancora disponibile — "
            "lo strato di archiviazione che la supporta è in costruzione."
        ),
    }
    template = templates.get(code) or templates[DEFAULT_LANGUAGE]
    return template.format(kind=kind)
