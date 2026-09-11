"""Guard against other programs' traffic reaching Wintermute's LLM pipeline.

Fix C proved where the "phantom requests" came from: Open WebUI and
similar front-ends fire their own background tasks (title generation,
follow-up suggestions, topic tagging) at the same ``/v1/chat/completions``
endpoint the user's real messages use. Their prompts are recognizable:
they start with explicit task instructions ("Generate a title",
"Suggest 3-5 follow-up questions") and embed whole chat histories.

Routing such a prompt through the analyzer produced nonsense requests —
and each one triggered a full routing run (dispatches, possibly live
ingestions) whose answer was thrown away by the front-end.

The guard is a two-layer contract:

* :data:`META_SENTINEL` — a module-level string a front-end can send as
  the message content to probe the API without paying for a routing run.
* :func:`prompt_is_meta` / :func:`meta_answer` — the classifier and the
  honest, immediate answer for recognized meta prompts.
"""

from __future__ import annotations

from typing import Final

# Probing sentinel: a client may send this exact content to check the
# endpoint is alive without triggering analysis, routing or LLM calls.
META_SENTINEL: Final[str] = "__WINTERMUTE_PING__"

# Recognized prefixes of auxiliary prompts sent by chat front-ends
# (lowercased, matched against the whitespace-collapsed start of the
# prompt). Open WebUI's title generation, follow-up suggestions and
# topic tagging all begin with an explicit "### Task:" instruction;
# the generic imperative forms catch other clients' variants.
_META_PREFIXES: Final[tuple[str, ...]] = (
    "### task:",
    "generate a concise",
    "generate 1-3 broad tags",
    "suggest 3-5 relevant follow-up",
    "suggest follow-up",
    "write a short title",
)

# Bounded length of the prompt excerpt embedded in the log line.
_LOG_SNIPPET_LIMIT = 120


def _snippet(text: str) -> str:
    """One-line, bounded excerpt for the log line."""
    return " ".join(text.split())[:_LOG_SNIPPET_LIMIT]


def prompt_is_meta(prompt: str) -> bool:
    """True when ``prompt`` is a front-end auxiliary task, not a user turn.

    The sentinel matches exactly (a user would have to type it on
    purpose); auxiliary prompts are matched on their opening instruction,
    which every observed front-end task carries.
    """
    stripped = prompt.strip()
    if stripped == META_SENTINEL:
        return True
    collapsed = " ".join(stripped.split()).lower()
    return collapsed.startswith(_META_PREFIXES)


def meta_answer() -> str:
    """The fixed, immediate answer for a recognized meta prompt.

    Honest (it names what happened), zero-cost (no analyzer, no routing,
    no LLM), and worded so a front-end that *displays* the reply instead
    of discarding it still shows something coherent.
    """
    return (
        "OK. (This was a background request from your front-end — title, "
        "tags or follow-up suggestions — not a message for Wintermute; "
        "no routing was performed.)"
    )
