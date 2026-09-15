"""CLI progress display — the pretty face of the deterministic pipelines.

The ingestion graph already narrates everything it does through the
context's trace events (``context.emit``); ``run_ingestion_file`` accepts
an ``on_event`` observer that receives each event as it happens. This
module is the console consumer of that stream: instead of raw logging
lines, the CLI shows a live checklist — one line per graph step, ticked,
timed and annotated with the step's own data — plus running warning/error
counters and a final summary block. The durable ``ingestion.log`` stays
the complete source of truth; only the console rendering changes.

Design rules:

* **Observer, never actor.** The display only reads events and prints;
  ``__call__`` swallows every internal error so a rendering bug can
  never fail a pipeline.
* **Step-agnostic.** The checklist grows from the ``step_started``
  events as they arrive (the graph defines order and names); nothing
  here hardcodes the step list, so graph evolutions show up for free.
* **Alive between events.** A daemon ticker repaints the running row a
  few times per second — the fan spins and the elapsed timer keeps
  increasing even when no event arrives (long MinerU or LLM calls), so
  a silent pipeline never looks stuck.
* **Graceful fallback.** Without ANSI (piped output, ``TERM=dumb``,
  ``WINTERMUTE_PLAIN=1``, old console), a plain mode prints one
  sequential line per terminal state instead of rewriting in place.

Terminal model (ANSI mode)::

    Ingesting 'Gazette.pdf' (origin=community, force=False)

      [✓] extraction               8.2s    17 page(s), 5 chapter(s)
      [✓] consolidation            0.0s    214 -> 96 block(s) (118 merged)
      [\] summarization           12s     unit 37/96
      [ ] indexing

      warnings: 1 · errors: 0  (full details: data/logs/ingestion.log)
    ✔ Ingestion completed (21.4s)

The removal display is simpler: ``remove_document`` is fast and returns
a complete report, so its checklist is rendered in one pass at the end.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

# -- label / duration helpers -------------------------------------------------

#: Full step name -> short label (display width is scarce; the mapping is
#: purely cosmetic — unknown steps fall back to their full name).
_STEP_ABBREV = {
    "content_extraction": "extraction",
    "extraction_validation": "extraction_valid",
    "hierarchical_summarization": "summarization",
    "knowledge_extraction": "knowledge_extract",
    "knowledge_validation": "knowledge_valid",
    "source_registration": "registration",
    "source_indexing": "indexing",
    "check_and_merge": "check_n_merge",
}

#: Width of the step-label column (shortened names must fit).
_LABEL_WIDTH = 22


def _shorten(step: str) -> str:
    return _STEP_ABBREV.get(step, step)[:_LABEL_WIDTH]


def _fmt_duration(seconds: float) -> str:
    """Compact human duration (``8.2s``, ``1m04s``, ``12.4m``)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _fmt_running(seconds: float) -> str:
    """Whole-second duration for the RUNNING row.

    Flooring to seconds makes the text change exactly once per second —
    the calm "timer is counting" feedback — while the fan spins faster
    beside it.
    """
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    minutes, sec = divmod(s, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


# -- ANSI capability ----------------------------------------------------------

# Escape sequences (kept minimal on purpose).
_HIDE_CURSOR = "\x1b[?25l"
_SHOW_CURSOR = "\x1b[?25h"
_CLEAR_LINE = "\x1b[2K"
_MOVE_UP = "\x1b[{n}A"
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_YELLOW = "\x1b[33m"
_RED = "\x1b[31m"
_GREEN = "\x1b[32m"
_CYAN = "\x1b[36m"
_RESET = "\x1b[0m"

# The "working fan" the user asked for: spins while a step runs so a
# silent pipeline never looks stuck. The em-dash frame degrades safely
# through the UnicodeEncodeError guard on legacy codepages.
_SPIN_FRAMES_ANSI = ["\\", "|", "/", "—"]
_SPIN_FRAMES_PLAIN = ["|", "/", "-", "\\"]

#: Delay between two animation frames of the daemon ticker (seconds).
_TICK_INTERVAL = 0.25


def _ansi_capable(stream: Any) -> bool:
    """True when the stream can take ANSI control sequences.

    Heuristic, fail-plain: a Windows 10+ console answers ``os.system("")``
    by enabling VT processing; a pipe or ``TERM=dumb`` does not.
    """
    if os.environ.get("WINTERMUTE_PLAIN"):
        return False
    if os.environ.get("TERM") == "dumb" or not stream.isatty():
        return False
    if os.name == "nt":
        os.system("")  # enable VT processing on Windows 10+ terminals
    return True


# -- checklist rows ------------------------------------------------------------


class _StepRow:
    """Mutable rendering state of one graph step."""

    __slots__ = ("name", "state", "started", "duration", "detail", "spin")

    def __init__(self, name: str) -> None:
        self.name = name
        self.state = "pending"  # pending|running|done|skipped|failed
        self.started: Optional[float] = None
        self.duration: Optional[float] = None
        self.detail: Optional[str] = None
        self.spin = -1  # first bump lands on frame 0

    def _glyph(self, ansi: bool) -> str:
        if self.state == "done":
            return "[✓]" if ansi else "[x]"
        if self.state == "running":
            frames = _SPIN_FRAMES_ANSI if ansi else _SPIN_FRAMES_PLAIN
            return f"[{frames[self.spin % len(frames)]}]"
        if self.state == "skipped":
            return "[-]"
        if self.state == "failed":
            return "[✗]" if ansi else "[!]"
        return "[ ]"

    def _color(self, text: str) -> str:
        if self.state == "failed":
            return f"{_RED}{text}{_RESET}"
        if self.state == "skipped":
            return f"{_DIM}{text}{_RESET}"
        if self.state == "running":
            return f"{_CYAN}{text}{_RESET}"
        return text

    def render(self, ansi: bool) -> str:
        label = _shorten(self.name)
        if self.state == "running" and self.started is not None:
            dur = _fmt_running(time.monotonic() - self.started)
        elif self.duration is not None:
            dur = _fmt_duration(self.duration)
        else:
            dur = ""
        prefix = self._glyph(ansi)
        if ansi:
            prefix = self._color(prefix)
        line = f"  {prefix} {label:<{_LABEL_WIDTH}} {dur:>7}"
        if self.detail:
            line += f"  {_DIM}{self.detail}{_RESET}" if ansi else f"  {self.detail}"
        return line


# -- ingestion display ----------------------------------------------------------


class IngestionDisplay:
    """Live checklist renderer fed by the graph's trace events.

    Installed as the ``on_event`` observer of ``run_ingestion_file``.
    Events arrive on the caller's thread (the orchestrator runs
    synchronously); a daemon ticker repaints the running row every
    ``_TICK_INTERVAL`` so the fan spins and the elapsed timer grows even
    between events. All rendering — events and ticks — is serialized by
    one lock so concurrent writes can never interleave.
    """

    def __init__(self, stream: Any, *, title: str, ansi: bool) -> None:
        self._stream = stream
        self._ansi = ansi
        self._title = title
        self._rows: List[_StepRow] = []
        self._by_name: Dict[str, _StepRow] = {}
        self._warnings = 0
        self._errors = 0
        # Cursor accounting: how many checklist lines and footer lines are
        # currently on screen below the title (ANSI mode only).
        self._rendered_rows = 0
        self._footer_lines = 0
        # Sub-progress counters, keyed by the step that owns them.
        self._sum_total: Optional[int] = None
        self._sum_done = 0
        self._know_total: Optional[int] = None
        self._know_done = 0
        # Animation plumbing: one lock serializes event redraws (caller
        # thread) and ticker redraws (daemon thread); the stop event ends
        # the ticker cleanly at finish().
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ticker: Optional[threading.Thread] = None

    # -- construction / output ------------------------------------------------

    @classmethod
    def create(cls, *, title: str, plain: bool = False) -> Optional["IngestionDisplay"]:
        """Build a display on stdout, or ``None`` when rendering is off.

        ``None`` means "keep the historical raw-log console": piped
        output and quiet mode simply do not install an observer.
        """
        stream = sys.stdout
        try:
            if not stream.isatty():
                return None
        except (AttributeError, ValueError, OSError):
            return None
        return cls(stream, title=title, ansi=not plain and _ansi_capable(stream))

    def _write(self, text: str) -> None:
        try:
            self._stream.write(text)
        except UnicodeEncodeError:
            # A redirected stream on a legacy codepage: degrade instead of
            # crashing the pipeline over one glyph.
            enc = getattr(self._stream, "encoding", None) or "ascii"
            self._stream.write(text.encode(enc, "replace").decode(enc, "replace"))
        self._stream.flush()

    def start(self) -> None:
        """Print the title bar (call once, before the run)."""
        if self._ansi:
            self._write(f"\n{_BOLD}{self._title}{_RESET}\n\n")
            self._write(_HIDE_CURSOR)
            self._start_ticker()
        else:
            self._write(f"\n{self._title}\n")

    # -- animation ticker -------------------------------------------------------

    def _start_ticker(self) -> None:
        """Launch the daemon animation thread (ANSI mode only)."""
        if not self._ansi or self._ticker is not None:
            return
        self._stop_event.clear()
        self._ticker = threading.Thread(
            target=self._tick_loop, name="cli-progress-ticker", daemon=True,
        )
        self._ticker.start()

    def _tick_loop(self) -> None:
        """Repaint the running row every ``_TICK_INTERVAL`` until stopped.

        The fan advances and the elapsed timer grows even when no event
        arrives — the "nothing is stuck" feedback. Every iteration is
        exception-guarded: the display must never break a pipeline.
        """
        while not self._stop_event.wait(_TICK_INTERVAL):
            try:
                with self._lock:
                    row = self._running_row()
                    if row is None:
                        continue  # nothing alive to animate yet
                    row.spin += 1
                    self._redraw()
            except Exception:  # noqa: BLE001 — animation must never crash
                pass

    def _stop_ticker(self) -> None:
        """Stop the animation thread; idempotent, safe before finish()."""
        self._stop_event.set()
        ticker = self._ticker
        self._ticker = None
        if ticker is not None and ticker.is_alive():
            ticker.join(timeout=2)

    def close(self) -> None:
        """Release the animation thread (idempotent; tests and teardown)."""
        self._stop_ticker()

    # -- redraw (ANSI mode) -----------------------------------------------------

    def _redraw(self) -> None:
        if not self._ansi:
            return
        # Move the cursor back above the currently rendered block, then
        # rewrite it: rows first, then the counters footer.
        up = self._rendered_rows + self._footer_lines
        if up:
            self._write(_MOVE_UP.format(n=up))
        for row in self._rows:
            self._write(_CLEAR_LINE + row.render(ansi=True) + "\n")
        self._rendered_rows = len(self._rows)
        self._write(_CLEAR_LINE + "\n")
        self._write(_CLEAR_LINE + self._footer() + "\n")
        self._footer_lines = 2

    def _footer(self) -> str:
        counters = f"warnings: {self._warnings} · errors: {self._errors}"
        return f"  {counters}  {_DIM}(full details: data/logs/ingestion.log){_RESET}"

    # -- event entry point --------------------------------------------------------

    def __call__(self, event: Dict[str, Any]) -> None:
        """``on_event`` observer: must never raise, must never block."""
        try:
            # The lock serializes with the daemon ticker's redraws: two
            # writers on the same terminal region would interleave.
            with self._lock:
                self._dispatch(event)
        except Exception:  # noqa: BLE001 — display must never break a pipeline
            pass

    def _dispatch(self, event: Dict[str, Any]) -> None:
        phase = event.get("phase")
        kind = event.get("kind", "")
        data = event.get("data") or {}
        if phase == "pipeline":
            self._on_pipeline(kind, data, event.get("message", ""))
        elif phase == "task":
            self._on_task(kind, data)

    # -- pipeline (graph step lifecycle) -----------------------------------------

    def _on_pipeline(self, kind: str, data: Dict[str, Any], message: str) -> None:
        step = data.get("step")
        if not isinstance(step, str):
            return

        if kind == "step_started":
            row = self._by_name.get(step)
            if row is None:
                row = _StepRow(step)
                self._by_name[step] = row
                self._rows.append(row)
            row.state = "running"
            row.started = time.monotonic()
            row.spin += 1
            if self._ansi:
                self._redraw()
            return

        row = self._by_name.get(step)
        if row is None:
            return
        terminal: Optional[_StepRow] = None
        if kind == "step_done":
            row.state = "done"
            if row.started is not None:
                row.duration = time.monotonic() - row.started
            terminal = row
        elif kind == "step_skipped":
            row.state = "skipped"
            if row.started is not None:
                row.duration = time.monotonic() - row.started
            terminal = row
        elif kind == "step_retry":
            self._errors += 1
            row.detail = (f"retry {data.get('attempt', '?')}"
                          f" ({data.get('domain', '')})")
        elif kind == "step_failed":
            row.state = "failed"
            self._errors += 1
            if row.started is not None:
                row.duration = time.monotonic() - row.started
            row.detail = message.split("failed: ", 1)[-1][:60]
            terminal = row
        else:
            return

        if self._ansi:
            self._redraw()
        elif terminal is not None:
            # Plain mode prints one sequential line per terminal state.
            self._write(terminal.render(ansi=False) + "\n")

    # -- task events (sub-progress + counters) --------------------------------------

    def _on_task(self, kind: str, data: Dict[str, Any]) -> None:
        detail: Optional[str] = None

        # -- content extraction -----------------------------------------------
        if kind == "extracting":
            detail = "MinerU working…"
        elif kind == "extracted":
            detail = (f"{data.get('pages', '?')} page(s), "
                      f"{data.get('chapters', '?')} chapter(s)")
        elif kind == "checkpoint_hit" or kind == "already_done":
            detail = "resume: extraction checkpoint reused"
        elif kind == "resumed":
            detail = "resumed from extracted JSON"

        # -- warnings / errors counters (enumerated degradations first) --------
        elif kind.endswith("_warning"):
            self._warnings += 1
        elif kind in ("summarization_skipped", "knowledge_unit_skipped",
                      "consolidation_cache_failed", "knowledge_save_failed",
                      "canonical_save_failed", "resume_failed"):
            self._warnings += 1
        elif kind == "structural_issue":
            self._errors += 1

        # -- consolidation ------------------------------------------------------
        elif kind == "consolidation_done":
            payload = data.get("data") or {}
            before, after = payload.get("blocks_before"), payload.get("blocks_after")
            if before is not None and after is not None:
                detail = (f"{before} -> {after} block(s) "
                          f"({payload.get('merged_away', 0)} merged)")

        # -- summarization -------------------------------------------------------
        elif kind == "summarization_start":
            total = data.get("units_total")
            self._sum_total = total if isinstance(total, int) else None
            self._sum_done = 0
            detail = "starting…"
        elif kind == "summarized":
            self._sum_done += 1
            detail = self._unit_progress(self._sum_done, self._sum_total,
                                         data.get("label"))

        # -- indexing --------------------------------------------------------------
        elif kind == "indexing_chunks_built":
            detail = f"{data.get('chunk_count', '?')} chunk(s) built"
        elif kind == "indexing_embedding":
            done, total = data.get("done"), data.get("total")
            if done is not None and total:
                detail = f"embedding {done}/{total}"

        # -- knowledge extraction ----------------------------------------------------
        elif kind == "knowledge_start":
            units = data.get("units")
            self._know_total = units if isinstance(units, int) else None
            self._know_done = 0
            detail = (f"{units} unit(s) · {data.get('granularity', '?')}"
                      if units is not None else None)
        elif kind == "knowledge_unit":
            self._know_done += 1
            detail = self._unit_progress(self._know_done, self._know_total,
                                         data.get("label"))
        elif kind == "knowledge_saved":
            detail = "knowledge cached"

        else:
            return

        row = self._running_row()
        if row is None:
            return
        if detail is not None:
            row.detail = detail
        self._redraw()

    @staticmethod
    def _unit_progress(done: int, total: Optional[int],
                       label: Any) -> str:
        if total:
            return f"unit {done}/{total}"
        if isinstance(label, str) and label:
            return f"unit: {label[:34]}"
        return "working…"

    def _running_row(self) -> Optional[_StepRow]:
        for row in self._rows:
            if row.state == "running":
                return row
        return None

    # -- final summary -----------------------------------------------------------

    def finish(self, ok: bool, message: str) -> None:
        """Print the closing block (call once, after the run returns)."""
        # Stop the animation before the final render: no competing writer,
        # no repaint over the summary.
        self._stop_ticker()
        durations = [r.duration for r in self._rows if r.duration is not None]
        total = sum(durations) if durations else 0.0
        with self._lock:
            if self._ansi:
                self._redraw()
                self._write("\n")
                glyph = f"{_GREEN}✔{_RESET}" if ok else f"{_RED}✘{_RESET}"
                self._write(f"{glyph} {message} ({_fmt_duration(total)})\n")
                self._write(_SHOW_CURSOR)
            else:
                status = "OK" if ok else "FAILED"
                self._write(f"\n{status}: {message} ({_fmt_duration(total)})\n")
                self._write(f"  warnings: {self._warnings} · errors: {self._errors}\n")
                self._write("  full details: data/logs/ingestion.log\n")


# -- removal display ------------------------------------------------------------


class RemovalDisplay:
    """Checklist renderer for the removal report (post-hoc, no live events).

    ``remove_document`` runs quickly and returns a complete report dict;
    the removal display renders that report in one pass at the end — no
    engine change, no event wiring.
    """

    #: report key -> display label (order = display order).
    _STEPS = [
        ("vector", "vector chunks"),
        ("job_files", "checkpoints"),
        ("json_files", "stored JSONs"),
        ("knowledge_base", "knowledge base"),
        ("source_registration", "source registration"),
        ("mineru", "MinerU folder"),
    ]

    def __init__(self, stream: Any, *, title: str, ansi: bool) -> None:
        self._stream = stream
        self._ansi = ansi
        self._title = title

    @classmethod
    def create(cls, *, title: str, plain: bool = False) -> Optional["RemovalDisplay"]:
        stream = sys.stdout
        try:
            if not stream.isatty():
                return None
        except (AttributeError, ValueError, OSError):
            return None
        return cls(stream, title=title, ansi=not plain and _ansi_capable(stream))

    def _write(self, text: str) -> None:
        try:
            self._stream.write(text)
        except UnicodeEncodeError:
            # A redirected stream on a legacy codepage: degrade instead of
            # crashing the pipeline over one glyph.
            enc = getattr(self._stream, "encoding", None) or "ascii"
            self._stream.write(text.encode(enc, "replace").decode(enc, "replace"))
        self._stream.flush()

    def render(self, report: Dict[str, Any]) -> None:
        bold = _BOLD if self._ansi else ""
        reset = _RESET if self._ansi else ""
        dim = _DIM if self._ansi else ""
        self._write(f"\n{bold}{self._title}{reset}\n")
        steps = report.get("steps") or {}
        for key, label in self._STEPS:
            entry = steps.get(key)
            if entry is None:
                continue
            ok = bool(entry.get("ok"))
            if self._ansi:
                glyph = f"{_GREEN}[✓]{_RESET}" if ok else f"{_YELLOW}[!]{_RESET}"
            else:
                glyph = "[x]" if ok else "[!]"
            detail = self._step_detail(key, entry)
            line = f"  {glyph} {label:<20}"
            if detail:
                line += f"  {dim}{detail}{reset}" if self._ansi else f"  {detail}"
            self._write(line + "\n")

        doc_id = report.get("doc_id")
        status = report.get("status")
        self._write("\n")
        if status == "removed":
            glyph = f"{_GREEN}✔{_RESET}" if self._ansi else "OK:"
            kept = " - the source file is kept." if self._ansi \
                else " - the source file is kept."
            self._write(f"{glyph} Removed from the corpus"
                        + (f" ({doc_id})" if doc_id else "")
                        + kept + "\n")
        elif status == "partial":
            glyph = f"{_RED}✘{_RESET}" if self._ansi else "PARTIAL:"
            self._write(f"{glyph} Removal partially failed: "
                        f"{report.get('reason', '?')}\n")
            self._write("  Re-run the same command once fixed: "
                        "removal is idempotent.\n")

    @staticmethod
    def _step_detail(key: str, entry: Dict[str, Any]) -> Optional[str]:
        if key == "vector":
            return f"{entry.get('deleted', 0)} chunk(s) deleted"
        if key == "job_files":
            return f"{len(entry.get('removed') or [])} checkpoint(s) removed"
        if key == "json_files":
            return f"{len(entry.get('removed') or [])} file(s) removed"
        if key == "knowledge_base":
            return (f"{entry.get('purged_files', 0)} file(s) updated, "
                    f"{entry.get('deleted_files', 0)} emptied")
        if key == "source_registration":
            return "removed" if entry.get("removed") else "not registered"
        if key == "mineru":
            return "folder removed" if entry.get("removed") else "no folder"
        return None
