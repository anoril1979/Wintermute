"""Tests for the CLI progress displays (src/tools/cli_progress.py).

Hermetic: a fake tty stream captures the ANSI renderings, a StringIO
drives the plain fallback; the ingestion display is fed hand-built
events shaped exactly like the graph's trace events. No real pipeline,
no real console.
"""

from __future__ import annotations

import io
import time
import unittest
from unittest import mock

from src.tools.cli_progress import (
    _SPIN_FRAMES_ANSI,
    IngestionDisplay,
    RemovalDisplay,
    _ansi_capable,
    _fmt_duration,
    _fmt_running,
    _shorten,
)


class FakeTty(io.StringIO):
    """A StringIO that claims to be a terminal."""

    def isatty(self) -> bool:
        return True


def _pipeline_event(kind: str, step: str, **data):
    return {"phase": "pipeline", "kind": kind, "message": f"step '{step}' {kind}",
            "data": {"step": step, **data}}


def _task_event(kind: str, **data):
    return {"phase": "task", "kind": kind, "message": kind, "data": data}


class HelpersTest(unittest.TestCase):
    def test_shorten_maps_known_steps(self):
        self.assertEqual(_shorten("hierarchical_summarization"), "summarization")
        self.assertEqual(_shorten("content_extraction"), "extraction")

    def test_shorten_passes_unknown_steps_through(self):
        self.assertEqual(_shorten("brand_new_step"), "brand_new_step")

    def test_duration_formatting(self):
        self.assertEqual(_fmt_duration(0.4), "0.4s")
        self.assertEqual(_fmt_duration(8.24), "8.2s")
        self.assertEqual(_fmt_duration(64), "1m04s")
        self.assertEqual(_fmt_duration(3724), "1h02m")

    def test_running_duration_is_whole_seconds(self):
        # The running row counts whole seconds: the text changes once per
        # second, exactly the "timer is counting" feedback.
        self.assertEqual(_fmt_running(0.9), "0s")
        self.assertEqual(_fmt_running(2.4), "2s")
        self.assertEqual(_fmt_running(64.7), "1m04s")
        self.assertEqual(_fmt_running(3724.2), "1h02m")


class AnsiCapabilityTest(unittest.TestCase):
    def test_pipe_is_not_ansi_capable(self):
        stream = io.StringIO()
        with mock.patch.dict("os.environ", {"WINTERMUTE_PLAIN": ""}, clear=False):
            self.assertFalse(_ansi_capable(stream))

    def test_env_var_forces_plain(self):
        stream = FakeTty()
        with mock.patch.dict("os.environ", {"WINTERMUTE_PLAIN": "1"}, clear=False):
            self.assertFalse(_ansi_capable(stream))


class IngestionDisplayAnsiTest(unittest.TestCase):
    """ANSI mode through a fake tty: the checklist is rewritten in place."""

    def setUp(self):
        self.stream = FakeTty()
        self.display = IngestionDisplay(self.stream, title="Ingesting 'x.pdf'",
                                        ansi=True)
        self.display.start()

    def _feed(self, *events):
        for event in events:
            self.display(event)

    def test_step_lifecycle_renders_running_then_done(self):
        self._feed(_pipeline_event("step_started", "consolidation"))
        out = self.stream.getvalue()
        # The fan glyph (any frame) with the elapsed whole-second timer.
        self.assertTrue(any(f"[{f}]" in out for f in _SPIN_FRAMES_ANSI), out)
        self.assertIn("consolidation", out)
        self.assertRegex(out, r"\d+s")

        self._feed(_pipeline_event("step_done", "consolidation"))
        self.assertIn("[✓]", self.stream.getvalue())

    def test_skipped_state(self):
        self._feed(_pipeline_event("step_started", "indexing"),
                   _pipeline_event("step_skipped", "indexing"))
        self.assertIn("[-]", self.stream.getvalue())

    def test_failed_state_carries_message_excerpt(self):
        event = _pipeline_event("step_started", "summarization")
        failure = {"phase": "pipeline", "kind": "step_failed",
                   "message": "step 'summarization' failed: LLM down",
                   "data": {"step": "summarization"}}
        self._feed(event, failure)
        out = self.stream.getvalue()
        self.assertIn("[✗]", out)
        self.assertIn("LLM down", out)

    def test_retry_counts_as_error_and_shows_detail(self):
        self._feed(
            _pipeline_event("step_started", "summarization"),
            _pipeline_event("step_retry", "summarization",
                            attempt=1, domain="llm_response"),
        )
        out = self.stream.getvalue()
        self.assertIn("retry 1 (llm_response)", out)
        self.assertIn("errors: 1", out)

    def test_task_details_land_on_running_row(self):
        self._feed(
            _pipeline_event("step_started", "content_extraction"),
            _task_event("extracted", pages=17, chapters=5),
        )
        self.assertIn("17 page(s), 5 chapter(s)", self.stream.getvalue())

    def test_extraction_checkpoint_resume_detail(self):
        self._feed(
            _pipeline_event("step_started", "content_extraction"),
            _task_event("checkpoint_hit"),
        )
        self.assertIn("checkpoint reused", self.stream.getvalue())

    def test_consolidation_counts(self):
        self._feed(
            _pipeline_event("step_started", "consolidation"),
            _task_event("consolidation_done",
                        data={"blocks_before": 214, "blocks_after": 96,
                              "merged_away": 118}),
        )
        self.assertIn("214 -> 96 block(s) (118 merged)", self.stream.getvalue())

    def test_indexing_embedding_progress(self):
        self._feed(
            _pipeline_event("step_started", "source_indexing"),
            _task_event("indexing_embedding", done=6, total=12),
        )
        self.assertIn("embedding 6/12", self.stream.getvalue())

    def test_knowledge_unit_progress_with_total(self):
        self._feed(
            _pipeline_event("step_started", "knowledge_extraction"),
            _task_event("knowledge_start", units=5, granularity="section"),
            _task_event("knowledge_unit", label="section 0 (page 1)", found=1),
            _task_event("knowledge_unit", label="section 1 (page 1)", found=0),
        )
        out = self.stream.getvalue()
        self.assertIn("5 unit(s) · section", out)
        self.assertIn("unit 2/5", out)

    def test_warning_counter_increments(self):
        self._feed(
            _pipeline_event("step_started", "extraction_validation"),
            _task_event("consistency_warning", message="blank block"),
            _task_event("summarization_warning", message="oversize"),
        )
        self.assertIn("warnings: 2", self.stream.getvalue())

    def test_summarized_events_count_units(self):
        self._feed(
            _pipeline_event("step_started", "hierarchical_summarization"),
            _task_event("summarization_start", units_total=3),
            _task_event("summarized", label="block 0", copied=True),
            _task_event("summarized", label="block 1", copied=False),
            _task_event("summarized", label="chapter 1", copied=False),
        )
        self.assertIn("unit 3/3", self.stream.getvalue())

    def test_summarization_skipped_counts_as_warning(self):
        self._feed(
            _pipeline_event("step_started", "hierarchical_summarization"),
            _task_event("summarization_skipped", message="blank content"),
        )
        self.assertIn("warnings: 1", self.stream.getvalue())

    def test_observer_never_raises(self):
        # Malformed events must be swallowed silently.
        self.display({"phase": "pipeline"})  # no kind, no data
        self.display({"phase": "task", "kind": "unknown_kind", "data": None})
        self.display({"kind": "step_started"})  # no phase
        self.display(None)  # type: ignore[arg-type] — hostile input on purpose

    def test_finish_prints_summary_and_restores_cursor(self):
        self._feed(
            _pipeline_event("step_started", "consolidation"),
            _pipeline_event("step_done", "consolidation"),
        )
        self.display.finish(ok=True, message="Ingestion completed")
        out = self.stream.getvalue()
        self.assertIn("Ingestion completed (0.0s)", out)
        self.assertIn("\x1b[32m\u2714\x1b[0m", out)  # green check
        self.assertIn("\x1b[?25h", out)  # cursor restored


class TickerTest(unittest.TestCase):
    """The daemon fan: repaints the running row between events."""

    def test_ticker_spins_running_row_between_events(self):
        stream = FakeTty()
        display = IngestionDisplay(stream, title="t", ansi=True)
        with mock.patch("src.tools.cli_progress._TICK_INTERVAL", 0.05):
            display.start()
            display(_pipeline_event("step_started", "content_extraction"))
            stream.seek(0)
            stream.truncate(0)
            before = stream.getvalue()
            time.sleep(0.35)  # several ticks at the patched interval
            after = stream.getvalue()
            self.assertNotEqual(before, after)  # the fan kept repainting
            self.assertIn("extraction", after)  # the running row, shortened
            self.assertRegex(after, r"\d+s")     # elapsed timer present
            display.close()

    def test_ticker_does_not_advance_terminal_rows(self):
        stream = FakeTty()
        display = IngestionDisplay(stream, title="t", ansi=True)
        with mock.patch("src.tools.cli_progress._TICK_INTERVAL", 0.05):
            display.start()
            display(_pipeline_event("step_started", "consolidation"))
            display(_pipeline_event("step_done", "consolidation"))
            stream.seek(0)
            stream.truncate(0)
            time.sleep(0.2)
            # No running row left: the ticker has nothing to animate and
            # must not rewrite the finished checklist.
            self.assertEqual(stream.getvalue(), "")
            display.close()

    def test_finish_stops_the_ticker(self):
        stream = FakeTty()
        display = IngestionDisplay(stream, title="t", ansi=True)
        with mock.patch("src.tools.cli_progress._TICK_INTERVAL", 0.05):
            display.start()
            display(_pipeline_event("step_started", "consolidation"))
            display.finish(ok=True, message="done")
            self.assertIsNone(display._ticker)
            stream.seek(0)
            stream.truncate(0)
            time.sleep(0.2)
            self.assertEqual(stream.getvalue(), "")  # animation stopped

    def test_plain_mode_has_no_ticker(self):
        stream = io.StringIO()
        display = IngestionDisplay(stream, title="t", ansi=False)
        display.start()
        display(_pipeline_event("step_started", "consolidation"))
        self.assertIsNone(display._ticker)
        display.close()


class IngestionDisplayPlainTest(unittest.TestCase):
    """Plain mode: sequential lines, no cursor control, no spinners."""

    def setUp(self):
        self.stream = io.StringIO()
        self.display = IngestionDisplay(self.stream, title="Ingesting 'x.pdf'",
                                        ansi=False)
        self.display.start()

    def _feed(self, *events):
        for event in events:
            self.display(event)

    def test_no_ansi_sequences_anywhere(self):
        self._feed(
            _pipeline_event("step_started", "consolidation"),
            _task_event("extracted", pages=3, chapters=1),
            _pipeline_event("step_done", "consolidation"),
            _pipeline_event("step_started", "hierarchical_summarization"),
            _task_event("summarization_start", units_total=2),
            _task_event("summarized", label="block 0", copied=True),
            _pipeline_event("step_done", "hierarchical_summarization"),
            _pipeline_event("step_failed", "source_indexing",
                            message="step 'source_indexing' failed: boom"),
        )
        self.display.finish(ok=False, message="boom")
        out = self.stream.getvalue()
        for forbidden in ("\x1b[", "▸", "▾", "◂", "▴"):
            self.assertNotIn(forbidden, out)

    def test_terminal_states_are_printed_once(self):
        self._feed(
            _pipeline_event("step_started", "consolidation"),
            _task_event("extracted", pages=3, chapters=1),  # mid-run detail
            _pipeline_event("step_done", "consolidation"),
        )
        out = self.stream.getvalue()
        # The plain mode prints only the terminal state, not the start.
        self.assertEqual(out.count("[x]"), 1)
        self.assertNotIn("[/]", out)
        self.assertIn("3 page(s), 1 chapter(s)", out)

    def test_finish_plain_summary(self):
        self._feed(_pipeline_event("step_started", "consolidation"),
                   _pipeline_event("step_done", "consolidation"))
        self.display.finish(ok=True, message="done")
        out = self.stream.getvalue()
        self.assertIn("OK: done", out)
        self.assertIn("warnings: 0 · errors: 0", out)
        self.assertIn("full details: data/logs/ingestion.log", out)


class RemovalDisplayTest(unittest.TestCase):
    def _report(self):
        return {
            "status": "removed",
            "document": "meow.pdf",
            "doc_id": "doc:36a911e2",
            "steps": {
                "vector": {"ok": True, "deleted": 12},
                "job_files": {"ok": True, "removed": ["a", "b"]},
                "json_files": {"ok": True, "removed": ["a.json"]},
                "knowledge_base": {"ok": True, "purged_files": 2,
                                   "deleted_files": 1},
                "source_registration": {"ok": True, "removed": True},
                "mineru": {"ok": True, "removed": None},
            },
        }

    def test_ansi_rendering(self):
        stream = FakeTty()
        display = RemovalDisplay(stream, title="Removing 'meow.pdf'", ansi=True)
        display.render(self._report())
        out = stream.getvalue()
        self.assertIn("Removing 'meow.pdf'", out)
        self.assertIn("12 chunk(s) deleted", out)
        self.assertIn("2 checkpoint(s) removed", out)
        self.assertIn("2 file(s) updated, 1 emptied", out)
        self.assertIn("Removed from the corpus (doc:36a911e2)", out)
        self.assertIn("the source file is kept", out)

    def test_plain_rendering(self):
        stream = io.StringIO()
        display = RemovalDisplay(stream, title="Removing 'meow.pdf'", ansi=False)
        display.render(self._report())
        out = stream.getvalue()
        self.assertNotIn("\x1b[", out)
        self.assertIn("OK: Removed from the corpus", out)

    def test_partial_status(self):
        stream = io.StringIO()
        display = RemovalDisplay(stream, title="Removing 'meow.pdf'", ansi=False)
        report = self._report()
        report["status"] = "partial"
        report["reason"] = "vector store locked"
        display.render(report)
        out = stream.getvalue()
        self.assertIn("PARTIAL: Removal partially failed: vector store locked", out)
        self.assertIn("removal is idempotent", out)

    def test_missing_steps_are_skipped(self):
        stream = io.StringIO()
        display = RemovalDisplay(stream, title="t", ansi=False)
        display.render({"status": "removed", "steps": {}})
        self.assertNotIn("vector chunks", stream.getvalue())


class CreateFactoryTest(unittest.TestCase):
    def test_create_returns_none_on_piped_output(self):
        with mock.patch("sys.stdout", new=io.StringIO()):
            self.assertIsNone(IngestionDisplay.create(title="t"))
            self.assertIsNone(RemovalDisplay.create(title="t"))

    def test_create_builds_displays_on_tty(self):
        with mock.patch("sys.stdout", new=FakeTty()):
            with mock.patch.dict("os.environ", {"WINTERMUTE_PLAIN": "1"},
                                 clear=False):
                # WINTERMUTE_PLAIN forces the plain renderer but still builds.
                ingest = IngestionDisplay.create(title="t")
                self.assertIsNotNone(ingest)
                self.assertFalse(ingest._ansi)  # type: ignore[union-attr]
                removal = RemovalDisplay.create(title="t")
                self.assertIsNotNone(removal)
                self.assertFalse(removal._ansi)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
