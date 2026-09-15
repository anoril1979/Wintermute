"""Tests for the CharacterExtractionAgent (knowledge_extraction step).

Hermetic: the LLM is stubbed (the protocol is one ``complete`` call) and
every store lives in a throwaway temp dir — no live Ollama, no writes to
the real data/cache/knowledge folder.

Covers: unit walk at each granularity, LLM cycle (prompts, dedup, call
count), answer parsing (fenced JSON, error marker, malformed answers),
failure mapping (LLM_RESPONSE → retryable), cache persistence and the
validate() hook.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.agents.agents.entity_extraction_agent import (
    CharacterExtractionAgent,
    EntityExtractionAgent,
)

#: The character pass's context payload key (derived from ENTITY_TYPE).
OUTPUT_KEY = "knowledge_characters"
from src.agents.contexts import IngestionContext
from src.agents.llm_roles import MissingLLMRoleError
from src.extraction.models import (
    Chapter,
    DocumentExtract,
    PageContent,
    Section,
    TocEntry,
)
from src.knowledge.character_cache import KnowledgeJsonError, load_knowledge


def make_block(block_id: int, text: str, page: int = 1):
    from src.extraction.models import TextBlock

    return TextBlock(block_id=block_id, page_number=page, bbox=(0, 0, 1, 1), raw_text=text)


def make_section(section_id: int, texts: list[str], page: int = 1, title: str | None = None):
    return Section(
        section_id=section_id,
        blocks=[make_block(i, t, page) for i, t in enumerate(texts)],
        page_number=page,
        bbox=(0, 0, 1, 1),
        raw_text="\n".join(texts),
        section_title=title,
    )


def make_document() -> DocumentExtract:
    """Two chapters / one page each / two sections each — every section has
    text so the section-granularity walk yields 4 units."""
    sections1 = [
        make_section(0, ["Dantès entered the harbor.", "Mercedes waved."], page=1, title="Arrival"),
        make_section(1, ["The Count smiled.", "He knew the secret."], page=1, title="Revelation"),
    ]
    sections2 = [
        make_section(0, ["Villefort read the letter."], page=2, title="The Trial"),
        make_section(1, ["Faria whispered about the treasure."], page=2, title="The Cell"),
    ]
    pages = [
        PageContent(page_number=1, width=100, height=100, raw_text="page one",
                    sections=sections1),
        PageContent(page_number=2, width=100, height=100, raw_text="page two",
                    sections=sections2),
    ]
    chapters = [
        Chapter(toc_entry=TocEntry(level=1, title="Chapter I", page_number=1, page_index=0),
                pages=[pages[0]], full_text="page one"),
        Chapter(toc_entry=TocEntry(level=1, title="Chapter II", page_number=2, page_index=1),
                pages=[pages[1]], full_text="page two"),
    ]
    return DocumentExtract(
        source_path="gazette.pdf", title="Gazette", author="A", subject="S",
        total_pages=2, chapters=chapters,
    )


class StubKnowledgeAgent(CharacterExtractionAgent):
    """CharacterExtractionAgent with a scripted LLM.

    ``replies``: one string (every call returns it) or a list (calls
    consume the entries in order, cycling). ``prompts`` records every
    prompt for assertions.
    """

    def __init__(self, replies: str | list[str], **kwargs):
        kwargs.setdefault("output_dir", Path(tempfile.mkdtemp()))
        super().__init__(**kwargs)
        self.replies = list(replies) if isinstance(replies, list) else [replies]
        self.calls = 0
        self.prompts: list[str] = []

    def llm_client(self):  # noqa: D102 - protocol stand-in
        stub = self

        class _Client:
            def complete(self, prompt: str, max_tokens: int = 1024) -> str:
                text = stub.replies[stub.calls % len(stub.replies)]
                stub.calls += 1
                stub.prompts.append(prompt)
                return text

        return _Client()


def run_agent(agent, document=None, document_path="gazette.pdf"):
    context = IngestionContext(document_path=Path(document_path))
    if document is not None:
        context.outputs["content_extraction"] = document
    result = agent.run(context)
    return result, context


class UnitWalkTest(unittest.TestCase):
    """The granularity setting drives which units are fed to the LLM."""

    def test_section_granularity_walks_all_sections(self):
        agent = StubKnowledgeAgent("{}", granularity="section")
        units = list(agent._iter_units(make_document()))
        self.assertEqual(len(units), 4)  # 2 sections x 2 chapters

    def test_page_granularity_walks_pages(self):
        agent = StubKnowledgeAgent("{}", granularity="page")
        units = list(agent._iter_units(make_document()))
        self.assertEqual(len(units), 2)
        self.assertTrue(all(hasattr(u, "sections") and hasattr(u, "width") for u in units))

    def test_chapter_granularity_walks_chapters_and_uses_full_text(self):
        agent = StubKnowledgeAgent("{}", granularity="chapter")
        units = list(agent._iter_units(make_document()))
        self.assertEqual(len(units), 2)
        for unit in units:
            self.assertEqual(agent.unit_text(unit), unit.full_text)

    def test_blank_units_are_not_sent_to_the_llm(self):
        agent = StubKnowledgeAgent("{}", granularity="section")
        document = make_document()
        for section in document.chapters[0].pages[0].sections:
            section.raw_text = "   "
            section.blocks = []
        units = [u for u in agent._iter_units(document) if agent.unit_text(u)]
        self.assertEqual(len(units), 2)

    def test_invalid_granularity_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            StubKnowledgeAgent("{}", granularity="block")

    def test_orphan_pages_are_walked(self):
        agent = StubKnowledgeAgent("{}", granularity="section")
        document = make_document()
        document.orphan_pages.append(
            PageContent(page_number=3, width=100, height=100, raw_text="orphan",
                        sections=[make_section(0, ["An orphan page."], page=3)])
        )
        units = list(agent._iter_units(document))
        self.assertEqual(len(units), 5)


class ExtractionCycleTest(unittest.TestCase):
    """The LLM cycle: one call per unit, dedup, persistence, payload."""

    def test_extracts_and_merges_cross_unit_duplicates(self):
        replies = [
            json.dumps({"characters": [
                {"full_name": "Edmond Dantès", "short_name": "Dantès",
                 "aliases": ["Edmond"]},
            ]}),
            json.dumps({"characters": [
                {"full_name": "Edmond Dantès", "short_name": "Dantès",
                 "aliases": ["the Count", "Lord Wilmore"]},
            ]}),
            json.dumps({"characters": []}),
            json.dumps({"characters": []}),
        ]
        agent = StubKnowledgeAgent(replies)
        result, context = run_agent(agent, make_document())

        self.assertEqual(result.status.value, "ok")
        self.assertEqual(agent.calls, 4)  # one per section
        characters = result.payload["characters"]
        self.assertEqual(len(characters), 1)
        self.assertEqual(characters[0]["full_name"], "Edmond Dantès")
        # Union of aliases, order-preserving.
        self.assertEqual(characters[0]["aliases"], ["Edmond", "the Count", "Lord Wilmore"])
        self.assertEqual(result.payload["merged"], 1)
        self.assertEqual(result.payload["cache_path"], str(Path(agent._output_dir) / "gazette.json"))

    def test_cache_file_written_and_readable(self):
        agent = StubKnowledgeAgent(json.dumps({"characters": [
            {"full_name": "Villefort", "short_name": None, "aliases": []},
        ]}))
        result, _ = run_agent(agent, make_document())
        path = Path(result.payload["cache_path"])
        self.assertTrue(path.exists())
        payload = load_knowledge(path)
        self.assertEqual(payload["characters"][0]["full_name"], "Villefort")
        # short_name None is kept as null — the cache mirrors the LLM answer.
        self.assertIsNone(payload["characters"][0]["short_name"])

    def test_cache_file_is_plain_json_no_envelope(self):
        agent = StubKnowledgeAgent(json.dumps({"characters": []}))
        result, _ = run_agent(agent, make_document())
        data = json.loads(Path(result.payload["cache_path"]).read_text(encoding="utf-8"))
        self.assertEqual(set(data.keys()), {"characters"})

    def test_every_prompt_carries_the_unit_text(self):
        agent = StubKnowledgeAgent(json.dumps({"characters": []}))
        run_agent(agent, make_document())
        self.assertEqual(len(agent.prompts), 4)
        self.assertIn("Dantès entered the harbor.", agent.prompts[0])
        self.assertIn("Villefort read the letter.", agent.prompts[2])
        self.assertIn("<<<<TEXT>>>>", agent.prompts[0])

    def test_context_output_payload_written(self):
        agent = StubKnowledgeAgent(json.dumps({"characters": []}))
        result, context = run_agent(agent, make_document())
        payload = context.outputs[OUTPUT_KEY]
        self.assertIn("characters", payload)
        self.assertEqual(payload["llm_calls"], 4)

    def test_save_failure_is_reported_not_fatal(self):
        agent = StubKnowledgeAgent(json.dumps({"characters": []}))
        with mock.patch(
            "src.agents.agents.entity_extraction_agent.save_knowledge",
            side_effect=OSError("disk full"),
        ):
            result, context = run_agent(agent, make_document())
        self.assertEqual(result.status.value, "ok")
        self.assertIn("knowledge cache not saved", context.errors[OUTPUT_KEY])
        self.assertIsNone(result.payload["cache_path"])


class ParsingAndFailureTest(unittest.TestCase):
    """Answer parsing and the failure mapping (LLM_RESPONSE, retryable)."""

    def test_fenced_json_answer_is_accepted(self):
        raw = 'Voici :\n```json\n{"characters": [{"full_name": "Faria"}]}\n```'
        entries = CharacterExtractionAgent._parse_answer(
            CharacterExtractionAgent(llm_role="knowledge_extractor"), raw, "label"
        )
        self.assertEqual(entries[0]["full_name"], "Faria")

    def test_error_marker_skips_the_unit_and_continues(self):
        # The Gazette case: a section with no characters must NOT fail the
        # step (a retry could not fix it and would re-call every unit).
        # The unit is skipped with a warning; the walk continues.
        raw = json.dumps({"characters": [], "error": "no characters mentioned"})
        agent = StubKnowledgeAgent(raw)
        result, context = run_agent(agent, make_document())
        self.assertEqual(result.status.value, "ok")
        self.assertEqual(agent.calls, 4)  # every unit was still attempted
        self.assertEqual(result.payload["characters"], [])
        skipped = [e for e in context.events if e["kind"] == "knowledge_unit_skipped"]
        self.assertEqual(len(skipped), 4)
        self.assertIn("no characters mentioned", skipped[0]["message"])

    def test_error_marker_unit_then_valid_units_still_collects(self):
        # First unit reports unreadable, the others yield entries: the
        # entries from the healthy units are kept.
        replies = [
            json.dumps({"characters": [], "error": "garbled"}),
            json.dumps({"characters": [{"full_name": "Faria"}]}),
            json.dumps({"characters": []}),
            json.dumps({"characters": []}),
        ]
        agent = StubKnowledgeAgent(replies)
        result, _ = run_agent(agent, make_document())
        self.assertEqual(result.status.value, "ok")
        self.assertEqual(result.payload["characters"][0]["full_name"], "Faria")

    def test_empty_answer_is_never_a_failure(self):
        agent = StubKnowledgeAgent(json.dumps({"characters": []}))
        result, _ = run_agent(agent, make_document())
        self.assertEqual(result.status.value, "ok")

    def test_non_json_answer_fails_with_llm_response_domain(self):
        agent = StubKnowledgeAgent("I cannot comply, here is a haiku instead")
        result, context = run_agent(agent, make_document())
        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.failure_domain.value, "llm_response")
        self.assertIn("knowledge_failed", [e["kind"] for e in context.events])

    def test_wrong_entry_shape_fails(self):
        agent = StubKnowledgeAgent(json.dumps({"characters": [{"aliases": []}]}))
        result, _ = run_agent(agent, make_document())
        self.assertEqual(result.status.value, "failed")
        self.assertIn("full_name", result.detail)

    def test_aliases_are_stripped_and_blanks_dropped(self):
        raw = json.dumps({"characters": [
            {"full_name": "  Mercedes  ", "aliases": ["  Mercédès ", "", "  "]},
        ]})
        agent = StubKnowledgeAgent(raw)
        result, _ = run_agent(agent, make_document())
        self.assertEqual(result.payload["characters"][0]["full_name"], "Mercedes")
        self.assertEqual(result.payload["characters"][0]["aliases"], ["Mercédès"])

    def test_llm_transport_failure_maps_to_llm_response(self):
        agent = StubKnowledgeAgent("{}")

        def boom(prompt, max_tokens=1024):
            raise RuntimeError("connection refused")

        with mock.patch.object(agent, "llm_client", return_value=type("C", (), {"complete": staticmethod(boom)})):
            result, _ = run_agent(agent, make_document())
        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.failure_domain.value, "llm_response")

    def test_no_extraction_output_is_skipped(self):
        agent = StubKnowledgeAgent("{}")
        result, _ = run_agent(agent, document=None)
        self.assertEqual(result.status.value, "skipped")


class ValidateHookTest(unittest.TestCase):
    """validate(): the cache must exist on disk after a successful run."""

    def test_validate_ok_when_cache_written(self):
        agent = StubKnowledgeAgent(json.dumps({"characters": []}))
        result, context = run_agent(agent, make_document())
        self.assertIsNone(agent.validate(context))

    def test_validate_fails_when_cache_missing(self):
        agent = StubKnowledgeAgent(json.dumps({"characters": []}))
        _, context = run_agent(agent, make_document())
        context.outputs[OUTPUT_KEY]["cache_path"] = None
        Path(context.outputs[OUTPUT_KEY]["cache_path"] or agent._cache_path(Path("gazette.pdf"))).unlink()
        validation = agent.validate(context)
        self.assertIsNotNone(validation)
        self.assertEqual(validation.failure_domain.value, "input_data")

    def test_validate_none_before_any_run(self):
        agent = StubKnowledgeAgent("{}")
        context = IngestionContext(document_path=Path("gazette.pdf"))
        self.assertIsNone(agent.validate(context))


class CacheStoreTest(unittest.TestCase):
    """The knowledge cache store's own contract (load path)."""

    def test_load_rejects_malformed_json(self):
        path = Path(tempfile.mkdtemp()) / "x.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(KnowledgeJsonError):
            load_knowledge(path)

    def test_load_rejects_non_object_root(self):
        path = Path(tempfile.mkdtemp()) / "x.json"
        path.write_text("[1, 2]", encoding="utf-8")
        with self.assertRaises(KnowledgeJsonError):
            load_knowledge(path)

    def test_load_rejects_bad_characters_shape(self):
        path = Path(tempfile.mkdtemp()) / "x.json"
        path.write_text(json.dumps({"characters": "nope"}), encoding="utf-8")
        with self.assertRaises(KnowledgeJsonError):
            load_knowledge(path)

    def test_load_reports_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_knowledge(Path(tempfile.mkdtemp()) / "nope.json")


class RoleResolutionTest(unittest.TestCase):
    """The knowledge_extractor role is resolved STRICTLY at construction."""

    def test_missing_role_raises_at_construction(self):
        with mock.patch(
            "src.agents.llm_roles.require_llm_role",
            side_effect=MissingLLMRoleError("no such role"),
        ):
            with self.assertRaises(MissingLLMRoleError):
                CharacterExtractionAgent()


if __name__ == "__main__":
    unittest.main()
