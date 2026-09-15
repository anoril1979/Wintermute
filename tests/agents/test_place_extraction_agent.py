"""Tests for the Place knowledge pass (extraction → validation → resolution).

The places pipeline is the exact sibling of the characters one on the
shared ``EntityExtractionAgent`` / ``EntityValidatorAgent`` /
``EntityResolverAgent`` machinery: these tests pin the PLACE wiring —
prompt file, ``{"places": [...]}`` answer contract, ``knowledge_places``
context key, ``data/knowledge/places`` markdown output — and the key
regression this slice fixed: a subclass payload written under the
module-level ``OUTPUT_KEY`` constant instead of its own key.

Hermetic: scripted LLM, temp cache and temp knowledge base. No live
Ollama, no real data/ writes.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.agents.agents.entity_extraction_agent import PlaceExtractionAgent
from src.agents.agents.entity_resolver_agent import PlaceResolver
from src.agents.agents.knowledge_validation_agent import PlaceValidatorAgent
from src.agents.contexts import IngestionContext
from src.agents.protocols import AgentStatus
from src.extraction.models import (
    Chapter,
    DocumentExtract,
    PageContent,
    Section,
    TextBlock,
    TocEntry,
)


def make_document() -> DocumentExtract:
    """One chapter / one page / two sections — section granularity → 2 units."""
    sections1 = [
        Section(section_id=1, page_number=1, bbox=(0, 0, 10, 10),
                blocks=[TextBlock(block_id=1, page_number=1, bbox=(0, 0, 10, 10),
                                  raw_text="La citadelle d'Aras domine la vallee.")],
                raw_text="La citadelle d'Aras domine la vallee."),
        Section(section_id=2, page_number=1, bbox=(0, 0, 10, 10),
                blocks=[TextBlock(block_id=2, page_number=1, bbox=(0, 0, 10, 10),
                                  raw_text="Le col de Brume reste infranchissable l'hiver.")],
                raw_text="Le col de Brume reste infranchissable l'hiver."),
    ]
    doc = DocumentExtract(
        source_path="gazette.pdf", title="Gazette", author="A", subject="S",
        total_pages=1,
        chapters=[Chapter(
            toc_entry=TocEntry(level=1, title="Chapter I",
                               page_number=1, page_index=0),
            pages=[PageContent(page_number=1, width=100, height=100,
                               raw_text="places", sections=sections1)],
            full_text="places",
        )],
    )
    return doc


class StubPlaceAgent(PlaceExtractionAgent):
    """PlaceExtractionAgent with a scripted LLM (one reply per unit)."""

    def __init__(self, replies: str | list[str]):
        super().__init__(output_dir=Path(tempfile.mkdtemp()))
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


class PlaceExtractionTest(unittest.TestCase):
    def test_contract_and_payload_key(self):
        """The place pass runs the shared walk with the place contract:
        place prompt, ``{"places": [...]}`` entries, ``knowledge_places``
        key — NOT the characters key."""
        agent = StubPlaceAgent([
            json.dumps({"places": [{"full_name": "Citadelle d'Aras",
                                    "short_name": "Aras",
                                    "aliases": ["Pic de Fer"]}]}),
            json.dumps({"places": [{"full_name": "Col de Brume"}]}),
        ])
        context = IngestionContext(document_path=Path("gazette.pdf"))
        context.outputs["content_extraction"] = make_document()

        result = agent.run(context)

        self.assertEqual(result.status, AgentStatus.OK)
        self.assertEqual(result.payload["places"], [
            {"full_name": "Citadelle d'Aras", "short_name": "Aras",
             "aliases": ["Pic de Fer"],
             "source_ids": result.payload["places"][0]["source_ids"]},
            {"full_name": "Col de Brume", "short_name": None,
             "aliases": [],
             "source_ids": result.payload["places"][1]["source_ids"]},
        ])
        # The subclass key, not the historical characters one.
        self.assertIn("knowledge_places", context.outputs)
        self.assertNotIn("knowledge_characters", context.outputs)
        # Two units fed, both through the place prompt.
        self.assertEqual(agent.calls, 2)
        for prompt in agent.prompts:
            self.assertIn("place", prompt.lower())

    def test_derived_class_members(self):
        """The subclass declares only ENTITY_TYPE (+name): the prompt path,
        entry key and payload key derive from it — and the place prompt
        file exists."""
        self.assertEqual(PlaceExtractionAgent.ENTITY_TYPE, "place")
        self.assertEqual(PlaceExtractionAgent.llm_role, "knowledge_extractor")
        agent = PlaceExtractionAgent()
        self.assertEqual(agent.OUTPUT_ENTRY_KEY, "places")
        self.assertEqual(agent.OUTPUT_KEY, "knowledge_places")
        self.assertEqual(
            agent._prompt_path, Path("prompts/knowledge/place_extraction.md")
        )
        self.assertTrue(agent._prompt_path.is_file())

    def test_base_requires_entity_type(self):
        """A subclass without ENTITY_TYPE is a wiring error."""
        from src.agents.agents.entity_extraction_agent import (
            EntityExtractionAgent,
        )

        class Incomplete(EntityExtractionAgent):
            pass

        with self.assertRaises(ValueError):
            Incomplete()


class PlaceResolutionTest(unittest.TestCase):
    def test_places_land_in_their_own_knowledge_folder(self):
        """Full trio: extracted places validate and resolve into
        ``<base>/places/<slug>.md`` files plus the ``places.md`` sidecar
        listing — and the lookup finds them by alias."""
        from src.knowledge.character_markdown_store import index_path_for, load_index
        from src.knowledge.entity_lookup import resolve_entity

        base = Path(tempfile.mkdtemp())
        agent = StubPlaceAgent([
            json.dumps({"places": [{"full_name": "Citadelle d'Aras",
                                    "short_name": "Aras",
                                    "aliases": ["Pic de Fer"]}]}),
            json.dumps({"places": []}),
        ])
        context = IngestionContext(document_path=Path("gazette.pdf"))
        context.outputs["content_extraction"] = make_document()

        self.assertEqual(agent.run(context).status, AgentStatus.OK)
        self.assertEqual(PlaceValidatorAgent().run(context).status,
                         AgentStatus.OK)
        result = PlaceResolver(base_dir=base).run(context)

        self.assertEqual(result.status, AgentStatus.OK)
        self.assertEqual(result.payload["created"], 1)
        entity_file = base / "places" / "citadelle-d-aras.md"
        self.assertTrue(entity_file.is_file())
        text = entity_file.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# Place : Citadelle d'Aras"))
        self.assertIn("- Pic de Fer", text)
        index = load_index(index_path_for(base, "places"))
        self.assertEqual(index, [{"full_name": "Citadelle d'Aras",
                                  "aliases": ["Pic de Fer"]}])
        # The retrieval-side lookup resolves the new type by alias.
        match = resolve_entity("Pic de Fer", base_dir=base)
        self.assertIsNotNone(match)
        self.assertEqual(match.full_name, "Citadelle d'Aras")


if __name__ == "__main__":
    unittest.main()
