"""Tests for the reply-language management (detect once, enforce at reply).

Hermetic: no LLM, no config — the language is a plain string extracted by
the analyzer and carried to the reply agents. Covers the normalization
seam, the localized deterministic fallbacks, the analyzer-model wiring,
the orchestrator's metadata/result carrying and the two reply agents'
prompt injection.
"""

from __future__ import annotations

import unittest
import unittest.mock

from src.agents.agents.answer_agent import AnswerAgent
from src.agents.agents.general_task_agent import GeneralTaskAgent
from src.agents.contexts import RetrievalContext, RoutingContext
from src.indexing.chunks import VectorChunk
from src.routing.language import (
    DEFAULT_LANGUAGE,
    dormant_corpus_reply,
    entity_unknown_reply,
    nothing_found_reply,
    normalize_language,
    unserved_kind_reply,
)
from src.routing.models import AnalysisResult, GeneralRequest, parse_analysis


# ---------------------------------------------------------------------------
# normalize_language
# ---------------------------------------------------------------------------

class NormalizeLanguageTest(unittest.TestCase):
    def test_canonical_codes_pass_through(self):
        self.assertEqual(normalize_language("fr"), "fr")
        self.assertEqual(normalize_language("EN"), "en")

    def test_language_names_map_to_codes(self):
        self.assertEqual(normalize_language("French"), "fr")
        self.assertEqual(normalize_language("français"), "fr")
        self.assertEqual(normalize_language("English"), "en")
        self.assertEqual(normalize_language("deutsch"), "de")

    def test_regional_codes_normalize_on_primary_subtag(self):
        self.assertEqual(normalize_language("fr-FR"), "fr")
        self.assertEqual(normalize_language("en_US"), "en")
        self.assertEqual(normalize_language("pt-BR"), "pt")

    def test_unknown_label_falls_back_to_english(self):
        self.assertEqual(normalize_language("klingon"), DEFAULT_LANGUAGE)

    def test_none_and_blank_fall_back_to_english(self):
        self.assertEqual(normalize_language(None), DEFAULT_LANGUAGE)
        self.assertEqual(normalize_language("   "), DEFAULT_LANGUAGE)

    def test_unknown_nonblank_value_logs_a_warning(self):
        with self.assertLogs("src.routing.language", level="WARNING"):
            normalize_language("klingon")


# ---------------------------------------------------------------------------
# Localized deterministic fallbacks
# ---------------------------------------------------------------------------

class LocalizedFallbacksTest(unittest.TestCase):
    def test_nothing_found_localizes(self):
        self.assertIn("Ingérez", nothing_found_reply("fr"))
        self.assertIn("Ingest", nothing_found_reply("en"))
        self.assertEqual(nothing_found_reply("fr-FR"), nothing_found_reply("fr"))

    def test_nothing_found_unknown_language_falls_back_to_english(self):
        self.assertEqual(nothing_found_reply("klingon"), nothing_found_reply("en"))

    def test_dormant_localizes(self):
        self.assertIn("sommeil", dormant_corpus_reply("fr"))
        self.assertIn("dormant", dormant_corpus_reply("en"))

    def test_unserved_kind_interpolates_and_localizes(self):
        fr = unserved_kind_reply("relationship", "fr")
        self.assertIn("relationship", fr)
        self.assertTrue(fr.startswith("Ce type"))
        en = unserved_kind_reply("relationship", "en")
        self.assertIn("not served yet", en)

    def test_entity_unknown_interpolates_and_localizes(self):
        fr = entity_unknown_reply("Rorg", "fr")
        self.assertIn("Rorg", fr)
        self.assertTrue(fr.startswith("Aucune entité"))
        en = entity_unknown_reply("Rorg", "en")
        self.assertIn("No entity named 'Rorg'", en)

    def test_entity_unknown_lists_candidates_when_given(self):
        reply = entity_unknown_reply("le", "en", ["Joe le Clodo",
                                                   "Bobby le Frelon"])
        self.assertIn("Did you mean one of these?", reply)
        self.assertIn("+ Joe le Clodo", reply)
        self.assertIn("+ Bobby le Frelon", reply)
        # Localized candidates line.
        fr = entity_unknown_reply("le", "fr", ["Joe le Clodo"])
        self.assertIn("Vouliez-vous dire", fr)

    def test_entity_unknown_without_candidates_has_no_list(self):
        reply = entity_unknown_reply("Rorg", "en", [])
        self.assertNotIn("Did you mean", reply)


# ---------------------------------------------------------------------------
# AnalysisResult.language (the analyzer's detection)
# ---------------------------------------------------------------------------

class AnalysisLanguageTest(unittest.TestCase):
    def test_language_parsed_from_analyzer_payload(self):
        result = parse_analysis(
            '{"language": "fr", "retrieval": [], "general": '
            '[{"question": "bonjour", "utterance": "bonjour"}]}'
        )
        self.assertEqual(result.language, "fr")

    def test_language_absent_defaults_to_english(self):
        result = parse_analysis('{"retrieval": [], "general": []}')
        self.assertEqual(result.language, DEFAULT_LANGUAGE)

    def test_language_free_label_is_normalized(self):
        result = parse_analysis(
            '{"language": "French", "retrieval": [], "general": []}'
        )
        self.assertEqual(result.language, "fr")

    def test_language_garbage_fails_open_to_english(self):
        result = parse_analysis(
            '{"language": "the user speaks French", "retrieval": [], "general": []}'
        )
        self.assertEqual(result.language, DEFAULT_LANGUAGE)

    def test_empty_analysis_defaults_to_english(self):
        self.assertEqual(AnalysisResult().language, DEFAULT_LANGUAGE)


# ---------------------------------------------------------------------------
# Orchestrator carrying
# ---------------------------------------------------------------------------

class OrchestratorLanguageTest(unittest.TestCase):
    def _analyze(self, language="fr"):
        analysis = AnalysisResult(language=language)
        analysis.general.append(GeneralRequest(utterance="coucou", question="coucou"))
        return analysis

    def test_metadata_and_result_carry_the_language(self):
        from src.routing.routing_orchestrator import run_routing

        analyzer = unittest.mock.Mock()
        analyzer.analyze.return_value = self._analyze("fr")

        class _NoopAgent:
            name = "noop"

            def run(self, context, request):
                from src.agents.protocols import AgentResult, AgentStatus

                return AgentResult(agent_name=self.name, status=AgentStatus.OK)

            def validate(self, context, request):
                return None

        result = run_routing("coucou", agents={"general_task": _NoopAgent()},
                             analyzer=analyzer)

        self.assertEqual(result["language"], "fr")
        trace = next(e for e in result["traces"] if e["kind"] == "understood")
        self.assertEqual(trace["data"]["language"], "fr")

    def test_analysis_error_result_has_no_language_yet(self):
        from src.routing.routing_orchestrator import STATUS_ANALYSIS_ERROR, run_routing

        analyzer = unittest.mock.Mock()
        from src.routing.request_analyzer import RequestAnalysisError

        analyzer.analyze.side_effect = RequestAnalysisError("boom", cause="llm_request")
        result = run_routing("coucou", agents={}, analyzer=analyzer)
        self.assertEqual(result["status"], STATUS_ANALYSIS_ERROR)
        self.assertNotIn("language", result)


# ---------------------------------------------------------------------------
# Reply agents: the authoritative "Reply language" key
# ---------------------------------------------------------------------------

class GeneralAgentLanguageTest(unittest.TestCase):
    class _FakeLLM:
        def __init__(self):
            self.prompts: list[str] = []

        def complete(self, prompt, max_tokens=None):
            self.prompts.append(prompt)
            return "Réponse glacée."

    def test_prompt_carries_detected_language(self):
        llm = self._FakeLLM()
        context = RoutingContext(request="test")
        context.metadata["language"] = "fr"
        agent = GeneralTaskAgent(allow_missing_role=True, llm=llm)
        request = GeneralRequest(utterance="Salut", question="Salut")
        result = agent.run(context, request)

        self.assertEqual(result.status.value, "ok")
        self.assertIn("Reply language:\nfr", llm.prompts[0])

    def test_prompt_defaults_to_english_without_metadata(self):
        llm = self._FakeLLM()
        context = RoutingContext(request="test")  # no language metadata
        agent = GeneralTaskAgent(allow_missing_role=True, llm=llm)
        agent.run(context, GeneralRequest(utterance="Hi", question="Hi"))
        self.assertIn(f"Reply language:\n{DEFAULT_LANGUAGE}", llm.prompts[0])


class AnswerAgentLanguageTest(unittest.TestCase):
    def _hit(self, **meta):
        metadata = {"doc_title": "Gazette", "page_number": 1,
                    "origin": "canon", "kind": "content", **meta}
        return VectorChunk(id="doc:x::chp:1::pg:1::sec:1::txt:1",
                           text="Le roi s'enfuit.", metadata=metadata, score=0.9)

    class _FakeLLM:
        def __init__(self):
            self.prompts: list[str] = []

        def complete(self, prompt):
            self.prompts.append(prompt)
            return "Le roi s'enfuit [1]."

    def test_prompt_carries_detected_language(self):
        llm = self._FakeLLM()
        context = RetrievalContext(question="Où est le roi ?")
        context.metadata["language"] = "fr"
        context.outputs["hits"] = [self._hit()]
        agent = AnswerAgent(allow_missing_role=True, llm=llm)
        result = agent.run(context)

        self.assertEqual(result.status.value, "ok")
        self.assertIn("Reply language:\nfr", llm.prompts[0])
        self.assertIn("Où est le roi ?", llm.prompts[0])

    def test_no_hit_fallback_is_localized(self):
        context = RetrievalContext(question="Où est le roi ?")
        context.metadata["language"] = "fr"
        agent = AnswerAgent(allow_missing_role=True, llm=self._FakeLLM())
        result = agent.run(context)

        self.assertEqual(result.status.value, "ok")
        self.assertTrue(result.payload["no_answer"])
        self.assertIn("Ingérez", result.payload["answer"])
        self.assertEqual(context.outputs["answer"], result.payload["answer"])


# ---------------------------------------------------------------------------
# Retrieval task agent: forwards the language to the pipeline
# ---------------------------------------------------------------------------

class RetrievalTaskAgentLanguageTest(unittest.TestCase):
    def test_language_forwarded_to_the_runner(self):
        from src.agents.agents.retrieval_task_agent import RetrievalTaskAgent

        captured = {}

        def runner(requests, **kwargs):
            captured.update(kwargs)
            return {"status": "ok", "requests": [], "hits": [], "traces": []}

        context = RoutingContext(request="test")
        context.metadata["language"] = "fr"
        from src.routing.models import RetrievalRequest

        agent = RetrievalTaskAgent(runner=runner)
        result = agent.run(
            context, RetrievalRequest(utterance="q", question="q")
        )

        self.assertEqual(result.status.value, "ok")
        self.assertEqual(captured.get("language"), "fr")


if __name__ == "__main__":
    unittest.main()
