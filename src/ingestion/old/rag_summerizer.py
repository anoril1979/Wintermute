# summarizer.py
"""
SectionSummarizer – Étape de résumé dans le pipeline RAG.

Deux implémentations sont fournies, toutes deux conformes à
SummarizerProtocol (défini dans rag_pipeline.py) :

	1. LLMSectionSummarizer      → appelle un LLM (via un client injecté)
	2. ExtractiveSectionSummarizer → résumé local, sans API, sans dépendance

Le client LLM n'est PAS fourni ici : ce module reste agnostique du provider
(OpenAI, Anthropic, un endpoint interne Ubisoft, etc.). Il suffit d'implémenter
LLMClientProtocol pour brancher le provider de son choix.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from src.ingestion.old import rag_config
from src.ingestion.old.rag_models import Chapter

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# ---------------------------------------------------------------------------
# Contrat du client LLM (à implémenter par l'appelant)
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMClientProtocol(Protocol):
	"""
	Interface minimale attendue pour tout client LLM branché sur le summarizer.
	Exemple d'implémentation : wrapper autour de l'API OpenAI, d'un endpoint
	interne Ubisoft, d'un modèle local via Ollama, etc.
	"""

	def complete(self, prompt: str, max_tokens: int = 300) -> str:
		"""
		Envoie un prompt au LLM et retourne le texte généré.

		Args:
			prompt: texte du prompt complet
			max_tokens: budget de tokens pour la réponse

		Returns:
			Texte généré par le modèle (déjà nettoyé, sans métadonnées)
		"""
		...

@runtime_checkable
class SummarizerProtocol(Protocol):
	def summarize(self, section: Section) -> str:
		"""Retourne un résumé textuel d'une section."""
		...



# ---------------------------------------------------------------------------
# Implémentation 1 : résumé via LLM (map-reduce si section trop longue)
# ---------------------------------------------------------------------------

@dataclass
class LLMSectionSummarizer:
	"""
	Résume une Section via un LLM.

	Stratégie :
		- Si le texte de la section tient dans max_input_chars → résumé direct.
		- Sinon → découpage en fragments, résumé de chaque fragment (map),
		  puis résumé du résumé combiné (reduce).

	Paramètres :
		llm_client       : implémentation de LLMClientProtocol
		max_input_chars   : taille max de texte envoyée en un seul appel
		max_summary_tokens: budget de tokens pour chaque résumé généré
		max_retries       : nombre de tentatives en cas d'échec d'appel
		retry_backoff_sec : délai (secondes) entre tentatives, multiplié par le rang
		prompt_template   : template utilisé pour le résumé direct/map
		reduce_prompt_template : template utilisé pour l'étape reduce
	"""

	llm_client: LLMClientProtocol
	max_input_chars: int = rag_config.MAX_INPUT_CHAR # 6000
	max_summary_tokens: int = rag_config.MAX_REQUEST_TOKENS # 4000
	max_retries: int = rag_config.MAX_SUMMARY_ATTEMPTS # 2
	retry_backoff_sec: float = rag_config.MAX_TIMEOUT_IN_SEC # 1.5

	prompt_template: str = (
		"Résume le texte suivant, extrait du chapitre \"{title}\" "
		"(pages {start_page}-{end_page}) d'un document de jeu.\n"
		"Le résumé doit être factuel, concis (5 à 8 phrases maximum), "
		"et conserver les termes spécifiques importants.\n\n"
		"Texte :\n{text}"
	)

	reduce_prompt_template: str = (
		"Voici plusieurs résumés partiels du chapitre \"{title}\". "
		"Fusionne-les en un résumé unique, cohérent, sans répétition, "
		"en 6 à 10 phrases maximum.\n\n"
		"Résumés partiels :\n{text}"
	)

	def summarize(self, chapitre: Chapter) -> str:
		text = chapitre.full_text.strip()

		if not text:
			logger.warning(
				"Chapitre '%s' (pages %d-%d) vide — pas de résumé généré.",
				chapitre.toc_entry.title,
				chapitre.start_page,
				chapitre.end_page,
			)
			return ""

		if len(text) <= self.max_input_chars:
			prompt = self.prompt_template.format(
				title=chapitre.toc_entry.title,
				start_page=chapitre.start_page,
				end_page=chapitre.end_page,
				text=text,
			)
			return self._call_llm(prompt)

		# ── Map-reduce pour les chapitres trop longues ──────────────────
		logger.info(
			"Chapitre '%s' trop longue (%d caractères) — map-reduce activé.",
			chapitre.toc_entry.title,
			len(text),
		)
		fragments = self._split_text(text, self.max_input_chars)

		partial_summaries = []
		for i, fragment in enumerate(fragments):
			logger.debug(
				"  Résumé du fragment %d/%d (%d caractères)",
				i + 1, len(fragments), len(fragment),
			)
			prompt = self.prompt_template.format(
				title=chapitre.toc_entry.title,
				start_page=chapitre.start_page,
				end_page=chapitre.end_page,
				text=fragment,
			)
			partial_summaries.append(self._call_llm(prompt))

		combined = "\n\n".join(
			f"[Partie {i+1}] {s}" for i, s in enumerate(partial_summaries)
		)
		reduce_prompt = self.reduce_prompt_template.format(
			title=chapitre.toc_entry.title,
			text=combined,
		)
		return self._call_llm(reduce_prompt)

	def _call_llm(self, prompt: str) -> str:
		"""Appelle le LLM avec retry simple en cas d'échec."""
		last_error: Exception | None = None

		for attempt in range(1, self.max_retries + 2):
			try:
				result = self.llm_client.complete(
					prompt, max_tokens=self.max_summary_tokens
				)
				return result.strip()
			except Exception as exc:  # noqa: BLE001 — on veut logguer tout type d'échec
				last_error = exc
				logger.warning(
					"Échec d'appel LLM (tentative %d/%d) : %s",
					attempt, self.max_retries + 1, exc,
				)
				if attempt <= self.max_retries:
					time.sleep(self.retry_backoff_sec * attempt)

		logger.error("Résumé impossible après %d tentatives.", self.max_retries + 1)
		raise RuntimeError(f"Échec du résumé via LLM : {last_error}") from last_error

	@staticmethod
	def _split_text(text: str, max_chars: int) -> list[str]:
		"""Découpe le texte en fragments respectant les limites de paragraphes."""
		paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
		fragments = []
		current, current_len = [], 0

		for para in paragraphs:
			if current_len + len(para) > max_chars and current:
				fragments.append("\n\n".join(current))
				current, current_len = [], 0
			current.append(para)
			current_len += len(para)

		if current:
			fragments.append("\n\n".join(current))

		return fragments


# ---------------------------------------------------------------------------
# Implémentation 2 : résumé extractif local (sans LLM, sans dépendance externe)
# ---------------------------------------------------------------------------

@dataclass
class ExtractiveSectionSummarizer:
	"""
	Fallback sans API ni dépendance externe : sélectionne les phrases
	les plus représentatives du texte via un score de fréquence de mots.

	Utile quand :
		- aucun accès LLM n'est disponible,
		- on veut un résumé rapide/gratuit avant une passe LLM plus coûteuse,
		- on veut tester le pipeline de bout en bout sans dépendance externe.

	Ce n'est pas un résumé "intelligent" — c'est une sélection des phrases
	les plus denses en mots-clés du texte (méthode extractive classique,
	proche de TextRank simplifié).
	"""

	max_sentences: int = 5
	min_sentence_len: int = 20  # caractères, filtre les fragments trop courts

	# Mots vides français/anglais basiques, à étendre si besoin
	stopwords: frozenset[str] = field(
		default_factory=lambda: frozenset({
			"le", "la", "les", "de", "des", "du", "un", "une", "et", "en",
			"que", "qui", "dans", "pour", "sur", "avec", "par", "est",
			"sont", "au", "aux", "ce", "cette", "ces", "il", "elle", "ils",
			"elles", "à", "d", "l", "the", "a", "an", "of", "and", "to",
			"in", "for", "on", "with", "is", "are", "this", "that",
		})
	)

	def summarize(self, section: Section) -> str:
		text = section.full_text.strip()
		if not text:
			return ""

		sentences = self._split_sentences(text)
		sentences = [s for s in sentences if len(s) >= self.min_sentence_len]

		if len(sentences) <= self.max_sentences:
			return " ".join(sentences)

		scores = self._score_sentences(sentences)

		# Sélectionne les meilleures phrases en conservant l'ordre d'origine
		top_indices = sorted(
			range(len(sentences)),
			key=lambda i: scores[i],
			reverse=True,
		)[: self.max_sentences]

		top_indices.sort()  # remet dans l'ordre d'apparition dans le texte
		selected = [sentences[i] for i in top_indices]

		return " ".join(selected)

	def _score_sentences(self, sentences: list[str]) -> list[float]:
		"""
		Score chaque phrase par la somme des fréquences de ses mots
		significatifs (hors stopwords), normalisée par la longueur.
		"""
		word_freq: dict[str, int] = {}

		for sentence in sentences:
			for word in self._tokenize(sentence):
				if word not in self.stopwords and len(word) > 2:
					word_freq[word] = word_freq.get(word, 0) + 1

		if not word_freq:
			return [0.0] * len(sentences)

		max_freq = max(word_freq.values())
		normalized_freq = {w: f / max_freq for w, f in word_freq.items()}

		scores = []
		for sentence in sentences:
			words = self._tokenize(sentence)
			significant_words = [w for w in words if w in normalized_freq]

			if not significant_words:
				scores.append(0.0)
				continue

			raw_score = sum(normalized_freq[w] for w in significant_words)
			# Normalisation par longueur pour ne pas favoriser les phrases longues
			score = raw_score / (len(words) ** 0.5) if words else 0.0
			scores.append(score)

		return scores

	@staticmethod
	def _tokenize(text: str) -> list[str]:
		return re.findall(r"\b\w+\b", text.lower())

	@staticmethod
	def _split_sentences(text: str) -> list[str]:
		"""
		Découpage simple par ponctuation forte.
		Suffisant pour un fallback ; pour un usage plus robuste,
		on pourrait brancher un tokenizer type nltk/spacy (mais on
		évite volontairement d'ajouter cette dépendance ici).
		"""
		# Nettoyage des retours à la ligne multiples avant découpe
		cleaned = re.sub(r"\s+", " ", text).strip()
		raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ü])", cleaned)
		return [s.strip() for s in raw_sentences if s.strip()]


# ---------------------------------------------------------------------------
# Factory helper — sélection de l'implémentation selon la config
# ---------------------------------------------------------------------------

def build_summarizer(
	llm_client: LLMClientProtocol | None = None,
	**kwargs,
) -> LLMSectionSummarizer | ExtractiveSectionSummarizer:
	"""
	Retourne l'implémentation de summarizer adaptée selon la disponibilité
	d'un client LLM.

	Args:
		llm_client: client LLM à injecter (None → fallback extractif)
		**kwargs: paramètres additionnels passés à l'implémentation choisie

	Returns:
		Une instance conforme à SummarizerProtocol
	"""
	if llm_client is not None:
		logger.info("Summarizer sélectionné : LLMSectionSummarizer")
		return LLMSectionSummarizer(llm_client=llm_client, **kwargs)

	logger.info("Summarizer sélectionné : ExtractiveSectionSummarizer (fallback local)")
	return ExtractiveSectionSummarizer(**kwargs)


# ---------------------------------------------------------------------------
# Exemple d'implémentation de LLMClientProtocol (à titre illustratif)
# ---------------------------------------------------------------------------

class DummyEchoClient:
	"""
	Client factice pour tests unitaires — ne fait aucun appel réseau.
	Retourne les N premiers mots du prompt suivi d'une mention explicite.
	Sert uniquement à valider le câblage du pipeline sans dépendance API.
	"""

	def complete(self, prompt: str, max_tokens: int = 300) -> str:
		preview = " ".join(prompt.split()[:30])
		return f"[RÉSUMÉ SIMULÉ] {preview}..."


# ---------------------------------------------------------------------------
# Point d'entrée de test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
	from src.ingestion.old.rag_models import Chapter, TocEntry
	from src.llm.llm_client_ollama import LLMClientOllama as OllamaLLMClient

	logging.basicConfig(level=logging.INFO)

	# Section de test
	fake_toc = TocEntry(level=1, title="Chapitre Test", page_number=1, page_index=0)
	fake_section = Chapter(
		toc_entry=fake_toc,
		full_text=(
			"PyMuPDF est une bibliothèque Python permettant de lire et "
			"manipuler des fichiers PDF. Elle offre un accès direct aux "
			"métadonnées, au texte, aux images et à la structure du document. "
			"Contrairement à d'autres outils, elle extrait nativement la table "
			"des matières sans traitement additionnel. Son installation est "
			"simple via pip et ne nécessite aucune dépendance système lourde. "
			"Elle est largement utilisée dans les pipelines de traitement "
			"documentaire et de RAG pour sa rapidité et sa fiabilité."
		),
	)

	print("=== Test ExtractiveSectionSummarizer ===")
	extractive = ExtractiveSectionSummarizer(max_sentences=2)
	print(extractive.summarize(fake_section))

	print("\n=== Test LLMSectionSummarizer (avec client factice) ===")
	llm_summarizer = LLMSectionSummarizer(llm_client=DummyEchoClient())
	print(llm_summarizer.summarize(fake_section))

	print("\n=== Test LLMSectionSummarizer (avec client Ollama) ===")
	ollama_summarizer = LLMSectionSummarizer(llm_client=OllamaLLMClient(model=rag_config.OLLAMA_MODEL_SUMMARY, enable_thinking=False))
	print(ollama_summarizer.summarize(fake_section))