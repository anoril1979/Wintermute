import os
import logging
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# --- Configuration ---
DB_PATH = "./chroma_db"
LLM_MODEL = "qwen3"
EMBEDDING_MODEL = "nomic-embed-text"

PROMPT_SYSTEM = """Tu es un assistant documentaire. Utilise uniquement le contexte fourni pour répondre.
Si la réponse n'est pas dans le contexte, dis clairement que tu ne trouves pas cette information dans les documents.
Ne fabrique pas de données chiffrées ou de fait absent du contexte.
Contexte :
{context}"""

# --- Initialisation unique de la chaîne RAG ---
# La chaîne est construite une seule fois au démarrage du serveur,
# pas à chaque requête, pour éviter de recharger ChromaDB en boucle.
_rag_chain = None

def _initialiser_chaine() -> bool:
    """
    Charge ChromaDB et construit la chaîne RAG.
    Retourne True si l'initialisation a réussi, False sinon.
    """
    global _rag_chain

    if not os.path.exists(DB_PATH):
        logger.error(
            f"Base de données introuvable dans '{DB_PATH}' — "
            "retrieval stays dormant until an ingestion is completed."
        )
        return False

    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        vector_db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings,
            collection_name="mes_documents"
        )

        retriever = vector_db.as_retriever(search_kwargs={"k": 6})

        llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_SYSTEM),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        _rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        logger.info("Chaîne RAG initialisée avec succès.")
        return True

    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de la chaîne RAG : {e}")
        return False


def answer(question: str) -> str:
    """
    Point d'entrée principal appelé par api.py.
    Prend une question en texte, retourne la réponse du LLM enrichie du contexte RAG.
    """
    global _rag_chain

    # Initialisation paresseuse : si la chaîne n'est pas encore chargée
    if _rag_chain is None:
        success = _initialiser_chaine()
        if not success:
            logger.warning(
                "RAG chain dormant (no ChromaDB or Ollama down); answering in-band."
            )
            return (
                "⚠️ My retrieval memory is dormant: no document base is loaded "
                "right now. I can still talk, but for questions about your "
                "documents, an ingestion must be completed first — ask me to "
                "ingest one and try again afterwards."
            )

    try:
        response = _rag_chain.invoke({"input": question})
        answer_text = response.get("answer", "Aucune réponse générée.")

        # Construction du bloc sources (optionnel mais utile dans le chat)
        sources = []
        seen = set()
        for doc in response.get("context", []):
            src = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            key = f"{src}:{page}"
            if key not in seen:
                seen.add(key)
                sources.append(f"- Page {page} — {os.path.basename(src)}")

        if sources:
            sources_block = "\n\n📄 **Sources utilisées :**\n" + "\n".join(sources)
            return answer_text + sources_block

        return answer_text

    except Exception as e:
        logger.error(f"Erreur lors de la requête RAG : {e}")
        return (
            f"⚠️ Une erreur est survenue lors du traitement de ta question : {e}\n"
            "Vérifiez qu'Ollama est actif (`ollama serve`), puis réessaie."
        )