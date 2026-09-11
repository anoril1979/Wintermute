import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration ---
DB_PATH = "./chroma_db"
LLM_MODEL = "qwen3"
EMBEDDING_MODEL = "nomic-embed-text"

PROMPT_SYSTEM = """Tu es un assistant documentaire. Utilise uniquement le contexte fourni pour répondre.
Si la réponse n'est pas dans le contexte, dis clairement que tu ne trouves pas cette information dans les documents.
Ne fabrique pas de données chiffrées ou de faits absents du contexte.

Contexte :
{context}"""

def creer_chaine_rag():
    """Charge la base vectorielle existante et construit la chaîne RAG."""
    if not os.path.exists(DB_PATH):
        print(f"❌ Base de données introuvable dans '{DB_PATH}'.")
        print("Lancez d'abord 'python ingest.py' pour indexer vos documents.")
        return None

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="mes_documents"
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_SYSTEM),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


if __name__ == "__main__":
    chain = creer_chaine_rag()
    if chain:
        print("\n🤖 Assistant prêt. Posez vos questions (tapez 'exit' pour quitter).")
        while True:
            query = input("\n❓ Question : ").strip()
            if query.lower() in ["exit", "quit", "quitter"]:
                break
            if not query:
                continue

            try:
                response = chain.invoke({"input": query})
                print(f"\n💡 Réponse :\n{response['answer']}")
                print("\n📄 Sources utilisées :")
                seen = set()
                for doc in response.get("context", []):
                    src = doc.metadata.get("source", "?")
                    page = doc.metadata.get("page", "?")
                    key = f"{src}:{page}"
                    if key not in seen:
                        seen.add(key)
                        print(f"  - Page {page} du fichier {os.path.basename(src)}")
            except Exception as e:
                print(f"\n❌ Erreur lors de la requête : {e}")
                print("Vérifiez qu'Ollama est actif (lancez 'ollama serve' si nécessaire).")
