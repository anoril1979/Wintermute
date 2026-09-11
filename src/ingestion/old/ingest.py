import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# --- Configuration ---
DOSSIER_DOCS = "./documents"
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"

def charger_et_indexer():
    """Charge les PDF, les découpe et les indexe dans ChromaDB."""
    print(f"🔍 Chargement des fichiers depuis '{DOSSIER_DOCS}'...")

    if not os.path.exists(DOSSIER_DOCS) or not os.listdir(DOSSIER_DOCS):
        print("❌ Le dossier est vide ou inexistant. Ajoutez des PDF et relancez.")
        return

    loader = DirectoryLoader(DOSSIER_DOCS, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        print("❌ Aucun document PDF trouvé. Vérifiez le dossier.")
        return

    print(f"✅ {len(documents)} pages chargées depuis les PDF.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️  {len(chunks)} morceaux créés.")

    print("💾 Indexation dans ChromaDB (peut prendre quelques minutes)...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH,
        collection_name="mes_documents"
    )

    print(f"✅ Base de données créée dans '{DB_PATH}'. Prêt à interroger !")


if __name__ == "__main__":
    if not os.path.exists(DOSSIER_DOCS):
        os.makedirs(DOSSIER_DOCS)
        print(f"📁 Dossier '{DOSSIER_DOCS}' créé. Ajoutez vos PDF dedans et relancez.")
    else:
        charger_et_indexer()
