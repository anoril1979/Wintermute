import sys
import os
import subprocess

def check_python_version():
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} détecté.")
    else:
        print(f"❌ Python {version.major}.{version.minor} trop ancien. Installez Python 3.10+.")

def check_libraries():
    required = [
        "langchain",
        "fastapi",
        "uvicorn",
        "pymupdf",
        "python_dotenv",
        "pydantics",
        "open_webui",
        "langchain",
        "langchain_ollama",
        "langchain_community",
        "langchain_core",
        "langchain_text_splitters",
        "langchain_chroma",
        "chromadb",
        "mineru"
    ]
    all_ok = True
    for lib in required:
        try:
            __import__(lib)
            print(f"✅ Bibliothèque '{lib}' trouvée.")
        except ImportError:
            print(f"❌ Bibliothèque '{lib}' manquante. Lancez : pip install {lib}")
            all_ok = False
    return all_ok

def check_ollama_running():
    try:
        import httpx
        response = httpx.get("http://localhost:11434", timeout=3)
        if response.status_code == 200:
            print("✅ Ollama est actif et répond sur le port 11434.")
            return True
    except Exception:
        pass
    print("❌ Ollama ne répond pas. Lancez 'ollama serve' dans un terminal séparé.")
    return False

def check_ollama_models():
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        output = result.stdout
        for model in ["llama3", "ministral", "qwen3", "qwen3-embedding", "nomic-embed-text"]:
            if model in output:
                print(f"✅ Modèle '{model}' disponible.")
            else:
                print(f"❌ Modèle '{model}' manquant. Lancez : ollama pull {model}")
    except FileNotFoundError:
        print("❌ Commande 'ollama' introuvable. Ollama est-il installé ?")
    except subprocess.TimeoutExpired:
        print("⚠️  Timeout lors de la vérification des modèles Ollama.")

def check_chroma_db():
    db_path = "./chroma_db"
    if os.path.exists(db_path) and os.listdir(db_path):
        print(f"✅ Base ChromaDB trouvée dans '{db_path}'.")
    else:
        print(f"⚠️  Aucune base ChromaDB trouvée. Lancez 'python ingest.py' après avoir ajouté des PDF.")

def check_documents_folder():
    docs_path = "./data/sources"
    if not os.path.exists(docs_path):
        print(f"⚠️  Dossier '{docs_path}' inexistant. C'est là que l'ingestion cherche les fichiers !")
    else:
        pdfs = [f for f in os.listdir(docs_path+"/pdf") if f.endswith(".pdf")]
        if pdfs:
            print(f"✅ {len(pdfs)} fichier(s) PDF trouvé(s) dans '{docs_path}'.")
        else:
            print(f"⚠️  Dossier '{docs_path}' vide. Ajoutez des PDF avant de lancer ingest.py.")

if __name__ == "__main__":
    print("=" * 50)
    print("  Diagnostic RAG Local Python")
    print("=" * 50)
    check_python_version()
    print()
    check_libraries()
    print()
    check_ollama_running()
    check_ollama_models()
    print()
    check_documents_folder()
    check_chroma_db()
    print()
    print("=" * 50)
    print("  Diagnostic terminé. Corrigez les ❌ avant de continuer.")
    print("=" * 50)
