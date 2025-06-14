from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from .config import settings
from .logger import log_chunks_to_file
from pathlib import Path


CHROMA_DIR = Path("data/vector_db")
SAMPLE_CVS_DIR = Path("data/sample_cvs")

embedding_model = OllamaEmbeddings(model=settings.EMBEDDING_MODEL)

def chunk_by_blank_lines(texts: list[str], metadatas: list[dict]) -> list[Document]:
    documents = []

    for text, metadata in zip(texts, metadatas):
        sections = [section.strip() for section in text.split("\n\n") if section.strip()]
        
        for section in sections:
            documents.append(Document(page_content=section, metadata=metadata))

    return documents


def get_cv_documents():
    texts = []
    metadatas = []

    for file in SAMPLE_CVS_DIR.glob("*.txt"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                texts.append(content)
                metadatas.append({"filename": file.name})
        except Exception as e:
            print(f"❌ Failed to read {file.name}: {e}")

    docs = chunk_by_blank_lines(texts, metadatas)
    log_chunks_to_file(docs)
    return docs



def create_chroma():
    try:
        vectorstore = Chroma(
            embedding_function=embedding_model,
            persist_directory=str(CHROMA_DIR)
        )

        # Fetch and delete all IDs manually
        all_ids = vectorstore._collection.get()["ids"]
        if all_ids:
            vectorstore._collection.delete(ids=all_ids)
            print(f"🧹 Deleted {len(all_ids)} record(s) from ChromaDB.")
        else:
            print("ℹ️ No records found to delete.")

    except Exception as e:
        print(f"⚠️ Could not load existing DB (may be empty): {e}")
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore = None

    # Load current documents
    documents = get_cv_documents()
    if not documents:
        print("⚠️ No valid CVs to embed.")
        return None

    # Rebuild ChromaDB with clean data
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(CHROMA_DIR)
    )
    print(f"✅ Rebuilt ChromaDB with {len(documents)} document(s).")

    return vectorstore

def load_chroma():
    return Chroma(
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR)
    )