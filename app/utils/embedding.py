from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from .config import settings
from pathlib import Path
import shutil
import os
import re

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


def log_chunks_to_file(documents: list[Document], log_path="data/chunks_log.txt"):
    with open(log_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(documents):
            fname = doc.metadata.get("filename", f"doc_{i}")
            f.write(f"\n--- Chunk {i+1} from {fname} ---\n")
            f.write(doc.page_content + "\n")
            f.write(f"Length: {len(doc.page_content)} characters\n")


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



def create_or_update_chroma():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    documents = get_cv_documents()

    if not documents:
        print("⚠️ No documents found to embed.")
        return None

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(CHROMA_DIR)
    )

    print(f"✅ Stored {len(documents)} CVs in ChromaDB.")
    return vectorstore

def load_chroma():
    return Chroma(
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR)
    )

def delete_chroma_db():
    """
    1. Deletes all records from the default (or current) collection.
    2. Removes every subdirectory under CHROMA_DIR (these are the per-collection folders).
    3. Leaves CHROMA_DIR itself and its root-level DB/index files in place.
    """
    if not CHROMA_DIR.exists():
        print("ℹ️ ChromaDB directory does not exist.")
        return

    # Step 1 – clear the collection’s records
    try:
        vectorstore = Chroma(
            embedding_function=embedding_model,
            persist_directory=str(CHROMA_DIR)
        )
        vectorstore.delete_collection()
        print("🗑️ All ChromaDB records deleted.")
    except Exception as e:
        print(f"⚠️ Could not delete collection records: {e}")

    # Step 2 – remove every sub-folder (collections) but keep root DB files
    removed_dirs = 0
    for path in CHROMA_DIR.iterdir():
        if path.is_dir():
            try:
                shutil.rmtree(path, ignore_errors=True)
                removed_dirs += 1
            except Exception as e:
                print(f"⚠️ Failed to remove {path}: {e}")

    print(f"📁 Removed {removed_dirs} collection folder(s); root DB preserved.")




