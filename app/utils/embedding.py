from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from app.utils.config import settings
from app.utils.logger import log_chunks_to_file
from pathlib import Path

embedding_model = OllamaEmbeddings(model=settings.EMBEDDING_MODEL)

def chunk_by_blank_lines(texts: list[str], metadatas: list[dict]) -> list[Document]:
    documents = []

    for text, metadata in zip(texts, metadatas):
        sections = [section.strip() for section in text.split("\n\n") if section.strip()]
        
        for section in sections:
            documents.append(Document(page_content=section, metadata=metadata))

    return documents

def chunk_cvs(texts: list[str], metadatas: list[dict], chunk_size: int = 50, overlap: int = 10) -> list[Document]:
    assert len(texts) == len(metadatas), "texts and metadatas must have the same length"
    assert 0 <= overlap < chunk_size, "overlap must be non-negative and smaller than chunk_size"

    def split_into_chunks(text: str) -> list[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            if len(words) - start < chunk_size // 2 and chunks:
                chunks[-1].extend(words[start:])
                break
            chunk = words[start:end]
            chunks.append(chunk)
            if end >= len(words): break
            start += chunk_size - overlap
        return [' '.join(chunk) for chunk in chunks]

    all_docs = []
    for text, metadata in zip(texts, metadatas):
        for chunk in split_into_chunks(text):
            print(metadata)
            all_docs.append(Document(page_content=chunk, metadata=metadata))
    
    return all_docs

def get_cv_documents():
    texts = []
    metadatas = []

    for file in settings.CVS_DIR.glob("*.txt"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                texts.append(content)
                metadatas.append({'candidate_name':Path(file.name).stem})
        except Exception as e:
            print(f"❌ Failed to read {file.name}: {e}")

    docs = chunk_cvs(texts, metadatas)
    log_chunks_to_file(docs)
    return docs

def create_chroma():
    try:
        vectorstore = Chroma(
            embedding_function=embedding_model,
            persist_directory=str(settings.DB_DIR)
        )

        all_ids = vectorstore._collection.get()["ids"]
        if all_ids:
            vectorstore._collection.delete(ids=all_ids)
            print(f"🧹 Deleted {len(all_ids)} record(s) from ChromaDB.")
        else:
            print("ℹ️ No records found to delete.")

    except Exception as e:
        print(f"⚠️ Could not load existing DB (may be empty): {e}")
        settings.DB_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore = None

    documents = get_cv_documents()
    if not documents:
        print("⚠️ No valid CVs to embed.")
        return None

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(settings.DB_DIR)
    )
    print(f"✅ Rebuilt ChromaDB with {len(documents)} document(s).")

    return vectorstore

def load_chroma():
    return Chroma(
        embedding_function=embedding_model,
        persist_directory=str(settings.DB_DIR)
    )

