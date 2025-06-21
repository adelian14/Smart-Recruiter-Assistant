import datetime
from pathlib import Path
from langchain_core.documents import Document
from app.utils.config import settings

PROMPT_LOG_FILE = settings.LOG_DIR / Path("debug_logs.txt")
CHUNKS_LOG_FILE = settings.LOG_DIR / Path("chunks_log.txt")

def log_debug(question: str, context: str):
    timestamp = datetime.datetime.now().isoformat()
    PROMPT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(PROMPT_LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n==== {timestamp} ====\n")
        log.write(f"Question:\n{question.strip()}\n\n")
        log.write("Retrieved Context:\n")
        log.write(context.strip() + "\n")
        
def log_chunks_to_file(documents: list[Document]):
    with open(CHUNKS_LOG_FILE, "w", encoding="utf-8") as f:
        for i, doc in enumerate(documents):
            fname = doc.metadata.get("candidate_name", f"doc_{i}")
            f.write(f"\n--- Chunk {i+1} from {fname} ---\n")
            f.write(doc.page_content + "\n")
            f.write(f"Length: {len(doc.page_content)} characters\n")
