from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

class Settings:
    def __init__(self):
        self.LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
        self.EMBEDDING_MODEL= os.getenv("EMBEDDING_MODEL")
        
        self.UPLOAD_DIR = Path("data/uploads")
        self.CVS_DIR = Path("data/txt_cvs")
        self.LOG_DIR = Path("data/logs")
        self.DB_DIR = Path("data/vector_db")
        
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.CVS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.DB_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
