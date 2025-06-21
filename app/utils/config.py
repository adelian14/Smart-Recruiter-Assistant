from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
    EMBEDDING_MODEL= os.getenv("EMBEDDING_MODEL")

settings = Settings()
