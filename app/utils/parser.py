import fitz
import docx2txt
from typing import List
from pathlib import Path
from app.utils.llm_extractor import get_candidate_name
from app.utils.config import settings
import re

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]

def parse_pdf(file_path: Path) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def parse_docx(file_path: Path) -> str:
    return docx2txt.process(file_path)

def parse_txt(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def clean_text(text: str) -> str:
    return ' '.join(text.strip().split())

def parse_cv(file_path: Path) -> dict:
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    if ext == ".pdf":
        raw_text = parse_pdf(file_path)
    elif ext == ".docx":
        raw_text = parse_docx(file_path)
    elif ext == ".txt":
        raw_text = parse_txt(file_path)
    else:
        raw_text = ""

    cleaned = clean_text(raw_text)
    return {
        "filename": file_path.name,
        "text": cleaned
    }

def parse_multiple(files: List[Path]) -> List[dict]:
    parsed_cvs = []
    for f in files:
        try:
            parsed = parse_cv(f)
            parsed_cvs.append(parsed)
        except Exception as e:
            print(f"❌ Failed to parse {f.name}: {e}")
    return parsed_cvs

def structure_and_save(parsed_cvs: List[dict]) -> List[Path]:
    saved_paths = []

    for parsed in parsed_cvs:
        try:
            candidate_name = get_candidate_name(parsed["text"])
            if not candidate_name:
                candidate_name = parsed['filename']
            parsed['candidate_name']=candidate_name
                
            out_name = Path(parsed['candidate_name']).stem + ".txt"
            out_path = settings.CVS_DIR / out_name

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(parsed['text'])

            saved_paths.append(out_path)

        except Exception as e:
            print(f"❌ Failed to structure {parsed['filename']}: {e}")

    return saved_paths
