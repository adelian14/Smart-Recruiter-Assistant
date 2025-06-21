from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import re
import pandas as pd
import math
from app.utils.config import settings

def read_cvs_from_directory() -> dict:
    directory = Path(settings.CVS_DIR)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Invalid directory: {settings.CVS_DIR}")
    
    cv_dict = {}
    for file in directory.glob("*.txt"):
        stem = file.stem
        text = file.read_text(encoding="utf-8")
        cv_dict[stem] = text
    return cv_dict

def preprocess(text: str) -> str:
    # Lowercase everything
    text = text.lower()
    
    # Replace underscores, dashes, commas with spaces
    text = re.sub(r"[_\-,]", " ", text)
    
    # Remove dots at the end of sentences (but not in e.g., ".net" or "node.js")
    text = re.sub(r"\.(?=\s|$)", "", text)  # removes . only if followed by whitespace or end of line

    # Keep only alphanumerics and relevant symbols (+, #, .) — discard the rest
    text = re.sub(r"[^a-z0-9+#. ]+", " ", text)

    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()

    return text

def tokenize_cv(text: str) -> list:
    return text.split()

def compute_tfidf_matrix(cv_dict: dict):
    cv_ids = list(cv_dict.keys())
    preprocessed = [preprocess(cv_dict[cv_id]) for cv_id in cv_ids]
    
    vectorizer = TfidfVectorizer(ngram_range=(1,4), use_idf=False, norm=None, tokenizer=tokenize_cv)
    tfidf_matrix = vectorizer.fit_transform(preprocessed)
    
    return vectorizer, tfidf_matrix, cv_ids

def compute_skill_score(tfidf_matrix, vectorizer, skill: str, cv_id: str, cv_ids: list) -> float:
    skill = skill.lower()
    if skill not in vectorizer.vocabulary_:
        return 0.0
    if cv_id not in cv_ids:
        raise ValueError(f"CV ID '{cv_id}' not found in the provided list.")

    index = vectorizer.vocabulary_[skill]
    row = cv_ids.index(cv_id)
    return float(tfidf_matrix[row, index])

def normalize_frequencies(freqs: list[int]) -> list[float]:
    log_scaled = []
    for f in freqs:
        if f <= 0:
            log_scaled.append(0.0)
        else:
            log_scaled.append(math.log(f + 1))

    max_val = max(log_scaled)
    if max_val == 0:
        return [0.0] * len(freqs)

    normalized = [round(val, 4) for val in log_scaled]
    return normalized

def skill_scoring(skill_input: str):
    skills = [s.strip() for s in skill_input.split(",") if s.strip()]
    skills = [preprocess(s) for s in skills]
    if not skills:
        return pd.DataFrame(columns=["Candidate"] + skills)

    cv_dict = read_cvs_from_directory()
    vectorizer, tfidf_matrix, cv_ids = compute_tfidf_matrix(cv_dict)

    raw_data = []
    for cv_id in cv_ids:
        row = [compute_skill_score(tfidf_matrix, vectorizer, skill, cv_id, cv_ids) for skill in skills]
        raw_data.append(row)

    df = pd.DataFrame(raw_data, columns=skills)
    df.insert(0, "Candidate", cv_ids)

    return df
