import gradio as gr
from chatbot import stream_answer
from summarizer import stream_summary
import os
import shutil
from pathlib import Path
from utils.parser import parse_multiple, structure_and_save
from utils.embedding import create_chroma

UPLOAD_DIR = Path("data/uploads")
CV_DIR = Path("data/sample_cvs")
uploaded_files = []
parsed_files = []

def stream_chat_interface(question, history):
    history = history or []
    history.append((question, "..."))
    yield history, history, ""

    response = ""
    for token in stream_answer(question):
        response += token
        history[-1] = (question, response)
        yield history, history, ""

def upload_and_process_files(files):
    global uploaded_files, parsed_files

    if not files:
        return "⚠️ No files uploaded."

    # Upload
    uploaded_files = []
    for f in files:
        filename = os.path.basename(f.name)
        file_path = UPLOAD_DIR / filename
        shutil.copy(f.name, file_path)
        uploaded_files.append(file_path)

    # Process
    parsed_files = parse_multiple(uploaded_files)
    if not parsed_files:
        return f"❌ Parsing failed."

    return f"🟡 Uploaded {len(uploaded_files)} file(s)."

def store_structured_files():
    upload_paths = list(UPLOAD_DIR.glob("*"))

    if not upload_paths:
        return "⚠️ No uploaded files found."

    parsed = parse_multiple(upload_paths)

    if not parsed:
        return "❌ Parsing failed for uploaded files."

    structured_paths = structure_and_save(parsed)

    if not structured_paths:
        return "❌ Structuring failed."

    return f"✅ Structured and saved {len(structured_paths)} CV(s) from uploaded files."

def store_to_vector_db():
    vectorstore = create_chroma()
    if vectorstore is None:
        return "❌ No documents found to embed."

    return f"✅ Stored {len(vectorstore._collection.get()['documents'])} Chunk(s) in ChromaDB."

def clear_uploads():
    global uploaded_files, parsed_files
    uploaded_files = []
    parsed_files = []

    upload_removed = 0
    processed_removed = 0

    for f in UPLOAD_DIR.glob("*"):
        try:
            f.unlink()
            upload_removed += 1
        except Exception as e:
            print(f"⚠️ Could not delete {f.name}: {e}")

    for f in CV_DIR.glob("*"):
        try:
            f.unlink()
            processed_removed += 1
        except Exception as e:
            print(f"⚠️ Could not delete {f.name}: {e}")

    upload_msg = f"🧹 Cleared {upload_removed} uploaded file(s)." if upload_removed else "ℹ️ No uploaded files to clear."
    processed_msg = f"🧹 Cleared {processed_removed} processed CV(s)." if processed_removed else "ℹ️ No processed CVs to clear."

    return upload_msg, processed_msg

def get_file_stems() -> list[str]:
    path = Path(CV_DIR)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Invalid directory: {CV_DIR}")

    stems = [file.stem for file in path.iterdir() if file.is_file()]
    return stems

def retrieve_candidate_context(candidate_name: str) -> str:
    file_path = Path(CV_DIR) / f"{candidate_name}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Context file not found for: {candidate_name}")
    
    return file_path.read_text(encoding="utf-8")


def stream_summary_response(candidate_name: str):
    # Step 1: Retrieve full context for the candidate
    cv_text = retrieve_candidate_context(candidate_name)  # ← assume this exists

    yield [(f"Summarizing {candidate_name} CV",'...')]
    response = ''
    # Step 3: Stream the answer
    for chunk in stream_summary(cv=cv_text):
        response += chunk
        yield [(f'{candidate_name} CV Summary',response)]
