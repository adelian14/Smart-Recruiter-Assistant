import pandas as pd
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import shutil
from pathlib import Path
from app.chatbot import stream_answer
from app.summarizer import stream_summary
from app.skill_assessor import skill_scoring
from app.utils.parser import parse_multiple, structure_and_save
from app.utils.embedding import create_chroma
from app.utils.config import settings

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

    uploaded_files = []
    for f in files:
        filename = os.path.basename(f.name)
        file_path = settings.UPLOAD_DIR / filename
        shutil.copy(f.name, file_path)
        uploaded_files.append(file_path)

    parsed_files = parse_multiple(uploaded_files)
    if not parsed_files:
        return f"❌ Parsing failed."

    return f"🟡 Uploaded {len(uploaded_files)} file(s)."

def store_structured_files():
    upload_paths = list(settings.UPLOAD_DIR.glob("*"))

    if not upload_paths:
        return "⚠️ No uploaded files found."

    parsed = parse_multiple(upload_paths)

    if not parsed:
        return "❌ Parsing failed for uploaded files."

    structured_paths = structure_and_save(parsed)

    if not structured_paths:
        return "❌ Structuring failed."

    return f"✅ Porcessed {len(structured_paths)} CV(s) from uploaded files."

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

    for f in settings.UPLOAD_DIR.glob("*"):
        try:
            f.unlink()
            upload_removed += 1
        except Exception as e:
            print(f"⚠️ Could not delete {f.name}: {e}")

    for f in settings.CVS_DIR.glob("*"):
        try:
            f.unlink()
            processed_removed += 1
        except Exception as e:
            print(f"⚠️ Could not delete {f.name}: {e}")

    upload_msg = f"🧹 Cleared {upload_removed} uploaded file(s)." if upload_removed else "ℹ️ No uploaded files to clear."
    processed_msg = f"🧹 Cleared {processed_removed} processed CV(s)." if processed_removed else "ℹ️ No processed CVs to clear."

    return upload_msg, processed_msg

def get_file_stems() -> list[str]:
    path = Path(settings.CVS_DIR)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Invalid directory: {settings.CVS_DIR}")

    stems = [file.stem for file in path.iterdir() if file.is_file()]
    return stems

def retrieve_candidate_context(candidate_name: str) -> str:
    file_path = Path(settings.CVS_DIR) / f"{candidate_name}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Context file not found for: {candidate_name}")
    
    return file_path.read_text(encoding="utf-8")

def stream_summary_response(candidate_name: str):
    # Retrieve full context for the candidate
    cv_text = retrieve_candidate_context(candidate_name)

    yield [(f"Summarizing {candidate_name} CV",'...')]
    response = ''
    # Stream the answer
    for chunk in stream_summary(cv=cv_text):
        response += chunk
        yield [(f'{candidate_name} CV Summary',response)]

def update_choices(skill_input: str):
    skills = [s.strip() for s in skill_input.split(",") if s.strip()]
    candidates = get_file_stems()
    return  gr.update(choices=skills, value=skills[0] if skills else None), gr.update(choices=candidates, value=candidates[0] if candidates else None)

def skill_scoring_interface_single_skill(skill_input: str, selected_skill: str):
    df = skill_scoring(skill_input)
    
    if df.empty or "Candidate" not in df.columns:
        return px.bar(title="No data available.")

    if selected_skill not in df.columns:
        return px.bar(title=f"No mentions of skill: {selected_skill}")


    df_plot = df[["Candidate", selected_skill]].copy()
    df_plot.sort_values(by=selected_skill, ascending=True, inplace=True)

    fig = px.bar(
        df_plot,
        x=selected_skill,
        y="Candidate",
        text=selected_skill,
        orientation="h",
        title=f"Mentions of {selected_skill} per Candidate",
        labels={selected_skill: "Mentions"},
    )

    fig.update_traces(
        marker_color="#4c78a8",
        textposition="outside"
    )

    fig.update_layout(
        height=max(400, 40 * len(df_plot)),
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="white"),
        xaxis_title="Mentions",
        yaxis_title="Candidate",
        margin=dict(t=60, b=60, l=40, r=40),
        xaxis=dict(tickformat=".0f")
    )
    
    return fig

def skill_scoring_interface_single_candidate(skill_input: str, selected_candidate: str):
    df = skill_scoring(skill_input)
    
    if df.empty or "Candidate" not in df.columns:
        return px.bar(title="No data available.")

    if selected_candidate not in df["Candidate"].values:
        return px.bar(title=f"No data for candidate: {selected_candidate}")

    # Extract data for the selected candidate
    row = df[df["Candidate"] == selected_candidate].squeeze()
    skills = row.index.drop("Candidate")
    freqs = row[skills].values

    df_plot = pd.DataFrame({
        "Skill": skills,
        "Mentions": freqs
    }).sort_values(by="Skill")  # Sort alphabetically or remove for original order

    # Vertical bar chart (skills on x-axis)
    fig = px.bar(
        df_plot,
        x="Skill",
        y="Mentions",
        text="Mentions",
        title=f"Skill Mentions for {selected_candidate}",
        labels={"Mentions": "Mentions", "Skill": "Skill"},
    )

    fig.update_traces(
        marker_color="#4c78a8",
        textposition="outside"
    )

    fig.update_layout(
        height=550,
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="white"),
        xaxis_title="Skill",
        yaxis_title="Mentions",
        margin=dict(t=60, b=100, l=40, r=40),  # extra bottom margin for long skill names
        xaxis=dict(tickangle=0),
        yaxis=dict(tickformat=".0f")
    )

    return fig

def update_candidate_choices():
    return gr.update(choices=get_file_stems(), value=None)
