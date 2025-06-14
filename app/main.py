import gradio as gr
from chatbot import stream_answer
import os
import shutil
from pathlib import Path
from utils.parser import parse_multiple, structure_and_save
from utils.embedding import create_chroma


# === Paths ===
UPLOAD_DIR = Path("data/uploads")
CV_DIR = Path("data/sample_cvs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CV_DIR.mkdir(parents=True, exist_ok=True)

# === Global in-memory state ===
uploaded_files = []
parsed_files = []

# === Chatbot logic ===
def stream_chat_interface(question, history):
    history = history or []
    history.append((question, "..."))
    yield history, history, ""

    response = ""
    for token in stream_answer(question):
        response += token
        history[-1] = (question, response)
        yield history, history, ""

# === Step 1: Upload Files ===
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

# === Step 2: Structure + Save ===
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


# === Step 3: Store in vector db ===
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

# === UI ===
with gr.Blocks(title="Smart Recruiter Assistant", css="""
#status-row {
    display: flex;
    justify-content: space-between;
    margin-top: 1rem;
}
.status-box {
    width: 32%;
    font-size: 14px;
    border: 1px solid #ccc;
    padding: 10px;
    border-radius: 6px;
    min-height: 40px;
    text-align: center;
}
#input-row {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
}
#chat-input {
    flex: 0 0 83%;
    height: 42px;
    font-size: 16px;
}
#send-btn {
    flex: 0 0 15%;
    height: 42px;
}
""") as demo:

    gr.Markdown("## 🤖 Smart Recruiter Assistant")

    with gr.Tabs():
        # === Tab 1: Upload & Process ===
        with gr.Tab("📁 Upload & Process CVs"):
            file_upload = gr.File(
                label="Upload CVs",
                file_types=[".pdf", ".docx", ".txt"],
                file_count="multiple"
            )

            with gr.Row(elem_id="status-row"):
                upload_status = gr.Label(value="", label="Upload", elem_classes=["status-box"])
                process_status = gr.Label(value="", label="Process", elem_classes=["status-box"])
                store_status = gr.Label(value="", label="Storage", elem_classes=["status-box"])

            process_btn = gr.Button("🔄 Process Files")
            store_btn = gr.Button("🧠 Save to Database")
            clear_btn = gr.Button("🧹 Clear All", variant="stop")

            file_upload.upload(upload_and_process_files, inputs=file_upload, outputs=upload_status)
            process_btn.click(store_structured_files, inputs=None, outputs=process_status)
            store_btn.click(store_to_vector_db, inputs=None, outputs=store_status)
            clear_btn.click(clear_uploads, inputs=None, outputs=[upload_status,process_status])

        # === Tab 2: Chatbot ===
        with gr.Tab("💬 CV Chatbot"):
            chatbot = gr.Chatbot(height=600)
            with gr.Row(elem_id="input-row"):
                txt = gr.Textbox(
                    placeholder="Ask something like: Who graduated from Cairo University?",
                    show_label=False,
                    lines=1,
                    elem_id="chat-input",
                    container=False
                )
                send_btn = gr.Button("Send", elem_id="send-btn")

            state = gr.State([])

            send_btn.click(
                fn=stream_chat_interface,
                inputs=[txt, state],
                outputs=[chatbot, state, txt]
            )

            txt.submit(
                fn=stream_chat_interface,
                inputs=[txt, state],
                outputs=[chatbot, state, txt]
            )

if __name__ == "__main__":
    demo.launch()
