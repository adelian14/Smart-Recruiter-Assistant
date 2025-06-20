import gradio as gr
from chatbot import stream_answer
import os
import shutil
from pathlib import Path
from utils.parser import parse_multiple, structure_and_save
from utils.embedding import create_chroma
from utils.callbacks import stream_chat_interface, upload_and_process_files, store_structured_files, store_to_vector_db, clear_uploads, get_file_stems, stream_summary_response, UPLOAD_DIR, CV_DIR, uploaded_files, parsed_files

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CV_DIR.mkdir(parents=True, exist_ok=True)

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
        
        with gr.Tab("📄 CV Summarizer"):
            with gr.Row():
                candidate_dropdown = gr.Dropdown(
                    label="Select a Candidate",
                    choices=get_file_stems(),
                    interactive=True,
                    elem_id="candidate-dropdown"
                )
                summarize_btn = gr.Button("📝 Summarize")
                
            cv_summary_display = gr.Chatbot(
                label="Summary Output",
                height=500,
                elem_id="summary-display"
            )


            # Streamed response — same pattern as chatbot
            summarize_btn.click(
                fn=stream_summary_response,  # ← define later
                inputs=[candidate_dropdown],
                outputs=[cv_summary_display]
            )


if __name__ == "__main__":
    demo.launch()
