import gradio as gr
from app.utils.callbacks import stream_chat_interface, upload_and_process_files, store_structured_files, store_to_vector_db, clear_uploads, stream_summary_response, update_choices, skill_scoring_interface_single_skill, skill_scoring_interface_single_candidate, update_candidate_choices
        

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
""") as app:

    gr.Markdown("## 🤖 Smart Recruiter Assistant")

    with gr.Tabs():
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
        
        with gr.Tab("📄 CV Summarizer") as summarizer_tab:
            with gr.Row():
                candidate_dropdown = gr.Dropdown(
                    label="Select a Candidate",
                    choices=[],
                    interactive=True,
                    elem_id="candidate-dropdown",
                    scale=10
                )
                summarize_btn = gr.Button("📝 Summarize",scale=2)

                
            cv_summary_display = gr.Chatbot(
                label="Summary Output",
                height=500,
                elem_id="summary-display"
            )
            
            summarizer_tab.select(
                fn=update_candidate_choices,
                inputs=None,
                outputs=candidate_dropdown
            )
            
            summarize_btn.click(
                fn=stream_summary_response,
                inputs=[candidate_dropdown],
                outputs=[cv_summary_display]
            )
        
        with gr.Tab("📊 Skill Assessor"):
            
            gr.Markdown("### Estimate Skill Relevance for Each CV")
            with gr.Row():
                skill_input = gr.Textbox(
                    label="Enter Skills (comma-separated)", 
                    placeholder="e.g., Python, TensorFlow, Docker",
                    scale=10
                )
                score_button = gr.Button("Assess Candidate Skills", scale=2)
                
            with gr.Tab('🧠 Skill Focus View'):
                skill_selector = gr.Dropdown(choices=[], label="Select Skill")
                plot_output = gr.Plot()
                
                skill_selector.change(
                    fn=skill_scoring_interface_single_skill,
                    inputs=[skill_input,skill_selector],
                    outputs=plot_output
                )
            with gr.Tab('👤 Candidate Focus View'):
                candidate_dropdown_skills = gr.Dropdown(label="Select a Candidate",choices=[])
                plot_output_candidate = gr.Plot()
                
                candidate_dropdown_skills.change(
                    fn = skill_scoring_interface_single_candidate,
                    inputs=[skill_input, candidate_dropdown_skills],
                    outputs=plot_output_candidate
                )
                
            score_button.click(
                fn=update_choices,
                inputs=skill_input,
                outputs=[skill_selector, candidate_dropdown_skills]
            )

            gr.Markdown("""
            #### 📌 How to Interpret the Bar Chart

            - The chart shows **how often each candidate mentioned a specific skill**.
            - Use the skill dropdown to select one skill at a time.
            - Bars represent candidates; the length of the bar shows mention frequency.
            - The count is based on exact and related keyword appearances in the CV.

            This is helpful for **quickly spotting which candidates emphasize certain skills**.
            """)

            gr.Markdown("""
            #### ⚠️ Disclaimer
            This scoring system uses traditional NLP techniques (not deep learning) and relies on term frequency patterns in the CV text.

            It **does not guarantee actual proficiency** or intent. A high score means the skill is mentioned more prominently — not necessarily that the candidate is highly skilled in it.

            Always combine automated scores with human judgment.
            """, elem_classes=["text-xs", "text-gray-500"])


            
            