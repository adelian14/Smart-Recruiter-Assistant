from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from app.utils.config import settings
from app.utils.logger import log_debug

llm = ChatOllama(
    model=settings.LLM_MODEL,
    temperature=0.27,
    streaming=True
)

prompt = ChatPromptTemplate.from_template("""
You are an expert recruitment assistant. Read the full CV of a candidate and generate a professional, well-structured summary.

Begin with a paragraph-style summary that narratively summarizes the candidate's background, education, experience, and professional focus. Avoid copying text directly from the CV — instead, distill and rephrase the information in a natural, human tone.

Then, present the following structured sections. Provide your answer right away without any leading or trailing text:

---

**Skills**  
List the candidate's technical and soft skills as bullet points, grouped naturally if possible (e.g., Programming Languages, Frameworks, Tools, etc.).

**Extracted Insights**  
Highlight meaningful observations from the CV in two parts:
- **Strengths**: What stands out? What is the candidate good at?
- **Areas for Improvement**: Any gaps, weak spots, or missing experiences that could be developed?

---

CV:
{cv}

Summary:
""")

summary_chain = prompt | llm

def stream_summary(cv: str):
    log_debug("Summarizing CV", cv[:500])
    for chunk in summary_chain.stream({"cv": cv}):
        yield chunk.content
