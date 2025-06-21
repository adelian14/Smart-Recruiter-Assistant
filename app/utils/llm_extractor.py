from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from app.utils.config import settings

llm = ChatOllama(
    model=settings.LLM_MODEL,
    temperature=0.1
)

prompt = ChatPromptTemplate.from_template("""
You will be given the text of a candidate's CV.

Extract **only** the candidate's full name, as it appears in the CV. Do **not** include any labels, explanations, punctuation, or additional text. Return only the name.

CV:
{cv_text}

Name:
""")


chain = prompt | llm

def get_candidate_name(cv_text: str) -> str:
    response = chain.invoke({"cv_text": cv_text})
    return response.content.strip()
