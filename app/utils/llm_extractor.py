from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from .config import settings

llm = ChatOllama(
    model=settings.LLM_MODEL,
    temperature=0.1
)

SYSTEM_PROMPT = """You are a CV parsing agent. Given the raw text of a resume, extract the candidate's information into a structured format suitable for use in a retrieval-based system.

Follow these rules:
- Use clear section headers: Name, Email, Phone, LinkedIn, GitHub, Education, Experience, Skills, and Projects.
- Repeat the candidate's full name naturally in each section when applicable.
- Use bullet points for items within each section.
- Output only the parsed CV in the specified format — do not include any commentary, notes, explanations, or introductory/concluding text.

Output format:

Personal Information:
<full name>
<email address>
<phone number>
<linkedIn URL>
<github URL>
<full name> graduated from:
- <Degree>, <University>, <Year>
- ...
...

Experience:
<full name> worked at:
- <Role> at <Company> (<Years>)
- ...

Skills:
<full name> has the following skills:
- <Skill 1>, <Skill 2>, <Skill 3>, ...

Projects:
<full name> worked on the following projects:
- <Project Title>: <Brief Description>
- ...
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "{cv_text}")
])

chain = prompt | llm

def extract_structured_cv(cv_text: str) -> str:
    response = chain.invoke({"cv_text": cv_text})
    return response.content.strip()
