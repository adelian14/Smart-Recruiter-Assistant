# app/chatbot.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from utils.config import settings
from utils.embedding import load_chroma  # ← Loads the vector store
from utils.logger import log_debug
from langchain_core.documents import Document
from collections import defaultdict

llm = ChatOllama(
    model=settings.LLM_MODEL,
    temperature=0.2,
    streaming=True
)

vectorstore = load_chroma()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 15})

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are a skilled recruitment assistant. The context below contains partial excerpts from candidate CVs.

Your task is to answer the question using **only** the information in the context. Be specific and factual.

Include any relevant supporting details found in the context to make the answer clear and informative. However, do **not** refer to the context or mention that the answer is based on the context.

Avoid filler phrases. Just state the facts plainly. If the information is missing, say so clearly.

Context:
{context}

Question:
{question}

Answer:
""")



# Chain
chain = prompt | llm

def stream_answer(question: str):
    # Step 1: Retrieve documents
    docs: list[Document] = retriever.invoke(question)

    # Step 2: Group chunks by candidate filename
    grouped_docs = defaultdict(list)
    for doc in docs:
        key = doc.metadata.get("candidate_name", "unknown")
        grouped_docs[key].append(doc.page_content)

    # Step 3: Construct context block for each candidate
    full_context = ''
    for candidate, texts in grouped_docs.items():
        block = "\n".join(texts)
        full_context = full_context + '\n' +f'------The following chunk belongs to {candidate}------' + '\n'
        full_context = full_context + block + '\n\n'
    
    log_debug(question, full_context)
    
    # Step 5: Stream the response from LLM
    for chunk in chain.stream({"question": question, "context": full_context}):
        yield chunk.content

