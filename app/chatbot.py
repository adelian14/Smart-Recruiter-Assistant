# app/chatbot.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from utils.config import settings
from utils.embedding import load_chroma  # ← Loads the vector store
from utils.logger import log_debug
from langchain_core.documents import Document

llm = ChatOllama(
    model=settings.LLM_MODEL,
    temperature=0.2,
    streaming=True
)

vectorstore = load_chroma()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are a helpful recruitment assistant. Use the following candidate profiles to answer the question. Be precise and use only the provided context.

Context:
{context}

Question:
{question}
""")

# Chain
chain = prompt | llm

from collections import defaultdict

def stream_answer(question: str):
    # Step 1: Retrieve documents
    docs: list[Document] = retriever.invoke(question)

    # Step 2: Group chunks by candidate filename
    grouped_docs = defaultdict(list)
    for doc in docs:
        key = doc.metadata.get("filename", "unknown")
        grouped_docs[key].append(doc.page_content)

    # Step 3: Construct context block for each candidate
    context_blocks = []
    for candidate, texts in grouped_docs.items():
        block = "\n".join(texts)
        context_blocks.append(block)

    # Step 4: Join all context blocks clearly
    full_context = "\n\n---\n\n".join(context_blocks)
    
    log_debug(question, full_context)
    
    # Step 5: Stream the response from LLM
    for chunk in chain.stream({"question": question, "context": full_context}):
        yield chunk.content

