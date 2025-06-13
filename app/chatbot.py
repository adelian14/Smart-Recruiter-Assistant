# app/chatbot.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from utils.config import settings
from utils.embedding import load_chroma  # ← Loads the vector store
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

def stream_answer(question: str):
    # Step 1: Retrieve documents
    docs: list[Document] = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Step 2: Generate response stream
    for chunk in chain.stream({"question": question, "context": context}):
        yield chunk.content
