"""
backend/rag.py
──────────────
Retrieval-Augmented Generation pipeline.
Accepts a user question, retrieves relevant chunks from Pinecone,
and returns an answer from the LLM.

Usage (from Python or Streamlit):
    from backend.rag import get_answer
    answer = get_answer("What should I do on Day 1 in Paris?")
"""

import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY")
INDEX_NAME        = os.getenv("PINECONE_INDEX_NAME", "travel-guide")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN          = os.getenv("HUGGINGFACE_API_TOKEN")

# Free HF LLM — swap for "mistralai/Mistral-7B-Instruct-v0.2" if preferred
LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"

# ── Prompt Template ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a knowledgeable Paris travel assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have that information in the travel guide."

Context:
{context}

Question: {question}

Answer (be specific and helpful):"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)

# ── Lazy singletons (loaded once per session) ─────────────────────────────────
_qa_chain = None


def _build_chain():
    """Build and cache the RAG chain."""
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Vector store retriever
    pc        = Pinecone(api_key=PINECONE_API_KEY)
    index     = pc.Index(INDEX_NAME)
    vectorstore = LangchainPinecone(index, embeddings, "text")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # LLM (Hugging Face Inference API — free tier)
    llm = HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
        temperature=0.3,
    )

    # Chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )
    return chain


def get_answer(question: str) -> dict:
    """
    Ask a question and get an answer + source chunks.

    Returns:
        {
            "answer": str,
            "sources": [{"text": str, "page": int}, ...]
        }
    """
    global _qa_chain
    if _qa_chain is None:
        _qa_chain = _build_chain()

    result = _qa_chain.invoke({"query": question})

    sources = [
        {
            "text": doc.page_content[:300],
            "page": doc.metadata.get("page", "?"),
        }
        for doc in result.get("source_documents", [])
    ]

    return {
        "answer":  result["result"].strip(),
        "sources": sources,
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    q = "What are the must-see attractions in Paris?"
    print(f"Q: {q}\n")
    r = get_answer(q)
    print(f"A: {r['answer']}\n")
    print("Sources:")
    for s in r["sources"]:
        print(f"  [Page {s['page']}] {s['text'][:100]}…")
