# Streamlit chat UI for the travel assistant.

import os
from typing import Any, List, Optional

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from langchain.chains import RetrievalQA
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate

from ingest import build_or_load_vectorstore

load_dotenv()

st.set_page_config(page_title="Travel Assistant", page_icon="✈️", layout="wide")


HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# models that route through HF 
LLM_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
]
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_K = 4

# tells the LLM to only use info from the retrieved chunks
PROMPT = """You are a helpful travel assistant. Answer the question using ONLY
the context below from the loaded travel guide. If the answer isn't in the
context, say so.

For itinerary questions, format as Day 1 / Day 2 / ... with Morning,
Afternoon, Evening blocks. Include hours and prices where available.

Context:
{context}

Question: {question}

Answer:"""


class HFChatLLM(LLM):
    """LangChain LLM wrapper around huggingface_hub.InferenceClient.chat_completion.
    """

    model: str
    token: str
    temperature: float = 0.3
    max_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "huggingface-chat"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        client = InferenceClient(token=self.token)
        resp = client.chat_completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=max(self.temperature, 0.01),
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""


@st.cache_resource(show_spinner="Loading vector store...")
def get_vectorstore():
    return build_or_load_vectorstore()


@st.cache_resource(show_spinner="Connecting to LLM...")
def get_qa_chain(model_id, hf_token, temperature, top_k):
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": top_k})

    llm = HFChatLLM(
        model=model_id,
        token=hf_token,
        temperature=temperature,
        max_tokens=512,
    )

    prompt = PromptTemplate(template=PROMPT, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )


# sidebar — user-facing only
with st.sidebar:
    st.title("✈️ Travel Assistant")
    st.caption("Group Project #4")

    st.subheader("Sample questions")
    samples = [
        "Plan a 2-day itinerary.",
        "I'm traveling with kids — give me a one-day plan.",
        "Which restaurants are budget-friendly?",
        "What attractions are open today?",
        "Build a 3-day itinerary that ends with a fancy dinner.",
    ]
    sample_choice = st.radio("Pick one:", samples, index=None)

    # tucked away — most users won't need these
    with st.expander("Advanced settings"):
        model_id = st.selectbox("LLM", LLM_MODELS, index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.1)
        top_k = st.slider("Top-k chunks", 1, 8, DEFAULT_TOP_K, 1)

    st.divider()
    st.caption("Embeddings: all-MiniLM-L6-v2 · Vector store: FAISS")


# main pane
st.header("Plan your trip ✈️")
st.write("Ask anything from the loaded travel guide.")

# config check — shown only if the Space wasn't set up correctly
if not HF_TOKEN:
    st.error(
        "Server is missing the HF token. Set "
        "`HUGGINGFACEHUB_API_TOKEN` ."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for i, doc in enumerate(msg["sources"], 1):
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Source {i}** — page {page}")
                    st.text(doc.page_content)

typed = st.chat_input("Ask about your trip...")
user_input = typed or sample_choice

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                chain = get_qa_chain(model_id, HF_TOKEN, temperature, top_k)
                result = chain.invoke({"query": user_input})
                answer = result["result"].strip()
                sources = result.get("source_documents", [])
            except Exception as e:
                answer = f"⚠️ Model call failed: `{e}`"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for i, doc in enumerate(sources, 1):
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Source {i}** — page {page}")
                    st.text(doc.page_content)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
