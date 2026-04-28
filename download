"""
frontend/app.py
───────────────
Streamlit UI for the AI Travel Itinerary Assistant.
Run with: streamlit run frontend/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from backend.rag import get_answer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Paris Travel Assistant",
    page_icon="🗼",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #fafaf8; }
    .stChatMessage { border-radius: 12px; }
    .source-box {
        background: #f0f4ff;
        border-left: 3px solid #4f6ef7;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.82rem;
        color: #444;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🗼 Paris Travel Assistant")
st.caption("Powered by RAG · LangChain · Pinecone · Hugging Face")
st.markdown("Ask me anything about visiting Paris — attractions, dining, itineraries, tips!")
st.divider()

# ── Suggested questions ───────────────────────────────────────────────────────
SUGGESTIONS = [
    "What should I do on Day 1 in Paris?",
    "What are the best restaurants in Paris?",
    "How much does the Eiffel Tower cost?",
    "Plan a family-friendly itinerary.",
    "What time does the Louvre open?",
]

st.markdown("**💡 Try asking:**")
cols = st.columns(len(SUGGESTIONS))
for col, suggestion in zip(cols, SUGGESTIONS):
    if col.button(suggestion, use_container_width=True):
        st.session_state["prefill"] = suggestion

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources from travel guide"):
                for s in msg["sources"]:
                    st.markdown(
                        f'<div class="source-box">📄 Page {s["page"]}: {s["text"]}…</div>',
                        unsafe_allow_html=True,
                    )

# ── Input ─────────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input("Ask about Paris…", key="chat_input") or prefill

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get AI answer
    with st.chat_message("assistant"):
        with st.spinner("Searching the travel guide…"):
            try:
                result = get_answer(user_input)
                answer  = result["answer"]
                sources = result["sources"]
            except Exception as e:
                answer  = f"⚠️ Error: {e}\n\nMake sure your `.env` is configured and `ingest.py` has been run."
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("📚 Sources from travel guide"):
                for s in sources:
                    st.markdown(
                        f'<div class="source-box">📄 Page {s["page"]}: {s["text"]}…</div>',
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
This assistant uses **RAG (Retrieval-Augmented Generation)**:
1. Your question is converted to an embedding
2. Similar chunks are retrieved from **Pinecone**
3. An LLM generates an answer grounded in the guide

**Tech Stack**
- 🦜 LangChain
- 🌲 Pinecone
- 🤗 Hugging Face
- ⚡ Streamlit
    """)

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()
