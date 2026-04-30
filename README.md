---
title: Travel Assistant
emoji: ✈️
colorFrom: blue
colorTo: pink
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
python_version: "3.11"
pinned: false
license: mit
---

# Travel Assistant

RAG app that answers travel questions using a PDF guide as the
knowledge base. Currently loaded with a Paris guide; swap in any
travel PDF to point it elsewhere. Built for Group Project #4.

## Components

- **Frontend** — Streamlit (`app.py`)
- **Backend** — LangChain `RetrievalQA` chain
- **Vector DB** — FAISS, built from `data/paris_guide.pdf`
- **Hosting** — Hugging Face Spaces

Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
LLM: Hugging Face Inference API (Mistral-7B-Instruct by default).

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # paste your HF token
python ingest.py                   # optional, builds the index
streamlit run app.py
```

First run downloads the embedding model (~90 MB) and builds the FAISS
index. Takes about a minute. After that it's instant.

## Deploy to Hugging Face Spaces

1. Create a new Streamlit Space at huggingface.co/new-space.
2. In **Settings → Variables and secrets**, add a **Secret** called
   `HUGGINGFACEHUB_API_TOKEN` (your read-token). The app reads it
   server-side; it is never displayed in the UI.
3. Push everything in this folder (including `data/paris_guide.pdf`):
   ```
   git remote add space https://huggingface.co/spaces/<user>/travel-assistant
   git push space main
   ```
4. First build takes a few minutes.

The YAML at the top of this file tells Spaces to use the Streamlit SDK.

## Loading a different guide

Drop your own travel PDF into `data/`, update the filename in
`ingest.py` (`PDF_PATH = ...`), delete the `faiss_index/` folder, and
restart. The app re-indexes on first request.

## Sample questions

- Plan a 2-day itinerary.
- I'm traveling with kids — give me a one-day plan.
- Which restaurants are budget-friendly?
- What attractions are open today?
- Build a 3-day itinerary that ends with a fancy dinner.

## Files

```
travel-assistant/
├── app.py              # Streamlit UI + RAG chain
├── ingest.py           # PDF → chunks → FAISS
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
└── data/paris_guide.pdf
```
