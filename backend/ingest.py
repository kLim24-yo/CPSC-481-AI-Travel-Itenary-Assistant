"""
backend/ingest.py
─────────────────
Loads the Paris travel guide PDF, splits it into chunks,
generates embeddings, and uploads them to Pinecone.

Run once (or whenever the PDF changes):
    python backend/ingest.py
"""

import os
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
INDEX_NAME         = os.getenv("PINECONE_INDEX_NAME", "travel-guide")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
PDF_PATH           = os.getenv("PDF_PATH", "data/paris_guide.pdf")
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", 50))
EMBEDDING_DIM      = 384   # fixed for all-MiniLM-L6-v2


def load_and_split(pdf_path: str):
    """Load PDF and split into overlapping chunks."""
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()
    print(f"   → {len(pages)} pages loaded")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"   → {len(chunks)} chunks created (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def get_or_create_index(pc: Pinecone):
    """Create the Pinecone index if it doesn't already exist."""
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index: '{INDEX_NAME}'")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("   → Index created ✓")
    else:
        print(f"Index '{INDEX_NAME}' already exists")
    return pc.Index(INDEX_NAME)


def upsert_chunks(index, chunks, embeddings_model):
    """Embed each chunk and upsert into Pinecone in batches."""
    batch_size = 100
    texts    = [c.page_content for c in chunks]
    metadatas = [{"source": c.metadata.get("source", ""), "page": c.metadata.get("page", 0)} for c in chunks]

    print(f"\nEmbedding {len(texts)} chunks with '{EMBEDDING_MODEL}' …")
    vectors = embeddings_model.embed_documents(texts)

    print(f"⬆Upserting to Pinecone in batches of {batch_size} …")
    for i in tqdm(range(0, len(vectors), batch_size)):
        batch = [
            {
                "id":       f"chunk-{i + j}",
                "values":   vectors[i + j],
                "metadata": {**metadatas[i + j], "text": texts[i + j]},
            }
            for j in range(min(batch_size, len(vectors) - i))
        ]
        index.upsert(vectors=batch)

    print(f"\nDone! {len(vectors)} vectors stored in index '{INDEX_NAME}'")


def main():
    if not PINECONE_API_KEY:
        raise EnvironmentError("PINECONE_API_KEY is not set. Copy .env.example → .env and fill it in.")

    # 1. Load & chunk PDF
    chunks = load_and_split(PDF_PATH)

    # 2. Set up Pinecone
    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = get_or_create_index(pc)

    # 3. Set up embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 4. Embed + upsert
    upsert_chunks(index, chunks, embeddings)


if __name__ == "__main__":
    main()
