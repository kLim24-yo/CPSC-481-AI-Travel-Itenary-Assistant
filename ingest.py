# Builds the FAISS vector store from the Paris travel guide PDF.

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "data" / "paris_guide.pdf"
INDEX_DIR = ROOT / "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_and_chunk_pdf(pdf_path=PDF_PATH):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # one Document per page
    pages = PyPDFLoader(str(pdf_path)).load()

    # split into smaller overlapping chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    # tag chunks so we can show the source in the UI
    for c in chunks:
        c.metadata["source"] = pdf_path.name
    return chunks


def build_vectorstore(force_rebuild=False):
    embeddings = get_embeddings()

    if INDEX_DIR.exists() and not force_rebuild:
        return FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )

    print(f"Loading PDF: {PDF_PATH}")
    chunks = load_and_chunk_pdf(PDF_PATH)
    print(f"Got {len(chunks)} chunks")

    print(f"Embedding with {EMBEDDING_MODEL}...")
    vs = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(INDEX_DIR))
    print(f"Saved index to {INDEX_DIR}")
    return vs


def build_or_load_vectorstore():
    return build_vectorstore(force_rebuild=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rebuild even if index exists")
    args = parser.parse_args()

    vs = build_vectorstore(force_rebuild=args.force)
    print(f"Index has {vs.index.ntotal} vectors.")
    print("Done.")
