"""
Phase 3 — Embedding & FAISS Index

Reads cleaned chunks from Phase 2, embeds them with HuggingFace
all-MiniLM-L6-v2 (local, free), and stores in a FAISS vector index.

Usage:
    python build_index.py
"""

import json
import sys
import pickle

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

from config import (
    CHUNKS_PATH,
    DATA_DIR,
    HF_EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    BM25_INDEX_PATH,
)


def build_index():
    """Load chunks, embed with HuggingFace, build FAISS, and save to disk."""

    # Load chunks
    print("📖 Loading chunks...")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   {len(chunks)} chunks loaded")

    if not chunks:
        print("❌ No chunks found. Run chunk.py first.")
        sys.exit(1)

    # Convert to LangChain Document objects
    print("📄 Creating Document objects...")
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"],
        )
        documents.append(doc)

    # Embed with local HuggingFace model (free, no API key needed)
    print(f"🧮 Embedding {len(documents)} documents with {HF_EMBEDDING_MODEL}...")
    print("   (first run downloads ~80MB model, subsequent runs use cache)")
    embeddings = HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save to disk
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_DIR))

    # Stats
    index_files = list(FAISS_INDEX_DIR.iterdir())
    total_size = sum(f.stat().st_size for f in index_files if f.is_file())
    print(f"\n✅ FAISS index saved → {FAISS_INDEX_DIR}")
    print(f"   {len(documents)} vectors | {total_size / 1024 / 1024:.1f} MB on disk")
    print(f"   Files: {[f.name for f in index_files]}")

    print("\n📚 Building BM25 index (sparse retrieval)...")
    bm25_retriever = BM25Retriever.from_documents(documents)
    
    print(f"✅ Saving BM25 index → {BM25_INDEX_PATH}")
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)


if __name__ == "__main__":
    build_index()

