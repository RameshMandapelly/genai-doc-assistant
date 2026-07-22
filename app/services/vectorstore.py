import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "documents"

_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_vectorstore() -> Chroma:
    """Returns a Chroma vector store instance, persisted to disk."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings,
        persist_directory=PERSIST_DIR,
    )


def add_chunks(chunks: list[str], filename: str) -> int:
    """
    Embeds and stores text chunks in ChromaDB with metadata (filename, chunk index).
    Returns the number of chunks stored.
    """
    store = get_vectorstore()
    metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]
    ids = [f"{filename}_{i}" for i in range(len(chunks))]

    store.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
    return len(chunks)