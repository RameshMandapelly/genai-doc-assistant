import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "documents"

_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
_store = None  # singleton, built lazily on first call


def get_vectorstore() -> Chroma:
    """
    Returns a single shared Chroma client for the whole process.

    IMPORTANT: this used to construct a *new* Chroma() on every call. That meant
    ingestion (add_chunks) and retrieval (RetrieverAgent, rag.py) each held their
    own separate client instance pointed at the same persist_directory - and those
    instances could go out of sync, so newly uploaded chunks weren't visible to a
    Retriever that was instantiated earlier in the process. Caching one instance
    here means every caller shares the exact same in-memory client, so writes are
    always immediately visible to reads.
    """
    global _store
    if _store is None:
        _store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=_embeddings,
            persist_directory=PERSIST_DIR,
        )
    return _store


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