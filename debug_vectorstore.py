"""
Debug script - run this from your project root (venv activated) to diagnose
why /ask-agent is returning 0 retrieved chunks.

Usage:cd
    python debug_vectorstore.py
"""

from app.services.vectorstore import get_vectorstore

QUESTION = "candidate's educational background"

store = get_vectorstore()

# 1. How many documents are actually in the store right now?
all_docs = store.get()
print(f"Total chunks currently stored: {len(all_docs['ids'])}")
if all_docs["metadatas"]:
    filenames = {m.get("filename") for m in all_docs["metadatas"]}
    print(f"Filenames present: {filenames}")
print()

# 2. What does a raw similarity search return, ignoring the relevance threshold?
print(f"Raw similarity search for: {QUESTION!r}\n")
results = store.similarity_search_with_score(QUESTION, k=5)

if not results:
    print("No results at all -> the store is empty or the collection name doesn't match.")
else:
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"#{rank}  distance={score:.4f}  filename={doc.metadata.get('filename')}")
        print(f"     text: {doc.page_content[:120]}...\n")

    print("If you see relevant-looking chunks above with distance >= 1.0,")
    print("the RELEVANCE_THRESHOLD in retriever_agent.py is filtering them out.")
    print("Try raising it (e.g. to 1.5) and re-test.")