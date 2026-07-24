from app.services.vectorstore import get_vectorstore

RELEVANCE_THRESHOLD = 1.8  # lower distance = more similar (matches rag.py's threshold)


class RetrieverAgent:
    """Executes the Planner's sub_queries against ChromaDB and returns a deduped, ranked chunk list."""

    def __init__(self):
        self.store = get_vectorstore()

    def run(self, plan: dict) -> list[dict]:
        if not plan.get("needs_retrieval", True):
            return []

        seen_ids = set()
        retrieved = []

        for sub_query in plan.get("sub_queries", []):
            results = self.store.similarity_search_with_score(
                sub_query, k=plan.get("top_k", 4)
            )
            for doc, score in results:
                if score >= RELEVANCE_THRESHOLD:
                    continue

                doc_id = f"{doc.metadata.get('filename')}_{doc.metadata.get('chunk_index')}"
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)

                retrieved.append(
                    {
                        "text": doc.page_content,
                        "filename": doc.metadata.get("filename"),
                        "chunk_index": doc.metadata.get("chunk_index"),
                        "score": score,
                        "matched_query": sub_query,
                    }
                )

        retrieved.sort(key=lambda r: r["score"])
        return retrieved
