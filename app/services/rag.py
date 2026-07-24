import os
from dotenv import load_dotenv
from groq import Groq

from app.services.vectorstore import get_vectorstore

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
_MODEL = "llama-3.1-8b-instant"


def answer_question(question: str, top_k: int = 4) -> dict:
    store = get_vectorstore()
    results_with_scores = store.similarity_search_with_score(question, k=top_k)
    RELEVANCE_THRESHOLD = 1.8  # lower distance = more similar
   
    results = [doc for doc, score in results_with_scores if score < RELEVANCE_THRESHOLD]

    if not results:
        return {
            "answer": "I don't know based on the uploaded documents.",
            "sources": [],
        }

    context = "\n\n".join(doc.page_content for doc in results)

    completion = _client.chat.completions.create(
        model=_MODEL,
        messages=[           
            {
                "role": "system",
                "content": (
                    "You answer questions strictly from the provided context. "
                    "If the context does not contain the answer, reply with exactly: "
                    "\"I don't know based on the uploaded documents.\" "
                    "Never use outside knowledge. Never output partial words or fragments."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        max_tokens=512,
        temperature=0.2,
    )
    answer = completion.choices[0].message.content

    sources = [
        {
            "filename": doc.metadata.get("filename"),
            "chunk_index": doc.metadata.get("chunk_index"),
            "preview": doc.page_content[:150],
        }
        for doc in results
    ]

    return {"answer": answer, "sources": sources}