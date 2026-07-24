from app.utils.llm_client import call_llm

SYSTEM_PROMPT = """You answer questions using the reasoning notes and context provided.

If the reasoning notes contain ANY fact that answers the question - even a short one like a
name, date, or single detail - state that fact clearly and directly as the answer. Do not
refuse just because the notes also mention that further detail or elaboration is missing;
a short factual answer is still a valid answer.

Only reply with exactly "I don't know based on the uploaded documents." if the notes contain
NO fact that is actually relevant to the question.

Never use outside knowledge. Write a clear, direct answer for the end user - no bullet points
unless the question specifically asks for a list.
"""


class ResponderAgent:
    """Turns the Reasoner's notes into a final, user-facing answer."""

    def run(self, question: str, reasoning_notes: str, retrieved_chunks: list[dict]) -> str:
        if not retrieved_chunks:
            return "I don't know based on the uploaded documents."

        context = "\n\n".join(c["text"] for c in retrieved_chunks)
        user_prompt = (
            f"Question: {question}\n\n"
            f"Reasoning notes:\n{reasoning_notes}\n\n"
            f"Context:\n{context}"
        )
        return call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.2, max_tokens=512)
