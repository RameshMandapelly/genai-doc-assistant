from app.utils.llm_client import call_llm

SYSTEM_PROMPT = """You are a reasoning agent. You are given a user question and a set of
retrieved document excerpts. Your job is NOT to write the final answer - it is to analyze
the excerpts and produce short bullet-point notes.

Rules:
- If any excerpt contains a fact that directly answers the question, state it clearly under
  "Facts extracted". A short, direct fact (e.g. a name, a date, a college) IS a sufficient
  answer on its own - do not also flag the context as insufficient just because no further
  detail or elaboration is given.
- Only write "Insufficient information" if NONE of the excerpts contain any fact relevant to
  the question. Never write both "Facts extracted" with a real answer AND "Insufficient
  information" about that same answer - that is a contradiction. Pick one.
- Flag "Contradiction" only if two excerpts state genuinely conflicting facts.

Do not use outside knowledge. Only reason over the text given to you.
Respond in plain text bullet points, no more than 6 bullets.
"""


class ReasonerAgent:
    """Forces an explicit 'what do these chunks actually say' step before the final answer is written."""

    def run(self, question: str, retrieved_chunks: list[dict]) -> str:
        if not retrieved_chunks:
            return "No relevant context was retrieved."

        context = "\n\n".join(
            f"[{c['filename']} #{c['chunk_index']}] {c['text']}" for c in retrieved_chunks
        )
        user_prompt = f"Question: {question}\n\nRetrieved excerpts:\n{context}"
        return call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.1, max_tokens=400)
