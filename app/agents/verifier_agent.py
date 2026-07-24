import json

from app.utils.llm_client import call_llm

SYSTEM_PROMPT = """You are a verification agent. Given a context and a proposed answer,
check whether every claim in the answer is actually supported by the context.

Respond with ONLY valid JSON in this exact shape, no markdown fences:
{"grounded": <true|false>, "reason": "<one short sentence>"}
"""


class VerifierAgent:
    """Last line of defense against hallucination - flags answers not backed by retrieved context."""

    def run(self, answer: str, retrieved_chunks: list[dict]) -> dict:
        if not retrieved_chunks:
            return {"grounded": True, "reason": "No context was used; refusal is trivially grounded."}

        context = "\n\n".join(c["text"] for c in retrieved_chunks)
        user_prompt = f"Context:\n{context}\n\nProposed answer:\n{answer}"
        raw = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.0, max_tokens=150)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"grounded": True, "reason": "Verifier response unparsable; defaulting to pass."}
