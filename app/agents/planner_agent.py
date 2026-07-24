import json

from app.utils.llm_client import call_llm

SYSTEM_PROMPT = """You are a planning agent for a document question-answering system.
Given a user's question, break it down into a short retrieval plan.

Respond with ONLY valid JSON, no markdown fences, no explanation, in this exact shape:
{
  "sub_queries": ["<one or more focused search queries derived from the question>"],
  "top_k": <int between 2 and 6>,
  "needs_retrieval": <true|false>
}

Rules:
- If the question is a simple greeting or not answerable from documents (e.g. "hello",
  "what can you do"), set needs_retrieval to false and sub_queries to [].
- If the question has multiple parts (e.g. "compare X and Y"), create one sub_query per part.
- Otherwise return a single sub_query equal to a cleaned-up version of the question.
"""


class PlannerAgent:
    """Decides *what* to search for and *how many* results to pull, before any retrieval happens."""

    def run(self, question: str) -> dict:
        raw = call_llm(SYSTEM_PROMPT, question, temperature=0.0, max_tokens=300)

        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            # Model didn't return clean JSON -> fall back to a safe default plan
            # rather than failing the whole pipeline.
            plan = {"sub_queries": [question], "top_k": 4, "needs_retrieval": True}

        plan.setdefault("sub_queries", [question])
        plan.setdefault("top_k", 4)
        plan.setdefault("needs_retrieval", True)
        return plan
