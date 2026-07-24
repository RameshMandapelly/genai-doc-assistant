import logging

from app.agents.planner_agent import PlannerAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.reasoner_agent import ReasonerAgent
from app.agents.responder_agent import ResponderAgent
from app.agents.verifier_agent import VerifierAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Runs the Planner -> Retriever -> Reasoner -> Responder -> Verifier pipeline.

    Each stage is wrapped individually: if one agent's LLM call fails (rate limit,
    network blip, malformed response), the orchestrator logs it and falls back to
    a safe default for that stage instead of raising and 500-ing the whole request.
    The full trace (including which stages failed) is still returned, so failures
    are visible in the response rather than hidden.
    """

    def __init__(self):
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent()
        self.reasoner = ReasonerAgent()
        self.responder = ResponderAgent()
        self.verifier = VerifierAgent()

    def run(self, question: str) -> dict:
        trace = {}

        try:
            plan = self.planner.run(question)
        except Exception:
            logger.exception("Planner failed for question=%r; using fallback plan", question)
            plan = {"sub_queries": [question], "top_k": 4, "needs_retrieval": True}
            trace["planner_failed"] = True
        trace["plan"] = plan

        try:
            retrieved = self.retriever.run(plan)
        except Exception:
            logger.exception("Retriever failed for question=%r; continuing with no context", question)
            retrieved = []
            trace["retriever_failed"] = True
        trace["retrieved_count"] = len(retrieved)
        trace["retrieved"] = retrieved

        try:
            reasoning_notes = self.reasoner.run(question, retrieved)
        except Exception:
            logger.exception("Reasoner failed for question=%r", question)
            reasoning_notes = "Reasoning step failed; treating context as insufficient."
            trace["reasoner_failed"] = True
        trace["reasoning_notes"] = reasoning_notes

        try:
            answer = self.responder.run(question, reasoning_notes, retrieved)
        except Exception:
            logger.exception("Responder failed for question=%r", question)
            answer = "I don't know based on the uploaded documents."
            trace["responder_failed"] = True
        trace["draft_answer"] = answer

        try:
            verification = self.verifier.run(answer, retrieved)
        except Exception:
            logger.exception("Verifier failed for question=%r; defaulting to pass-through", question)
            verification = {"grounded": True, "reason": "Verifier step failed; defaulting to pass."}
            trace["verifier_failed"] = True
        trace["verification"] = verification

        if not verification.get("grounded", True):
            answer = (
                "I don't know based on the uploaded documents "
                "(the draft answer could not be verified against the retrieved context)."
            )
            trace["final_answer_overridden"] = True

        logger.info(
            "Pipeline complete: question=%r retrieved=%d grounded=%s",
            question, len(retrieved), verification.get("grounded"),
        )

        return {
            "answer": answer,
            "sources": [
                {
                    "filename": c["filename"],
                    "chunk_index": c["chunk_index"],
                    "preview": c["text"][:150],
                }
                for c in retrieved
            ],
            "trace": trace,
        }
