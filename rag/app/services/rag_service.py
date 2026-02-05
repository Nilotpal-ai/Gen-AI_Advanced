from typing import Dict, Any
from loguru import logger

from app.services.retrieval_service import RetrievalService


class RAGService:
    """
    RAG Answering Service:
    - Uses retrieval results
    - Grounds LLM strictly in context
    """

    def __init__(
        self,
        collection_name: str,
        llm
    ):
        self.retriever = RetrievalService(collection_name)
        self.llm = llm

        logger.info("RAGService initialized")

    # ---------------------------------------------------------
    # PROMPT (STRICTLY GUARDED)
    # ---------------------------------------------------------

    @staticmethod
    def _build_prompt(context: str, question: str) -> str:
        return f"""
You are an insurance assistant.

RULES:
- Answer ONLY using the context below.
- If the answer is not present, say:
  "The provided document does not contain this information."
- Do NOT use prior knowledge.
- Do NOT guess.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    # ---------------------------------------------------------
    # PUBLIC RAG METHOD
    # ---------------------------------------------------------

    def answer(self, query: str):
        logger.info("RAG answering started", extra={"query": query})

        retrieved_chunks = self.retriever.retrieve(query)

        if not retrieved_chunks:
            return {
                "answer": "The provided document does not contain this information.",
                "sources": []
            }

        # 🔥 SHORT-CIRCUIT FOR QA
        top_chunk = retrieved_chunks[0]

        if top_chunk["type"] == "qa":
            content = top_chunk["content"]

            # Extract answer from "Q: ... A: ..."
            if "A:" in content:
                answer = content.split("A:", 1)[1].strip()
            else:
                answer = content.strip()

            logger.info("Returned extractive QA answer")

            return {
                "answer": answer,
                "sources": [
                    {
                        "page": top_chunk["page"],
                        "type": "qa"
                    }
                ]
            }

        # ---------- LLM PATH (TEXT CHUNKS ONLY) ----------

        context_text = "\n\n".join(
            chunk["content"] for chunk in retrieved_chunks
        )

        prompt = self._build_prompt(context_text, query)

        response = self.llm(prompt)

        return {
            "answer": response,
            "sources": [
                {
                    "page": c["page"],
                    "type": c["type"]
                }
                for c in retrieved_chunks
            ]
        }

