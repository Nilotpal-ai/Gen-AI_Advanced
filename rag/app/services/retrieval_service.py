from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from loguru import logger


class RetrievalService:
    """
    Hybrid retrieval service:
    1. QA-first fuzzy matching
    2. Vector similarity fallback
    """

    def __init__(
        self,
        collection_name: str,
        qdrant_url: str = "http://localhost:6333"
    ):
        self.collection_name = collection_name

        self.client = QdrantClient(
            url=qdrant_url,
            check_compatibility=False
        )

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        logger.info(
            "RetrievalService initialized",
            extra={"collection": collection_name}
        )

    # ---------------------------------------------------------
    # QA-FIRST SEARCH (DETERMINISTIC)
    # ---------------------------------------------------------

    def _qa_first_search(
        self,
        query: str,
        threshold: int = 80
    ) -> List[Dict[str, Any]]:
        """
        Perform fuzzy matching against QA chunks only.
        """

        results: List[Dict[str, Any]] = []

        # Scroll through QA points (bounded, safe)
        scroll_result, _ = self.client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            limit=1000
        )

        for point in scroll_result:
            payload = point.payload or {}

            if payload.get("type") != "qa":
                continue

            question = payload.get("question", "")
            score = fuzz.token_set_ratio(
                query.lower(),
                question.lower()
            )

            if score >= threshold:
                results.append({
                    "score": score,
                    "content": payload.get("content"),
                    "page": payload.get("page"),
                    "type": "qa"
                })

        results.sort(key=lambda x: x["score"], reverse=True)

        logger.info(
            "QA-first retrieval completed",
            extra={"matches": len(results)}
        )

        return results

    # ---------------------------------------------------------
    # VECTOR FALLBACK SEARCH (SEMANTIC)
    # ---------------------------------------------------------

    def _vector_search(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search using embeddings.
        """

        query_vector = self.embedding_model.encode(query).tolist()

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )

        results: List[Dict[str, Any]] = []

        for hit in hits:
            payload = hit.payload or {}

            results.append({
                "score": hit.score,
                "content": payload.get("content"),
                "page": payload.get("page"),
                "type": payload.get("type")
            })

        logger.info(
            "Vector retrieval completed",
            extra={"matches": len(results)}
        )

        return results

    # ---------------------------------------------------------
    # PUBLIC RETRIEVAL METHOD (FASTAPI CALLS THIS)
    # ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval:
        1. QA-first fuzzy match
        2. Vector fallback
        """

        logger.info(
            "Retrieval started",
            extra={"query": query}
        )

        # Step 1: QA-first
        qa_results = self._qa_first_search(query)

        if qa_results:
            logger.info(
                "Returning QA-first results",
                extra={"count": len(qa_results)}
            )
            return qa_results[:top_k]

        # Step 2: Vector fallback
        logger.info("QA-first failed, using vector search")

        return self._vector_search(query, limit=top_k)
