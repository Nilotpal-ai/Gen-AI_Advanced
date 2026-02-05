from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from loguru import logger
import uuid


class VectorStoreService:
    """
    Handles vector indexing and storage using Qdrant.
    """

    def __init__(
        self,
        collection_name: str,
        qdrant_url: str = "http://localhost:6333",
    ):
        self.collection_name = collection_name

        self.client = QdrantClient(
            url=qdrant_url,
            check_compatibility=False  # suppress version warning safely
        )

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vector_size = 384

        logger.info(
            "VectorStoreService initialized",
            extra={"collection": collection_name}
        )

    # ---------------------------------------------------------
    # Collection Management
    # ---------------------------------------------------------

    def create_collection(self) -> None:
        """
        Create Qdrant collection if it does not exist.
        Safe to call multiple times.
        """
        try:
            self.client.get_collection(self.collection_name)
            logger.info(
                "Qdrant collection already exists",
                extra={"collection": self.collection_name}
            )
        except Exception:
            logger.info(
                "Creating Qdrant collection",
                extra={"collection": self.collection_name}
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    # ---------------------------------------------------------
    # Chunking Logic
    # ---------------------------------------------------------

    @staticmethod
    def _chunk_text(
        text: str,
        chunk_size: int = 500,
        overlap: int = 100
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        """
        chunks = []
        start = 0
        length = len(text)

        while start < length:
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap

            if start < 0:
                start = 0

        return chunks

    # ---------------------------------------------------------
    # Indexing Logic
    # ---------------------------------------------------------

    def index_document(self, ingestion_data: Dict[str, Any]) -> None:
        """
        Index structured ingestion output into Qdrant.
        Supports QA-first and text fallback indexing.
        """

        texts: List[str] = []
        payloads: List[Dict[str, Any]] = []

        for page in ingestion_data["pages"]:
            page_number = page["page"]

            # ---- QA Pages ----
            if page["type"] == "qa":
                for qa in page["qa_pairs"]:
                    content = f"Q: {qa['question']} A: {qa['answer']}"

                    texts.append(content)
                    payloads.append({
                        "page": page_number,
                        "type": "qa",
                        "question": qa["question"],
                        "source": "pdf"
                    })

            # ---- Text Pages ----
            else:
                chunks = self._chunk_text(page["content"])

                for chunk in chunks:
                    texts.append(chunk)
                    payloads.append({
                        "page": page_number,
                        "type": "text",
                        "source": "pdf"
                    })

        if not texts:
            logger.warning("No content found to index")
            return

        # ---- Embedding ----
        logger.info(
            "Generating embeddings",
            extra={"chunks": len(texts)}
        )

        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True
        )

        # ---- Upsert into Qdrant ----
        points = []

        for idx, vector in enumerate(embeddings):
            points.append({
                "id": str(uuid.uuid4()),
                "vector": vector.tolist(),
                "payload": {
                    **payloads[idx],
                    "content": texts[idx]
                }
            })

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        logger.info(
            "Document indexed successfully",
            extra={
                "collection": self.collection_name,
                "points": len(points)
            }
        )
