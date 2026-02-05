from fastapi import APIRouter
from pydantic import BaseModel

from app.services.rag_service import RAGService
from app.services.llm_service import LLMService

router = APIRouter()

llm = LLMService()
rag = RAGService(
    collection_name="motor_insurance",
    llm=llm
)


class QueryRequest(BaseModel):
    query: str


@router.post("/rag/query")
def rag_query(request: QueryRequest):
    return rag.answer(request.query)
