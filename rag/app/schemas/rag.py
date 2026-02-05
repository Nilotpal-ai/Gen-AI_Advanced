from pydantic import BaseModel, Field

class RagQueryRequest(BaseModel):
    user_id: str = Field(..., example="discord_123")
    query: str = Field(..., example="What is motor insurance?")

class RagQueryResponse(BaseModel):
    answer: str
    latency_ms: float
