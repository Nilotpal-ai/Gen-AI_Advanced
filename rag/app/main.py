from fastapi import FastAPI
from app.api.routes import health, rag
from app.core.logging import setup_logging
from dotenv import load_dotenv
load_dotenv()


setup_logging()

app = FastAPI(title="RAG Backend")

app.include_router(
    health.router,
    prefix="/health",
    tags=["Health"]
)

app.include_router(
    rag.router,
    prefix="/api",
    tags=["RAG"]
)
