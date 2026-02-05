from app.services.vector_store_service import VectorStoreService
from app.services.ingestion_service import DocumentIngestionService

ingestor = DocumentIngestionService("Motor_Insurance_Handbook.pdf")
data = ingestor.ingest()

store = VectorStoreService("motor_insurance")
store.create_collection()
store.index_document(data)
