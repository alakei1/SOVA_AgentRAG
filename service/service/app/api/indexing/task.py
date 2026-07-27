from celery import Celery

from service.rag.document_processor import DocumentProcessor
from service.rag.rag_pipeline import RAGPipeline
from service.utils.config import settings

celery_app = Celery('tasks', broker=settings.REDIS_URL)

@celery_app.task
def process_file(bucket_name, object_name, collection_name, tenant_id):
    rag = RAGPipeline()
    doc = DocumentProcessor()
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(
        doc.process_file_from_minio(bucket_name, object_name, collection_name, tenant_id, rag)
    )
    return result
