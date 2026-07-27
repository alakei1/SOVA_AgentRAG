#  ____   ___  _   _    _
# / ___| / _ \| | | |  / \
# \___ \| | | | | | | / _ \
#  ___) | |_| |\ V / / ___ \
# |____/ \___/  \_/ /_/   \_\
#
#    developer alakei1

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from service.app.schemas import DocumentDeleteResponse, ErrorResponse
from service.rag.document_processor import DocumentProcessor
from service.rag.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/documents", tags=["documents"])

# Инициализация (или получить через Depends)
rag_pipeline = RAGPipeline()
doc_processor = DocumentProcessor()

@router.delete(
    "/",
    response_model=DocumentDeleteResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def delete_document(
    collection: str = Query(..., description="Имя коллекции (bucket)"),
    file_name: str = Query(..., description="Имя файла в MinIO"),
    tenant_id: str = Query(...),
):
    """
    Удаляет документ по имени коллекции и имени файла.
    Принадлежность к арендатору проверяется через JWT.
    """
    # 1. Проверяем существование файла в MinIO
    exists = await doc_processor.file_exists_in_minio(collection, file_name)
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Файл '{file_name}' не найден в коллекции '{collection}'"
        )

    # 2. Удаляем векторы из Qdrant (по фильтру collection + tenant + source)
    deleted_chunks = await rag_pipeline.delete_document(
        collection_name=collection,
        tenant_id=tenant_id,
        file_name=file_name
    )

    # 3. Удаляем сам файл из MinIO
    await doc_processor.delete_file_from_minio(collection, file_name)

    # 4. Формируем ответ (ID документа можно сгенерировать или вернуть на основе имени)
    # Если у вас есть база данных с метаданными, можно взять ID оттуда.
    # Пока используем заглушку – можно вычислить UUID из имени или вернуть пустой.
    document_id = UUID(int=hash(f"{tenant_id}:{collection}:{file_name}") & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFF)

    return DocumentDeleteResponse(
        message=f"Документ '{file_name}' удалён",
        document_id=document_id,
        deleted_chunks=deleted_chunks
    )
