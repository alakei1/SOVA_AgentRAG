#  ____   ___  _   _    _
# / ___| / _ \| | | |  / \
# \___ \| | | | | | | / _ \
#  ___) | |_| |\ V / / ___ \
# |____/ \___/  \_/ /_/   \_\
#
#    developer alakei1

import os
import tempfile
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from service.core.document_processor import DocumentProcessor
from service.core.rag_pipeline import RAGPipeline

router = APIRouter()

# Инициализация (можно вынести в зависимости)
rag_pipeline = RAGPipeline()
doc_processor = DocumentProcessor()

@router.post("/upload")
async def upload_file(
    file: Annotated[UploadFile, File(...)],
    collection: Annotated[str, Form(...)],
    tenant_id:Annotated[str, Form(...)],
):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        if file.filename is None:
            raise HTTPException(
                status_code=400,
                detail="Имя файла отсутствует или не корректен "
                                )
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        success = await doc_processor.upload_file(
            bucket_name=collection,
            object_name=file.filename,
            file_path=tmp_path
        )
        if not success:
            raise HTTPException(status_code=500, detail="Upload to MinIO failed")

        chunks_count = await doc_processor.process_file_from_minio(
            bucket_name=collection,
            object_name=file.filename,
            collection_name=collection,
            tenant_id=tenant_id,
            rag_pipeline=rag_pipeline
        )
        return {
            "message": f"File processed, {chunks_count} chunks indexed",
            "file": file.filename,
            "collection": collection
        }
    finally:
        os.unlink(tmp_path)
