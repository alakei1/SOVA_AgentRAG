from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, PositiveInt


# ---------- Запросы ----------
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Вопрос пользователя")
    collection: str = Field(..., description="Имя коллекции")
    tenant_id: str = Field(..., description="Идентификатор арендатора")
    top_k: PositiveInt = Field(5, le=20, description="Количество чанков для поиска")

# ---------- Ответы ----------
class Source(BaseModel):
    id: str = Field(..., description="ID точки в Qdrant")
    score: float = Field(..., ge=0, le=1, description="Оценка релевантности")
    source: str = Field(..., description="Исходный файл")
    text: str = Field(..., description="Текст чанка")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Ответ LLM")
    sources: list[Source] = Field(..., description="Использованные источники")
    context: str | None = Field(None, description="Полный контекст (опционально)")

    # Дополнительные метаданные
    total_chunks: int = Field(..., description="Количество возвращённых чанков")
    average_score: float = Field(..., ge=0, le=1, description="Средняя оценка релевантности среди всех источников")
    top_k_requested: int = Field(..., description="Запрошенное количество чанков (top_k)")
    processing_time_ms: float = Field(..., description="Время выполнения запроса в миллисекундах")

# ---------- Документы ----------
class Document(BaseModel):
    id: UUID = Field(..., description="ID документа")
    tenant_id: str
    collection: str
    file_name: str
    file_size: int
    uploaded_at: datetime
    status: str
    chunks_count: int | None = None
    error_message: str | None = None

class UploadResponse(BaseModel):
    id: UUID = Field(..., description="ID документа")
    message: str
    file: str
    collection: str
    status: str  # "processing" | "indexed"
    chunks: int | None = None

class DocumentStatusResponse(BaseModel):
    id: UUID
    status: str
    chunks: int | None = None
    error: str | None = None

class DocumentListResponse(BaseModel):
    documents: list[Document]
    total: int

class DocumentDeleteResponse(BaseModel):
    message: str
    document_id: UUID
    deleted_chunks: int

class ErrorResponse(BaseModel):
    detail: str
    status_code: int
