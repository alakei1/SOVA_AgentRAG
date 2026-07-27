import io
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from miniopy_async.error import MinioException

from service.rag.document_processor import DocumentProcessor

# -------------------- Фикстуры --------------------


@pytest.fixture
def mock_minio():
    """Создаёт мок-объект MinIO с поддержкой асинхронности."""
    with patch("service.rag.document_processor.Minio") as mock:
        minio_instance = AsyncMock()
        mock.return_value = minio_instance
        yield minio_instance


@pytest.fixture
def mock_rag_pipeline():
    """Создаёт мок RAG-пайплайна."""
    pipeline = AsyncMock()
    pipeline.run_from_text = AsyncMock(return_value=5)  # возвращает 5 чанков
    return pipeline


@pytest.fixture
def processor(mock_minio):
    """Экземпляр DocumentProcessor с замоканным MinIO и настройками."""
    with patch("service.rag.document_processor.settings") as mock_settings:
        mock_settings.MINIO_URL = "test-minio:9000"
        mock_settings.MINIO_ACCESS_KEY = "test-key"
        mock_settings.MINIO_SECRET_KEY = "test-secret"
        mock_settings.MINIO_SECURE = False
        mock_settings.MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
        return DocumentProcessor()


# -------------------- Тесты _extract_text_from_bytes --------------------


def test_extract_text_from_bytes_pdf(processor):
    """Проверяет извлечение текста из PDF (через мок fitz)."""
    with patch("service.rag.document_processor.fitz") as mock_fitz:
        mock_doc = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1 text\n"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Page 2 text\n"
        mock_doc.__iter__.return_value = [mock_page1, mock_page2]
        mock_fitz.open.return_value.__enter__.return_value = mock_doc

        # Вызываем метод через экземпляр processor
        result = processor._extract_text_from_bytes(b"fake_pdf_data", ".pdf")
        assert result == "Page 1 text\n\nPage 2 text\n\n"
        mock_fitz.open.assert_called_once_with(stream=b"fake_pdf_data", filetype="pdf")


def test_extract_text_from_bytes_docx(processor):
    """Проверяет извлечение текста из DOCX (через мок Document)."""
    with patch("service.rag.document_processor.Document") as mock_document:
        # 1. Настраиваем фейковые параграфы для DOCX
        mock_paragraph1 = MagicMock()
        mock_paragraph1.text = "First paragraph"
        mock_paragraph2 = MagicMock()
        mock_paragraph2.text = "Second paragraph"

        mock_doc_instance = MagicMock()
        mock_doc_instance.paragraphs = [mock_paragraph1, mock_paragraph2]
        mock_document.return_value = mock_doc_instance

        # 2. Вызываем метод через экземпляр processor
        result = processor._extract_text_from_bytes(b"fake_docx_data", ".docx")

        # 3. ИСПРАВЛЕНО: Проверяем правильный текст (стыковка параграфов)
        assert result == "First paragraph\nSecond paragraph"

        # 4. Проверка передачи потока байт напрямую в конструктор Document(...)
        mock_document.assert_called_once_with(ANY)
        called_stream = mock_document.call_args[0][0]
        assert isinstance(called_stream, io.BytesIO)
        assert called_stream.getvalue() == b"fake_docx_data"


def test_extract_text_from_bytes_unsupported(processor):
    """Проверяет возврат пустой строки для неподдерживаемого формата."""
    result = processor._extract_text_from_bytes(b"data", ".txt")
    assert result == ""


def test_extract_text_from_bytes_exception(processor, caplog):
    """Проверяет обработку исключения при парсинге."""
    with patch("service.rag.document_processor.fitz.open", side_effect=Exception("PDF error")):
        result = processor._extract_text_from_bytes(b"data", ".pdf")
        assert result == ""
        assert "Ошибка парсинга" in caplog.text


# -------------------- Тесты process_file_from_minio --------------------


@pytest.mark.asyncio
async def test_process_file_success_pdf(processor, mock_minio, mock_rag_pipeline):
    """Успешная обработка PDF-файла из MinIO."""
    mock_response = AsyncMock()
    mock_response.read = AsyncMock(return_value=b"fake_pdf_content")
    mock_response.close = MagicMock()
    mock_response.release_conn = MagicMock()

    mock_minio.get_object.return_value = mock_response

    with patch.object(
        DocumentProcessor, "_extract_text_from_bytes", return_value="Extracted text from PDF"
    ) as mock_extract:
        result = await processor.process_file_from_minio(
            bucket_name="test-bucket",
            object_name="doc.pdf",
            collection_name="coll",
            tenant_id="tenant1",
            rag_pipeline=mock_rag_pipeline,
        )

    assert result == 5
    mock_minio.get_object.assert_called_once_with("test-bucket", "doc.pdf")
    mock_response.read.assert_called_once()
    mock_extract.assert_called_once_with(b"fake_pdf_content", ".pdf")
    mock_rag_pipeline.run_from_text.assert_called_once_with(
        text="Extracted text from PDF",
        source_name="doc.pdf",
        collection_name="coll",
        tenant_id="tenant1",
    )


@pytest.mark.asyncio
async def test_process_file_success_docx(processor, mock_minio, mock_rag_pipeline):
    """Успешная обработка DOCX-файла из MinIO."""
    mock_response = AsyncMock()
    mock_response.read = AsyncMock(return_value=b"fake_docx_content")
    mock_response.close = MagicMock()
    mock_response.release_conn = MagicMock()
    mock_minio.get_object.return_value = mock_response

    with patch.object(
        DocumentProcessor, "_extract_text_from_bytes", return_value="Extracted text from DOCX"
    ) as mock_extract:
        result = await processor.process_file_from_minio(
            bucket_name="test-bucket",
            object_name="doc.docx",
            collection_name="coll",
            tenant_id="tenant1",
            rag_pipeline=mock_rag_pipeline,
        )

    assert result == 5
    mock_extract.assert_called_once_with(b"fake_docx_content", ".docx")


@pytest.mark.asyncio
async def test_process_file_too_large(processor, mock_minio):
    """Файл превышает лимит размера -> возврат 0."""
    mock_response = AsyncMock()
    mock_response.read = AsyncMock(return_value=b"x" * (60 * 1024 * 1024))  # 60 MB
    mock_response.close = MagicMock()
    mock_response.release_conn = MagicMock()
    mock_minio.get_object.return_value = mock_response

    result = await processor.process_file_from_minio(
        bucket_name="bucket",
        object_name="large.pdf",
        collection_name="coll",
        tenant_id="t1",
        rag_pipeline=MagicMock(),
    )
    assert result == 0
    mock_response.read.assert_called_once()


@pytest.mark.asyncio
async def test_process_file_unsupported_extension(processor, mock_minio):
    """Неподдерживаемое расширение -> возврат 0 без вызова парсера."""
    mock_response = AsyncMock()
    mock_response.read = AsyncMock(return_value=b"some content")
    mock_response.close = MagicMock()
    mock_response.release_conn = MagicMock()
    mock_minio.get_object.return_value = mock_response

    with patch.object(DocumentProcessor, "_extract_text_from_bytes") as mock_extract:
        result = await processor.process_file_from_minio(
            bucket_name="bucket",
            object_name="file.txt",
            collection_name="coll",
            tenant_id="t1",
            rag_pipeline=MagicMock(),
        )
        assert result == 0
        mock_extract.assert_not_called()


@pytest.mark.asyncio
async def test_process_file_empty_text(processor, mock_minio, mock_rag_pipeline):
    """Извлечённый текст пуст или состоит из пробелов -> возврат 0."""
    mock_response = AsyncMock()
    mock_response.read = AsyncMock(return_value=b"content")
    mock_response.close = MagicMock()
    mock_response.release_conn = MagicMock()
    mock_minio.get_object.return_value = mock_response

    with patch.object(DocumentProcessor, "_extract_text_from_bytes", return_value="    "):
        result = await processor.process_file_from_minio(
            bucket_name="bucket",
            object_name="empty.pdf",
            collection_name="coll",
            tenant_id="t1",
            rag_pipeline=mock_rag_pipeline,
        )
    assert result == 0
    mock_rag_pipeline.run_from_text.assert_not_called()


@pytest.mark.asyncio
async def test_process_file_minio_exception(processor, mock_minio):
    """Ошибка MinIO во время скачивания -> возврат 0."""
    mock_minio.get_object.side_effect = MinioException("Connection error")

    result = await processor.process_file_from_minio(
        bucket_name="bucket",
        object_name="doc.pdf",
        collection_name="coll",
        tenant_id="t1",
        rag_pipeline=MagicMock(),
    )
    assert result == 0


@pytest.mark.asyncio
async def test_process_file_generic_exception(processor, mock_minio, caplog):
    """Непредвиденная ошибка -> возврат 0 и логирование критической ошибки."""
    mock_response = AsyncMock()
    mock_response.read = AsyncMock(side_effect=RuntimeError("Unexpected"))
    mock_response.close = MagicMock()
    mock_response.release_conn = MagicMock()
    mock_minio.get_object.return_value = mock_response

    result = await processor.process_file_from_minio(
        bucket_name="bucket",
        object_name="doc.pdf",
        collection_name="coll",
        tenant_id="t1",
        rag_pipeline=MagicMock(),
    )
    assert result == 0
    assert "Критическая ошибка" in caplog.text


# -------------------- Тесты upload_file --------------------


@pytest.mark.asyncio
async def test_upload_file_success(processor, mock_minio):
    """Успешная загрузка файла, если бакет уже создан."""
    mock_minio.bucket_exists = AsyncMock(return_value=True)
    mock_minio.fput_object = AsyncMock()
    mock_minio.make_bucket = AsyncMock()

    result = await processor.upload_file("bucket", "obj.txt", "/path/file.txt")
    assert result is True
    mock_minio.bucket_exists.assert_called_once_with("bucket")
    mock_minio.make_bucket.assert_not_called()

    # ИСПРАВЛЕНО: проверяем позиционные аргументы вместо именованных
    mock_minio.fput_object.assert_called_once_with("bucket", "obj.txt", "/path/file.txt")


@pytest.mark.asyncio
async def test_upload_file_bucket_created(processor, mock_minio):
    """Создание бакета «на лету», если он отсутствует в MinIO."""
    mock_minio.bucket_exists = AsyncMock(return_value=False)
    mock_minio.make_bucket = AsyncMock()
    mock_minio.fput_object = AsyncMock()

    result = await processor.upload_file("new-bucket", "obj.txt", "/path/file.txt")
    assert result is True
    mock_minio.make_bucket.assert_called_once_with("new-bucket")


@pytest.mark.asyncio
async def test_upload_file_minio_exception(processor, mock_minio):
    """Корректный перехват MinioException при отправке файла."""
    mock_minio.bucket_exists = AsyncMock(return_value=True)
    mock_minio.fput_object = AsyncMock(side_effect=MinioException("Upload failed"))

    result = await processor.upload_file("bucket", "obj.txt", "/path/file.txt")
    assert result is False


@pytest.mark.asyncio
async def test_upload_file_generic_exception(processor, mock_minio):
    """Перехват непредвиденных исключений (RuntimeError) при проверке бакета."""
    mock_minio.bucket_exists = AsyncMock(side_effect=RuntimeError("Unexpected error"))

    result = await processor.upload_file("bucket", "obj.txt", "/path/file.txt")
    assert result is False