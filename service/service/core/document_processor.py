import asyncio
import io
import os
from typing import Any

import fitz
from docx import Document
from miniopy_async import Minio
from miniopy_async.error import MinioException

from service.utils.config import settings
from service.utils.log import logger


class DocumentProcessor:
    def __init__(self):
        self.minio = Minio(
            endpoint=settings.MINIO_URL,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        logger.info("DocumentProcessor: инициализирован асинхронный клиент MinIO (miniopy_async)")

    @staticmethod
    def _extract_text_from_bytes(file_data: bytes, file_extension: str) -> str:
        """
        Извлекает текст из байтов (синхронная CPU-операция).
        """
        try:
            if file_extension == ".pdf":
                text = ""
                with fitz.open(stream=file_data, filetype="pdf") as doc:
                    for page in doc:
                        text += page.get_text() + "\n"
                return text
            elif file_extension == ".docx":
                doc_stream = io.BytesIO(file_data)
                doc = Document(doc_stream)
                return "\n".join([p.text for p in doc.paragraphs])
            else:
                logger.warning(f"Неподдерживаемый формат {file_extension}")
                return ""
        except Exception as e:
            logger.error(f"Ошибка парсинга {file_extension}: {e}")
            return ""

    async def process_file_from_minio(
        self,
        bucket_name: str,
        object_name: str,
        collection_name: str,
        tenant_id: str,
        rag_pipeline: Any,
    ) -> int:
        """
        Скачивает файл из MinIO, извлекает текст, передаёт в RAG-пайплайн.

        Возвращает количество созданных чанков или 0 в случае ошибки.
        """
        response = None
        try:
            logger.info(f"Обработка файла {object_name} из {bucket_name}")

            # Получаем объект
            response = await self.minio.get_object(bucket_name, object_name)

            # Ограничение размера (по умолчанию 50 МБ)
            max_size = getattr(settings, "MAX_FILE_SIZE_BYTES", 50 * 1024 * 1024)
            file_data = await response.read()
            if len(file_data) > max_size:
                logger.error(f"Файл {object_name} превышает лимит {max_size} байт")
                return 0

            # Проверка расширения
            _, ext = os.path.splitext(object_name.lower())
            if ext not in (".pdf", ".docx"):
                logger.warning(f"Недопустимое расширение {ext} для файла {object_name}")
                return 0

            # Извлечение текста (CPU‑интенсивная операция – выносим в поток)
            raw_text = await asyncio.to_thread(self._extract_text_from_bytes, file_data, ext)

            if not raw_text.strip():
                logger.warning(f"Файл {object_name} не содержит текста")
                return 0

            # Передача в пайплайн
            chunks_count = await rag_pipeline.run_from_text(
                text=raw_text,
                source_name=object_name,
                collection_name=collection_name,
                tenant_id=tenant_id,
            )
            logger.info(f"Файл {object_name} обработан, создано чанков: {chunks_count}")
            return chunks_count

        except MinioException as e:
            logger.error(f"Ошибка MinIO при обработке {object_name}: {e}")
            return 0
        except Exception as e:
            logger.error(f"Критическая ошибка при обработке {object_name}: {e}", exc_info=True)
            return 0
        finally:
            if response:
                response.close()

    async def upload_file(self, bucket_name: str, object_name: str, file_path: str) -> bool:
        """
        Загружает локальный файл в MinIO.
        Возвращает True при успехе, False при ошибке.
        """
        try:
            if not await self.minio.bucket_exists(bucket_name):
                await self.minio.make_bucket(bucket_name)
                logger.info(f"Создан бакет {bucket_name}")

            await self.minio.fput_object(bucket_name, object_name, file_path)
            logger.info(f"Файл {file_path} загружен как {object_name}")
            return True
        except MinioException as e:
            logger.error(f"Ошибка загрузки файла {file_path} в MinIO: {e}")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка при загрузке {file_path}: {e}", exc_info=True)
            return False
