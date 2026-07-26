#  ____   ___  _   _    _
# / ___| / _ \| | | |  / \
# \___ \| | | | | | | / _ \
#  ___) | |_| |\ V / / ___ \
# |____/ \___/  \_/ /_/   \_\
#
#    developer alakei1

import asyncio
import time
import uuid
from typing import Any

from qdrant_client.http import models

#from service.core.ai_chunker import LocalChunker
from service.core.chunker import TextChunker
from service.core.embedder import Embedder
from service.core.llm_client import LLMClient
from service.core.vector_store import VectorStore
from service.utils.config import settings
from service.utils.log import logger


class RAGPipeline:
    def __init__(self):
        #self.chunker = LocalChunker() <- Тех долг
        self.chunker = TextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=settings.CHUNK_SEPARATORS,
        )
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.llm_client = LLMClient()

    async def run_from_text(self, text: str, source_name: str, collection_name: str, tenant_id: str) -> int:
        #chunks = await self.chunker.split_text(text) <- тех-долг

        chunks = await asyncio.to_thread(self.chunker.split_text, text)

        if not chunks:
            logger.warning("No chunks generated")
            return 0

        self.vector_store.create_collection(collection_name, self.embedder.vector_size)

        vectors = self.embedder.embed(chunks)
        if not vectors:
            logger.error("Embedding failed")
            return 0

        points = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            point_id = str(uuid.uuid4())
            payload = {
                "text": chunk,
                "source": source_name,
                "tenant_id": tenant_id,
                "collection": collection_name
            }
            points.append({
                "id": point_id,
                "vector": vec,
                "payload": payload
            })

        self.vector_store.upsert_points(collection_name, points)
        logger.info(f"Indexed {len(points)} chunks from {source_name}")
        return len(points)

    async def query(self, query_text: str, collection_name: str, tenant_id: str, top_k: int = 5) -> dict[str, Any]:
        start = time.perf_counter()

        query_vector = self.embedder.embed(query_text)
        if query_vector is None:
            return {
                "answer": "Ошибка эмбеддинга запроса",
                "sources": [],
                "total_chunks": 0,
                "average_score": 0.0,
                "top_k_requested": top_k,
                "processing_time_ms": 0.0,
            }

        filter_cond = {"must": [{"key": "tenant_id", "match": {"value": tenant_id}}]}

        # 1. Проверяем структуру query_vector перед вызовом search
        if query_vector and isinstance(query_vector[0], list):
            # Если внутри лежит еще один список, значит это батч. Берем первый вектор.
            single_query_vector: list[float] = query_vector[0]
        else:
            # Если внутри лежат числа (float), значит это уже готовый одиночный вектор.
            single_query_vector: list[float] = query_vector  # type: ignore

        # 2. Передаем в метод гарантированно одномерный вектор
        results = self.vector_store.search(
            collection_name=collection_name,
            query_vector=single_query_vector,  # Теперь PyCharm полностью доволен!
            limit=top_k,
            filter_conditions=filter_cond,
            score_threshold=0.5,
        )

        context_chunks = []
        sources = []
        for scored in results:
            payload = scored.payload
            if payload and "text" in payload:
                context_chunks.append(payload["text"])
                sources.append(
                    {
                        "id": scored.id,
                        "score": scored.score,
                        "source": payload.get("source", "unknown"),
                        "text": payload.get("text", ""),
                    }
                )

        total = len(sources)
        avg_score = sum(s["score"] for s in sources) / total if total > 0 else 0.0
        context = "\n\n---\n\n".join(context_chunks) if context_chunks else "Информация не найдена."

        answer = await self.llm_client.generate(
            question=query_text,
            context=context,
            temperature=0.2,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "answer": answer,
            "sources": sources,
            "context": context,
            "total_chunks": total,
            "average_score": round(avg_score, 4),
            "top_k_requested": top_k,
            "processing_time_ms": round(elapsed_ms, 2),
        }

    async def delete_document(self, collection_name: str, tenant_id: str, file_name: str) -> int:
        """
        Удаляет все чанки, относящиеся к указанному файлу в данной коллекции и тенанте.
        Возвращает количество удалённых чанков.
        """
        point_ids = []
        offset = None
        while True:
            result = await asyncio.to_thread(
                self.vector_store.client.scroll,
                collection_name=collection_name,
                scroll_filter=models.Filter(  # <-- ИСПРАВЛЕНО
                    must=[
                        models.FieldCondition(key="collection", match=models.MatchValue(value=collection_name)),
                        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
                        models.FieldCondition(key="source", match=models.MatchValue(value=file_name)),
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            points_batch = result[0]
            if not points_batch:
                break
            point_ids.extend([p.id for p in points_batch])
            offset = result[1]
            if offset is None:
                break

        if point_ids:
            self.vector_store.delete_points(collection_name, point_ids)

        logger.info(f"Удалено {len(point_ids)} чанков для документа {file_name} (коллекция {collection_name})")
        return len(point_ids)
