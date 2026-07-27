#  ____   ___  _   _    _
# / ___| / _ \| | | |  / \
# \___ \| | | | | | | / _ \
#  ___) | |_| |\ V / / ___ \
# |____/ \___/  \_/ /_/   \_\
#
#    developer alakei1

from typing import Any, cast

from qdrant_client import QdrantClient, models

from service.utils.config import settings
from service.utils.log import logger


class VectorStore:
    """
    Клиент для работы с векторным хранилищем Qdrant.

    Предоставляет методы для создания коллекций, вставки/обновления точек,
    поиска с фильтрацией, удаления и получения точек по ID.
    """

    def __init__(self) -> None:
        """
        Инициализирует подключение к Qdrant.

        Параметры подключения берутся из глобального конфига (settings):
        - QDRANT_HOST
        - QDRANT_PORT
        - QDRANT_GPC_PORT (используется как grpc_port)
        - QDRANT_PREFER_GRPC
        """
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            grpc_port=settings.QDRANT_GRPC_PORT,
            prefer_grpc=settings.QDRANT_PREFER_GRPC,
        )

    def create_collection(self, collection_name: str, vector_size: int, distance: str = "Cosine") -> None:
        """
        Создаёт коллекцию, если она ещё не существует.
        Включает On-Disk хранение векторов и payload для предотвращения OOM.
        """
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=getattr(models.Distance, distance.upper()),
                    # НАСТРОЙКА 1: Хранить сами векторы на диске, а не в RAM
                    on_disk=True,
                ),
                # НАСТРОЙКА 2: Хранить метаданные (payload) на диске
                hnsw_config=models.HnswConfigDiff(on_disk=True),
            )
            # Создаем индекс для поля tenant_id, чтобы фильтрация была быстрой
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="tenant_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.debug(f"Коллекция {collection_name} создана с On-Disk индексами")
        else:
            logger.debug(f"Коллекция {collection_name} уже существует")

    def upsert_points(self, collection_name: str, points: list[dict[str, Any]]) -> None:
        qdrant_points = [
            models.PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p.get("payload", {}),
            )
            for p in points
        ]

        # Изменяем wait=True на wait=False
        # Qdrant примет батч в память/буфер и проиндексирует его в фоне,
        # что сгладит пиковые нагрузки на RAM и CPU вашего ПК
        self.client.upsert(collection_name=collection_name, points=qdrant_points, wait=False)
        logger.debug(f"Добавлено/Обновлено {len(points)} точек в {collection_name}")

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        filter_conditions: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[models.ScoredPoint]:
        """
        Выполняет поиск ближайших соседей по вектору с возможной фильтрацией.

        Аргументы:
            collection_name (str): Имя коллекции.
            query_vector (list[float]): Вектор запроса.
            limit (int): Максимальное количество возвращаемых результатов. По умолчанию 10.
            filter_conditions (dict | None): Условия фильтрации в формате Qdrant Filter.
                Например: {"must": [{"key": "tenant_id", "match": {"value": "123"}}]}.
            score_threshold (float | None): Минимальный порог оценки (score) для включения в результат.

        Возвращает:
            list[models.ScoredPoint]: Список найденных точек с полями id, vector, payload, score.
        """
        qdrant_filter = None
        if filter_conditions:
            qdrant_filter = models.Filter(**filter_conditions)

        response = self.client.query_points(
            collection_name=collection_name,
            query=cast(Any, query_vector),
            limit=limit,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        results = response.points
        return results

    def delete_points(self, collection_name: str, point_ids: list[str]) -> None:
        """
        Удаляет точки из коллекции по списку идентификаторов.

        Аргументы:
            collection_name (str): Имя коллекции.
            point_ids (list[str]): Список ID точек для удаления.
        """
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=point_ids),
        )
        logger.info(f"Удалено {len(point_ids)} точек из {collection_name}")

    def get_point(self, collection_name: str, point_id: str) -> dict | None:
        """
        Получает одну точку по её ID вместе с payload.

        Аргументы:
            collection_name (str): Имя коллекции.
            point_id (str): Идентификатор точки.

        Возвращает:
            dict | None: Словарь с данными точки (поля id, vector, payload) или None, если точка не найдена.
        """
        results = self.client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_payload=True,
        )
        return results[0].model_dump() if results else None
