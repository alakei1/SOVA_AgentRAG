"""
Модуль с unit-тестами для VectorStore.

Тесты используют моки (unittest.mock.patch) для изоляции от реального Qdrant.
Проверяется корректность инициализации, создания коллекций, вставки/обновления точек,
поиска с фильтрацией, удаления и получения точек по ID.
"""

from unittest.mock import patch

import pytest
from qdrant_client import models

from service.core.vector_store import VectorStore


@pytest.fixture
def mock_client():
    """
    Фикстура для подмены QdrantClient и настроек во всех тестах (кроме test_init).

    Мокает:
    - QdrantClient в модуле service.core.vector_store,
    - объект settings с фиктивными значениями для всех переменных QDRANT_*.

    Возвращает класс-мок QdrantClient, через который можно управлять поведением
    экземпляра клиента (mock_client.return_value).
    """
    with patch("service.core.vector_store.QdrantClient") as MockClient:
        with patch("service.core.vector_store.settings") as mock_settings:
            mock_settings.QDRANT_HOST = "localhost"
            mock_settings.QDRANT_PORT = 6333
            mock_settings.QDRANT_GRPC_PORT = 6334
            mock_settings.QDRANT_PREFER_GRPC = False
            yield MockClient


def test_init():
    """
    Проверяет, что конструктор VectorStore вызывает QdrantClient с правильными параметрами.

    Использует собственные патчи, чтобы гарантировать изоляцию.
    Проверяет, что QdrantClient был вызван ровно один раз с ожидаемыми аргументами.
    """
    with patch("service.core.vector_store.QdrantClient") as MockClient:
        with patch("service.core.vector_store.settings") as mock_settings:
            mock_settings.QDRANT_HOST = "localhost"
            mock_settings.QDRANT_PORT = 6333
            mock_settings.QDRANT_GRPC_PORT = 6334
            mock_settings.QDRANT_PREFER_GRPC = False

            store = VectorStore()  # noqa: F841
            MockClient.assert_called_once_with(
                host="localhost",
                port=6333,
                grpc_port=6334,
                prefer_grpc=False,
            )


def test_create_collection_new(mock_client):
    """
    Проверяет создание новой коллекции, когда она ещё не существует.

    Сценарий:
    - collection_exists возвращает False.
    - Вызывается create_collection с правильными параметрами размера и метрики.
    - Создаётся индекс по полю tenant_id.
    """
    mock_client.return_value.collection_exists.return_value = False
    store = VectorStore()
    store.create_collection("test", 128, "Cosine")
    mock_client.return_value.create_collection.assert_called_once_with(
        collection_name="test",
        vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE),
    )
    mock_client.return_value.create_payload_index.assert_called_once_with(
        collection_name="test",
        field_name="tenant_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def test_create_collection_exists(mock_client):
    """
    Проверяет поведение, когда коллекция уже существует.

    Ожидается, что create_collection и create_payload_index не вызываются.
    """
    mock_client.return_value.collection_exists.return_value = True
    store = VectorStore()
    store.create_collection("test", 128)
    mock_client.return_value.create_collection.assert_not_called()
    mock_client.return_value.create_payload_index.assert_not_called()


def test_upsert_points(mock_client):
    """
    Проверяет вставку/обновление точек.

    Убеждается, что точки преобразуются в PointStruct с правильными полями:
    - id, vector, payload (если передан, иначе пустой словарь).
    - Вызов upsert происходит с wait=True и правильным именем коллекции.
    """
    store = VectorStore()
    points = [
        {"id": "1", "vector": [0.1, 0.2], "payload": {"tenant": "a"}},
        {"id": "2", "vector": [0.3, 0.4]},
    ]
    store.upsert_points("test_col", points)
    mock_client.return_value.upsert.assert_called_once()
    args, kwargs = mock_client.return_value.upsert.call_args
    assert kwargs["collection_name"] == "test_col"
    assert kwargs["wait"] is True
    assert len(kwargs["points"]) == 2
    assert kwargs["points"][0].id == "1"
    assert kwargs["points"][0].vector == [0.1, 0.2]
    assert kwargs["points"][0].payload == {"tenant": "a"}
    assert kwargs["points"][1].id == "2"
    assert kwargs["points"][1].payload == {}


def test_search(mock_client):
    """
    Проверяет поиск по вектору с фильтрацией и порогом оценки.

    Мокается ответ query_points, содержащий один ScoredPoint.
    Проверяется, что:
    - Фильтр преобразуется в объект models.Filter.
    - Передаются все параметры (collection_name, query, limit, score_threshold, with_payload).
    - Результат содержит ожидаемую точку.
    """
    mock_response = type("MockResponse", (), {"points": []})()
    mock_response.points = [
        models.ScoredPoint(id="1", version=0, score=0.9, vector=[0.1], payload={})
    ]
    mock_client.return_value.query_points.return_value = mock_response

    store = VectorStore()
    filter_cond = {"must": [{"key": "tenant", "match": {"value": "x"}}]}
    results = store.search(
        "col", [0.1, 0.2], limit=5, filter_conditions=filter_cond, score_threshold=0.8
    )

    mock_client.return_value.query_points.assert_called_once_with(
        collection_name="col",
        query=[0.1, 0.2],
        limit=5,
        score_threshold=0.8,
        query_filter=models.Filter(**filter_cond),
        with_payload=True,
    )
    assert len(results) == 1
    assert results[0].id == "1"


def test_delete_points(mock_client):
    """
    Проверяет удаление точек по списку ID.

    Убеждается, что delete вызывается с правильным именем коллекции
    и объектом PointIdsList, содержащим переданные ID.
    """
    store = VectorStore()
    store.delete_points("col", ["id1", "id2"])
    mock_client.return_value.delete.assert_called_once_with(
        collection_name="col",
        points_selector=models.PointIdsList(points=["id1", "id2"]),
    )


def test_get_point_found(mock_client):
    """
    Проверяет получение существующей точки по ID.

    Мокается retrieve, возвращающий список с одним Record.
    Проверяется, что результат преобразован в словарь с ключами id, vector, payload
    (служебные поля игнорируются).
    """
    mock_record = models.Record(
        id="1",
        vector=[0.1],
        payload={"x": 1},
        version=0,
        order_value=None,
        shard_key=None,
    )
    mock_client.return_value.retrieve.return_value = [mock_record]

    store = VectorStore()
    result = store.get_point("col", "1")
    mock_client.return_value.retrieve.assert_called_once_with(
        collection_name="col", ids=["1"], with_payload=True
    )
    assert result["id"] == "1"
    assert result["vector"] == [0.1]
    assert result["payload"] == {"x": 1}


def test_get_point_not_found(mock_client):
    """
    Проверяет, что get_point возвращает None, если точка не найдена.

    Мокается retrieve, возвращающий пустой список.
    """
    mock_client.return_value.retrieve.return_value = []
    store = VectorStore()
    result = store.get_point("col", "1")
    assert result is None
