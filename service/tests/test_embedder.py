"""
Модуль с unit-тестами для Embedder.

Тесты используют моки (unittest.mock.patch) для изоляции от реального Redis.
Проверяется корректность инициализации, генерации ключей кэша,
одиночного и пакетного эмбеддинга, работы с кэшем,
а также интеграционные тесты с реальным Redis.
"""

import hashlib
import json
from typing import cast
from unittest.mock import Mock, patch

import numpy as np
import pytest
import redis
from sentence_transformers import SentenceTransformer

from service.rag.embedder import Embedder
from service.utils.config import settings

# ==================== Сессионная фикстура с моделью (загружается один раз) ====================

@pytest.fixture(scope="session")
def model():
    """
    Загружает модель один раз для всех тестов.

    Это самая тяжёлая операция, поэтому она выполняется только один раз
    за всю сессию тестирования.
    """
    return SentenceTransformer("sergeyzh/LaBSE-ru-sts")


@pytest.fixture(scope="session")
def embedder_without_cache(model):
    """
    Фикстура с эмбеддером без кэша.

    Модель загружается один раз для всех тестов.
    """
    embedder = Embedder(model_name="sergeyzh/LaBSE-ru-sts", use_cache=False)
    embedder.model = model
    embedder.vector_size = model.get_embedding_dimension()
    return embedder


@pytest.fixture(scope="session")
def embedder_with_mock(model):
    """
    Фикстура с эмбеддером и замоканным методом encode (очень быстрый).

    Используется для тестов, где не нужно реальное вычисление векторов.
    """
    embedder = Embedder(model_name="sergeyzh/LaBSE-ru-sts", use_cache=False)
    embedder.model = model
    embedder.vector_size = model.get_embedding_dimension()

    def mock_encode(text, convert_to_numpy=True, *args, **kwargs):
        # Создаём чередующийся вектор для тестов с кэшем
        base_pattern = [0.1, 0.2, 0.3]
        vector = (base_pattern * (embedder.vector_size // len(base_pattern) + 1))[: embedder.vector_size]

        if isinstance(text, str):
            return np.array(vector)
        return np.array([vector for _ in range(len(text))])

    embedder.model.encode = mock_encode
    return embedder

# ==================== Обычные фикстуры ====================

@pytest.fixture
def mock_redis_client():
    """Фикстура с моком Redis клиента."""
    mock_client = Mock(spec=redis.Redis)
    mock_client.get = Mock(return_value=None)
    mock_client.setex = Mock()
    return mock_client


@pytest.fixture
def embedder_with_cache(mock_redis_client, embedder_without_cache):
    """
    Фикстура с эмбеддером и моком Redis.

    Использует уже загруженную модель из сессионной фикстуры.
    """
    with patch("service.rag.embedder.redis.Redis.from_url", return_value=mock_redis_client):
        embedder = Embedder(model_name="sergeyzh/LaBSE-ru-sts", use_cache=True)
        # Копируем уже загруженную модель
        embedder.model = embedder_without_cache.model
        embedder.vector_size = embedder_without_cache.vector_size
        embedder.redis_client = mock_redis_client
        return embedder


class TestEmbedder:
    """Тесты для класса Embedder."""

    # ==================== Тесты инициализации ====================

    def test_init_default_model(self):
        """
        Проверяет инициализацию с моделью по умолчанию.

        Ожидается, что:
        - Используется модель sergeyzh/LaBSE-ru-sts.
        - Размерность вектора равна 768.
        - Кэш включён (use_cache=True).
        - TTL кэша равен 3600 секунд.
        """
        embedder = Embedder()
        assert embedder.model_name == "sergeyzh/LaBSE-ru-sts"
        assert isinstance(embedder.model, SentenceTransformer)
        assert embedder.vector_size == 768
        assert embedder.use_cache is True
        assert embedder.cache_ttl == 3600

    def test_init_custom_model(self):
        """
        Проверяет инициализацию с кастомной моделью.

        Ожидается, что:
        - Используется указанная модель all-MiniLM-L6-v2.
        - Размерность вектора равна 384.
        """
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.vector_size == 384

    def test_init_without_cache(self):
        """
        Проверяет инициализацию с отключенным кэшем.

        Ожидается, что:
        - use_cache равен False.
        - redis_client равен None.
        """
        embedder = Embedder(use_cache=False)
        assert embedder.use_cache is False
        assert embedder.redis_client is None

    def test_init_custom_cache_ttl(self):
        """
        Проверяет инициализацию с кастомным TTL.

        Ожидается, что cache_ttl равен 7200 секунд.
        """
        embedder = Embedder(cache_ttl=7200)
        assert embedder.cache_ttl == 7200

    # ==================== Тесты _make_cache_key ====================

    def test_make_cache_key(self):
        """
        Проверяет генерацию ключа для Redis.

        Ожидается, что ключ формируется по шаблону:
        emb:{model_name}:{md5_hash}
        """
        embedder = Embedder()
        text = "Привет, мир!"
        key = embedder._make_cache_key(text)
        expected_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        expected_key = f"emb:{embedder.model_name}:{expected_hash}"
        assert key == expected_key

    def test_make_cache_key_different_models(self):
        """
        Проверяет, что ключи для разных моделей различаются.

        Ожидается, что для одной и той же строки ключи будут разными,
        если используются разные модели.
        """
        embedder1 = Embedder(model_name="sergeyzh/LaBSE-ru-sts")
        embedder2 = Embedder(model_name="all-MiniLM-L6-v2")
        text = "test"
        key1 = embedder1._make_cache_key(text)
        key2 = embedder2._make_cache_key(text)
        assert key1 != key2
        assert "sergeyzh/LaBSE-ru-sts" in key1
        assert "all-MiniLM-L6-v2" in key2

    # ==================== Тесты _embed_single ====================

    def test_embed_single_without_cache(self, embedder_with_mock):
        """
        Проверяет эмбеддинг одного текста без кэша (быстрый тест).

        Ожидается, что:
        - Вектор не None.
        - Вектор является списком float.
        - Размерность вектора соответствует модели.
        """
        vector = embedder_with_mock._embed_single("Тестовый текст")
        assert vector is not None
        assert isinstance(vector, list)
        assert len(vector) == embedder_with_mock.vector_size
        assert all(isinstance(x, float) for x in vector)

    def test_embed_single_empty_text(self, embedder_without_cache):
        """
        Проверяет эмбеддинг пустого текста.

        Ожидается, что возвращается None.
        """
        vector = embedder_without_cache._embed_single("")
        assert vector is None

    def test_embed_single_with_cache_hit(self, embedder_with_cache, mock_redis_client):
        """
        Проверяет эмбеддинг с попаданием в кэш.

        Сценарий:
        - Вектор уже есть в Redis.
        - Метод должен вернуть его без повторного вычисления.
        - Вызов get должен быть с правильным ключом.
        """
        text = "Кэшируемый текст"
        expected_vector = [0.1, 0.2, 0.3] * 256
        cache_key = embedder_with_cache._make_cache_key(text)

        mock_redis_client.get.return_value = json.dumps(expected_vector)

        vector = embedder_with_cache._embed_single(text)

        mock_redis_client.get.assert_called_once_with(cache_key)
        assert vector == expected_vector

    def test_embed_single_with_cache_miss(self, embedder_with_cache, mock_redis_client):
        """
        Проверяет эмбеддинг с промахом кэша.

        Сценарий:
        - Вектора нет в Redis.
        - Метод должен вычислить вектор и сохранить его.
        - Вызов get должен быть с правильным ключом.
        - Вызов setex должен сохранить вектор.
        """
        text = "Новый текст"
        cache_key = embedder_with_cache._make_cache_key(text)

        mock_redis_client.get.return_value = None

        vector = embedder_with_cache._embed_single(text)

        mock_redis_client.get.assert_called_once_with(cache_key)
        mock_redis_client.setex.assert_called_once()
        assert vector is not None
        assert len(vector) == embedder_with_cache.vector_size

    # ==================== Тесты _embed_batch ====================

    def test_embed_batch_without_cache(self, embedder_with_mock):
        """
        Проверяет пакетный эмбеддинг без кэша (быстрый тест).

        Ожидается, что:
        - Возвращается список векторов.
        - Количество векторов равно количеству текстов.
        - Размерность каждого вектора соответствует модели.
        """
        texts = ["текст1", "текст2", "текст3"]
        vectors = embedder_with_mock._embed_batch(texts)

        assert vectors is not None
        assert len(vectors) == len(texts)
        for v in vectors:
            assert isinstance(v, list)
            assert len(v) == embedder_with_mock.vector_size

    def test_embed_batch_empty_list(self, embedder_without_cache):
        """
        Проверяет пакетный эмбеддинг с пустым списком.

        Ожидается, что возвращается None.
        """
        vectors = embedder_without_cache._embed_batch([])
        assert vectors is None

    def test_embed_batch_with_mixed_cache(self, embedder_with_cache, mock_redis_client):
        """
        Проверяет пакетный эмбеддинг с частичным попаданием в кэш.

        Сценарий:
        - Первый текст есть в кэше.
        - Остальных текстов в кэше нет.
        - Ожидается, что первый вектор берётся из кэша,
          остальные вычисляются.
        """
        texts = ["текст1", "текст2", "текст3"]
        cached_vector = [0.1, 0.2, 0.3] * 256

        def get_side_effect(key):
            if "текст1" in key:
                return json.dumps(cached_vector)
            return None

        mock_redis_client.get.side_effect = get_side_effect

        vectors = embedder_with_cache._embed_batch(texts)

        assert vectors is not None
        assert len(vectors) == len(texts)
        assert vectors[0] == cached_vector
        for v in vectors[1:]:
            assert len(v) == embedder_with_cache.vector_size

    # ==================== Тесты основного метода embed ====================

    def test_embed_single_text(self, embedder_with_mock):
        """
        Проверяет основной метод с одним текстом (быстрый тест).

        Ожидается, что:
        - Возвращается вектор (список float).
        - Размерность вектора соответствует модели.
        """
        result = embedder_with_mock.embed("Тестовый текст")
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == embedder_with_mock.vector_size

    def test_embed_list_texts(self, embedder_with_mock):
        """
        Проверяет основной метод со списком текстов (быстрый тест).

        Ожидается, что:
        - Возвращается список векторов.
        - Количество векторов равно количеству текстов.
        - Размерность каждого вектора соответствует модели.
        """
        result = embedder_with_mock.embed(["текст1", "текст2"])
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 2
        for v in result:
            assert len(v) == embedder_with_mock.vector_size

    def test_embed_empty_text(self, embedder_without_cache):
        """
        Проверяет основной метод с пустым текстом.

        Ожидается, что возвращается None.
        """
        result = embedder_without_cache.embed("")
        assert result is None

    def test_embed_empty_list(self, embedder_without_cache):
        """
        Проверяет основной метод с пустым списком.

        Ожидается, что возвращается None.
        """
        result = embedder_without_cache.embed([])
        assert result is None

    def test_embed_with_cache_enabled(self, embedder_with_cache, mock_redis_client):
        """
        Проверяет основной метод с включённым кэшем.

        Ожидается, что:
        - Происходит обращение к Redis.
        - Вектор вычисляется и сохраняется при промахе кэша.
        """
        text = "Тестовый текст"
        mock_redis_client.get.return_value = None

        result = embedder_with_cache.embed(text)

        assert result is not None
        mock_redis_client.get.assert_called_once()
        mock_redis_client.setex.assert_called_once()

    # ==================== Интеграционные тесты ====================

    @pytest.mark.integration
    def test_integration_real_redis(self):
        """
        Интеграционный тест с реальным Redis.

        Проверяет:
        - Подключение к Redis работает.
        - Первый вызов вычисляет вектор и сохраняет в кэш.
        - Второй вызов возвращает тот же вектор из кэша.
        - Размерность вектора соответствует модели.
        """
        try:
            test_redis = redis.Redis.from_url(settings.redis_url)
            test_redis.ping()
        except redis.ConnectionError:
            pytest.skip("Redis недоступен")

        embedder = Embedder(use_cache=True)
        text = "Интеграционный тест"

        vector1 = cast(list[float], embedder.embed(text))
        vector2 = cast(list[float], embedder.embed(text))

        assert vector1 is not None
        assert vector2 is not None
        assert vector1 == vector2
        assert len(vector1) == embedder.vector_size

    # ==================== Тесты производительности ====================

    @pytest.mark.benchmark
    def test_embed_single_performance(self, benchmark, embedder_without_cache):
        """
        Тест производительности для одного текста.
        """
        def run():
            return embedder_without_cache.embed("Тестовый текст для замера производительности")

        result = benchmark(run)
        assert result is not None
        assert len(result) == embedder_without_cache.vector_size

    @pytest.mark.benchmark
    def test_embed_batch_performance(self, benchmark, embedder_without_cache):
        """
        Тест производительности для пакетной обработки.
        """
        texts = [f"текст_{i}" for i in range(10)]

        def run():
            return embedder_without_cache.embed(texts)

        result = benchmark(run)
        assert result is not None
        assert len(result) == len(texts)

    @pytest.mark.benchmark
    def test_cache_performance(self, benchmark, embedder_with_cache, mock_redis_client):
        """
        Тест производительности с кэшем.
        """
        text = "Кэшируемый текст"
        expected_vector = [0.1] * embedder_with_cache.vector_size
        mock_redis_client.get.return_value = json.dumps(expected_vector)

        def run():
            return embedder_with_cache.embed(text)

        result = benchmark(run)
        assert result == expected_vector


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
