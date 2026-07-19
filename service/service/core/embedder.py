import hashlib
import json

import redis
from sentence_transformers import SentenceTransformer

from service.utils.config import settings
from service.utils.log import logger


class Embedder:
    """
    Генерирует векторные представления текста с кэшированием в Redis.

    Использует модели Sentence Transformers (sbert). Результаты кэшируются
    по хешу текста, что ускоряет повторные запросы при индексации или поиске.

    Особенности:
        - Автоматически определяет размерность вектора модели.
        - Кэш хранится в Redis с настраиваемым TTL.
        - При недоступности Redis кэш отключается, работа продолжается.
        - Поддерживает как одиночные тексты, так и пакетную обработку.

    Примеры использования:
        # Базовое использование
        embedder = Embedder()
        vector = embedder.embed("Привет, мир!")          # список float
        print(len(vector))  # 768 (размерность модели LaBSE)

        # Использование с другой моделью
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        vectors = embedder.embed(["текст1", "текст2"])   # список списков
        print(len(vectors))  # 2
        print(len(vectors[0]))  # 384 (размерность MiniLM)

        # Отключение кэширования
        embedder = Embedder(use_cache=False)
        vector = embedder.embed("Без кэша")

    Атрибуты:
        model_name (str): Имя загруженной модели.
        model (SentenceTransformer): Экземпляр модели.
        vector_size (int): Размерность выходных векторов.
        use_cache (bool): Флаг использования кэша.
        cache_ttl (int): Время жизни кэша в секундах.
        redis_client (redis.Redis | None): Клиент Redis или None.
    """
    def __init__(self,
                 model_name : str | None = None,
                 use_cache : bool = True,
                 cache_ttl : int = 3600):
        """
        Инициализирует модель эмбеддинга и (опционально) подключение к Redis.

        Аргументы:
            model_name (str | None): Имя модели из каталога Sentence Transformers.
                Если не указано, берётся из настроек `settings.EMBEDDER_MODEL`.
                По умолчанию: 'sergeyzh/LaBSE-ru-sts' (оптимизирована для русского языка).
            user_cache (bool): Включать ли кэширование в Redis. По умолчанию True.
            cache_ttl (int): Время жизни кэша в секундах. По умолчанию 3600 (1 час).

        Примеры:
            # Использование настроек по умолчанию
            embedder = Embedder()

            # Без кэширования
            embedder = Embedder(use_cache=False)

            # Кэш на 2 часа
            embedder = Embedder(cache_ttl=7200)

            # С указанием конкретной модели
            embedder = Embedder(model_name="all-MiniLM-L6-v2")

        Исключения:
            Не выбрасывает исключений при проблемах с Redis – просто отключает кэш.
        """
        self.model_name = model_name or getattr(settings, "EMBEDDER_MODEL","sergeyzh/LaBSE-ru-sts")
        self.model = SentenceTransformer(self.model_name)
        self.vector_size = self.model[0].auto_model.config.hidden_size
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl

        if self.use_cache:
            self.redis_client = redis.Redis(
                settings.redis_url, decode_responses=True
            )
            logger.info("Кеширование эмбеддингов  в Redis включено")
        else:
            self.redis_client = None
            logger.warning("Кеширование эмбеддингов в Redis  ОТКЛЮЧЕНО !!!!")
        logger.info(f"Загружена модель эмбеддинга: {self.model_name}"
                    f"Размерность: {self.vector_size}")


    def _make_cache_key(self, text: str) -> str:
        """
        Генерирует ключ для Redis на основе текста и имени модели.

        Формат: emb:{model_name}:{md5_hash}

        Аргументы:
            text (str): Исходный текст.

        Возвращает:
            str: Ключ для Redis.

        Пример:
            embedder = Embedder()
            key = embedder._make_cache_key("Привет")
            # Результат: 'emb:sergeyzh/LaBSE-ru-sts:8b1a9953c4611296a827abf8c47804d7'
        """
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"emb:{self.model_name}:{text_hash}"

    def _embed_single(self,text: str) -> list[float] | None:
        """
        Внутренний метод для одного текста с проверкой кэша.

        Аргументы:
            text (str): Текст для эмбеддинга.

        Возвращает:
            list[float] | None: Вектор или None, если текст пуст.

        Пример:
            embedder = Embedder()
            vector = embedder._embed_single("Тестовый текст")
            # vector будет списком float размерностью модели
        """
        if not text:
            return None
        cache_key = self._make_cache_key(text)
        if self.use_cache and self.redis_client:
            cached = self.redis_client.get(cache_key)
            if cached:
                logger.debug(f"Эмбеддинг взят из кеша дл текста длинной {len(text)}")
                return  json.loads(cached)
        #вычисляем векторы
        vector = self.model.encode(text, convert_to_numpy = True).tolist()

        if self.use_cache and self.redis_client:
            self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(vector))
        return vector

    def _embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        """
        Внутренний метод для пакетной обработки с кэшированием каждого текста.

        Аргументы:
            texts (list[str]): Список текстов.

        Возвращает:
            list[list[float]] | None: Список векторов или None, если список пуст.

        Пример:
            embedder = Embedder()
            vectors = embedder._embed_batch(["текст1", "текст2", "текст3"])
            # vectors будет списком из 3 векторов
            # каждый вектор - список float размерностью модели
        """
        if not texts:
            return None

        # Инициализируем переменные один раз на самом верху
        results: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        has_cache = self.use_cache and self.redis_client is not None

        if has_cache and self.redis_client:
            for i, t in enumerate(texts):
                cache_key = self._make_cache_key(t)
                cached = self.redis_client.get(cache_key)
                if cached:
                    results[i] = json.loads(cached)
                else:
                    missing_indices.append(i)
                    missing_texts.append(t)  # ИСПРАВЛЕНО
        else:
            # Если кэша нет, то все индексы и тексты отправляются на кодирование
            missing_indices = list(range(len(texts)))
            missing_texts = texts.copy()

        if missing_texts:
            vectors = self.model.encode(missing_texts, convert_to_numpy=True).tolist()  # ИСПРАВЛЕНО
            if len(missing_indices) != len(vectors):
                min_len = min(len(missing_indices), len(vectors))
                missing_indices = missing_indices[:min_len]
                vectors = vectors[:min_len]

            for idx, vec in zip(missing_indices, vectors,strict = False):
                if has_cache and self.redis_client:
                    self.redis_client.setex(self._make_cache_key(texts[idx]), self.cache_ttl, json.dumps(vec))
                results[idx] = vec

        return [v for v in results if v is not None]

    def embed(self, text: str | list[str]) -> list[float] | list[list[float]] | None:
        """
        Преобразует текст(ы) в вектор(ы) с использованием кэша (если включён).

        Аргументы:
            text (str | list[str]): Один текст или список текстов для эмбеддинга.

        Возвращает:
            list[float] – для одного текста.
            list[list[float]] – для списка текстов.
            None – если входной текст пуст или произошла ошибка.

        Примеры:
            # Одиночный текст
            embedder = Embedder()
            vector = embedder.embed("Привет, мир!")
            # vector: список float (например, размерность 768)

            # Список текстов
            vectors = embedder.embed(["текст1", "текст2"])
            # vectors: список из 2 векторов

            # Пустой текст
            result = embedder.embed("")
            # result: None

            # Без кэширования
            embedder_no_cache = Embedder(use_cache=False)
            vector = embedder_no_cache.embed("Без кэша")

        Примечание:
            При пустом тексте возвращает None и логирует предупреждение.
        """
        if isinstance(text,str):
            results : list[float] | None = self._embed_single(text)
        else:
            results = self._embed_batch(text)
        if results is None:
            logger.error(f"Ошибка!!! Параметр text =  {text}")
            return None
        return results
