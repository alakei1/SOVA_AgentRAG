service/                                   # Корень проекта
│
├── app/                                   # Основной код приложения
│   ├── __init__.py                        # Маркер пакета
│   ├── main.py                            # Точка входа FastAPI
│   │
│   ├── api/                               # API слой (эндпоинты)
│   │   ├── __init__.py
│   │   ├── routes.py                      # Все эндпоинты
│   │   └── dependencies.py                # Зависимости (авторизация)
│   │
│   ├── core/                              # Основная бизнес-логика
│   │   ├── __init__.py
│   │   ├── auth.py                        # JWT авторизация
│   │   ├── embedder.py                    # Эмбеддинги (LaBSE-ru-sts)
│   │   ├── vector_store.py                # Qdrant клиент
│   │   ├── llm_client.py                  # DeepSeek API
│   │   ├── chunker.py                     # Разбиение текста на чанки
│   │   └── document_processor.py          # Обработка документов (PDF, DOCX)
│   │
│   ├── models/                            # Pydantic модели
│   │   ├── __init__.py
│   │   ├── schemas.py                     # Схемы запросов/ответов
│   │   └── db_models.py                   # Модели для БД (если нужны)
│   │
│   ├── services/                          # Сервисный слой
│   │   ├── __init__.py
│   │   ├── indexing.py                    # Индексация документов
│   │   ├── query.py                       # Обработка вопросов
│   │   └── tasks.py                       # Фоновые задачи (RQ)
│   │
│   └── utils/                             # Утилиты
│       ├── __init__.py
│       ├── config.py                      # Конфигурация (.env)
│       ├── logger.py                      # Настройка логирования
│       └── validators.py                  # Валидаторы
│
├── docker/                                # Docker файлы
│   ├── Dockerfile                         # Образ для AI сервиса
│   ├── docker-compose.yml                 # Все сервисы (Qdrant, Redis, MinIO)
│   └── .dockerignore
│
├── data/                                  # Локальные данные (не в Git)
│   ├── qdrant_storage/                    # Данные Qdrant
│   ├── minio_data/                        # Данные MinIO
│   ├── pg_data/                           # Данные PostgreSQL
│   └── redis_data/                        # Данные Redis
│
├── tests/                                 # Тесты
│   ├── __init__.py
│   ├── test_api.py                        # Тесты API
│   ├── test_indexing.py                   # Тесты индексации
│   ├── test_embedder.py                   # Тесты эмбеддингов
│   └── conftest.py                        # Фикстуры pytest
│
├── scripts/                               # Вспомогательные скрипты
│   ├── download_models.py                 # Скачивание моделей
│   └── seed_test_data.py                  # Загрузка тестовых данных
│
├── .venv/                                 # Виртуальное окружение (создается uv)
│
├── pyproject.toml                         # Зависимости и настройки проекта
├── uv.lock                                # Зафиксированные версии (создается uv)
├── .env                                   # Переменные окружения (НЕ в Git!)
├── .gitignore                             # Что не коммитить

